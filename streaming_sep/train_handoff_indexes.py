from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

from streaming_sep.data import StreamingSepHandoffDataset, collate_sep
from streaming_sep.train import build_model, move_batch
from streaming_sep.model import separation_loss


DEFAULT_DATA_ROOT = "/cfs4/folkswei/test/ft_local/streaming_sep_offload"
DEFAULT_STEMS = ["vocal", "bass", "drums", "other"]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_index_paths(data_root: str | Path) -> dict[str, Path]:
    root = Path(data_root)
    other_main = root / "encodec_scnet_other_pairs_gpu_prefetch" / "index.json"
    other_retry = root / "encodec_scnet_other_pairs_gpu_prefetch_retry_segmented" / "index.json"
    return {
        "vocal": first_existing(
            root / "encodec_vocal_pairs_gpu_prefetch" / "index-mnt.json",
            root / "encodec_vocal_pairs_gpu_prefetch" / "index.json",
            root / "encodec_pairs_gpu_prefetch" / "index-mnt.json",
            root / "encodec_pairs_gpu_prefetch" / "index.json",
        ),
        "bass": first_existing(
            root / "encodec_bass_pairs_gpu_prefetch_clean" / "index-mnt.json",
            root / "encodec_bass_pairs_gpu_prefetch_clean" / "index.json",
        ),
        "drums": first_existing(
            root / "encodec_drums_pairs_gpu_prefetch_clean" / "index.json",
            root / "encodec_drums_pairs_gpu_prefetch_clean" / "index-mnt.json",
        ),
        "other": other_retry if other_retry.exists() and not other_main.exists() else other_main,
    }


def overlay_other_retry(index_paths: dict[str, Path], data_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    retry = Path(data_root) / "encodec_scnet_other_pairs_gpu_prefetch_retry_segmented" / "index.json"
    if not retry.exists() or not index_paths["other"].exists():
        return index_paths

    with open(index_paths["other"], "r", encoding="utf-8") as f:
        main_payload = json.load(f)
    with open(retry, "r", encoding="utf-8") as f:
        retry_payload = json.load(f)

    def items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        for key in ("items", "records", "encoded_pairs"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValueError("Cannot find item list while overlaying other retry index")

    merged = {str(item["relative_path"]): item for item in items(main_payload)}
    merged.update({str(item["relative_path"]): item for item in items(retry_payload)})
    out = Path(out_dir) / "encodec_scnet_other_pairs_gpu_prefetch_merged_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"items": list(merged.values())}, ensure_ascii=False), encoding="utf-8")
    index_paths["other"] = out
    return index_paths


def split_keys(keys: list[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    valid_count = max(1, int(round(len(keys) * valid_ratio))) if len(keys) > 1 else 0
    return keys[valid_count:], keys[:valid_count]


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_sep,
    )


def train_epoch(model, loader, optimizer, scaler, cfg, device, step_offset: int) -> int:
    model.train()
    residual_weight = float(cfg["model"].get("residual_loss_weight", 0.5))
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    log_interval = int(cfg["train"].get("log_interval", 20))
    step = step_offset
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(
                mixture_codes=batch["mixture_codes"],
                source_codes=batch["source_codes"],
                mask=batch["mask"],
                stem_id=batch["stem_id"],
            )
            loss, metrics = separation_loss(output, batch["source_codes"], batch["mask"], residual_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % log_interval == 0:
            print(
                f"step={step} loss={metrics['loss']:.4f} coarse={metrics['coarse_loss']:.4f} "
                f"res={metrics['residual_loss']:.4f} coarse_acc={metrics['coarse_acc']:.4f}"
            )
    return step


@torch.no_grad()
def validate(model, loader, cfg, device) -> dict[str, float]:
    model.eval()
    residual_weight = float(cfg["model"].get("residual_loss_weight", 0.5))
    totals = {"loss": 0.0, "coarse_loss": 0.0, "residual_loss": 0.0, "coarse_acc": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            mixture_codes=batch["mixture_codes"],
            source_codes=batch["source_codes"],
            mask=batch["mask"],
            stem_id=batch["stem_id"],
        )
        _, metrics = separation_loss(output, batch["source_codes"], batch["mask"], residual_weight)
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/encodec_sep_200m.yaml")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--stems", nargs="+", default=DEFAULT_STEMS)
    parser.add_argument("--residual-predictor", choices=["mtp", "parallel"], default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.residual_predictor is not None:
        cfg["model"]["residual_predictor"] = args.residual_predictor
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_paths = overlay_other_retry(default_index_paths(args.data_root), args.data_root, cfg["paths"]["checkpoints"])
    missing = {stem: str(path) for stem, path in index_paths.items() if stem in args.stems and not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing index files: {missing}")

    probe = StreamingSepHandoffDataset(
        index_paths=index_paths,
        data_root=args.data_root,
        stems=args.stems,
        max_frames=cfg["data"].get("max_frames"),
        crop_frames=cfg["data"].get("crop_frames", cfg["data"].get("max_frames")),
        random_crop=False,
    )
    train_keys, valid_keys = split_keys(probe.keys, args.valid_ratio, seed)
    train_ds = StreamingSepHandoffDataset(
        index_paths=index_paths,
        data_root=args.data_root,
        keys=train_keys,
        stems=args.stems,
        max_frames=cfg["data"].get("max_frames"),
        crop_frames=cfg["data"].get("crop_frames", cfg["data"].get("max_frames")),
        random_crop=True,
    )
    valid_ds = StreamingSepHandoffDataset(
        index_paths=index_paths,
        data_root=args.data_root,
        keys=valid_keys,
        stems=args.stems,
        max_frames=cfg["data"].get("max_frames"),
        crop_frames=cfg["data"].get("crop_frames", cfg["data"].get("max_frames")),
        random_crop=False,
    )
    print(f"tracks train={len(train_keys)} valid={len(valid_keys)} rows train={len(train_ds)} valid={len(valid_ds)}")

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    valid_loader = make_loader(
        valid_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(cfg["train"].get("amp", True)))
    start_epoch = 1
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        step = int(ckpt.get("step", 0))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"model_params={param_count / 1e6:.1f}M device={device}")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        step = train_epoch(model, train_loader, optimizer, scaler, cfg, device, step)
        if epoch % int(cfg["train"].get("valid_interval", 1)) == 0:
            metrics = validate(model, valid_loader, cfg, device)
            print(
                f"epoch={epoch} valid_loss={metrics['loss']:.4f} "
                f"valid_coarse_acc={metrics['coarse_acc']:.4f}"
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "step": step,
                "param_count": param_count,
                "index_paths": {stem: str(path) for stem, path in index_paths.items()},
                "stems": args.stems,
            },
            ckpt_dir / f"handoff_epoch_{epoch:04d}.pt",
        )


if __name__ == "__main__":
    main()
