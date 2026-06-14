from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from streaming_sep.data import EncodecSepDataset, collate_sep
from streaming_sep.model import DualStreamEncodecSeparator, separation_loss


def build_model(cfg: dict) -> DualStreamEncodecSeparator:
    audio = cfg["audio"]
    model = cfg["model"]
    return DualStreamEncodecSeparator(
        codebook_size=int(audio.get("codebook_size", 1024)),
        num_codebooks=int(audio.get("num_codebooks", 8)),
        d_model=int(model["d_model"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        ffn_dim=int(model["ffn_dim"]),
        dropout=float(model.get("dropout", 0.0)),
        sliding_window=model.get("sliding_window", None),
        prompt_dim=int(model.get("prompt_dim", model["d_model"])),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


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
                audio_prompt_codes=batch.get("audio_prompt_codes"),
                audio_prompt_mask=batch.get("audio_prompt_mask"),
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
            audio_prompt_codes=batch.get("audio_prompt_codes"),
            audio_prompt_mask=batch.get("audio_prompt_mask"),
        )
        _, metrics = separation_loss(output, batch["source_codes"], batch["mask"], residual_weight)
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/encodec_sep_200m.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EncodecSepDataset(cfg["data"]["train_cache"], max_frames=cfg["data"].get("max_frames"))
    valid_ds = EncodecSepDataset(cfg["data"]["valid_cache"], max_frames=cfg["data"].get("max_frames"))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate_sep,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate_sep,
    )

    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model_params={param_count / 1e6:.1f}M device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and bool(cfg["train"].get("amp", True)))
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
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
            },
            ckpt_dir / f"epoch_{epoch:04d}.pt",
        )


if __name__ == "__main__":
    main()

