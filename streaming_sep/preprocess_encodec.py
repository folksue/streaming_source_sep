from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torchaudio
import yaml
from tqdm import tqdm

from streaming_sep.codec import CausalEncodecTokenizer
from streaming_sep.data import SepItem, load_manifest, save_cache


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.mean(dim=0)


def resolve_path(raw_path: str, manifest_path: str | Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((Path(manifest_path).parent / path).resolve())


def build_cache(cfg: dict[str, Any], manifest_path: str, out_path: str, device: str) -> None:
    audio_cfg = cfg["audio"]
    sample_rate = int(audio_cfg["sample_rate"])
    tokenizer = CausalEncodecTokenizer(
        model_name=str(audio_cfg.get("encodec_model_name", "facebook/encodec_24khz")),
        bandwidth=audio_cfg.get("encodec_bandwidth", 6.0),
        device=device,
    )
    if tokenizer.sample_rate != sample_rate:
        raise ValueError(f"Config sample_rate={sample_rate} but EnCodec expects {tokenizer.sample_rate}")

    entries = load_manifest(manifest_path)
    items: list[SepItem] = []
    for entry in tqdm(entries, desc=f"encodec {Path(manifest_path).name}"):
        utt_id = str(entry.get("utt_id", len(items)))
        mixture_path = resolve_path(str(entry["mixture_path"]), manifest_path)
        source_path = resolve_path(str(entry["source_path"]), manifest_path)
        mixture = tokenizer.encode(load_audio(mixture_path, sample_rate), sample_rate).codes
        source = tokenizer.encode(load_audio(source_path, sample_rate), sample_rate).codes

        prompt_codes = None
        if entry.get("audio_prompt_path"):
            prompt_path = resolve_path(str(entry["audio_prompt_path"]), manifest_path)
            prompt_codes = tokenizer.encode(load_audio(prompt_path, sample_rate), sample_rate).codes

        length = min(mixture.size(0), source.size(0))
        if length <= 0:
            continue
        items.append(
            SepItem(
                utt_id=utt_id,
                mixture_codes=mixture[:length],
                source_codes=source[:length],
                audio_prompt_codes=prompt_codes,
            )
        )

    if not items:
        raise RuntimeError(f"No usable examples in manifest: {manifest_path}")
    save_cache(
        out_path,
        items,
        meta={
            "sample_rate": sample_rate,
            "encodec_model_name": audio_cfg.get("encodec_model_name", "facebook/encodec_24khz"),
            "encodec_bandwidth": audio_cfg.get("encodec_bandwidth", 6.0),
            "codebook_size": int(audio_cfg.get("codebook_size", 1024)),
            "num_codebooks": int(items[0].mixture_codes.size(1)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/encodec_sep_200m.yaml")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.split in {"train", "both"}:
        build_cache(cfg, cfg["data"]["train_manifest"], cfg["data"]["train_cache"], args.device)
    if args.split in {"valid", "both"}:
        build_cache(cfg, cfg["data"]["valid_manifest"], cfg["data"]["valid_cache"], args.device)


if __name__ == "__main__":
    main()

