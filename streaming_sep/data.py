from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


CACHE_FORMAT = "encodec_streaming_source_sep"
CACHE_VERSION = 1


@dataclass
class SepItem:
    utt_id: str
    mixture_codes: torch.Tensor  # [T, K]
    source_codes: torch.Tensor  # [T, K]
    audio_prompt_codes: torch.Tensor | None = None  # [P, K]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    return data


def save_cache(path: str | Path, items: list[SepItem], meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "cache_format": CACHE_FORMAT,
            "cache_version": CACHE_VERSION,
            **meta,
        },
        "items": [
            {
                "utt_id": item.utt_id,
                "mixture_codes": item.mixture_codes.cpu().long(),
                "source_codes": item.source_codes.cpu().long(),
                "audio_prompt_codes": None
                if item.audio_prompt_codes is None
                else item.audio_prompt_codes.cpu().long(),
            }
            for item in items
        ],
    }
    torch.save(payload, path)


def load_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta", {})
    if meta.get("cache_format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported cache format in {path}: {meta.get('cache_format')}")
    if int(meta.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported cache version in {path}: {meta.get('cache_version')}")
    return payload


class EncodecSepDataset(Dataset):
    def __init__(self, cache_path: str | Path, max_frames: int | None = None) -> None:
        payload = load_cache(cache_path)
        self.meta = payload["meta"]
        self.items = payload["items"]
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SepItem:
        raw = self.items[index]
        mix = raw["mixture_codes"].long()
        src = raw["source_codes"].long()
        prompt = raw.get("audio_prompt_codes")
        prompt = None if prompt is None else prompt.long()
        length = min(mix.size(0), src.size(0))
        if self.max_frames is not None:
            length = min(length, int(self.max_frames))
        return SepItem(
            utt_id=str(raw["utt_id"]),
            mixture_codes=mix[:length],
            source_codes=src[:length],
            audio_prompt_codes=None if prompt is None else prompt,
        )


def collate_sep(batch: list[SepItem]) -> dict[str, torch.Tensor | list[str]]:
    bsz = len(batch)
    max_t = max(item.mixture_codes.size(0) for item in batch)
    num_codebooks = batch[0].mixture_codes.size(1)
    mixture = torch.zeros(bsz, max_t, num_codebooks, dtype=torch.long)
    source = torch.zeros(bsz, max_t, num_codebooks, dtype=torch.long)
    mask = torch.zeros(bsz, max_t, dtype=torch.bool)

    prompt_max = max((0 if item.audio_prompt_codes is None else item.audio_prompt_codes.size(0)) for item in batch)
    prompt = torch.zeros(bsz, prompt_max, num_codebooks, dtype=torch.long) if prompt_max > 0 else None
    prompt_mask = torch.zeros(bsz, prompt_max, dtype=torch.bool) if prompt_max > 0 else None

    utt_ids: list[str] = []
    for i, item in enumerate(batch):
        t = item.mixture_codes.size(0)
        mixture[i, :t] = item.mixture_codes
        source[i, :t] = item.source_codes
        mask[i, :t] = True
        if item.audio_prompt_codes is not None and prompt is not None and prompt_mask is not None:
            p = item.audio_prompt_codes.size(0)
            prompt[i, :p] = item.audio_prompt_codes
            prompt_mask[i, :p] = True
        utt_ids.append(item.utt_id)

    out: dict[str, torch.Tensor | list[str]] = {
        "utt_id": utt_ids,
        "mixture_codes": mixture,
        "source_codes": source,
        "mask": mask,
    }
    if prompt is not None and prompt_mask is not None:
        out["audio_prompt_codes"] = prompt
        out["audio_prompt_mask"] = prompt_mask
    return out

