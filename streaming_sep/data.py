from __future__ import annotations

from collections import OrderedDict
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
    stem_id: int = 0
    stem_name: str = "source"


STEM_TO_ID = {"vocal": 0, "bass": 1, "drums": 2, "other": 3}
ID_TO_STEM = {value: key for key, value in STEM_TO_ID.items()}


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
                "stem_id": int(item.stem_id),
                "stem_name": item.stem_name,
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
            stem_id=int(raw.get("stem_id", 0)),
            stem_name=str(raw.get("stem_name", raw.get("stem", "source"))),
        )


def _load_index_items(index_path: str | Path) -> list[dict[str, Any]]:
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    for key in ("items", "records", "encoded_pairs"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError(f"Cannot find index item list in {index_path}")


def _by_relative_path(index_path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in _load_index_items(index_path):
        rel = item.get("relative_path") or item.get("utt_id") or item.get("id")
        if rel is None:
            raise ValueError(f"Index item has no relative_path in {index_path}: {item.keys()}")
        out[str(rel)] = item
    return out


def _resolve_existing_path(path_value: str | Path, data_root: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    data_root = Path(data_root)
    raw = str(path_value).replace("\\", "/")
    for prefix in ("/mnt/streaming_sep_offload/", "/tmp/streaming_sep_offload/"):
        if raw.startswith(prefix):
            candidate = data_root / raw[len(prefix) :]
            if candidate.exists():
                return candidate

    parts = Path(raw).parts
    for idx, part in enumerate(parts):
        if part.startswith("encodec_"):
            candidate = data_root.joinpath(*parts[idx:])
            if candidate.exists():
                return candidate
    return path


class StreamingSepHandoffDataset(Dataset):
    def __init__(
        self,
        index_paths: dict[str, str | Path],
        data_root: str | Path,
        keys: list[str] | None = None,
        stems: list[str] | None = None,
        max_frames: int | None = None,
        crop_frames: int | None = None,
        random_crop: bool = True,
        shard_cache_size: int = 2,
    ) -> None:
        self.data_root = Path(data_root)
        self.max_frames = max_frames
        self.crop_frames = crop_frames
        self.random_crop = random_crop
        self.shard_cache_size = int(shard_cache_size)
        self._shard_cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

        self.stems = stems or ["vocal", "bass", "drums", "other"]
        unknown = sorted(set(self.stems) - set(STEM_TO_ID))
        if unknown:
            raise ValueError(f"Unknown stems: {unknown}; expected {sorted(STEM_TO_ID)}")

        self.indices: dict[str, dict[str, dict[str, Any]]] = {
            stem: _by_relative_path(index_paths[stem]) for stem in self.stems
        }
        common = set.intersection(*(set(items) for items in self.indices.values()))
        self.keys = sorted(common if keys is None else common.intersection(keys))
        if not self.keys:
            raise ValueError("No common relative_path keys across requested stem indices")
        self.rows = [(key, stem) for key in self.keys for stem in self.stems]

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, shard_path: str | Path) -> dict[str, Any]:
        path = _resolve_existing_path(shard_path, self.data_root)
        cached = self._shard_cache.get(path)
        if cached is not None:
            self._shard_cache.move_to_end(path)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._shard_cache[path] = payload
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> SepItem:
        key, stem = self.rows[index]
        item = self.indices[stem][key]
        payload = self._load_shard(item["shard_path"])
        record = payload["records"][int(item["local_index"])]
        mix = record["mixture_codes"].long()
        src = record["separated_codes"].long()
        length = min(mix.size(0), src.size(0))
        if self.max_frames is not None:
            length = min(length, int(self.max_frames))
        mix = mix[:length]
        src = src[:length]

        if self.crop_frames is not None and length > int(self.crop_frames):
            crop = int(self.crop_frames)
            start = torch.randint(0, length - crop + 1, ()).item() if self.random_crop else 0
            mix = mix[start : start + crop]
            src = src[start : start + crop]

        return SepItem(
            utt_id=f"{key}::{stem}",
            mixture_codes=mix,
            source_codes=src,
            stem_id=STEM_TO_ID[stem],
            stem_name=stem,
        )


def collate_sep(batch: list[SepItem]) -> dict[str, torch.Tensor | list[str]]:
    bsz = len(batch)
    max_t = max(item.mixture_codes.size(0) for item in batch)
    num_codebooks = batch[0].mixture_codes.size(1)
    mixture = torch.zeros(bsz, max_t, num_codebooks, dtype=torch.long)
    source = torch.zeros(bsz, max_t, num_codebooks, dtype=torch.long)
    mask = torch.zeros(bsz, max_t, dtype=torch.bool)
    stem_id = torch.zeros(bsz, dtype=torch.long)

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
        stem_id[i] = int(item.stem_id)
        utt_ids.append(item.utt_id)

    out: dict[str, torch.Tensor | list[str]] = {
        "utt_id": utt_ids,
        "mixture_codes": mixture,
        "source_codes": source,
        "mask": mask,
        "stem_id": stem_id,
    }
    if prompt is not None and prompt_mask is not None:
        out["audio_prompt_codes"] = prompt
        out["audio_prompt_mask"] = prompt_mask
    return out

