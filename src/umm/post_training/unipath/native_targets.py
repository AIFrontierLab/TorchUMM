from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer_image_reference(row: dict[str, Any]) -> str | None:
    answer = row.get("answer")
    answer_image_path = row.get("answer_image_path")
    for value in (answer_image_path, answer):
        if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            return value
    return None


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples: int | None,
    sample_selection: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if max_samples is not None and max_samples <= 0:
        max_samples = None
    if sample_selection == "random":
        rng.shuffle(rows)
        return rows[:max_samples] if max_samples is not None else rows
    if sample_selection == "stratified":
        shuffled = list(rows)
        rng.shuffle(shuffled)
        groups: dict[str, list[dict[str, Any]]] = {}
        level_order: list[str] = []
        for row in shuffled:
            level = str(row.get("reasoning_level") or row.get("path_level") or row.get("level") or "unknown").lower()
            if level not in groups:
                groups[level] = []
                level_order.append(level)
            groups[level].append(row)
        selected: list[dict[str, Any]] = []
        cursors = {level: 0 for level in level_order}
        limit = len(shuffled) if max_samples is None else max_samples
        while len(selected) < limit:
            progressed = False
            for level in level_order:
                cursor = cursors[level]
                bucket = groups[level]
                if cursor < len(bucket):
                    selected.append(bucket[cursor])
                    cursors[level] = cursor + 1
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
        return selected
    if sample_selection == "longest":
        rows = sorted(rows, key=lambda row: len(str(row.get("trajectory") or "")), reverse=True)
        return rows[:max_samples] if max_samples is not None else rows
    raise ValueError(f"Unsupported sample_selection={sample_selection}")


def iter_native_image_answer_rows(
    paths: list[Path],
    *,
    seed: int,
    max_samples: int | None = None,
    sample_selection: str = "random",
    require_cached_answer_latent: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    rows = [row for row in rows if str(row.get("answer_type") or "").lower() == "image" and normalize_answer_image_reference(row)]
    if require_cached_answer_latent:
        rows = [row for row in rows if answer_latent_relpath(row)]
    return select_rows(rows, seed=seed, max_samples=max_samples, sample_selection=sample_selection)


def answer_latent_relpath(row: dict[str, Any]) -> str | None:
    latent_index = row.get("latent_index")
    if not isinstance(latent_index, dict):
        return None
    answer_items = latent_index.get("answer")
    if not isinstance(answer_items, list):
        return None
    for item in answer_items:
        if isinstance(item, dict):
            latent_path = item.get("latent_path")
            if isinstance(latent_path, str) and latent_path:
                return latent_path
    return None


def load_cached_answer_latent(cache_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    relpath = answer_latent_relpath(row)
    if not relpath:
        raise FileNotFoundError(f"Missing latent_index.answer for row id={row.get('id')!r}")
    full_path = (cache_root / relpath).resolve()
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    payload = torch.load(full_path, map_location="cpu")
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor):
        raise ValueError(f"Latent payload missing tensor at {full_path}")
    return {
        "latent": latent.float(),
        "orig_size": payload.get("orig_size"),
        "transformed_size": payload.get("transformed_size"),
        "latent_path": relpath,
    }


def patchify_cached_latent(latent: torch.Tensor, latent_patch_size: int = 2) -> tuple[torch.Tensor, tuple[int, int]]:
    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape [C, H, W], got {tuple(latent.shape)}")
    channels, height, width = latent.shape
    p = latent_patch_size
    h = height // p
    w = width // p
    cropped = latent[:, : h * p, : w * p].reshape(channels, h, p, w, p)
    patchified = torch.einsum("chpwq->hwpqc", cropped).reshape(-1, p * p * channels).contiguous()
    return patchified, (h, w)


def build_native_answer_target(
    cache_root: Path,
    row: dict[str, Any],
    latent_patch_size: int = 2,
) -> dict[str, Any]:
    payload = load_cached_answer_latent(cache_root, row)
    patchified_latent, patch_shape = patchify_cached_latent(payload["latent"], latent_patch_size=latent_patch_size)
    return {
        "row_id": str(row.get("id") or ""),
        "latent_path": payload["latent_path"],
        "latent": payload["latent"],
        "patchified_latent": patchified_latent,
        "patch_shape": patch_shape,
        "transformed_size": payload.get("transformed_size"),
        "orig_size": payload.get("orig_size"),
    }
