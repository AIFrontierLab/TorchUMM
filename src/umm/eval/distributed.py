# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Distributed-evaluation plumbing shared by understanding-class eval CLIs.

These helpers are intentionally narrow: they cover process-group init/teardown,
rank-shard path naming, and JSONL shard merge/cleanup. Per-benchmark output
formatting and scoring stay in each CLI.

Single-card mode (``WORLD_SIZE <= 1``) is a noop everywhere — no process group
is initialized and ``rank_shard_path`` returns the base path unchanged, so the
caller's on-disk filenames are unaffected.

`torch` is imported lazily inside functions so that callers that never enter
distributed mode don't pay the import cost.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    enabled: bool  # True iff this process actually init'd a process group


def get_dist_info() -> DistInfo:
    """Read RANK / WORLD_SIZE / LOCAL_RANK from the environment. Does not init."""
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        enabled=False,
    )


def maybe_init_distributed() -> DistInfo:
    """Initialize ``torch.distributed`` when ``WORLD_SIZE > 1``; otherwise noop."""
    info = get_dist_info()
    if info.world_size <= 1:
        return info

    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return DistInfo(
        rank=info.rank,
        world_size=info.world_size,
        local_rank=info.local_rank,
        enabled=True,
    )


def cleanup_distributed(info: DistInfo) -> None:
    if not info.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if not info.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def sum_across_ranks(value: int, info: DistInfo) -> int:
    if not info.enabled:
        return value
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return value
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor([value], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def rank_shard_path(base: Path, rank: int, world_size: int) -> Path:
    """Return the rank-local shard path for ``base``.

    Single-card mode (``world_size <= 1``) returns ``base`` unchanged so the
    caller's existing on-disk filenames are preserved.
    """
    if world_size <= 1:
        return base
    return base.parent / f"{base.stem}.rank{rank}{base.suffix}"


def _shard_glob_pattern(base: Path) -> str:
    return f"{base.stem}.rank*{base.suffix}"


def load_shard_items(shard: Path) -> list[dict[str, Any]]:
    """Read a JSONL shard, return [] if missing."""
    items: list[dict[str, Any]] = []
    if not shard.exists():
        return items
    with shard.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def merge_shards(base: Path) -> list[dict[str, Any]]:
    """Merge all rank shards for ``base`` into a single sorted list.

    Globs ``{stem}.rank*{suffix}`` plus ``base`` itself if present, so a re-run
    that uses a smaller world_size doesn't silently drop rows from earlier
    larger runs. Items are sorted by ``_sample_idx`` (injected by the runner).
    """
    candidates: list[Path] = []
    if base.exists():
        candidates.append(base)
    candidates.extend(sorted(base.parent.glob(_shard_glob_pattern(base))))

    merged: list[tuple[int, dict[str, Any]]] = []
    seen_keys: set[tuple[int, int]] = set()
    for path in candidates:
        for item in load_shard_items(path):
            key = int(item.get("_sample_idx", 0))
            # Defensive: if both base and a shard have the same _sample_idx
            # (shouldn't happen in practice), keep the first occurrence.
            dedup = (key, id(item))
            if dedup in seen_keys:
                continue
            seen_keys.add(dedup)
            merged.append((key, item))
    merged.sort(key=lambda pair: pair[0])
    return [item for _, item in merged]


def cleanup_shards(base: Path) -> None:
    """Delete all rank shards matching ``{stem}.rank*{suffix}``.

    The base path itself is NOT touched — the caller is responsible for the
    final merged file (which often lives at a different timestamped name).
    """
    for shard in base.parent.glob(_shard_glob_pattern(base)):
        try:
            shard.unlink()
        except FileNotFoundError:
            pass
