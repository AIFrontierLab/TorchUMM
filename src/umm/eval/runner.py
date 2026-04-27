# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Sharded inference loop shared by understanding-class eval CLIs.

The runner is intentionally minimal: it iterates a caller-supplied ``samples``
iterable, sends one payload per assigned sample through ``infer_fn``, and
appends a JSONL line (with an injected ``_sample_idx``) to a per-rank shard.

Per-benchmark output formatting (Excel, scoring, calculation scripts) stays in
the calling CLI behind ``if rank == 0:`` — the runner does not own those.

Single-card mode is supported by passing a ``DistInfo`` with ``world_size <= 1``;
all rank/skip checks become noops.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Optional, TypeVar

from tqdm import tqdm

from umm.eval.distributed import DistInfo


T = TypeVar("T")


def run_sharded_inference(
    *,
    infer_fn: Callable[[dict[str, Any]], Any],
    dist_info: DistInfo,
    shard_path: Path,
    samples: Iterable[T],
    payload_fn: Callable[[T], dict[str, Any]],
    record_fn: Callable[[T, Any, int], dict[str, Any]],
    total: Optional[int] = None,
    sample_id_fn: Optional[Callable[[T], Hashable]] = None,
    done_ids: Optional[set[Hashable]] = None,
    max_samples: int = 0,
    log_prefix: str = "eval",
) -> int:
    """Iterate ``samples``; run ``infer_fn`` for samples assigned to this rank;
    append each result as a JSONL line to ``shard_path``.

    Loop semantics (sample_idx is 1-based, in iteration order):
      - ``max_samples > 0``: stop when ``sample_idx > max_samples`` (global cap).
      - World size > 1: skip when ``(sample_idx - 1) % world_size != rank``.
      - Resume: skip when ``sample_id_fn(sample) in done_ids``.
      - Otherwise: ``payload = payload_fn(sample); raw = infer_fn(payload)``;
        ``item = record_fn(sample, raw, sample_idx)``; ``item["_sample_idx"] =
        sample_idx``; write JSONL line + flush + fsync.

    Pre-filter samples that should be skipped *before* this runner sees them
    (e.g. missing-image rows in MME) by wrapping ``samples`` in a generator —
    the runner's contract is "every sample yielded gets processed or sharded".

    Returns the number of samples written by THIS rank's shard.
    """
    rank = dist_info.rank
    world_size = dist_info.world_size
    done = done_ids if done_ids is not None else set()

    n_written = 0
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("a", encoding="utf-8") as shard_writer:
        iterator = tqdm(
            samples,
            total=total,
            desc=log_prefix,
            file=sys.stdout,
            disable=(rank != 0),
        )
        for sample_idx, sample in enumerate(iterator, start=1):
            if max_samples > 0 and sample_idx > max_samples:
                break
            if world_size > 1 and (sample_idx - 1) % world_size != rank:
                continue
            if sample_id_fn is not None:
                sample_key = sample_id_fn(sample)
                if sample_key in done:
                    continue
            else:
                sample_key = None

            payload = payload_fn(sample)
            raw = infer_fn(payload)
            item = dict(record_fn(sample, raw, sample_idx))
            item["_sample_idx"] = sample_idx
            shard_writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            shard_writer.flush()
            os.fsync(shard_writer.fileno())
            n_written += 1
            if sample_key is not None:
                done.add(sample_key)

    return n_written
