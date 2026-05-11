from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from accelerate.utils.modeling import set_module_tensor_to_device
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

BAGEL_CODE_ROOT = add_bagel_code_to_sys_path(REPO_ROOT)

from accelerate import init_empty_weights
from data.data_utils import add_special_tokens, get_flattened_position_ids_extrapolate, patchify, pil_img2rgb, prepare_attention_mask_per_sample
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from umm.backbones.bagel.adapter import BagelBackbone
from umm.post_training.unipath.train import (
    STAGE_PRESETS,
    VISUAL_SEGMENT_TYPE_IDS,
    Segment,
    answers_match,
    build_context_text,
    compute_visual_summary_loss,
    estimate_segment_text_length,
    expected_tag_sequence_for_row,
    extract_answer_text,
    extract_structural_tag_sequence,
    format_matches_expected,
    is_main_process,
    log_main,
    maybe_load_visual_head,
    maybe_load_trainable_model_state,
    normalize_answer_text,
    preview_segments,
    read_jsonl,
    reduce_mean,
    resolve_jsonl_paths,
    row_to_segments,
    save_checkpoint_bundle,
    write_checkpoint_metadata,
)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        timeout_minutes = int(os.environ.get("UNIPATH_DDP_TIMEOUT_MIN", "120"))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=max(timeout_minutes, 10)))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def broadcast_float_from_main(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if is_distributed():
        dist.broadcast(tensor, src=0)
    return float(tensor.item())


def gather_objects_to_main(local_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_distributed():
        return list(local_items)
    gathered: list[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, list(local_items))
    if not is_main_process():
        return []
    merged: list[dict[str, Any]] = []
    for chunk in gathered:
        if isinstance(chunk, list):
            merged.extend(chunk)
    return merged


def stratified_limit_by_level(items: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(items) <= max_samples:
        return list(items)
    groups: dict[str, list[dict[str, Any]]] = {}
    level_order: list[str] = []
    for item in items:
        level = str(item.get("reasoning_level") or "unknown")
        if level not in groups:
            groups[level] = []
            level_order.append(level)
        groups[level].append(item)
    cursors = {level: 0 for level in level_order}
    selected: list[dict[str, Any]] = []
    while len(selected) < max_samples:
        progressed = False
        for level in level_order:
            idx = cursors[level]
            bucket = groups[level]
            if idx < len(bucket):
                selected.append(bucket[idx])
                cursors[level] = idx + 1
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
    return selected


def collect_eval_generate_candidates(
    dataloader: DataLoader,
    *,
    max_generate_samples: int,
    generate_sampling: str,
) -> list[dict[str, Any]]:
    if max_generate_samples <= 0:
        return []
    local_candidates: list[dict[str, Any]] = []
    for batch in dataloader:
        for idx in range(len(batch["row_id"])):
            local_candidates.append(
                {
                    "row_id": batch["row_id"][idx],
                    "reasoning_level": batch["reasoning_level"][idx],
                    "answer_text": batch["answer_text"][idx],
                    "context_text": batch["context_text"][idx],
                    "expected_tag_sequence": batch["expected_tag_sequence"][idx],
                    "image_path": batch["image_path"][idx],
                }
            )
    gathered = gather_objects_to_main(local_candidates)
    if not is_main_process():
        return []
    deduped: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    for item in gathered:
        row_id = str(item.get("row_id", ""))
        if row_id in seen_row_ids:
            continue
        seen_row_ids.add(row_id)
        deduped.append(item)
    if generate_sampling == "stratified":
        return stratified_limit_by_level(deduped, max_generate_samples)
    return deduped[:max_generate_samples]


def best_metric_is_higher(metric_name: str) -> bool:
    return metric_name in {"val_answer_accuracy", "val_format_accuracy", "val_answer_format_sum", "val_answer_or_format"}


def best_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "val_answer_format_sum":
        return float(metrics.get("val_answer_accuracy", 0.0) + metrics.get("val_format_accuracy", 0.0))
    if metric_name == "val_answer_or_format":
        return float(max(metrics.get("val_answer_accuracy", 0.0), metrics.get("val_format_accuracy", 0.0)))
    return float(metrics[metric_name])


def metric_improved(metric_value: float, best_value: float, metric_name: str) -> bool:
    return metric_value > best_value if best_metric_is_higher(metric_name) else metric_value < best_value


def should_stop_for_zero_metrics(metrics: dict[str, float]) -> bool:
    return float(metrics.get("val_answer_accuracy", 0.0)) == 0.0 or float(metrics.get("val_format_accuracy", 0.0)) == 0.0


def broadcast_bool_from_main(value: bool, device: torch.device) -> bool:
    return bool(round(broadcast_float_from_main(1.0 if value else 0.0, device)))


def cuda_memory_summary(device: torch.device) -> str:
    if not torch.cuda.is_available() or device.type != "cuda":
        return "cuda_mem=NA"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved(idx) / (1024 ** 3)
    return (
        f"cuda_mem_alloc_gb={alloc:.2f} cuda_mem_reserved_gb={reserved:.2f} "
        f"cuda_peak_alloc_gb={peak_alloc:.2f} cuda_peak_reserved_gb={peak_reserved:.2f}"
    )


def get_language_model_core(language_model: torch.nn.Module) -> torch.nn.Module:
    base_model = getattr(language_model, "base_model", None)
    if base_model is not None:
        nested = getattr(base_model, "model", None)
        if nested is not None:
            return nested
    return language_model


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples: int | None,
    sample_selection: str,
    thought_weight: float,
    answer_weight: float,
    think_mode: bool,
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
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in shuffled:
            level = str(row.get("reasoning_level") or row.get("path_level") or row.get("level") or "unknown")
            buckets.setdefault(level, []).append(row)
        levels = sorted(buckets)
        if max_samples is None:
            return [row for level in levels for row in buckets[level]]
        selected: list[dict[str, Any]] = []
        while len(selected) < max_samples and levels:
            next_levels: list[str] = []
            for level in levels:
                bucket = buckets[level]
                if bucket and len(selected) < max_samples:
                    selected.append(bucket.pop(0))
                if bucket:
                    next_levels.append(level)
            levels = next_levels
        return selected
    if sample_selection == "longest":
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            segments = row_to_segments(row, thought_weight=thought_weight, answer_weight=answer_weight, think_mode=think_mode)
            scored.append((estimate_segment_text_length(segments), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for _, row in scored]
        return selected[:max_samples] if max_samples is not None else selected
    raise ValueError(f"Unsupported sample_selection={sample_selection}")


def load_latent_summary(latent_cache_root: Path, latent_paths: list[str], latent_target_dim: int) -> torch.Tensor | None:
    vectors: list[torch.Tensor] = []
    for rel_path in latent_paths:
        full_path = (latent_cache_root / rel_path).resolve()
        payload = torch.load(full_path, map_location="cpu")
        latent = payload["latent"]
        summary = latent.float().mean(dim=(-1, -2))
        if summary.numel() != latent_target_dim:
            raise ValueError(f"Expected latent summary dim {latent_target_dim}, got {summary.numel()} from {full_path}")
        vectors.append(summary)
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def validate_rows_for_image_aware_understanding(rows: list[dict[str, Any]], latent_cache_root: Path | None) -> None:
    missing_primary_image: list[str] = []
    image_answer_rows: list[str] = []
    missing_visual_latent: list[str] = []
    for row in rows:
        row_id = str(row.get("id") or "<unknown>")
        image_path = str(row.get("image_path") or "").strip()
        if not image_path:
            missing_primary_image.append(row_id)
        if not normalize_answer_text(row.get("answer")):
            raise ValueError(f"[image_aware] missing textual answer for row {row_id}")
        if row.get("answer_type") == "image" or str(row.get("answer_image_path") or "").strip():
            image_answer_rows.append(row_id)
        latent_index = row.get("latent_index") or {}
        if row.get("vc_image_path") and latent_cache_root is None:
            missing_visual_latent.append(row_id)
        if row.get("vc_image_path") and not latent_index.get("vc"):
            missing_visual_latent.append(row_id)
        vh_paths = row.get("vh_image_paths") or []
        if vh_paths and latent_cache_root is None:
            missing_visual_latent.append(row_id)
        if vh_paths and not latent_index.get("vh"):
            missing_visual_latent.append(row_id)
    if missing_primary_image:
        raise ValueError(f"[image_aware] missing primary image_path for {len(missing_primary_image)} rows, sample={missing_primary_image[:8]}")
    if image_answer_rows:
        raise ValueError(f"[image_aware] image answers are not supported in understanding trainer: {len(image_answer_rows)} rows, sample={image_answer_rows[:8]}")
    if missing_visual_latent:
        raise ValueError(f"[image_aware] missing required visual latents for {len(missing_visual_latent)} rows, sample={missing_visual_latent[:8]}")


def resolve_image_aware_row_paths(row: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    resolved = dict(row)
    image_path = str(resolved.get("image_path") or "").strip()
    if image_path and not Path(image_path).is_absolute():
        resolved["image_path"] = str((dataset_root / image_path).resolve())
    return resolved


def parse_level_sample_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    weights: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid level weight {item!r}; expected level:weight")
        level, weight = item.split(":", 1)
        weights[level.strip().lower()] = max(float(weight), 0.0)
    return weights


def apply_level_sample_weights(rows: list[dict[str, Any]], level_weights: dict[str, float]) -> list[dict[str, Any]]:
    if not level_weights:
        return rows
    weighted_rows: list[dict[str, Any]] = []
    rng = random.Random(0)
    for row in rows:
        level = str(row.get("reasoning_level") or row.get("path_level") or row.get("level") or "unknown").lower()
        weight = level_weights.get(level, 1.0)
        repeats = int(math.floor(weight))
        weighted_rows.extend([row] * repeats)
        if rng.random() < weight - repeats:
            weighted_rows.append(row)
    return weighted_rows or rows


@dataclass
class ImageAwareExample:
    row_id: str
    image_path: str
    image: Image.Image
    text_piece_ids: list[list[int]]
    text_piece_weights: list[float]
    segment_types: list[str]
    visual_piece_indexes: list[int]
    visual_targets: list[list[float]]
    preview: str
    reasoning_level: str
    answer_text: str
    context_text: str
    expected_tag_sequence: list[str]
    seq_len_estimate: int


class ImageAwareUnderstandingDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        tokenizer: Qwen2Tokenizer,
        seed: int,
        thought_weight: float,
        answer_weight: float,
        latent_cache_root: Path | None,
        latent_target_dim: int,
        max_samples: int | None = None,
        sample_selection: str = "random",
        level_sample_weights: dict[str, float] | None = None,
        think_mode: bool = False,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for path in paths:
            dataset_root = path.resolve().parent
            rows.extend(resolve_image_aware_row_paths(row, dataset_root) for row in read_jsonl(path))
        validate_rows_for_image_aware_understanding(rows, latent_cache_root=latent_cache_root)
        rows = select_rows(
            rows,
            seed=seed,
            max_samples=max_samples,
            sample_selection=sample_selection,
            thought_weight=thought_weight,
            answer_weight=answer_weight,
            think_mode=think_mode,
        )
        rows = apply_level_sample_weights(rows, level_sample_weights or {})
        self.rows = rows
        self.tokenizer = tokenizer
        self.thought_weight = thought_weight
        self.answer_weight = answer_weight
        self.latent_cache_root = latent_cache_root
        self.latent_target_dim = latent_target_dim
        self.think_mode = think_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> ImageAwareExample:
        row = self.rows[idx]
        segments = row_to_segments(
            row,
            thought_weight=self.thought_weight,
            answer_weight=self.answer_weight,
            think_mode=self.think_mode,
        )
        pieces: list[list[int]] = []
        weights: list[float] = []
        segment_types: list[str] = []
        visual_piece_indexes: list[int] = []
        visual_targets: list[list[float]] = []
        for segment in segments:
            piece_ids = self.tokenizer.encode(segment.text, add_special_tokens=False)
            if not piece_ids:
                continue
            piece_index = len(pieces)
            pieces.append(piece_ids)
            weights.append(segment.loss_weight)
            segment_types.append(segment.segment_type)
            if segment.visual_paths:
                if self.latent_cache_root is None:
                    raise ValueError(f"[image_aware] latent cache required for row {row.get('id')}")
                summary = load_latent_summary(
                    self.latent_cache_root,
                    segment.visual_paths,
                    latent_target_dim=self.latent_target_dim,
                )
                if summary is None:
                    raise ValueError(f"[image_aware] missing latent summary for row {row.get('id')}")
                visual_piece_indexes.append(piece_index)
                visual_targets.append(summary.tolist())
        if not pieces:
            raise ValueError(f"[image_aware] no tokenized pieces for row {row.get('id')}")
        image_path = str(row["image_path"])
        with Image.open(image_path) as image_file:
            image = pil_img2rgb(image_file)
        seq_len_estimate = 2 + sum(len(piece) for piece in pieces) + 2
        return ImageAwareExample(
            row_id=str(row.get("id") or ""),
            image_path=image_path,
            image=image,
            text_piece_ids=pieces,
            text_piece_weights=weights,
            segment_types=segment_types,
            visual_piece_indexes=visual_piece_indexes,
            visual_targets=visual_targets,
            preview=preview_segments(segments),
            reasoning_level=str(row.get("reasoning_level") or row.get("level") or ""),
            answer_text=normalize_answer_text(row.get("answer")),
            context_text=build_context_text(row, think_mode=self.think_mode),
            expected_tag_sequence=expected_tag_sequence_for_row(row),
            seq_len_estimate=seq_len_estimate,
        )


class ImageAwareCollator:
    def __init__(self, tokenizer: Qwen2Tokenizer, vit_transform: ImageTransform, latent_target_dim: int, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.latent_target_dim = latent_target_dim
        self.max_length = max_length
        self.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        assert self.bos_token_id is not None
        assert self.eos_token_id is not None
        assert self.start_of_image is not None
        assert self.end_of_image is not None

    def __call__(self, batch: list[ImageAwareExample]) -> dict[str, Any]:
        sample_lens: list[int] = []
        nested_attention_masks: list[torch.Tensor] = []
        packed_text_ids: list[int] = []
        packed_text_indexes: list[int] = []
        packed_position_ids: list[int] = []
        packed_label_ids: list[int] = []
        ce_loss_indexes: list[int] = []
        ce_loss_weights: list[float] = []
        packed_vit_tokens: list[torch.Tensor] = []
        packed_vit_position_ids: list[torch.Tensor] = []
        vit_token_seqlens: list[int] = []
        packed_vit_token_indexes: list[int] = []
        visual_spans: list[list[list[int]]] = []
        visual_targets: list[list[list[float]]] = []
        visual_mask: list[list[float]] = []
        previews: list[str] = []
        reasoning_levels: list[str] = []
        answer_texts: list[str] = []
        context_texts: list[str] = []
        row_ids: list[str] = []
        expected_tag_sequences: list[list[str]] = []

        curr = 0
        max_visual_segments = max((len(item.visual_piece_indexes) for item in batch), default=0)

        for item in batch:
            image_tensor = self.vit_transform(item.image)
            vit_tokens = patchify(image_tensor, 14)
            vit_position_ids = get_flattened_position_ids_extrapolate(
                image_tensor.size(1),
                image_tensor.size(2),
                14,
                max_num_patches_per_side=70,
            )
            num_img_tokens = vit_tokens.shape[0]
            image_len = num_img_tokens + 2
            if image_len + 2 > self.max_length:
                raise ValueError(
                    f"[image_aware] max_length={self.max_length} is too small for image-only prefix "
                    f"(image_len={image_len}) on row {item.row_id}"
                )

            text_body: list[int] = []
            target_weights: list[float] = []
            visual_target_map = {piece_idx: target for piece_idx, target in zip(item.visual_piece_indexes, item.visual_targets)}
            visual_piece_candidates: list[tuple[int, int, list[float]]] = []
            text_offset = 0
            for piece_idx, (piece_ids, weight, piece_type) in enumerate(zip(item.text_piece_ids, item.text_piece_weights, item.segment_types)):
                piece_start = text_offset
                text_body.extend(piece_ids)
                target_weights.extend([weight] * len(piece_ids))
                text_offset += len(piece_ids)
                if piece_type == "visual":
                    piece_end = text_offset
                    if piece_idx not in visual_target_map:
                        raise ValueError(f"[image_aware] visual piece index {piece_idx} missing target")
                    visual_piece_candidates.append((piece_start, piece_end, visual_target_map[piece_idx]))
            if len(text_body) + image_len + 2 > self.max_length:
                available = self.max_length - image_len - 2
                text_body = text_body[:available]
                target_weights = target_weights[:available]
                kept_visual_candidates: list[tuple[int, int, list[float]]] = []
                for start, end, target in visual_piece_candidates:
                    if start >= available:
                        continue
                    kept_visual_candidates.append((start, min(end, available), target))
                visual_piece_candidates = kept_visual_candidates

            text_start = curr + image_len
            text_seq_ids = [self.bos_token_id] + text_body + [self.eos_token_id]
            packed_text_ids.append(self.start_of_image)
            packed_text_indexes.append(curr)
            packed_position_ids.append(0)
            curr += 1

            packed_vit_tokens.append(vit_tokens)
            packed_vit_position_ids.append(vit_position_ids)
            packed_vit_token_indexes.extend(range(curr, curr + num_img_tokens))
            packed_position_ids.extend([0] * num_img_tokens)
            vit_token_seqlens.append(num_img_tokens)
            curr += num_img_tokens

            packed_text_ids.append(self.end_of_image)
            packed_text_indexes.append(curr)
            packed_position_ids.append(0)
            curr += 1

            text_positions = list(range(curr, curr + len(text_seq_ids)))
            packed_text_ids.extend(text_seq_ids)
            packed_text_indexes.extend(text_positions)
            packed_position_ids.extend(range(1, len(text_seq_ids) + 1))
            ce_positions = text_positions[:-1]
            ce_targets = text_seq_ids[1:]
            ce_weights = target_weights + ([target_weights[-1]] if target_weights else [0.0])
            ce_loss_indexes.extend(ce_positions)
            packed_label_ids.extend(ce_targets)
            ce_loss_weights.extend(ce_weights)
            curr += len(text_seq_ids)

            sample_lens.append(image_len + len(text_seq_ids))
            nested_attention_masks.append(
                prepare_attention_mask_per_sample([image_len, len(text_seq_ids)], ["full", "causal"])
            )

            sample_visual_spans: list[list[int]] = []
            sample_visual_targets: list[list[float]] = []
            sample_visual_mask: list[float] = []
            for start, end, target in visual_piece_candidates:
                global_start = text_start + 1 + start
                global_end = text_start + 1 + end
                sample_visual_spans.append([global_start, global_end])
                sample_visual_targets.append(target)
                sample_visual_mask.append(1.0)
            while len(sample_visual_spans) < max_visual_segments:
                sample_visual_spans.append([-1, -1])
                sample_visual_targets.append([0.0] * self.latent_target_dim)
                sample_visual_mask.append(0.0)
            visual_spans.append(sample_visual_spans)
            visual_targets.append(sample_visual_targets)
            visual_mask.append(sample_visual_mask)

            previews.append(item.preview)
            reasoning_levels.append(item.reasoning_level)
            answer_texts.append(item.answer_text)
            context_texts.append(item.context_text)
            row_ids.append(item.row_id)
            expected_tag_sequences.append(item.expected_tag_sequence)

        return {
            "sequence_length": curr,
            "sample_lens": sample_lens,
            "nested_attention_masks": nested_attention_masks,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_label_ids": torch.tensor(packed_label_ids, dtype=torch.long),
            "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long),
            "ce_loss_weights": torch.tensor(ce_loss_weights, dtype=torch.float32),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int32),
            "visual_spans": torch.tensor(visual_spans, dtype=torch.long) if max_visual_segments > 0 else torch.zeros((len(batch), 0, 2), dtype=torch.long),
            "visual_targets": torch.tensor(visual_targets, dtype=torch.float32) if max_visual_segments > 0 else torch.zeros((len(batch), 0, self.latent_target_dim), dtype=torch.float32),
            "visual_mask": torch.tensor(visual_mask, dtype=torch.float32) if max_visual_segments > 0 else torch.zeros((len(batch), 0), dtype=torch.float32),
            "preview": previews,
            "reasoning_level": reasoning_levels,
            "answer_text": answer_texts,
            "context_text": context_texts,
            "row_id": row_ids,
            "expected_tag_sequence": expected_tag_sequences,
            "image_path": [item.image_path for item in batch],
        }


def load_full_bagel_for_training(model_path: Path, device: torch.device) -> tuple[Bagel, Qwen2Tokenizer]:
    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", None)
    rope_scaling = getattr(llm_config, "rope_scaling", None)
    if rope_scaling is None:
        llm_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type in (None, "default"):
            llm_config.rope_scaling = {**rope_scaling, "rope_type": "linear", "factor": rope_scaling.get("factor", 1.0)}
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.freeze_und = False

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    _, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, _, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens != 0:
        raise ValueError(f"[image_aware] tokenizer unexpectedly added {num_new_tokens} tokens during model build")
    model = model.to_empty(device=device)
    model_keys = set(model.state_dict().keys())
    ema_path = model_path / "ema.safetensors"
    loaded_keys: set[str] = set()
    with safe_open(str(ema_path), framework="pt", device=0) as f:
        for key in f.keys():
            if key not in model_keys:
                continue
            tensor = f.get_tensor(key)
            set_module_tensor_to_device(model, key, device, value=tensor)
            loaded_keys.add(key)
    missing = sorted(model_keys - loaded_keys)
    if missing:
        raise ValueError(f"[image_aware] missing model keys after load: {len(missing)} sample={missing[:8]}")
    return model, tokenizer


def load_bagel_backbone_for_training(
    model_path: Path,
    *,
    distributed_single_gpu: bool,
) -> tuple[Bagel, Qwen2Tokenizer, dict[str, int], ImageTransform, ImageTransform]:
    backbone = BagelBackbone(model_path=str(model_path), distributed_single_gpu=distributed_single_gpu)
    backbone.load({})
    inferencer = backbone.inferencer
    if inferencer is None:
        raise RuntimeError("[image_aware] BagelBackbone did not create an inferencer")
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids
    vit_transform = inferencer.vit_transform
    vae_transform = inferencer.vae_transform
    if model is None or tokenizer is None or new_token_ids is None or vit_transform is None or vae_transform is None:
        raise RuntimeError("[image_aware] BagelBackbone inferencer is missing model/tokenizer/transforms")
    return model, tokenizer, new_token_ids, vit_transform, vae_transform


def cast_model_for_training(model: Bagel, dtype: torch.dtype) -> Bagel:
    model = model.to(dtype=dtype)
    return model


def install_lora_on_language_model(
    model: Bagel,
    target_keywords: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    resume_adapter_path: str | None = None,
) -> tuple[Bagel, int, int]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model.language_model, "prepare_inputs_for_generation"):
        def _prepare_inputs_for_generation(input_ids=None, **kwargs):
            payload = dict(kwargs)
            if input_ids is not None:
                payload["input_ids"] = input_ids
            return payload
        model.language_model.prepare_inputs_for_generation = _prepare_inputs_for_generation
    if resume_adapter_path:
        adapter_dir = Path(resume_adapter_path).expanduser()
        if not adapter_dir.exists():
            raise FileNotFoundError(adapter_dir)
        peft_model = PeftModel.from_pretrained(model.language_model, str(adapter_dir), is_trainable=True)
        print(f"[image_aware] loaded_lora_adapter={adapter_dir}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_keywords,
        )
        peft_model = get_peft_model(model.language_model, lora_cfg)
    wrapped_qwen = peft_model.base_model.model
    inner_transformer = getattr(wrapped_qwen, "model", None)
    if inner_transformer is None:
        raise AttributeError("PEFT-wrapped Qwen2 model is missing inner transformer model")
    if not hasattr(wrapped_qwen, "embed_tokens"):
        if not hasattr(inner_transformer, "embed_tokens"):
            raise AttributeError("Inner transformer is missing embed_tokens")
        wrapped_qwen.embed_tokens = inner_transformer.embed_tokens
    if not hasattr(wrapped_qwen, "norm") and hasattr(inner_transformer, "norm"):
        wrapped_qwen.norm = inner_transformer.norm
    model.language_model = peft_model
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def install_partial_sft_on_language_model(
    model: Bagel,
    *,
    train_last_n_layers: int,
    train_lm_head: bool,
    train_final_norm: bool,
) -> tuple[Bagel, int, int]:
    for param in model.parameters():
        param.requires_grad = False
    num_layers = int(getattr(model.language_model.config, "num_hidden_layers", 0))
    start_layer = max(0, num_layers - max(train_last_n_layers, 0))
    for name, param in model.language_model.named_parameters():
        should_train = False
        for layer_idx in range(start_layer, num_layers):
            if name.startswith(f"model.layers.{layer_idx}."):
                should_train = True
        if train_final_norm and name.startswith("model.norm."):
            should_train = True
        if train_lm_head and name.startswith("lm_head."):
            should_train = True
        if should_train:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def forward_image_aware_understanding(model: Bagel, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    language_model_core = get_language_model_core(model.language_model)
    input_embeddings = language_model_core.get_input_embeddings()
    packed_text_embedding = input_embeddings(batch["packed_text_ids"])
    packed_sequence = packed_text_embedding.new_zeros((batch["sequence_length"], model.hidden_size))
    packed_sequence[batch["packed_text_indexes"]] = packed_text_embedding

    cu_seqlens = torch.nn.functional.pad(torch.cumsum(batch["vit_token_seqlens"], dim=0), (1, 0)).to(torch.int32)
    max_seqlen = torch.max(batch["vit_token_seqlens"]).item()
    vit_dtype = next(model.vit_model.parameters()).dtype
    packed_vit_token_embed = model.vit_model(
        packed_pixel_values=batch["packed_vit_tokens"].to(vit_dtype),
        packed_flattened_position_ids=batch["packed_vit_position_ids"],
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    packed_vit_token_embed = model.connector(packed_vit_token_embed)
    packed_vit_token_embed = packed_vit_token_embed + model.vit_pos_embed(batch["packed_vit_position_ids"])
    if packed_vit_token_embed.dtype != packed_sequence.dtype:
        packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
    packed_sequence[batch["packed_vit_token_indexes"]] = packed_vit_token_embed

    extra_inputs = {}
    if model.use_moe:
        packed_und_token_indexes = torch.cat([batch["packed_text_indexes"], batch["packed_vit_token_indexes"]], dim=0)
        extra_inputs["packed_und_token_indexes"] = packed_und_token_indexes

    hidden_states = language_model_core(
        packed_sequence=packed_sequence,
        sample_lens=batch["sample_lens"],
        attention_mask=batch["nested_attention_masks"],
        packed_position_ids=batch["packed_position_ids"],
        **extra_inputs,
    )
    output_head = language_model_core.get_output_embeddings()
    logits = output_head(hidden_states[batch["ce_loss_indexes"]])
    ce = F.cross_entropy(logits, batch["packed_label_ids"], reduction="none")
    return hidden_states, ce


def unpack_hidden_states(hidden_states: torch.Tensor, sample_lens: list[int]) -> torch.Tensor:
    max_len = max(sample_lens)
    unpacked = hidden_states.new_zeros((len(sample_lens), max_len, hidden_states.size(-1)))
    curr = 0
    for batch_idx, sample_len in enumerate(sample_lens):
        unpacked[batch_idx, :sample_len] = hidden_states[curr:curr + sample_len]
        curr += sample_len
    return unpacked


def compute_weighted_ce(ce: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-8)
    return (ce * weights).sum() / denom


@torch.no_grad()
def generate_understanding_text(
    model: Bagel,
    tokenizer: Qwen2Tokenizer,
    new_token_ids: dict[str, int],
    vit_transform: ImageTransform,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    was_training = model.training
    model.eval()
    vae_transform = ImageTransform(1024, 512, 16)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    result = inferencer(
        image=pil_img2rgb(image),
        text=prompt_text,
        understanding_output=True,
        think=False,
        do_sample=False,
        max_think_token_n=max_new_tokens,
    )
    generated = str(result.get("text") or "")
    if was_training:
        model.train()
    return generated


@torch.no_grad()
def evaluate_image_aware_model(
    model: torch.nn.Module,
    tokenizer: Qwen2Tokenizer,
    new_token_ids: dict[str, int],
    vit_transform: ImageTransform,
    eval_vit_transform: ImageTransform,
    dataloader: DataLoader,
    device: torch.device,
    visual_head: torch.nn.Module,
    visual_loss_weight: float,
    max_generate_samples: int,
    max_generate_new_tokens: int,
    generate_sampling: str = "stratified",
    fixed_generate_candidates: list[dict[str, Any]] | None = None,
    skip_loss_eval: bool = False,
    case_output_path: Path | None = None,
) -> dict[str, float]:
    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.train()
    unwrap_model(visual_head).train()
    total_loss = torch.zeros(1, device=device)
    total_lm = torch.zeros(1, device=device)
    total_visual = torch.zeros(1, device=device)
    total_steps = torch.zeros(1, device=device)
    level_loss_sum: dict[str, float] = {}
    level_count: dict[str, int] = {}
    gen_candidates: list[dict[str, Any]] = []
    for batch in dataloader:
        if not skip_loss_eval:
            batch = move_batch_to_device(batch, device)
            hidden_states, ce = forward_image_aware_understanding(raw_model, batch)
            hidden_states = unpack_hidden_states(hidden_states, batch["sample_lens"])
            lm_loss = compute_weighted_ce(ce, batch["ce_loss_weights"])
            visual_loss = compute_visual_summary_loss(
                hidden_states,
                batch["visual_spans"],
                batch["visual_targets"],
                batch["visual_mask"],
                visual_head,
            )
            loss = lm_loss + visual_loss_weight * visual_loss
            total_loss += loss.detach()
            total_lm += lm_loss.detach()
            total_visual += visual_loss.detach()
            total_steps += 1
        for level in batch.get("reasoning_level", []):
            key = str(level or "unknown")
            if not skip_loss_eval:
                level_loss_sum[key] = level_loss_sum.get(key, 0.0) + float(loss.item())
            level_count[key] = level_count.get(key, 0) + 1
        if fixed_generate_candidates is None:
            for idx in range(len(batch["row_id"])):
                if len(gen_candidates) >= max_generate_samples:
                    break
                gen_candidates.append(
                    {
                        "row_id": batch["row_id"][idx],
                        "reasoning_level": batch["reasoning_level"][idx],
                        "answer_text": batch["answer_text"][idx],
                        "context_text": batch["context_text"][idx],
                        "expected_tag_sequence": batch["expected_tag_sequence"][idx],
                        "image_path": batch["image_path"][idx],
                    }
                )
    if skip_loss_eval:
        metrics: dict[str, float] = {
            "val_loss": 0.0,
            "val_lm_loss": 0.0,
            "val_visual_loss": 0.0,
        }
    else:
        total_loss = reduce_mean(total_loss)
        total_lm = reduce_mean(total_lm)
        total_visual = reduce_mean(total_visual)
        total_steps = reduce_mean(total_steps)
        denom = max(float(total_steps.item()), 1.0)
        metrics = {
            "val_loss": float(total_loss.item() / denom),
            "val_lm_loss": float(total_lm.item() / denom),
            "val_visual_loss": float(total_visual.item() / denom),
        }
        for level, count in sorted(level_count.items()):
            metrics[f"val_loss_{level}"] = level_loss_sum[level] / max(count, 1)

    if max_generate_samples > 0 and fixed_generate_candidates is not None:
        gen_candidates = fixed_generate_candidates if is_main_process() else []
    elif max_generate_samples > 0 and gen_candidates:
        gen_candidates = gather_objects_to_main(gen_candidates)
        if is_main_process():
            deduped_candidates: list[dict[str, Any]] = []
            seen_row_ids: set[str] = set()
            for item in gen_candidates:
                row_id = str(item.get("row_id", ""))
                if row_id in seen_row_ids:
                    continue
                seen_row_ids.add(row_id)
                deduped_candidates.append(item)
            if generate_sampling == "stratified":
                gen_candidates = stratified_limit_by_level(deduped_candidates, max_generate_samples)
            else:
                gen_candidates = deduped_candidates[:max_generate_samples]
    if max_generate_samples > 0 and is_main_process() and gen_candidates:
        format_hits = 0
        answer_hits = 0
        level_bucket: dict[str, dict[str, int]] = {}
        cases: list[dict[str, Any]] = []
        for item in gen_candidates:
            image = pil_img2rgb(Image.open(item["image_path"]))
            generated_text = generate_understanding_text(
                raw_model,
                tokenizer,
                new_token_ids,
                eval_vit_transform,
                image,
                item["context_text"],
                max_generate_new_tokens,
            )
            level = str(item["reasoning_level"] or "unknown")
            bucket = level_bucket.setdefault(level, {"samples": 0, "format": 0, "answer": 0})
            bucket["samples"] += 1
            format_ok = format_matches_expected(generated_text, item["expected_tag_sequence"])
            answer_text = extract_answer_text(generated_text)
            answer_ok = answers_match(item["answer_text"], answer_text, item["context_text"])
            if format_ok:
                format_hits += 1
                bucket["format"] += 1
            if answer_ok:
                answer_hits += 1
                bucket["answer"] += 1
            cases.append(
                {
                    "row_id": item["row_id"],
                    "reasoning_level": level,
                    "image_path": item["image_path"],
                    "prompt_used": item["context_text"],
                    "expected_tag_sequence": item["expected_tag_sequence"],
                    "generated_tag_sequence": extract_structural_tag_sequence(generated_text),
                    "format_ok": format_ok,
                    "expected_answer": item["answer_text"],
                    "generated_answer": answer_text,
                    "answer_ok": answer_ok,
                    "generated_text": generated_text,
                }
            )
        metrics["val_format_accuracy"] = format_hits / max(len(gen_candidates), 1)
        metrics["val_answer_accuracy"] = answer_hits / max(len(gen_candidates), 1)
        for level, bucket in sorted(level_bucket.items()):
            samples = max(bucket["samples"], 1)
            metrics[f"val_format_accuracy_{level}"] = bucket["format"] / samples
            metrics[f"val_answer_accuracy_{level}"] = bucket["answer"] / samples
        if case_output_path is not None:
            case_output_path.parent.mkdir(parents=True, exist_ok=True)
            with case_output_path.open("w", encoding="utf-8") as f:
                for case in cases:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")

    if not was_training:
        raw_model.eval()
        unwrap_model(visual_head).eval()
    return metrics


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list) and value and all(isinstance(item, torch.Tensor) for item in value):
            moved[key] = [item.to(device) for item in value]
        else:
            moved[key] = value
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-aware UniPath understanding trainer on full Bagel visual_und path.")
    parser.add_argument("--train-jsonl", action="append", required=True)
    parser.add_argument("--val-jsonl", action="append", default=[])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-cache-root", default=None)
    parser.add_argument("--latent-target-dim", type=int, default=16)
    parser.add_argument("--visual-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-selection", choices=["random", "longest", "stratified"], default="random")
    parser.add_argument("--level-sample-weights", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1792)
    parser.add_argument("--vit-max-image-size", type=int, default=384)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--vit-image-stride", type=int, default=14)
    parser.add_argument("--eval-vit-max-image-size", type=int, default=None)
    parser.add_argument("--eval-vit-min-image-size", type=int, default=None)
    parser.add_argument("--eval-vit-image-stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument(
        "--save-best-metric",
        choices=["val_loss", "val_answer_accuracy", "val_format_accuracy", "val_answer_format_sum", "val_answer_or_format"],
        default="val_loss",
    )
    parser.add_argument("--best-require-format-no-degrade", action="store_true")
    parser.add_argument("--best-format-tolerance", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--min-steps-before-early-stop", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--eval-generate-max-samples", type=int, default=0)
    parser.add_argument("--eval-generate-sampling", choices=["head", "stratified"], default="stratified")
    parser.add_argument("--eval-generate-max-new-tokens", type=int, default=320)
    parser.add_argument("--skip-loss-eval", action="store_true")
    parser.add_argument("--save-eval-cases", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--loader-mode", choices=["manual", "bagel_backbone"], default="manual")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--finetune-mode", choices=["lora", "partial_sft", "none"], default="lora")
    parser.add_argument("--train-last-n-layers", type=int, default=4)
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--train-final-norm", action="store_true")
    parser.add_argument("--resume-adapter-path", default=None)
    parser.add_argument("--resume-visual-head-path", default=None)
    parser.add_argument("--stop-on-zero-metrics", action="store_true")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--stage", choices=sorted(STAGE_PRESETS), default="imitation")
    parser.add_argument("--thought-loss-weight", type=float, default=None)
    parser.add_argument("--answer-loss-weight", type=float, default=None)
    parser.add_argument("--think-mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    train_paths = resolve_jsonl_paths(args.train_jsonl, skip_missing=False)
    val_paths = resolve_jsonl_paths(args.val_jsonl, skip_missing=True)
    latent_cache_root = Path(args.latent_cache_root).expanduser() if args.latent_cache_root else None

    thought_weight = args.thought_loss_weight if args.thought_loss_weight is not None else STAGE_PRESETS[args.stage]["thought_loss_weight"]
    answer_weight = args.answer_loss_weight if args.answer_loss_weight is not None else STAGE_PRESETS[args.stage]["answer_loss_weight"]

    model_path = Path(args.model_path).expanduser()
    if args.loader_mode == "bagel_backbone":
        model, tokenizer, new_token_ids, vit_transform, backbone_vae_transform = load_bagel_backbone_for_training(
            model_path,
            distributed_single_gpu=world_size > 1,
        )
        if args.eval_vit_max_image_size is not None or args.eval_vit_min_image_size is not None or args.eval_vit_image_stride is not None:
            raise ValueError("[image_aware] eval vit override is not supported with loader_mode=bagel_backbone")
        eval_vit_transform = vit_transform
    else:
        model, tokenizer = load_full_bagel_for_training(model_path, device=device)
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        if num_new_tokens != 0:
            raise ValueError(f"[image_aware] tokenizer unexpectedly added {num_new_tokens} new tokens during training setup")
        vit_transform = ImageTransform(args.vit_max_image_size, args.vit_min_image_size, args.vit_image_stride)
        eval_vit_transform = ImageTransform(
            args.eval_vit_max_image_size if args.eval_vit_max_image_size is not None else args.vit_max_image_size,
            args.eval_vit_min_image_size if args.eval_vit_min_image_size is not None else args.vit_min_image_size,
            args.eval_vit_image_stride if args.eval_vit_image_stride is not None else args.vit_image_stride,
        )
    if args.bf16 and args.loader_mode == "manual":
        model = cast_model_for_training(model, torch.bfloat16)

    if args.finetune_mode == "lora":
        model, trainable, total = install_lora_on_language_model(
            model,
            target_keywords=args.target_modules,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            resume_adapter_path=args.resume_adapter_path,
        )
    elif args.finetune_mode == "partial_sft":
        model, trainable, total = install_partial_sft_on_language_model(
            model,
            train_last_n_layers=args.train_last_n_layers,
            train_lm_head=args.train_lm_head,
            train_final_norm=args.train_final_norm,
        )
        model = maybe_load_trainable_model_state(model, args.resume_adapter_path, allow_missing=False)
    else:
        for param in model.parameters():
            param.requires_grad = False
        trainable = 0
        total = sum(p.numel() for p in model.parameters())
    head_dtype = next(model.parameters()).dtype
    visual_head = torch.nn.Linear(model.hidden_size, args.latent_target_dim, bias=True).to(device=device, dtype=head_dtype)
    visual_head = maybe_load_visual_head(visual_head, args.resume_visual_head_path, allow_missing=False)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        visual_head = DDP(visual_head, device_ids=[local_rank], find_unused_parameters=True)

    train_dataset = ImageAwareUnderstandingDataset(
        train_paths,
        tokenizer=tokenizer,
        seed=args.seed,
        thought_weight=thought_weight,
        answer_weight=answer_weight,
        latent_cache_root=latent_cache_root,
        latent_target_dim=args.latent_target_dim,
        max_samples=args.max_samples,
        sample_selection=args.sample_selection,
        level_sample_weights=parse_level_sample_weights(args.level_sample_weights),
        think_mode=args.think_mode,
    )
    val_dataset = ImageAwareUnderstandingDataset(
        val_paths,
        tokenizer=tokenizer,
        seed=args.seed + 1,
        thought_weight=thought_weight,
        answer_weight=answer_weight,
        latent_cache_root=latent_cache_root,
        latent_target_dim=args.latent_target_dim,
        max_samples=None,
        sample_selection="random",
        think_mode=args.think_mode,
    ) if val_paths else None

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 and val_dataset is not None else None
    collator = ImageAwareCollator(tokenizer, vit_transform=vit_transform, latent_target_dim=args.latent_target_dim, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, collate_fn=collator, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, collate_fn=collator, num_workers=args.num_workers) if val_dataset is not None else None

    trainable_params = [p for p in list(model.parameters()) + list(visual_head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay) if trainable_params else None
    scaler_enabled = args.bf16

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_main(
        f"[image_aware] train_rows={len(train_dataset)} val_rows={len(val_dataset) if val_dataset is not None else 0} "
        f"trainable={trainable} trainable_pct={100.0 * trainable / max(total, 1):.4f}"
    )

    best_metric = -math.inf if best_metric_is_higher(args.save_best_metric) else math.inf
    best_metrics: dict[str, float] | None = None
    format_floor: float | None = None
    no_improve_evals = 0
    global_step = 0
    train_start_time = time.monotonic()
    last_step_time = train_start_time
    stop_training = False

    fixed_generate_candidates: list[dict[str, Any]] | None = None
    if val_loader is not None and args.eval_generate_max_samples > 0:
        fixed_generate_candidates = collect_eval_generate_candidates(
            val_loader,
            max_generate_samples=args.eval_generate_max_samples,
            generate_sampling=args.eval_generate_sampling,
        )
        if is_main_process():
            log_main(
                f"[image_aware] fixed_eval_generate_set size={len(fixed_generate_candidates)} "
                f"sampling={args.eval_generate_sampling}"
            )

    if args.initial_eval and val_loader is not None:
        metrics = evaluate_image_aware_model(
            model,
            tokenizer,
            new_token_ids,
            vit_transform,
            eval_vit_transform,
            val_loader,
            device,
            visual_head,
            visual_loss_weight=args.visual_loss_weight,
            max_generate_samples=args.eval_generate_max_samples,
            max_generate_new_tokens=args.eval_generate_max_new_tokens,
            generate_sampling=args.eval_generate_sampling,
            fixed_generate_candidates=fixed_generate_candidates,
            skip_loss_eval=args.skip_loss_eval,
            case_output_path=(output_dir / "eval_cases" / "step_0.jsonl") if args.save_eval_cases else None,
        )
        log_main(f"[image_aware] eval step=0 metrics={json.dumps(metrics, sort_keys=True)}")
        initial_metric_value = best_metric_value(metrics, args.save_best_metric) if is_main_process() else best_metric
        best_metric = broadcast_float_from_main(initial_metric_value, device)
        best_metrics = metrics
        if args.best_require_format_no_degrade and is_main_process():
            format_floor = float(metrics.get("val_format_accuracy", 0.0))
        format_floor_serialized = float(format_floor) if format_floor is not None else -1.0
        format_floor_serialized = broadcast_float_from_main(format_floor_serialized, device)
        format_floor = None if format_floor_serialized < 0.0 else format_floor_serialized
        if is_main_process():
            save_checkpoint_bundle(model, visual_head, output_dir / f"adapter_{args.stage}_best", global_step=0, stage=args.stage, kind="best", finetune_mode=args.finetune_mode, metrics=metrics)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        zero_stop = should_stop_for_zero_metrics(metrics) if is_main_process() else False
        zero_stop = broadcast_bool_from_main(zero_stop, device)
        if args.stop_on_zero_metrics and zero_stop:
            log_main("[image_aware] zero_metric_stop_triggered step=0")
            stop_training = True

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            model.train()
            unwrap_model(visual_head).train()
            batch = move_batch_to_device(batch, device)
            hidden_states, ce = forward_image_aware_understanding(unwrap_model(model), batch)
            hidden_states = unpack_hidden_states(hidden_states, batch["sample_lens"])
            lm_loss = compute_weighted_ce(ce, batch["ce_loss_weights"])
            visual_loss = compute_visual_summary_loss(
                hidden_states,
                batch["visual_spans"],
                batch["visual_targets"],
                batch["visual_mask"],
                visual_head,
            )
            loss = lm_loss + args.visual_loss_weight * visual_loss
            loss.backward()
            if optimizer is None:
                raise RuntimeError("optimizer is None during training; use --epochs 0 for baseline mode")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            now = time.monotonic()
            step_seconds = now - last_step_time
            total_seconds = now - train_start_time
            last_step_time = now
            if is_main_process():
                print(
                    f"[image_aware] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.6f} lm_loss={lm_loss.item():.6f} visual_loss={visual_loss.item():.6f} "
                    f"step_seconds={step_seconds:.2f} total_seconds={total_seconds:.2f} {cuda_memory_summary(device)}",
                    flush=True,
                )

            if val_loader is not None and args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                if val_sampler is not None:
                    val_sampler.set_epoch(epoch)
                metrics = evaluate_image_aware_model(
                    model,
                    tokenizer,
                    new_token_ids,
                    vit_transform,
                    eval_vit_transform,
                    val_loader,
                    device,
                    visual_head,
                    visual_loss_weight=args.visual_loss_weight,
                    max_generate_samples=args.eval_generate_max_samples,
                    max_generate_new_tokens=args.eval_generate_max_new_tokens,
                    generate_sampling=args.eval_generate_sampling,
                    fixed_generate_candidates=fixed_generate_candidates,
                    skip_loss_eval=args.skip_loss_eval,
                    case_output_path=(output_dir / "eval_cases" / f"step_{global_step}.jsonl") if args.save_eval_cases else None,
                )
                log_main(f"[image_aware] eval step={global_step} metrics={json.dumps(metrics, sort_keys=True)}")
                log_main(f"[image_aware] eval step={global_step} {cuda_memory_summary(device)}")
                metric_value = best_metric_value(metrics, args.save_best_metric) if is_main_process() else best_metric
                metric_value = broadcast_float_from_main(metric_value, device)
                format_gate_ok = True
                if args.best_require_format_no_degrade:
                    if format_floor is None and is_main_process():
                        format_floor = float(metrics.get("val_format_accuracy", 0.0))
                    format_floor_serialized = float(format_floor) if format_floor is not None else -1.0
                    format_floor_serialized = broadcast_float_from_main(format_floor_serialized, device)
                    format_floor = None if format_floor_serialized < 0.0 else format_floor_serialized
                    local_gate_ok = False
                    if is_main_process():
                        local_gate_ok = (
                            float(metrics.get("val_format_accuracy", 0.0))
                            >= float(format_floor) - max(args.best_format_tolerance, 0.0)
                        )
                    format_gate_ok = broadcast_bool_from_main(local_gate_ok, device)
                if best_metric_is_higher(args.save_best_metric):
                    metric_better = metric_value > best_metric + max(args.min_delta, 0.0)
                else:
                    metric_better = metric_value < best_metric - max(args.min_delta, 0.0)
                improved = format_gate_ok and metric_better
                if improved:
                    best_metric = metric_value
                    best_metrics = metrics
                    no_improve_evals = 0
                    if is_main_process():
                        save_checkpoint_bundle(model, visual_head, output_dir / f"adapter_{args.stage}_best", global_step=global_step, stage=args.stage, kind="best", finetune_mode=args.finetune_mode, metrics=metrics)
                else:
                    no_improve_evals += 1
                    if args.best_require_format_no_degrade and not format_gate_ok and is_main_process():
                        log_main(
                            f"[image_aware] best_gate_blocked step={global_step} "
                            f"val_format={float(metrics.get('val_format_accuracy', 0.0)):.6f} "
                            f"format_floor={float(format_floor or 0.0):.6f} tol={max(args.best_format_tolerance, 0.0):.6f}"
                        )
                zero_stop = should_stop_for_zero_metrics(metrics) if is_main_process() else False
                zero_stop = broadcast_bool_from_main(zero_stop, device)
                if args.stop_on_zero_metrics and zero_stop:
                    log_main(f"[image_aware] zero_metric_stop_triggered step={global_step}")
                    stop_training = True
                if (
                    args.early_stop_patience > 0
                    and global_step >= args.min_steps_before_early_stop
                    and no_improve_evals >= args.early_stop_patience
                ):
                    stop_training = True
            if stop_training:
                break
        if stop_training:
            break

    if is_main_process() and args.save_last:
        save_checkpoint_bundle(
            model,
            visual_head,
            output_dir / f"adapter_{args.stage}_last",
            global_step=global_step,
            stage=args.stage,
            kind="last",
            finetune_mode=args.finetune_mode,
            metrics=best_metrics,
        )
        write_checkpoint_metadata(
            output_dir,
            {
                "global_step": global_step,
                "best_metrics": best_metrics,
                "train_rows": len(train_dataset),
                "val_rows": len(val_dataset) if val_dataset is not None else 0,
            },
        )
    if torch.cuda.is_available():
        print(f"[image_aware] rank={rank} final {cuda_memory_summary(device)}", flush=True)
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
