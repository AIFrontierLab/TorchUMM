from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

BAGEL_ROOT = add_bagel_code_to_sys_path(REPO_ROOT)

from data.data_utils import add_special_tokens, get_flattened_position_ids_extrapolate, prepare_attention_mask_per_sample
from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from .native_targets import build_native_answer_target
from .train import _latent_paths_from_row, build_context_text, extract_trajectory_blocks, load_latent_summary, normalize_answer_text


@dataclass
class NativeAnswerExample:
    row_id: str
    packed: dict[str, Any]
    text_token_count: int
    image_token_count: int
    patch_shape: tuple[int, int]
    latent_path: str


def build_native_answer_pieces(row: dict[str, Any], latent_cache_root: Path | None, latent_target_dim: int) -> tuple[list[str], list[tuple[int, torch.Tensor]]]:
    parts = [build_context_text(row)]
    visual_targets: list[tuple[int, torch.Tensor]] = []
    trajectory = str(row.get("trajectory") or "").strip()
    latent_index = row.get("latent_index") or {}
    for tag, body in extract_trajectory_blocks(trajectory):
        if tag == "A":
            continue
        if tag == "TU":
            parts.append(f"Understanding:\n{body.strip()}\n\n")
        elif tag == "TR":
            parts.append(f"Reasoning:\n{body.strip()}\n\n")
        elif tag == "VC":
            parts.append("Visual:\n<image>\n\n")
            latent_paths = _latent_paths_from_row(row, "vc")
            summary = load_latent_summary(latent_cache_root, latent_paths, latent_target_dim=latent_target_dim) if latent_cache_root is not None and latent_paths else None
            if summary is not None:
                visual_targets.append((len(parts) - 1, summary))
        elif tag == "VH":
            parts.append("Hypothesis:\n<image>\n\n")
            latent_paths = _latent_paths_from_row(row, "vh")
            summary = load_latent_summary(latent_cache_root, latent_paths, latent_target_dim=latent_target_dim) if latent_cache_root is not None and latent_paths else None
            if summary is not None:
                visual_targets.append((len(parts) - 1, summary))
        else:
            parts.append(f"{tag}:\n{body.strip()}\n\n")
    parts.append("Answer:\n")
    return parts, visual_targets


def build_native_answer_text(row: dict[str, Any]) -> str:
    parts, _ = build_native_answer_pieces(row, None, latent_target_dim=16)
    return "".join(parts)


def load_tokenizer_with_special_tokens(model_path: Path) -> tuple[Qwen2Tokenizer, dict[str, int]]:
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    return tokenizer, new_token_ids


def build_single_native_answer_example(
    *,
    row: dict[str, Any],
    model_path: Path,
    latent_cache_root: Path,
    max_num_patches_per_side: int = 64,
    latent_patch_size: int = 2,
    vae_downsample: int = 16,
    latent_target_dim: int = 16,
) -> NativeAnswerExample:
    tokenizer, new_token_ids = load_tokenizer_with_special_tokens(model_path)
    native_target = build_native_answer_target(latent_cache_root, row, latent_patch_size=latent_patch_size)
    latent = native_target["patchified_latent"]
    patch_h, patch_w = native_target["patch_shape"]
    text_pieces, visual_target_refs = build_native_answer_pieces(row, latent_cache_root, latent_target_dim=latent_target_dim)
    piece_token_ids = [tokenizer.encode(piece, add_special_tokens=False) for piece in text_pieces]
    text_ids = [token_id for piece in piece_token_ids for token_id in piece]
    # The first piece is the prompt/context. It should condition generation but not
    # contribute CE loss; all subsequent pieces are target reasoning/answer text.
    text_target_weights = [
        weight
        for piece_idx, piece_ids in enumerate(piece_token_ids)
        for weight in ([0.0 if piece_idx == 0 else 1.0] * len(piece_ids))
    ]

    start_of_image = new_token_ids["start_of_image"]
    end_of_image = new_token_ids["end_of_image"]
    bos_token_id = new_token_ids["bos_token_id"]
    eos_token_id = new_token_ids["eos_token_id"]

    packed_text_ids: list[int] = []
    packed_text_indexes: list[int] = []
    packed_label_ids: list[int] = []
    ce_loss_indexes: list[int] = []
    ce_loss_weights: list[float] = []
    packed_position_ids: list[int] = []
    packed_vae_token_indexes: list[int] = []
    mse_loss_indexes: list[int] = []

    curr = 0
    shifted_text_ids = [bos_token_id] + text_ids
    packed_text_ids.extend(shifted_text_ids)
    packed_text_indexes.extend(range(curr, curr + len(shifted_text_ids)))
    ce_loss_indexes.extend(range(curr, curr + len(shifted_text_ids)))
    ce_loss_weights.extend(text_target_weights + [1.0])
    packed_label_ids.extend(text_ids + [start_of_image])
    packed_position_ids.extend(range(len(shifted_text_ids)))
    curr += len(shifted_text_ids)
    text_split_len = len(shifted_text_ids)

    visual_summary_spans: list[list[int]] = []
    visual_summary_targets: list[torch.Tensor] = []
    offset = 1  # BOS token before the first text piece.
    piece_spans: list[tuple[int, int]] = []
    for piece_ids in piece_token_ids:
        start = offset
        end = offset + len(piece_ids)
        piece_spans.append((start, end))
        offset = end
    for piece_idx, summary in visual_target_refs:
        start, end = piece_spans[piece_idx]
        if end <= start:
            continue
        visual_summary_spans.append([start, end])
        visual_summary_targets.append(summary.float())

    packed_text_ids.append(start_of_image)
    packed_text_indexes.append(curr)
    ce_loss_indexes.append(curr)
    ce_loss_weights.append(1.0)
    packed_label_ids.append(start_of_image)
    packed_position_ids.append(len(shifted_text_ids))
    curr += 1

    image_token_start = curr
    image_token_count = patch_h * patch_w
    packed_vae_token_indexes.extend(range(image_token_start, image_token_start + image_token_count))
    mse_loss_indexes.extend(range(image_token_start, image_token_start + image_token_count))
    packed_position_ids.extend([len(shifted_text_ids)] * image_token_count)
    curr += image_token_count

    packed_text_ids.append(end_of_image)
    packed_text_indexes.append(curr)
    ce_loss_indexes.append(curr)
    ce_loss_weights.append(1.0)
    packed_label_ids.append(eos_token_id)
    packed_position_ids.append(len(shifted_text_ids))
    curr += 1
    image_split_len = image_token_count + 2

    packed_latent_position_ids = get_flattened_position_ids_extrapolate(
        native_target["transformed_size"][0],
        native_target["transformed_size"][1],
        vae_downsample,
        max_num_patches_per_side=max_num_patches_per_side,
    )
    packed_timesteps = torch.randn(image_token_count, dtype=torch.float32)
    padded_latent = native_target["latent"].unsqueeze(0)

    packed = {
        "sequence_length": curr,
        "sample_lens": [curr],
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "packed_label_ids": torch.tensor(packed_label_ids, dtype=torch.long),
        "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long),
        "ce_loss_weights": torch.tensor(ce_loss_weights, dtype=torch.float32),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        "nested_attention_masks": [
            prepare_attention_mask_per_sample([text_split_len, image_split_len], ["causal", "noise"])
        ],
        "split_lens": [text_split_len, image_split_len],
        "attn_modes": ["causal", "noise"],
        "padded_latent": padded_latent,
        "patchified_vae_latent_shapes": [native_target["patch_shape"]],
        "packed_latent_position_ids": packed_latent_position_ids,
        "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
        "packed_timesteps": packed_timesteps,
        "mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long),
        "visual_summary_spans": torch.tensor(visual_summary_spans, dtype=torch.long) if visual_summary_spans else torch.zeros((0, 2), dtype=torch.long),
        "visual_summary_targets": torch.stack(visual_summary_targets, dim=0) if visual_summary_targets else torch.zeros((0, latent_target_dim), dtype=torch.float32),
    }

    return NativeAnswerExample(
        row_id=str(row.get("id") or ""),
        packed=packed,
        text_token_count=len(shifted_text_ids) + 2,
        image_token_count=image_token_count,
        patch_shape=native_target["patch_shape"],
        latent_path=native_target["latent_path"],
    )
