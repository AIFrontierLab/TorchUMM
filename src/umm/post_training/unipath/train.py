from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

BAGEL_CODE_ROOT = add_bagel_code_to_sys_path(REPO_ROOT)

from data.data_utils import add_special_tokens
from modeling.qwen2.configuration_qwen2 import Qwen2Config
from modeling.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from umm.post_training.unipath.prompts import resolve_level_prompt

TAG_PATTERN = re.compile(r"<(?P<tag>[A-Z]+)>(?P<body>.*?)</(?P=tag)>", re.DOTALL)
IMAGE_PATH_PATTERN = re.compile(r"\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
VISUAL_SEGMENT_TYPE_IDS = {
    "visual": 0,
    "visual_answer": 1,
}
STRUCTURE_TAGS = ("TU", "TR", "VC", "VH", "A")
TAG_LABEL_TO_XML = {
    "Understanding:": "TU",
    "Reasoning:": "TR",
    "Visual:": "VC",
    "Hypothesis:": "VH",
    "Answer:": "A",
}

STAGE_PRESETS = {
    "imitation": {"thought_loss_weight": 1.0, "answer_loss_weight": 1.5},
    "answer_refinement": {"thought_loss_weight": 0.35, "answer_loss_weight": 2.0},
}


@dataclass
class Segment:
    text: str
    loss_weight: float
    segment_type: str
    visual_paths: list[str] = field(default_factory=list)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_jsonl_paths(raw_paths: list[str], skip_missing: bool) -> list[Path]:
    resolved: list[Path] = []
    missing: list[str] = []
    for raw_path in raw_paths:
        matches = sorted(Path(p).expanduser() for p in glob.glob(raw_path))
        if matches:
            resolved.extend(matches)
            continue
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            resolved.append(candidate)
        elif skip_missing:
            missing.append(raw_path)
        else:
            raise FileNotFoundError(raw_path)
    if missing:
        print(f"[unipath] skipped_missing_paths={missing}")
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def normalize_answer_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip()


def is_local_image_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.startswith("http://") or text.startswith("https://"):
        return False
    return IMAGE_PATH_PATTERN.search(text) is not None


def extract_trajectory_blocks(trajectory: str) -> list[tuple[str, str]]:
    return [(m.group("tag"), m.group("body").strip()) for m in TAG_PATTERN.finditer(trajectory or "")]


def _extract_image_paths(block_text: str) -> list[str]:
    image_paths = re.findall(r"<image_path>(.*?)</image_path>", block_text, flags=re.DOTALL)
    if image_paths:
        return [item.strip() for item in image_paths if item.strip()]
    stripped = block_text.strip()
    return [stripped] if stripped else []


def _latent_paths_from_row(row: dict[str, Any], role: str) -> list[str]:
    latent_index = row.get("latent_index")
    if not isinstance(latent_index, dict):
        return []
    items = latent_index.get(role)
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            latent_path = item.get("latent_path")
            if isinstance(latent_path, str) and latent_path:
                out.append(latent_path)
    return out


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "<unknown>")


def row_uses_image_input(row: dict[str, Any]) -> bool:
    return is_local_image_path(row.get("image_path")) or is_local_image_path(row.get("input_image_path"))


def row_uses_visual_reasoning(row: dict[str, Any]) -> bool:
    if is_local_image_path(row.get("vc_image_path")):
        return True
    vh_paths = row.get("vh_image_paths") or []
    return isinstance(vh_paths, list) and any(is_local_image_path(item) for item in vh_paths)


def row_uses_image_answer(row: dict[str, Any]) -> bool:
    if is_local_image_path(row.get("answer_image_path")):
        return True
    if is_local_image_path(row.get("answer")):
        return True
    latent_index = row.get("latent_index") or {}
    return bool(isinstance(latent_index, dict) and latent_index.get("answer"))


def row_has_required_visual_latents(row: dict[str, Any]) -> bool:
    latent_index = row.get("latent_index") or {}
    if not isinstance(latent_index, dict):
        return False
    if is_local_image_path(row.get("vc_image_path")) and not latent_index.get("vc"):
        return False
    vh_paths = row.get("vh_image_paths") or []
    if isinstance(vh_paths, list) and any(is_local_image_path(item) for item in vh_paths) and not latent_index.get("vh"):
        return False
    return True


def validate_rows_for_text_trainer(rows: list[dict[str, Any]], latent_cache_root: Path | None) -> None:
    issues: dict[str, list[str]] = {
        "image_input_not_supported": [],
        "image_answer_not_supported": [],
        "visual_reasoning_missing_latent_index": [],
        "latent_cache_required_for_visual_reasoning": [],
    }
    for row in rows:
        row_id = _row_id(row)
        if row_uses_image_input(row):
            issues["image_input_not_supported"].append(row_id)
        if row_uses_image_answer(row):
            issues["image_answer_not_supported"].append(row_id)
        if row_uses_visual_reasoning(row):
            if latent_cache_root is None:
                issues["latent_cache_required_for_visual_reasoning"].append(row_id)
            elif not row_has_required_visual_latents(row):
                issues["visual_reasoning_missing_latent_index"].append(row_id)
    active = {key: ids for key, ids in issues.items() if ids}
    if not active:
        return
    lines = ["[unipath] incompatible dataset rows detected for text/summarized trainer:"]
    for key, ids in active.items():
        preview = ", ".join(ids[:8])
        suffix = "" if len(ids) <= 8 else f", ... (+{len(ids) - 8} more)"
        lines.append(f"- {key}: {len(ids)} rows [{preview}{suffix}]")
    lines.append(
        "This trainer does not consume raw image_path inputs or native image-answer targets. "
        "Use Bagel inferencer/native image-answer pipeline for those rows."
    )
    raise ValueError("\n".join(lines))


def build_context_prefix(row: dict[str, Any], think_mode: bool = False) -> list[Segment]:
    segments: list[Segment] = []
    _, level_prompt = resolve_level_prompt(row, think_mode=think_mode)
    if level_prompt:
        segments.append(Segment(text=f"{level_prompt}\n\n", loss_weight=0.0, segment_type="context"))
    prompt = str(row.get("prompt") or "").strip()
    question = str(row.get("question") or "").strip()
    if prompt:
        segments.append(Segment(text=f"Prompt:\n{prompt}\n\n", loss_weight=0.0, segment_type="context"))
    if question and question.lower() != "none":
        if question.startswith("Question:"):
            segments.append(Segment(text=f"{question}\n\n", loss_weight=0.0, segment_type="context"))
        else:
            segments.append(Segment(text=f"Question:\n{question}\n\n", loss_weight=0.0, segment_type="context"))
    return segments


def build_context_text(row: dict[str, Any], think_mode: bool = False) -> str:
    return "".join(segment.text for segment in build_context_prefix(row, think_mode=think_mode))


def expected_tag_sequence_for_row(row: dict[str, Any]) -> list[str]:
    trajectory = str(row.get("trajectory") or "").strip()
    sequence = [tag for tag, _ in extract_trajectory_blocks(trajectory) if tag in STRUCTURE_TAGS and tag != "A"]
    if normalize_answer_text(row.get("answer")):
        sequence.append("A")
    return sequence


def row_to_segments(row: dict[str, Any], thought_weight: float, answer_weight: float, think_mode: bool = False) -> list[Segment]:
    segments = build_context_prefix(row, think_mode=think_mode)
    trajectory = str(row.get("trajectory") or "").strip()
    answer = normalize_answer_text(row.get("answer"))

    blocks = extract_trajectory_blocks(trajectory)
    think_segments: list[Segment] = []
    for tag, body in blocks:
        if tag == "A":
            continue
        if tag == "TU":
            think_segments.append(Segment(text=f"Understanding:\n{body}\n\n", loss_weight=thought_weight, segment_type="thought"))
            continue
        if tag == "TR":
            think_segments.append(Segment(text=f"Reasoning:\n{body}\n\n", loss_weight=thought_weight, segment_type="thought"))
            continue
        if tag == "VC":
            think_segments.append(
                Segment(
                    text=f"Visual:\n{body}\n\n",
                    loss_weight=thought_weight,
                    segment_type="visual",
                    visual_paths=_latent_paths_from_row(row, "vc"),
                )
            )
            continue
        if tag == "VH":
            think_segments.append(
                Segment(
                    text=f"Hypothesis:\n{body}\n\n",
                    loss_weight=thought_weight,
                    segment_type="visual",
                    visual_paths=_latent_paths_from_row(row, "vh"),
                )
            )
            continue
        think_segments.append(Segment(text=f"{tag}:\n{body}\n\n", loss_weight=thought_weight, segment_type="thought"))

    if think_mode and think_segments:
        segments.append(Segment(text="<think>\n", loss_weight=thought_weight, segment_type="thought"))
        segments.extend(think_segments)
        segments.append(Segment(text="</think>\n\n", loss_weight=thought_weight, segment_type="thought"))
    else:
        segments.extend(think_segments)

    if answer:
        # Final image answers should not be routed through the visual-summary loss.
        # Intermediate visual supervision is only attached to VC/VH spans.
        segments.append(
            Segment(
                text=f"Answer:\n{answer}",
                loss_weight=answer_weight,
                segment_type="answer",
            )
        )
    return segments


def preview_segments(segments: list[Segment], max_chars: int = 280) -> str:
    preview = "".join(seg.text for seg in segments).strip()
    return preview[:max_chars]


def estimate_segment_text_length(segments: list[Segment]) -> int:
    return sum(len(segment.text) for segment in segments)


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples: int | None,
    sample_selection: str,
    thought_weight: float,
    answer_weight: float,
    think_mode: bool = False,
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
        scored = []
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
        if not full_path.exists():
            continue
        payload = torch.load(full_path, map_location="cpu")
        latent = payload.get("latent")
        if not isinstance(latent, torch.Tensor):
            continue
        summary = latent.float().mean(dim=(-1, -2))
        if summary.numel() != latent_target_dim:
            raise ValueError(f"Expected latent summary dim {latent_target_dim}, got {summary.numel()} from {full_path}")
        vectors.append(summary)
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


class UniPathTokenDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        tokenizer: Qwen2Tokenizer,
        seed: int,
        max_length: int,
        thought_weight: float,
        answer_weight: float,
        latent_cache_root: Path | None,
        latent_target_dim: int,
        max_samples: int | None = None,
        sample_selection: str = "random",
        think_mode: bool = False,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(read_jsonl(path))
        validate_rows_for_text_trainer(rows, latent_cache_root=latent_cache_root)
        rows = select_rows(
            rows,
            seed=seed,
            max_samples=max_samples,
            sample_selection=sample_selection,
            thought_weight=thought_weight,
            answer_weight=answer_weight,
            think_mode=think_mode,
        )

        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        self.examples: list[dict[str, Any]] = []
        for row in rows:
            segments = row_to_segments(row, thought_weight=thought_weight, answer_weight=answer_weight, think_mode=think_mode)
            if not segments:
                continue
            input_ids: list[int] = []
            labels: list[int] = []
            loss_weights: list[float] = []
            visual_spans: list[list[int]] = []
            visual_targets: list[list[float]] = []
            visual_types: list[int] = []
            if bos_token_id is not None:
                input_ids.append(bos_token_id)
                labels.append(-100)
                loss_weights.append(0.0)
            for segment in segments:
                piece_ids = tokenizer.encode(segment.text, add_special_tokens=False)
                if not piece_ids:
                    continue
                seg_start = len(input_ids)
                input_ids.extend(piece_ids)
                seg_end = len(input_ids)
                if segment.loss_weight > 0:
                    labels.extend(piece_ids)
                    loss_weights.extend([segment.loss_weight] * len(piece_ids))
                else:
                    labels.extend([-100] * len(piece_ids))
                    loss_weights.extend([0.0] * len(piece_ids))
                if latent_cache_root is not None and segment.visual_paths:
                    summary = load_latent_summary(latent_cache_root, segment.visual_paths, latent_target_dim=latent_target_dim)
                    if summary is not None:
                        visual_spans.append([seg_start, seg_end])
                        visual_targets.append(summary.tolist())
                        visual_types.append(VISUAL_SEGMENT_TYPE_IDS.get(segment.segment_type, 0))
            if eos_token_id is not None:
                input_ids.append(eos_token_id)
                labels.append(labels[-1] if labels and labels[-1] != -100 else -100)
                loss_weights.append(loss_weights[-1] if loss_weights else 0.0)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                loss_weights = loss_weights[:max_length]
                kept_spans: list[list[int]] = []
                kept_targets: list[list[float]] = []
                kept_types: list[int] = []
                for span, target, visual_type in zip(visual_spans, visual_targets, visual_types):
                    if span[0] >= max_length:
                        continue
                    kept_spans.append([span[0], min(span[1], max_length)])
                    kept_targets.append(target)
                    kept_types.append(visual_type)
                visual_spans = kept_spans
                visual_targets = kept_targets
                visual_types = kept_types
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_weights": loss_weights,
                    "preview": preview_segments(segments),
                    "visual_spans": visual_spans,
                    "visual_targets": visual_targets,
                    "visual_types": visual_types,
                    "row_id": str(row.get("id") or ""),
                    "reasoning_level": resolve_level_prompt(row, think_mode=think_mode)[0],
                    "answer_text": normalize_answer_text(row.get("answer")),
                    "answer_type": str(row.get("answer_type") or ""),
                    "context_text": build_context_text(row, think_mode=think_mode),
                    "expected_tag_sequence": expected_tag_sequence_for_row(row),
                    "seq_len": len(input_ids),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class WeightedCollator:
    def __init__(self, tokenizer: Qwen2Tokenizer, latent_target_dim: int) -> None:
        self.tokenizer = tokenizer
        self.latent_target_dim = latent_target_dim

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(item["input_ids"]) for item in batch)
        max_visual = max((len(item["visual_spans"]) for item in batch), default=0)

        input_ids = []
        attention_mask = []
        labels = []
        loss_weights = []
        previews: list[str] = []
        reasoning_levels: list[str | None] = []
        answer_types: list[str] = []
        answer_texts: list[str] = []
        context_texts: list[str] = []
        row_ids: list[str] = []
        expected_tag_sequences: list[list[str]] = []
        visual_spans = []
        visual_targets = []
        visual_mask = []
        visual_types = []

        for item in batch:
            cur_len = len(item["input_ids"])
            pad_len = max_len - cur_len
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * cur_len + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            loss_weights.append(item["loss_weights"] + [0.0] * pad_len)
            previews.append(str(item["preview"]))
            reasoning_levels.append(item.get("reasoning_level"))
            answer_types.append(str(item.get("answer_type") or ""))
            answer_texts.append(str(item.get("answer_text") or ""))
            context_texts.append(str(item.get("context_text") or ""))
            row_ids.append(str(item.get("row_id") or ""))
            expected_tag_sequences.append(list(item.get("expected_tag_sequence") or []))

            cur_visual_spans = [list(span) for span in item["visual_spans"]]
            cur_visual_targets = [list(target) for target in item["visual_targets"]]
            cur_visual_mask = [1.0] * len(cur_visual_spans)
            cur_visual_types = list(item.get("visual_types", []))
            while len(cur_visual_spans) < max_visual:
                cur_visual_spans.append([-1, -1])
                cur_visual_targets.append([0.0] * self.latent_target_dim)
                cur_visual_mask.append(0.0)
                cur_visual_types.append(-1)
            visual_spans.append(cur_visual_spans)
            visual_targets.append(cur_visual_targets)
            visual_mask.append(cur_visual_mask)
            visual_types.append(cur_visual_types)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "visual_spans": torch.tensor(visual_spans, dtype=torch.long) if max_visual > 0 else torch.zeros((len(batch), 0, 2), dtype=torch.long),
            "visual_targets": torch.tensor(visual_targets, dtype=torch.float32) if max_visual > 0 else torch.zeros((len(batch), 0, self.latent_target_dim), dtype=torch.float32),
            "visual_mask": torch.tensor(visual_mask, dtype=torch.float32) if max_visual > 0 else torch.zeros((len(batch), 0), dtype=torch.float32),
            "visual_types": torch.tensor(visual_types, dtype=torch.long) if max_visual > 0 else torch.zeros((len(batch), 0), dtype=torch.long),
            "preview": previews,
            "reasoning_level": reasoning_levels,
            "answer_type": answer_types,
            "answer_text": answer_texts,
            "context_text": context_texts,
            "row_id": row_ids,
            "expected_tag_sequence": expected_tag_sequences,
        }


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def log_main(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(unwrap_model(module).parameters()).dtype


def install_lora_layers(
    model: torch.nn.Module,
    target_keywords: list[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> tuple[torch.nn.Module, int, int]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("`peft` is required for UniPath LoRA training. Install it in the training environment first.") from exc

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_keywords,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def maybe_load_lora_adapter(model: torch.nn.Module, adapter_path: str | None, allow_missing: bool = False) -> torch.nn.Module:
    if not adapter_path:
        return model
    adapter_dir = Path(adapter_path).expanduser()
    if not adapter_dir.exists():
        if allow_missing:
            print(f"[unipath] resume_adapter_missing={adapter_dir} (ignored)")
            return model
        raise FileNotFoundError(adapter_dir)
    try:
        from peft import PeftModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("`peft` is required for loading a UniPath LoRA adapter.") from exc
    loaded = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    print(f"[unipath] loaded_lora_adapter={adapter_dir}")
    return loaded


def install_or_resume_lora(
    model: torch.nn.Module,
    target_keywords: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    resume_adapter_path: str | None,
    allow_missing_resume: bool,
) -> tuple[torch.nn.Module, int, int]:
    adapter_dir = Path(resume_adapter_path).expanduser() if resume_adapter_path else None
    if adapter_dir is not None and adapter_dir.exists():
        model = maybe_load_lora_adapter(model, str(adapter_dir), allow_missing=False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return model, trainable, total
    if adapter_dir is not None and allow_missing_resume and not adapter_dir.exists():
        print(f"[unipath] resume_adapter_missing={adapter_dir} (falling back to fresh LoRA)")
    model, trainable, total = install_lora_layers(
        model,
        target_keywords=target_keywords,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    return model, trainable, total


def install_partial_sft_layers(
    model: torch.nn.Module,
    *,
    train_last_n_layers: int,
    train_lm_head: bool = True,
    train_final_norm: bool = True,
) -> tuple[torch.nn.Module, int, int]:
    for param in model.parameters():
        param.requires_grad = False

    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    start_layer = max(0, num_layers - max(train_last_n_layers, 0))
    selected_prefixes: list[str] = []
    for layer_idx in range(start_layer, num_layers):
        selected_prefixes.append(f"model.layers.{layer_idx}.")
    if train_final_norm:
        selected_prefixes.append("model.norm.")
    if train_lm_head:
        selected_prefixes.append("lm_head.")

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in selected_prefixes):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


def _build_llm_config(model_path: Path) -> Qwen2Config:
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
    llm_config.freeze_und = True
    return llm_config


def _load_language_model_from_ema(model_path: Path, llm_config: Qwen2Config) -> Qwen2ForCausalLM:
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise RuntimeError("`safetensors` is required to load the Bagel language-model weights.") from exc
    try:
        from accelerate.utils.modeling import set_module_tensor_to_device
    except ModuleNotFoundError as exc:
        raise RuntimeError("`accelerate` with `set_module_tensor_to_device` is required for streaming LM weight load.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Non-dry-run UniPath training currently requires CUDA so LM weights can be loaded without CPU OOM.")

    ema_path = model_path / "ema.safetensors"
    if not ema_path.exists():
        raise FileNotFoundError(ema_path)

    device = torch.device("cuda", torch.cuda.current_device())
    with init_empty_weights():
        model = Qwen2ForCausalLM(llm_config)
    model = model.to_empty(device=device)
    model_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    with safe_open(str(ema_path), framework="pt", device=0) as f:
        for key in f.keys():
            if not key.startswith("language_model."):
                continue
            model_key = key[len("language_model."):]
            if model_key not in model_keys:
                continue
            tensor = f.get_tensor(key)
            set_module_tensor_to_device(model, model_key, device, value=tensor)
            loaded_keys.add(model_key)
            del tensor

    missing = sorted(model_keys - loaded_keys)
    unexpected: list[str] = []
    torch.cuda.empty_cache()
    print(f"[unipath] loaded_language_model_from={ema_path}")
    if missing:
        print(f"[unipath] missing_lm_keys={len(missing)}")
    if unexpected:
        print(f"[unipath] unexpected_lm_keys={len(unexpected)}")
    return model


def build_language_model(model_path: Path, meta_only: bool = False) -> tuple[Qwen2ForCausalLM, Qwen2Tokenizer]:
    llm_config = _build_llm_config(model_path)
    if meta_only:
        with init_empty_weights():
            model = Qwen2ForCausalLM(llm_config)
    else:
        model = _load_language_model_from_ema(model_path, llm_config)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, _, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
    return model, tokenizer


def weighted_language_model_loss(logits: torch.Tensor, labels: torch.Tensor, loss_weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    flat_weights = shift_weights.view(-1)
    valid = (shift_labels.view(-1) != -100).to(flat_weights.dtype)
    denom = (flat_weights * valid).sum().clamp_min(1e-8)
    return (flat_loss * flat_weights).sum() / denom


def build_bagel_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    mask_value = torch.finfo(dtype).min
    causal = torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len).clone()
    key_padding = (attention_mask[:, None, None, :] == 0)
    causal = causal.masked_fill(key_padding, mask_value)
    return causal


def compute_visual_summary_loss(
    hidden_states: torch.Tensor,
    visual_spans: torch.Tensor,
    visual_targets: torch.Tensor,
    visual_mask: torch.Tensor,
    visual_head: torch.nn.Module,
    visual_types: torch.Tensor | None = None,
    allowed_types: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if visual_spans.numel() == 0:
        dummy = next(unwrap_model(visual_head).parameters()).sum() * 0.0
        return hidden_states.sum() * 0.0 + dummy
    pooled_states = []
    pooled_targets = []
    for batch_idx in range(hidden_states.size(0)):
        for item_idx in range(visual_spans.size(1)):
            if visual_mask[batch_idx, item_idx] <= 0:
                continue
            if allowed_types is not None and visual_types is not None:
                current_type = int(visual_types[batch_idx, item_idx].item())
                if current_type not in allowed_types:
                    continue
            start = int(visual_spans[batch_idx, item_idx, 0].item())
            end = int(visual_spans[batch_idx, item_idx, 1].item())
            if start < 0 or end <= start:
                continue
            pooled_states.append(hidden_states[batch_idx, start:end].mean(dim=0))
            pooled_targets.append(visual_targets[batch_idx, item_idx])
    if not pooled_states:
        dummy = next(unwrap_model(visual_head).parameters()).sum() * 0.0
        return hidden_states.sum() * 0.0 + dummy
    pred = visual_head(torch.stack(pooled_states, dim=0))
    target = torch.stack(pooled_targets, dim=0).to(device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred, target)


def normalize_metric_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"^\(?([a-z])\)?[\s:.-]+", "", cleaned)
    cleaned = re.sub(r"[^\w\s./-]", "", cleaned)
    return cleaned.strip()


CHOICE_LINE_PATTERN = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$", re.MULTILINE)
LEADING_CHOICE_PATTERN = re.compile(r"^\(?([A-Z])\)?(?:[\s:.\-]+|$)", re.IGNORECASE)


def parse_question_choices(question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for match in CHOICE_LINE_PATTERN.finditer(question or ""):
        choices[match.group(1).upper()] = match.group(2).strip()
    return choices


def extract_choice_letter(text: str) -> str:
    match = LEADING_CHOICE_PATTERN.match((text or "").strip())
    if not match:
        return ""
    return match.group(1).upper()


def resolve_answer_to_choice_letter(answer: str, question: str) -> str:
    choices = parse_question_choices(question)
    if not choices:
        return ""
    letter = extract_choice_letter(answer)
    if letter and letter in choices:
        return letter
    normalized_answer = normalize_metric_text(answer)
    for key, value in choices.items():
        if normalize_metric_text(value) == normalized_answer:
            return key
    return ""


def answers_match(expected_answer: str, actual_answer: str, question: str) -> bool:
    expected = normalize_metric_text(expected_answer)
    actual = normalize_metric_text(actual_answer)
    if expected and actual and expected == actual:
        return True

    expected_choice = resolve_answer_to_choice_letter(expected_answer, question)
    actual_choice = resolve_answer_to_choice_letter(actual_answer, question)
    if expected_choice and actual_choice and expected_choice == actual_choice:
        return True

    choices = parse_question_choices(question)
    if choices and actual_choice:
        resolved_actual = normalize_metric_text(choices[actual_choice])
        if expected and resolved_actual == expected:
            return True
    if choices and expected_choice:
        resolved_expected = normalize_metric_text(choices[expected_choice])
        if actual and resolved_expected == actual:
            return True
    return False


def extract_answer_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r"Answer:\s*(.*)", text, flags=re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip()
    for heading in ("Understanding:", "Reasoning:", "Visual:", "Hypothesis:"):
        if heading in answer:
            answer = answer.split(heading)[0].strip()
    return answer


def extract_structural_tag_sequence(text: str) -> list[str]:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    xml_matches = [m.group("tag") for m in TAG_PATTERN.finditer(text or "") if m.group("tag") in STRUCTURE_TAGS]
    if xml_matches:
        return xml_matches
    sequence: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        for heading, tag in TAG_LABEL_TO_XML.items():
            if stripped == heading or stripped.startswith(f"{heading} "):
                sequence.append(tag)
                break
    return sequence


def format_matches_expected(text: str, expected_sequence: list[str]) -> bool:
    return extract_structural_tag_sequence(text) == expected_sequence


def evaluate_model(
    model: torch.nn.Module,
    visual_head: torch.nn.Module,
    loader: DataLoader,
    tokenizer: Qwen2Tokenizer,
    device: torch.device,
    visual_loss_weight: float,
    max_generate_samples: int = 0,
    max_generate_new_tokens: int = 192,
    case_output_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    visual_head.eval()
    total_loss = torch.zeros((), device=device)
    total_lm = torch.zeros((), device=device)
    total_visual = torch.zeros((), device=device)
    total_steps = torch.zeros((), device=device)
    level_loss_sum: dict[str, float] = {}
    level_count: dict[str, int] = {}
    gen_candidates: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            batch_attention_mask = build_bagel_attention_mask(
                batch["attention_mask"].to(device),
                dtype=module_dtype(model),
            )
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch_attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            lm_loss = weighted_language_model_loss(
                logits=outputs.logits,
                labels=batch["labels"].to(device),
                loss_weights=batch["loss_weights"].to(device),
            )
            visual_loss = compute_visual_summary_loss(
                hidden_states=outputs.hidden_states[-1],
                visual_spans=batch["visual_spans"].to(device),
                visual_targets=batch["visual_targets"].to(device),
                visual_mask=batch["visual_mask"].to(device),
                visual_head=visual_head,
                visual_types=batch["visual_types"].to(device),
            )
            loss = lm_loss + visual_loss_weight * visual_loss
            total_loss += loss.detach()
            total_lm += lm_loss.detach()
            total_visual += visual_loss.detach()
            total_steps += 1
            for level in batch.get("reasoning_level", []):
                key = str(level or "unknown")
                level_loss_sum[key] = level_loss_sum.get(key, 0.0) + float(loss.item())
                level_count[key] = level_count.get(key, 0) + 1
            if max_generate_samples > 0:
                for idx, answer_type in enumerate(batch.get("answer_type", [])):
                    if len(gen_candidates) >= max_generate_samples:
                        break
                    if str(answer_type).lower() == "image":
                        continue
                    gen_candidates.append(
                        {
                            "reasoning_level": str(batch["reasoning_level"][idx] or "unknown"),
                            "answer_text": str(batch["answer_text"][idx] or ""),
                            "context_text": str(batch["context_text"][idx] or ""),
                            "expected_tag_sequence": list(batch["expected_tag_sequence"][idx]),
                            "row_id": str(batch["row_id"][idx] or ""),
                        }
                    )
    total_loss = reduce_mean(total_loss)
    total_lm = reduce_mean(total_lm)
    total_visual = reduce_mean(total_visual)
    total_steps = reduce_mean(total_steps)
    denom = max(float(total_steps.item()), 1.0)
    metrics: dict[str, float] = {
        "val_loss": float(total_loss.item() / denom),
        "val_lm_loss": float(total_lm.item() / denom),
        "val_visual_loss": float(total_visual.item() / denom),
    }
    for level, count in sorted(level_count.items()):
        metrics[f"val_loss_{level}"] = level_loss_sum[level] / max(count, 1)

    if max_generate_samples > 0 and is_main_process() and gen_candidates:
        format_hits = 0
        exact_hits = 0
        per_level_hits: dict[str, dict[str, int]] = {}
        generated_cases: list[dict[str, Any]] = []
        for item in gen_candidates:
            encoded = tokenizer(item["context_text"], return_tensors="pt")
            prompt_ids = encoded.input_ids.to(device)
            prompt_attention_mask = encoded.attention_mask.to(device)
            generated = unwrap_model(model).generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_generate_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(generated[0][prompt_ids.shape[1]:], skip_special_tokens=True)
            level = item["reasoning_level"]
            bucket = per_level_hits.setdefault(level, {"samples": 0, "format": 0, "exact": 0})
            bucket["samples"] += 1
            if format_matches_expected(generated_text, item["expected_tag_sequence"]):
                format_hits += 1
                bucket["format"] += 1
            actual_answer_raw = extract_answer_text(generated_text)
            if answers_match(item["answer_text"], actual_answer_raw, item["context_text"]):
                exact_hits += 1
                bucket["exact"] += 1
            generated_cases.append(
                {
                    "row_id": item["row_id"],
                    "reasoning_level": level,
                    "expected_tag_sequence": item["expected_tag_sequence"],
                    "generated_tag_sequence": extract_structural_tag_sequence(generated_text),
                    "format_ok": format_matches_expected(generated_text, item["expected_tag_sequence"]),
                    "expected_answer": item["answer_text"],
                    "generated_answer": actual_answer_raw,
                    "answer_ok": answers_match(item["answer_text"], actual_answer_raw, item["context_text"]),
                    "generated_text": generated_text,
                }
            )
        metrics["val_format_accuracy"] = format_hits / max(len(gen_candidates), 1)
        metrics["val_answer_accuracy"] = exact_hits / max(len(gen_candidates), 1)
        for level, bucket in sorted(per_level_hits.items()):
            level_samples = max(bucket["samples"], 1)
            metrics[f"val_format_accuracy_{level}"] = bucket["format"] / level_samples
            metrics[f"val_answer_accuracy_{level}"] = bucket["exact"] / level_samples
        if case_output_path is not None:
            case_output_path.parent.mkdir(parents=True, exist_ok=True)
            with case_output_path.open("w", encoding="utf-8") as f:
                for case in generated_cases:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
    return metrics


def save_training_artifacts(
    model: torch.nn.Module,
    visual_head: torch.nn.Module,
    output_dir: Path,
    *,
    finetune_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    if finetune_mode == "lora":
        if hasattr(raw_model, "language_model") and hasattr(raw_model.language_model, "save_pretrained"):
            raw_model.language_model.save_pretrained(str(output_dir))
            adapter_meta = {
                "adapter_scope": "language_model",
                "wrapped_model_type": raw_model.__class__.__name__,
            }
            (output_dir / "adapter_scope.json").write_text(
                json.dumps(adapter_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            raw_model.save_pretrained(str(output_dir))
    else:
        trainable_state = {
            name: param.detach().cpu()
            for name, param in raw_model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state, output_dir / "trainable_model_state.pt")
    torch.save(unwrap_model(visual_head).state_dict(), output_dir / "visual_head.pt")


def maybe_load_visual_head(visual_head: torch.nn.Module, path: str | None, allow_missing: bool = False) -> torch.nn.Module:
    if not path:
        return visual_head
    head_path = Path(path).expanduser()
    if not head_path.exists():
        if allow_missing:
            print(f"[unipath] resume_visual_head_missing={head_path} (ignored)")
            return visual_head
        raise FileNotFoundError(head_path)
    state_dict = torch.load(head_path, map_location="cpu")
    visual_head.load_state_dict(state_dict)
    print(f"[unipath] loaded_visual_head={head_path}")
    return visual_head


def maybe_load_trainable_model_state(model: torch.nn.Module, path: str | None, allow_missing: bool = False) -> torch.nn.Module:
    if not path:
        return model
    state_path = Path(path).expanduser()
    if state_path.is_dir():
        state_path = state_path / "trainable_model_state.pt"
    if not state_path.exists():
        if allow_missing:
            print(f"[unipath] resume_trainable_state_missing={state_path} (ignored)")
            return model
        raise FileNotFoundError(state_path)
    state_dict = torch.load(state_path, map_location="cpu")
    named_params = dict(unwrap_model(model).named_parameters())
    loaded = 0
    skipped: list[str] = []
    with torch.no_grad():
        for name, tensor in state_dict.items():
            param = named_params.get(name)
            if param is None or tuple(param.shape) != tuple(tensor.shape):
                skipped.append(name)
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded += 1
    print(f"[unipath] loaded_trainable_state={state_path} tensors={loaded} skipped={len(skipped)}")
    if skipped:
        print(f"[unipath] skipped_trainable_state_sample={skipped[:8]}")
    return model


def write_checkpoint_metadata(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoint_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_checkpoint_bundle(
    model: torch.nn.Module,
    visual_head: torch.nn.Module,
    output_dir: Path,
    *,
    global_step: int,
    stage: str,
    kind: str,
    finetune_mode: str,
    metrics: dict[str, float] | None = None,
) -> None:
    save_training_artifacts(model, visual_head, output_dir, finetune_mode=finetune_mode)
    payload: dict[str, Any] = {
        "global_step": global_step,
        "stage": stage,
        "kind": kind,
        "finetune_mode": finetune_mode,
    }
    if metrics:
        payload["metrics"] = metrics
    write_checkpoint_metadata(output_dir, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniPath two-stage LoRA SFT scaffold for Bagel.")
    parser.add_argument("--train-jsonl", action="append", required=True)
    parser.add_argument("--val-jsonl", action="append", default=[])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-cache-root", default=None)
    parser.add_argument("--latent-target-dim", type=int, default=16)
    parser.add_argument("--visual-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-selection", choices=["random", "longest"], default="random")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1792)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--save-best-metric", choices=["val_loss"], default="val_loss")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--zero-metric-patience", type=int, default=0)
    parser.add_argument("--eval-generate-max-samples", type=int, default=0)
    parser.add_argument("--eval-generate-max-new-tokens", type=int, default=192)
    parser.add_argument("--save-eval-cases", action="store_true")
    parser.add_argument("--initial-eval", action="store_true")
    parser.add_argument("--max-eval-calls", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--finetune-mode", choices=["lora", "partial_sft"], default="lora")
    parser.add_argument("--train-last-n-layers", type=int, default=4)
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--train-final-norm", action="store_true")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--stage", choices=sorted(STAGE_PRESETS), default="imitation")
    parser.add_argument("--thought-loss-weight", type=float, default=None)
    parser.add_argument("--answer-loss-weight", type=float, default=None)
    parser.add_argument("--think-mode", action="store_true")
    parser.add_argument("--resume-adapter-path", default=None)
    parser.add_argument("--resume-visual-head-path", default=None)
    parser.add_argument("--skip-missing-paths", action="store_true")
    parser.add_argument("--save-last", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = Path(args.model_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    latent_cache_root = Path(args.latent_cache_root).expanduser() if args.latent_cache_root else None
    jsonl_paths = resolve_jsonl_paths(args.train_jsonl, skip_missing=args.skip_missing_paths)
    if not jsonl_paths:
        raise ValueError("No training jsonl paths resolved.")

    stage_defaults = STAGE_PRESETS[args.stage]
    thought_weight = args.thought_loss_weight if args.thought_loss_weight is not None else stage_defaults["thought_loss_weight"]
    answer_weight = args.answer_loss_weight if args.answer_loss_weight is not None else stage_defaults["answer_loss_weight"]

    try:
        model, tokenizer = build_language_model(model_path, meta_only=args.dry_run)
        dataset = UniPathTokenDataset(
            jsonl_paths,
            tokenizer=tokenizer,
            seed=args.seed,
            max_length=args.max_length,
            thought_weight=thought_weight,
            answer_weight=answer_weight,
            latent_cache_root=latent_cache_root,
            latent_target_dim=args.latent_target_dim,
            max_samples=args.max_samples,
            sample_selection=args.sample_selection,
            think_mode=args.think_mode,
        )
        if len(dataset) == 0:
            raise ValueError("No usable training samples found.")

        val_jsonl_paths = resolve_jsonl_paths(args.val_jsonl, skip_missing=args.skip_missing_paths) if args.val_jsonl else []
        val_dataset = None
        if val_jsonl_paths:
            val_dataset = UniPathTokenDataset(
                val_jsonl_paths,
                tokenizer=tokenizer,
                seed=args.seed,
                max_length=args.max_length,
                thought_weight=thought_weight,
                answer_weight=answer_weight,
                latent_cache_root=latent_cache_root,
                latent_target_dim=args.latent_target_dim,
                max_samples=args.max_samples,
                sample_selection=args.sample_selection,
                think_mode=args.think_mode,
            )
            if len(val_dataset) == 0:
                val_dataset = None

        if args.finetune_mode == "lora":
            model, trainable, total = install_or_resume_lora(
                model,
                target_keywords=args.target_modules,
                rank=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                resume_adapter_path=args.resume_adapter_path,
                allow_missing_resume=args.dry_run,
            )
        else:
            if args.resume_adapter_path:
                raise ValueError("resume_adapter_path is only supported for lora finetuning.")
            model, trainable, total = install_partial_sft_layers(
                model,
                train_last_n_layers=args.train_last_n_layers,
                train_lm_head=args.train_lm_head,
                train_final_norm=args.train_final_norm,
            )

        hidden_size = int(model.config.hidden_size)
        visual_head = torch.nn.Linear(hidden_size, args.latent_target_dim)
        visual_head = maybe_load_visual_head(visual_head, args.resume_visual_head_path, allow_missing=args.dry_run)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in visual_head.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in visual_head.parameters())
        num_visual_examples = sum(len(item["visual_spans"]) for item in dataset.examples)
        max_seq_len = max((item["seq_len"] for item in dataset.examples), default=0)
        avg_seq_len = sum(item["seq_len"] for item in dataset.examples) / max(len(dataset.examples), 1)

        log_main(
            f"[unipath] stage={args.stage} thought_loss_weight={thought_weight} "
            f"answer_loss_weight={answer_weight} visual_loss_weight={args.visual_loss_weight} "
            f"think_mode={args.think_mode}"
        )
        log_main(f"[unipath] world_size={world_size} rank={rank} local_rank={local_rank}")
        log_main(f"[unipath] train_jsonl={','.join(str(p) for p in jsonl_paths)}")
        if val_jsonl_paths:
            log_main(f"[unipath] val_jsonl={','.join(str(p) for p in val_jsonl_paths)}")
        if latent_cache_root is not None:
            log_main(f"[unipath] latent_cache_root={latent_cache_root}")
        if args.resume_adapter_path:
            log_main(f"[unipath] resume_adapter_path={Path(args.resume_adapter_path).expanduser()}")
        if args.resume_visual_head_path:
            log_main(f"[unipath] resume_visual_head_path={Path(args.resume_visual_head_path).expanduser()}")
        log_main(f"[unipath] dataset_size={len(dataset)} visual_segments={num_visual_examples}")
        log_main(f"[unipath] sample_selection={args.sample_selection} seq_len_avg={avg_seq_len:.1f} seq_len_max={max_seq_len}")
        if val_dataset is not None:
            log_main(f"[unipath] val_dataset_size={len(val_dataset)}")
            val_max_seq_len = max((item["seq_len"] for item in val_dataset.examples), default=0)
            val_avg_seq_len = sum(item["seq_len"] for item in val_dataset.examples) / max(len(val_dataset.examples), 1)
            log_main(f"[unipath] val_seq_len_avg={val_avg_seq_len:.1f} val_seq_len_max={val_max_seq_len}")
        log_main(f"[unipath] finetune_mode={args.finetune_mode}")
        log_main(f"[unipath] trainable={trainable} total_params={total} pct={100.0 * trainable / max(total, 1):.4f}")
        log_main(f"[unipath] sample_preview={dataset[0]['preview']!r}")

        if args.dry_run:
            log_main(f"[unipath] dry-run complete -> {output_dir / ('dry_run_' + args.stage)}")
            return 0

        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        visual_head = visual_head.to(device)

        if is_distributed():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            visual_head = DDP(visual_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        collator = WeightedCollator(tokenizer, latent_target_dim=args.latent_target_dim)
        train_sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if is_distributed() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=args.seed, drop_last=False) if is_distributed() and val_dataset is not None else None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collator,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        optimizer = torch.optim.AdamW(
            list(p for p in model.parameters() if p.requires_grad) + list(p for p in visual_head.parameters() if p.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        global_step = 0
        best_metric = math.inf
        no_improve_evals = 0
        zero_metric_evals = 0
        eval_calls = 0
        last_output_dir = output_dir / f"adapter_{args.stage}_last"
        best_output_dir = output_dir / f"adapter_{args.stage}_best"
        args.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
        total_optimizer_steps = math.ceil(len(loader) * args.epochs / args.gradient_accumulation_steps)
        if args.max_train_steps > 0:
            total_optimizer_steps = min(total_optimizer_steps, args.max_train_steps)
        log_main(f"[unipath] total_optimizer_steps={total_optimizer_steps}")

        stop_training = False
        if val_loader is not None and args.initial_eval:
            case_output_path = None
            if args.save_eval_cases and is_main_process():
                case_output_path = output_dir / "eval_cases" / "step_0.jsonl"
            metrics = evaluate_model(
                model=model,
                visual_head=visual_head,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                visual_loss_weight=args.visual_loss_weight,
                max_generate_samples=args.eval_generate_max_samples,
                max_generate_new_tokens=args.eval_generate_max_new_tokens,
                case_output_path=case_output_path,
            )
            eval_calls += 1
            summary_bits = [
                f"val_loss={metrics['val_loss']:.6f}",
                f"val_lm_loss={metrics['val_lm_loss']:.6f}",
                f"val_visual_loss={metrics['val_visual_loss']:.6f}",
            ]
            for key in sorted(metrics):
                if key.startswith("val_loss_l"):
                    summary_bits.append(f"{key}={metrics[key]:.4f}")
            if "val_answer_accuracy" in metrics:
                summary_bits.append(f"val_answer_accuracy={metrics['val_answer_accuracy']:.4f}")
            if "val_format_accuracy" in metrics:
                summary_bits.append(f"val_format_accuracy={metrics['val_format_accuracy']:.4f}")
            log_main("[unipath] eval step=0/" + str(total_optimizer_steps) + " " + " ".join(summary_bits))
            current_metric = metrics[args.save_best_metric]
            if current_metric < best_metric:
                best_metric = current_metric
                if is_main_process():
                    save_checkpoint_bundle(
                        model,
                        visual_head,
                        best_output_dir,
                        global_step=0,
                        stage=args.stage,
                        kind="best",
                        finetune_mode=args.finetune_mode,
                        metrics=metrics,
                    )
                    print(f"[unipath] saved_best_checkpoint={best_output_dir} metric={current_metric:.6f}", flush=True)
            if args.max_eval_calls > 0 and eval_calls >= args.max_eval_calls:
                stop_training = True
            if is_distributed():
                stop_tensor = torch.tensor([1 if stop_training else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(int(stop_tensor.item()))
                dist.barrier()
            model.train()
            visual_head.train()
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            visual_head.train()
            optimizer.zero_grad(set_to_none=True)
            for step_idx, batch in enumerate(loader, start=1):
                batch_attention_mask = build_bagel_attention_mask(
                    batch["attention_mask"].to(device),
                    dtype=module_dtype(model),
                )
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch_attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
                lm_loss = weighted_language_model_loss(
                    logits=outputs.logits,
                    labels=batch["labels"].to(device),
                    loss_weights=batch["loss_weights"].to(device),
                )
                visual_loss = compute_visual_summary_loss(
                    hidden_states=outputs.hidden_states[-1],
                    visual_spans=batch["visual_spans"].to(device),
                    visual_targets=batch["visual_targets"].to(device),
                    visual_mask=batch["visual_mask"].to(device),
                    visual_head=visual_head,
                    visual_types=batch["visual_types"].to(device),
                )
                loss = lm_loss + args.visual_loss_weight * visual_loss
                (loss / args.gradient_accumulation_steps).backward()

                if step_idx % args.gradient_accumulation_steps == 0 or step_idx == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    reduced_loss = float(reduce_mean(loss.detach()).item())
                    reduced_lm = float(reduce_mean(lm_loss.detach()).item())
                    reduced_visual = float(reduce_mean(visual_loss.detach()).item())
                    if global_step == 1 or global_step % 10 == 0:
                        progress = 100.0 * global_step / max(total_optimizer_steps, 1)
                        log_main(
                            f"[unipath] epoch={epoch} step={global_step}/{total_optimizer_steps} ({progress:.1f}%) "
                            f"loss={reduced_loss:.6f} lm_loss={reduced_lm:.6f} visual_loss={reduced_visual:.6f}"
                        )

                    if val_loader is not None and args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                        case_output_path = None
                        if args.save_eval_cases and is_main_process():
                            case_output_path = output_dir / "eval_cases" / f"step_{global_step}.jsonl"
                        metrics = evaluate_model(
                            model=model,
                            visual_head=visual_head,
                            loader=val_loader,
                            tokenizer=tokenizer,
                            device=device,
                            visual_loss_weight=args.visual_loss_weight,
                            max_generate_samples=args.eval_generate_max_samples,
                            max_generate_new_tokens=args.eval_generate_max_new_tokens,
                            case_output_path=case_output_path,
                        )
                        eval_calls += 1
                        summary_bits = [
                            f"val_loss={metrics['val_loss']:.6f}",
                            f"val_lm_loss={metrics['val_lm_loss']:.6f}",
                            f"val_visual_loss={metrics['val_visual_loss']:.6f}",
                        ]
                        for key in sorted(metrics):
                            if key.startswith("val_loss_l"):
                                summary_bits.append(f"{key}={metrics[key]:.4f}")
                        if "val_answer_accuracy" in metrics:
                            summary_bits.append(f"val_answer_accuracy={metrics['val_answer_accuracy']:.4f}")
                        if "val_format_accuracy" in metrics:
                            summary_bits.append(f"val_format_accuracy={metrics['val_format_accuracy']:.4f}")
                        log_main(f"[unipath] eval step={global_step}/{total_optimizer_steps} " + " ".join(summary_bits))
                        should_zero_stop = (
                            args.zero_metric_patience > 0
                            and metrics.get("val_answer_accuracy") == 0.0
                            and metrics.get("val_format_accuracy") == 0.0
                        )
                        if should_zero_stop:
                            zero_metric_evals += 1
                        else:
                            zero_metric_evals = 0
                        current_metric = metrics[args.save_best_metric]
                        if current_metric < best_metric:
                            best_metric = current_metric
                            no_improve_evals = 0
                            if is_main_process():
                                save_checkpoint_bundle(
                                    model,
                                    visual_head,
                                    best_output_dir,
                                    global_step=global_step,
                                    stage=args.stage,
                                    kind="best",
                                    finetune_mode=args.finetune_mode,
                                    metrics=metrics,
                                )
                                print(f"[unipath] saved_best_checkpoint={best_output_dir} metric={current_metric:.6f}", flush=True)
                        else:
                            no_improve_evals += 1
                            if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                                stop_training = True
                                log_main(
                                    f"[unipath] early_stop_triggered step={global_step} "
                                    f"no_improve_evals={no_improve_evals} best_{args.save_best_metric}={best_metric:.6f}"
                                )
                        zero_stop_now = args.zero_metric_patience > 0 and zero_metric_evals >= args.zero_metric_patience
                        if zero_stop_now:
                            log_main(
                                f"[unipath] zero_metric_stop_triggered step={global_step} "
                                f"zero_metric_evals={zero_metric_evals}"
                            )
                        if is_distributed():
                            stop_tensor = torch.tensor(
                                [1 if (stop_training or zero_stop_now) else 0],
                                device=device,
                                dtype=torch.int32,
                            )
                            if args.max_eval_calls > 0 and eval_calls >= args.max_eval_calls:
                                stop_tensor.fill_(1)
                            dist.broadcast(stop_tensor, src=0)
                            stop_training = bool(int(stop_tensor.item()))
                        elif zero_stop_now or (args.max_eval_calls > 0 and eval_calls >= args.max_eval_calls):
                            stop_training = True
                        if is_distributed():
                            dist.barrier()
                        model.train()
                        visual_head.train()

                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0 and is_main_process():
                        save_checkpoint_bundle(
                            model,
                            visual_head,
                            output_dir / f"checkpoint_step_{global_step}",
                            global_step=global_step,
                            stage=args.stage,
                            kind="step",
                            finetune_mode=args.finetune_mode,
                        )
                    if args.save_every_steps > 0 and global_step % args.save_every_steps == 0 and is_distributed():
                        dist.barrier()
                    if stop_training:
                        break
                    if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                        stop_training = True
                        log_main(f"[unipath] max_train_steps_reached step={global_step}")
                        break
            if stop_training:
                break

        if args.save_last and is_main_process():
            save_checkpoint_bundle(
                model,
                visual_head,
                last_output_dir,
                global_step=global_step,
                stage=args.stage,
                kind="last",
                finetune_mode=args.finetune_mode,
            )
            print(f"[unipath] saved_last_checkpoint={last_output_dir}", flush=True)
            if best_metric == math.inf:
                save_checkpoint_bundle(
                    model,
                    visual_head,
                    best_output_dir,
                    global_step=global_step,
                    stage=args.stage,
                    kind="best_fallback",
                    finetune_mode=args.finetune_mode,
                )
                print(f"[unipath] saved_best_checkpoint_fallback={best_output_dir}", flush=True)
        if args.save_last and is_distributed():
            dist.barrier()
        log_main(f"[unipath] training complete -> {best_output_dir if best_metric < math.inf else last_output_dir}")
        return 0
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
