#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import json
import os
import random
import re
import string
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from .choice_policy import (
    ChoiceConsensusConfig,
    aggregate_choice_log_probs,
    choice_confidence,
    should_exit_choice,
)
from .protocols import (
    MMBENCH_ANSWER_INSTRUCTIONS,
    VLM_EVAL_VIT_TRANSFORM,
)



def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
_ACTIVE_MODEL = None
_ROUTER_TASK_SAFE_RE = re.compile(r"[^a-z0-9]+")

_MATHVISTA_SINGLE_TYPE_SOURCE = {
    "A-OKVQA": "text",
    "AI2D": "text",
    "CLEVR-Math": "integer",
    "ScienceQA": "text",
    "Super-CLEVR": "integer",
    "TQA": "text",
    "TextVQA": "integer",
    "UniGeo": "text",
    "VQA-AS": "text",
    "VQA-RAD": "text",
    "VizWiz": "integer",
}


def _safe_router_task(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.lower().replace("&", "and")
    return _ROUTER_TASK_SAFE_RE.sub("_", text).strip("_") or "unknown"


def _metadata_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
        normalized = key.replace(".", "_")
        if normalized in item and item[normalized] not in (None, ""):
            return item[normalized]
        current: Any = item
        found = True
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current not in (None, ""):
            return current
    return None


def _mmmu_router_task(data_id: Any) -> str:
    text = str(data_id or "")
    parts = text.split("_")
    subject = "_".join(parts[1:-1]) if len(parts) > 2 else text
    return f"mmmu_subject_{_safe_router_task(subject)}" if subject else "mmmu"


def _normalize_mathvista_source(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "MathV360K:" in text:
        text = text.rsplit("MathV360K:", 1)[-1]
    elif ":" in text:
        text = text.rsplit(":", 1)[-1]
    aliases = {
        "AOKVQA": "A-OKVQA",
        "A-OKVQA": "A-OKVQA",
        "AI2D": "AI2D",
        "CLEVR-Math": "CLEVR-Math",
        "DVQA": "DVQA",
        "FigureQA": "FigureQA",
        "FunctionQA": "FunctionQA",
        "GEOS": "GEOS",
        "GeoQA+": "GeoQA+",
        "Geometry3K": "Geometry3K",
        "PlotQA": "PlotQA",
        "ScienceQA": "ScienceQA",
        "TabMWP": "TabMWP",
        "UniGeo": "UniGeo",
    }
    return aliases.get(text, text)


def _mathvista_router_task(problem: dict[str, Any]) -> str:
    source = _normalize_mathvista_source(
        _metadata_value(problem, "metadata.source", "metadata_source", "source")
    )
    answer_type = _metadata_value(
        problem,
        "answer_type",
        "metadata.answer_type",
        "metadata_answer_type",
    )
    if source in _MATHVISTA_SINGLE_TYPE_SOURCE:
        answer_type = _MATHVISTA_SINGLE_TYPE_SOURCE[source]
    if source and answer_type:
        return f"mathvista_{_safe_router_task([source, answer_type])}"
    return "mathvista"


def _mmstar_router_task(problem: dict[str, Any]) -> str:
    l2_category = _metadata_value(
        problem,
        "l2_category",
        "metadata.l2_category",
        "metadata_l2_category",
    )
    if l2_category:
        return f"mmstar_l2_{_safe_router_task(l2_category)}"
    return "mmstar"


def _expand_env_defaults(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default)

    return os.path.expandvars(_ENV_DEFAULT_RE.sub(replace, value))


def _expand_config_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_defaults(value)
    if isinstance(value, list):
        return [_expand_config_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_env(item) for key, item in value.items()}
    return value


def _resolve_path(value: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    raw = _expand_env_defaults(str(value))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_yaml(path: str) -> dict[str, Any]:
    with _resolve_path(path, must_exist=True).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Eval config must parse to a dictionary.")
    block = cfg.get("eval", cfg)
    if not isinstance(block, dict):
        raise ValueError("`eval` must be a dictionary when provided.")
    return _expand_config_env(block)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_image(value: str | Path) -> Image.Image:
    raw = str(value)
    if raw.startswith("http://") or raw.startswith("https://"):
        with urlopen(raw) as response:
            return Image.open(response).convert("RGB")
    return Image.open(_resolve_path(raw, must_exist=True)).convert("RGB")


def _normalize_image_size(value: Any, default: tuple[int, int] = (1024, 1024)) -> tuple[int, int]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("image_size must be a two-item [height, width] list.")
    return int(value[0]), int(value[1])


def _pick_first_present(record: dict[str, Any], keys: list[str], *, required: bool = True) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    if required:
        raise KeyError(f"None of these keys exist in record: {keys}")
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, dict):
            for key in ("data", "prompts", "items", "examples", "metadata"):
                if isinstance(data.get(key), list):
                    return [dict(item) for item in data[key]]
            return [dict(data)]
        raise ValueError(f"Unsupported JSON structure in {path}")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            return [
                {"id": f"{idx:05d}", "prompt": line.strip()}
                for idx, line in enumerate(f)
                if line.strip()
            ]

    raise ValueError(f"Unsupported metadata file type: {path}")


def _shard_spec(cfg: dict[str, Any]) -> tuple[int, int]:
    num_shards = int(cfg.get("num_shards") or 1)
    shard_index = int(cfg.get("shard_index") or 0)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}."
        )
    return shard_index, num_shards


def _shard_positions(length: int, cfg: dict[str, Any]) -> list[int]:
    shard_index, num_shards = _shard_spec(cfg)
    if num_shards == 1:
        return list(range(length))
    return list(range(shard_index, length, num_shards))


def _shard_sequence(items: list[Any], cfg: dict[str, Any]) -> list[Any]:
    positions = _shard_positions(len(items), cfg)
    if len(positions) == len(items):
        return items
    return [items[index] for index in positions]


def _shard_hf_dataset(dataset, cfg: dict[str, Any]):
    positions = _shard_positions(len(dataset), cfg)
    if len(positions) == len(dataset):
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(positions)
    return [dataset[index] for index in positions]


def _dataset_mode(
    dataset_cfg: dict[str, Any],
    cfg: dict[str, Any],
    *,
    default_score: bool = False,
) -> tuple[str, bool, bool]:
    default = (
        "full"
        if bool(cfg.get("run_scoring", default_score))
        else "generate"
    )
    mode = str(dataset_cfg.get("mode", default)).strip().lower()
    if mode not in {"full", "generate", "score"}:
        raise ValueError(
            "dataset.mode must be one of: full, generate, score."
        )
    return mode, mode in {"full", "generate"}, mode in {"full", "score"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_dual_inferencer(cfg: dict[str, Any]):
    import torch

    from .loading import load_dynamic_bagel

    backbone_cfg = dict(cfg.get("backbone_cfg") or {})
    model_path = backbone_cfg.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("`backbone_cfg.model_path` is required.")

    seed = int(backbone_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    checkpoint_path = backbone_cfg.get("checkpoint_path")
    depth_config = backbone_cfg.get(
        "dynamic_depth_config", "configs/router_posttrain.json"
    )
    configured_device_id = backbone_cfg.get("device_id")
    if configured_device_id is None and os.environ.get("FLEXUMM_DEVICE_ID"):
        configured_device_id = os.environ["FLEXUMM_DEVICE_ID"]
    runtime = load_dynamic_bagel(
        model_path=_resolve_path(model_path, must_exist=True),
        checkpoint_path=(
            _resolve_path(checkpoint_path, must_exist=True)
            if checkpoint_path
            else None
        ),
        depth_config=_resolve_path(depth_config, must_exist=True),
        max_memory_per_gpu=str(backbone_cfg.get("max_mem_per_gpu", "80GiB")),
        offload_folder=_resolve_path(
            str(backbone_cfg.get("offload_folder", "tmp/offload_eval"))
        ),
        device_id=(
            int(configured_device_id)
            if configured_device_id is not None
            else None
        ),
        vit_max_image_size=int(
            backbone_cfg.get(
                "vit_max_image_size",
                VLM_EVAL_VIT_TRANSFORM["max_image_size"],
            )
        ),
        vit_min_image_size=int(
            backbone_cfg.get(
                "vit_min_image_size",
                VLM_EVAL_VIT_TRANSFORM["min_image_size"],
            )
        ),
        vit_image_stride=int(
            backbone_cfg.get(
                "vit_image_stride",
                VLM_EVAL_VIT_TRANSFORM["image_stride"],
            )
        ),
        vit_max_pixels=int(
            backbone_cfg.get(
                "vit_max_pixels",
                VLM_EVAL_VIT_TRANSFORM["max_pixels"],
            )
        ),
    )
    depth_task = str(
        cfg.get("depth_task") or cfg.get("benchmark") or "understanding"
    )
    runtime.model.default_depth_task = depth_task
    runtime.inferencer.depth_task = depth_task
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = runtime.model
    return runtime.inferencer


def _write_routing_summary(cfg: dict[str, Any]) -> None:
    if _ACTIVE_MODEL is None or not hasattr(_ACTIVE_MODEL, "router_diagnostics"):
        return
    diagnostics = _ACTIVE_MODEL.router_diagnostics()
    if diagnostics is None:
        return
    if hasattr(_ACTIVE_MODEL, "tafe_diagnostics"):
        tafe_diagnostics = _ACTIVE_MODEL.tafe_diagnostics()
        if tafe_diagnostics is not None:
            diagnostics = dict(diagnostics)
            diagnostics["tafe"] = tafe_diagnostics
    output_dir = _resolve_path(
        str(cfg.get("output_dir", "eval_outputs/flexumm_benchmark"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "routing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)


def _understanding_telemetry_snapshot(inferencer) -> dict[str, Any]:
    model = inferencer.model
    diagnostics = model.tafe_diagnostics() if hasattr(model, "tafe_diagnostics") else {}
    understanding = (diagnostics or {}).get("understanding", {})
    actions = understanding.get("inference_actions", {})
    return {
        "enabled": bool(understanding),
        "actions": {
            name: int(actions.get(name, 0))
            for name in ("share", "decouple", "exit")
        },
        "subsets": [
            int(value)
            for value in understanding.get("inference_subset_counts", [])
        ],
        "calls": int(understanding.get("inference_calls", 0)),
        "tokens": int(understanding.get("inference_routed_tokens", 0)),
    }


def _attach_understanding_telemetry(
    inferencer,
    decision: Any,
    before: dict[str, Any],
) -> dict[str, Any]:
    after = _understanding_telemetry_snapshot(inferencer)
    if not before.get("enabled") and not after.get("enabled"):
        return decision if isinstance(decision, dict) else {}
    actions = {
        name: int(after["actions"].get(name, 0))
        - int(before["actions"].get(name, 0))
        for name in ("share", "decouple", "exit")
    }
    width = max(len(before["subsets"]), len(after["subsets"]))
    subsets = [
        (after["subsets"][index] if index < len(after["subsets"]) else 0)
        - (before["subsets"][index] if index < len(before["subsets"]) else 0)
        for index in range(width)
    ]
    total_actions = sum(actions.values())
    total_subsets = sum(subsets)
    controller = getattr(inferencer.model, "dynamic_depth_controller", None)
    plan = getattr(controller, "last_plan", None)
    depth = getattr(plan, "understanding_depth", None) or getattr(
        plan, "execution_depth", None
    )
    telemetry = {
        "actions": actions,
        "alpha": actions["decouple"] / total_actions if total_actions else None,
        "top_expert_share": (
            max(subsets) / total_subsets
            if total_subsets and subsets
            else None
        ),
        "subset_counts": subsets,
        "calls": int(after["calls"] - before["calls"]),
        "tokens": int(after["tokens"] - before["tokens"]),
        "depth": int(depth) if depth is not None else None,
    }
    if not isinstance(decision, dict):
        decision = {}
    else:
        decision = dict(decision)
    decision["telemetry"] = telemetry
    return decision


def _understanding(
    inferencer,
    images: list[Image.Image],
    prompt: str,
    params: dict[str, Any],
) -> str:
    telemetry_before = _understanding_telemetry_snapshot(inferencer)
    inference_api = str(
        params.get("inference_api", "interleave")
    ).strip().lower()
    if inference_api == "native_chat":
        if bool(params.get("think", False)):
            raise ValueError(
                "BAGEL native_chat benchmark parity mode does not support "
                "think=True."
            )
        model = inferencer.model
        output = model.chat(
            inferencer.tokenizer,
            inferencer.new_token_ids,
            inferencer.vit_transform,
            images=images,
            prompt=prompt,
            max_length=int(
                params.get(
                    "max_new_tokens",
                    params.get("max_think_token_n", 256),
                )
            ),
            do_sample=bool(params.get("do_sample", False)),
            temperature=float(params.get("text_temperature", 0.3)),
            depth_task=getattr(
                inferencer,
                "depth_task",
                "understanding",
            ),
        )
        qwen_model = getattr(
            getattr(model, "language_model", None),
            "model",
            None,
        )
        decision = copy.deepcopy(
            getattr(qwen_model, "last_prompt_router_decision", None)
        )
        fusion_decision = copy.deepcopy(
            getattr(qwen_model, "last_prompt_fusion_sealing_decision", None)
        )
        if fusion_decision is not None:
            if not isinstance(decision, dict):
                decision = {}
            decision["fusion_sealing"] = fusion_decision
        decision = _attach_understanding_telemetry(
            inferencer,
            decision,
            telemetry_before,
        )
        inferencer.last_prompt_router_decision = decision
        return str(output).strip()
    if inference_api == "torchumm_reference":
        if bool(params.get("think", False)):
            raise ValueError(
                "TorchUMM reference parity mode does not support think=True."
            )
        import torch

        from data.data_utils import pil_img2rgb

        model = inferencer.model
        route = (
            model.depth_route(
                inferencer.depth_task,
                kind="understanding",
            )
            if hasattr(model, "depth_route")
            else nullcontext()
        )
        gen_context = inferencer.init_gen_context()
        with route, torch.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
            dtype=torch.bfloat16,
        ):
            # Reproduce TorchUMM's published BAGEL adapter exactly: its
            # generic interleaved path first applies the VAE resize even for
            # understanding, then applies the 980/224/14 ViT transform while
            # preparing visual-understanding tokens.
            for image in images:
                resized = inferencer.vae_transform.resize_transform(
                    pil_img2rgb(image)
                )
                gen_context = inferencer.update_context_image(
                    resized,
                    gen_context,
                    vae=False,
                )
            gen_context = inferencer.update_context_text(prompt, gen_context)
            output = inferencer.gen_text(
                gen_context,
                max_length=int(
                    params.get(
                        "max_new_tokens",
                        params.get("max_think_token_n", 256),
                    )
                ),
                do_sample=bool(params.get("do_sample", False)),
                temperature=float(params.get("text_temperature", 0.3)),
                mode=str(params.get("text_mode", "und")),
                restrict_to_tokenizer_vocab=False,
            )
        qwen_model = getattr(
            getattr(model, "language_model", None),
            "model",
            None,
        )
        decision = copy.deepcopy(
            getattr(qwen_model, "last_prompt_router_decision", None)
        )
        fusion_decision = copy.deepcopy(
            getattr(qwen_model, "last_prompt_fusion_sealing_decision", None)
        )
        if fusion_decision is not None:
            if not isinstance(decision, dict):
                decision = {}
            decision["fusion_sealing"] = fusion_decision
        decision = _attach_understanding_telemetry(
            inferencer,
            decision,
            telemetry_before,
        )
        inferencer.last_prompt_router_decision = decision
        return str(output).strip()
    if inference_api != "interleave":
        raise ValueError(
            "understanding.inference_api must be 'interleave', "
            "'native_chat', or 'torchumm_reference'."
        )

    input_terms: list[str | Image.Image] = []
    input_terms.extend(images)
    input_terms.append(prompt)
    deliberation = dict(params.get("deliberation") or {})
    if bool(deliberation.get("enabled", False)):
        controller = getattr(
            inferencer.model,
            "dynamic_depth_controller",
            None,
        )
        config = getattr(controller, "config", None)
        candidates = tuple(
            getattr(config, "router_candidates_understanding", ())
        )
        draft_depth = int(
            deliberation.get(
                "draft_depth",
                candidates[0] if candidates else 26,
            )
        )
        result = inferencer.deliberate_understanding(
            input_terms,
            draft_depth=draft_depth,
            draft_tokens=int(deliberation.get("draft_tokens", 48)),
            draft_style=str(deliberation.get("draft_style", "answer")),
            final_tokens=int(
                params.get(
                    "max_new_tokens",
                    params.get("max_think_token_n", 256),
                )
            ),
            do_sample=bool(params.get("do_sample", False)),
            temperature=float(params.get("text_temperature", 0.3)),
            mode=str(params.get("text_mode", "und")),
        )
        output = [result]
    else:
        route = (
            inferencer.model.depth_route(
                inferencer.depth_task,
                kind="understanding",
            )
            if hasattr(inferencer.model, "depth_route")
            else nullcontext()
        )
        with route:
            output = inferencer.interleave_inference(
                input_terms,
                understanding_output=True,
                think=bool(params.get("think", False)),
                max_think_token_n=int(
                    params.get(
                        "max_new_tokens",
                        params.get("max_think_token_n", 256),
                    )
                ),
                do_sample=bool(params.get("do_sample", False)),
                text_temperature=float(params.get("text_temperature", 0.3)),
                text_mode=str(params.get("text_mode", "und")),
            )
    qwen_model = getattr(
        getattr(inferencer.model, "language_model", None), "model", None
    )
    decision = copy.deepcopy(
        getattr(
            qwen_model,
            "last_prompt_router_decision",
            None,
        )
    )
    fusion_decision = copy.deepcopy(
        getattr(qwen_model, "last_prompt_fusion_sealing_decision", None)
    )
    if fusion_decision is not None:
        if not isinstance(decision, dict):
            decision = {}
        decision["fusion_sealing"] = fusion_decision
    deliberation_trace = getattr(inferencer, "last_deliberation", None)
    if bool(deliberation.get("enabled", False)) and isinstance(decision, dict):
        decision["deliberation"] = copy.deepcopy(deliberation_trace)
    decision = _attach_understanding_telemetry(
        inferencer,
        decision,
        telemetry_before,
    )
    inferencer.last_prompt_router_decision = decision
    return next((item for item in output if isinstance(item, str)), "").strip()


def _understanding_choice(
    inferencer,
    images: list[Image.Image],
    prompt: str,
    params: dict[str, Any],
    choices: dict[str, str],
) -> str:
    """Apply one shared confidence policy to any explicit-choice prompt."""

    text_scoring_value = params.get("choice_text_scoring")
    if text_scoring_value and bool(text_scoring_value.get("enabled", False)):
        return _understanding_choice_text_scoring(
            inferencer,
            images,
            prompt,
            params,
            choices,
        )
    permutation_value = params.get("choice_permutation")
    if permutation_value and bool(permutation_value.get("enabled", False)):
        return _understanding_choice_permutation(
            inferencer,
            images,
            prompt,
            params,
            choices,
        )
    verifier_value = params.get("choice_verifier")
    if verifier_value and bool(verifier_value.get("enabled", False)):
        return _understanding_choice_verifier(
            inferencer,
            images,
            prompt,
            params,
            choices,
        )
    policy_value = params.get("choice_consensus")
    if not policy_value or not bool(policy_value.get("enabled", False)):
        return _understanding(inferencer, images, prompt, params)
    labels = list(choices)
    if len(labels) < 2:
        return _understanding(inferencer, images, prompt, params)

    policy = ChoiceConsensusConfig.from_dict(policy_value)
    input_terms: list[str | Image.Image] = [*images, prompt]
    trace: list[dict[str, Any]] = []
    selected_depth = policy.candidate_depths[-1]
    aggregate = None

    def stop_after_confident_exit(
        depth: int,
        depth_scores: dict[int, Any],
    ) -> bool:
        nonlocal aggregate, selected_depth
        candidate_index = policy.candidate_depths.index(depth)
        per_depth = [
            depth_scores[item]
            for item in policy.candidate_depths[: candidate_index + 1]
        ]
        aggregate = aggregate_choice_log_probs(
            per_depth,
            policy.depth_weights[: candidate_index + 1],
        )
        margin, entropy = choice_confidence(aggregate)
        top_index = int(aggregate.argmax().item())
        trace.append(
            {
                "depth": depth,
                "top_choice": labels[top_index],
                "margin": margin,
                "normalized_entropy": entropy,
                "choice_log_probs": {
                    label: float(score)
                    for label, score in zip(labels, aggregate.tolist())
                },
            }
        )
        should_stop = (
            candidate_index < len(policy.candidate_depths) - 1
            and should_exit_choice(
                per_depth,
                aggregate,
                margin_threshold=policy.early_exit_margin,
                max_entropy=policy.early_exit_max_entropy,
                require_agreement=policy.require_depth_agreement,
            )
        )
        if should_stop:
            selected_depth = depth
        return should_stop

    inferencer.score_choice_labels(
        input_terms,
        labels,
        list(policy.candidate_depths),
        should_stop=stop_after_confident_exit,
    )
    if aggregate is None:
        raise RuntimeError("Choice scorer did not return a candidate")

    selected_label = labels[int(aggregate.argmax().item())]
    inferencer.last_prompt_router_decision = {
        "task": "explicit_choice",
        "selected_depth": selected_depth,
        "trace": trace,
        "policy": "late_depth_consensus",
    }
    return selected_label


def _choice_text_scoring_prompt(
    prompt: str,
    choices: dict[str, str],
    policy: dict[str, Any],
) -> str:
    question_stem = _strip_choice_options_from_prompt(prompt, choices)
    option_lines = "\n".join(
        f"{label}. {text}" for label, text in choices.items()
    )
    instruction = str(
        policy.get(
            "instruction",
            (
                "Select the correct option by meaning. Answer with the exact "
                "text of the correct option, not the option letter."
            ),
        )
    ).strip()
    parts = [
        question_stem.strip(),
        "Options:",
        option_lines,
        instruction,
    ]
    return "\n".join(part for part in parts if part).strip()


def _understanding_choice_text_scoring(
    inferencer,
    images: list[Image.Image],
    prompt: str,
    params: dict[str, Any],
    choices: dict[str, str],
) -> str:
    """Choose by conditional likelihood of each option's text."""

    labels = list(choices)
    if len(labels) < 2:
        return _understanding(inferencer, images, prompt, params)
    policy = dict(params.get("choice_text_scoring") or {})
    candidate_depths = tuple(
        int(item) for item in policy.get("candidate_depths", [28])
    )
    if not candidate_depths:
        raise ValueError("choice_text_scoring.candidate_depths cannot be empty")
    depth_weights = tuple(
        float(item)
        for item in policy.get(
            "depth_weights",
            [1.0 / len(candidate_depths)] * len(candidate_depths),
        )
    )
    if len(depth_weights) != len(candidate_depths):
        raise ValueError(
            "choice_text_scoring.depth_weights must match candidate_depths"
        )
    scoring_prompt = _choice_text_scoring_prompt(prompt, choices, policy)
    input_terms: list[str | Image.Image] = [*images, scoring_prompt]
    scores = inferencer.score_choice_texts(
        input_terms,
        choices,
        list(candidate_depths),
        normalize=str(policy.get("normalize", "mean")),
        max_choice_tokens=(
            int(policy["max_choice_tokens"])
            if policy.get("max_choice_tokens") is not None
            else None
        ),
        mode=str(params.get("text_mode", "und")),
    )
    per_depth = [scores[int(depth)] for depth in candidate_depths]
    aggregate = aggregate_choice_log_probs(per_depth, depth_weights)
    selected_index = int(aggregate.argmax().item())
    selected_label = labels[selected_index]
    margin, entropy = choice_confidence(aggregate)
    inferencer.last_prompt_router_decision = {
        "task": "explicit_choice",
        "selected_depth": int(candidate_depths[-1]),
        "policy": "option_text_likelihood",
        "normalize": str(policy.get("normalize", "mean")),
        "selected_choice": selected_label,
        "margin": margin,
        "normalized_entropy": entropy,
        "trace": [
            {
                "depth": int(depth),
                "choice_log_probs": {
                    label: float(score)
                    for label, score in zip(
                        labels,
                        scores[int(depth)].tolist(),
                    )
                },
            }
            for depth in candidate_depths
        ],
        "aggregate_choice_log_probs": {
            label: float(score)
            for label, score in zip(labels, aggregate.tolist())
        },
    }
    return selected_label


def _extract_explicit_choice(
    response: str,
    choices: dict[str, str],
) -> str | None:
    """Extract a single explicit label without consulting the reference."""

    labels = list(choices)
    if not labels:
        return None
    stripped = str(response).strip()
    for label in labels:
        if re.match(
            rf"^\s*(?:the\s+)?(?:answer\s+is\s+|option\s+)?"
            rf"{re.escape(label)}(?:\b|[\s.):：])",
            stripped,
            flags=re.IGNORECASE,
        ):
            return label
    matches = [
        label
        for label in labels
        if re.search(
            rf"\b(?:answer\s+is|option)\s*"
            rf"{re.escape(label)}\b",
            stripped,
            flags=re.IGNORECASE,
        )
    ]
    if len(matches) == 1:
        return matches[0]
    normalized = re.sub(r"\s+", " ", stripped).strip().casefold()
    text_matches = [
        label
        for label, text in choices.items()
        if normalized == re.sub(r"\s+", " ", str(text)).strip().casefold()
    ]
    return text_matches[0] if len(text_matches) == 1 else None


def _cyclic_choice_relabeling(
    choices: dict[str, str],
    offset: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a cyclic relabeling and its new-label to original-label map."""

    labels = list(choices)
    if len(labels) < 2:
        raise ValueError("choice relabeling requires at least two options")
    normalized_offset = int(offset) % len(labels)
    relabeled: dict[str, str] = {}
    to_original: dict[str, str] = {}
    for new_index, new_label in enumerate(labels):
        original_label = labels[
            (new_index + normalized_offset) % len(labels)
        ]
        relabeled[new_label] = choices[original_label]
        to_original[new_label] = original_label
    return relabeled, to_original


def _strip_choice_options_from_prompt(
    prompt: str,
    choices: dict[str, str],
) -> str:
    """Remove the original option-letter list before relabeling choices."""

    labels = "".join(re.escape(label) for label in choices)
    suffixes = {
        instruction.casefold()
        for instruction in MMBENCH_ANSWER_INSTRUCTIONS.values()
    }
    suffixes.add(
        "answer with the option's letter from the given choices directly."
    )
    lines = []
    for line in str(prompt).splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if stripped.casefold() in suffixes:
            continue
        if re.match(
            rf"^(?:\([{labels}]\)|[{labels}]\s*[\.:：.)）])\s+",
            stripped,
        ):
            continue
        if stripped.casefold() in {"options:", "choices:"}:
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()

    # MMStar stores choices either inline after "Options: A: ... B: ..." or as
    # a "Choices:\n(A) ..." block. Keep only the question stem; the relabeled
    # list below becomes the sole option set.
    if re.search(r"\b(?:Options|Choices)\s*:", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(
            r"\b(?:Options|Choices)\s*:",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
    return cleaned or str(prompt).strip()


def _counterbalanced_choice_prompt(
    prompt: str,
    relabeled_choices: dict[str, str],
) -> str:
    question_stem = _strip_choice_options_from_prompt(
        prompt,
        relabeled_choices,
    )
    option_lines = "\n".join(
        f"{label}. {text}"
        for label, text in relabeled_choices.items()
    )
    return (
        f"{question_stem}\n"
        "Options:\n"
        f"{option_lines}\n"
        "Answer with the option's letter from the given choices directly."
    )


def _counterbalanced_majority(
    votes: list[str],
    direct_label: str | None,
) -> str | None:
    """Choose a unique repeated vote, using the direct pass for all ties."""

    if not votes:
        return direct_label
    counts = Counter(votes)
    best_count = max(counts.values())
    winners = [
        label for label, count in counts.items()
        if count == best_count
    ]
    if best_count >= 2 and len(winners) == 1:
        return winners[0]
    if direct_label is not None:
        return direct_label
    return winners[0] if len(winners) == 1 else None


def _understanding_choice_permutation(
    inferencer,
    images: list[Image.Image],
    prompt: str,
    params: dict[str, Any],
    choices: dict[str, str],
) -> str:
    """Counterbalance option-label bias with cyclic, mapped-back votes."""

    labels = list(choices)
    if len(labels) < 2:
        return _understanding(inferencer, images, prompt, params)
    policy = dict(params.get("choice_permutation") or {})
    max_passes = int(policy.get("max_passes", 5))
    if max_passes < 2:
        raise ValueError("choice_permutation.max_passes must be at least 2")
    num_passes = min(len(labels), max_passes)

    pass_params = copy.deepcopy(params)
    pass_params["choice_permutation"] = {"enabled": False}
    pass_params["choice_verifier"] = {"enabled": False}
    pass_params["choice_consensus"] = {"enabled": False}
    pass_params["deliberation"] = {"enabled": False}

    direct = _understanding(inferencer, images, prompt, pass_params)
    direct_route = copy.deepcopy(
        getattr(inferencer, "last_prompt_router_decision", None)
    )
    direct_label = _extract_explicit_choice(direct, choices)
    votes = [direct_label] if direct_label is not None else []
    trace = [
        {
            "offset": 0,
            "response": direct,
            "relabeled_choice": direct_label,
            "original_choice": direct_label,
            "routing": direct_route,
        }
    ]

    for offset in range(1, num_passes):
        relabeled, to_original = _cyclic_choice_relabeling(
            choices,
            offset,
        )
        relabeled_prompt = _counterbalanced_choice_prompt(
            prompt,
            relabeled,
        )
        response = _understanding(
            inferencer,
            images,
            relabeled_prompt,
            pass_params,
        )
        route = copy.deepcopy(
            getattr(inferencer, "last_prompt_router_decision", None)
        )
        relabeled_label = _extract_explicit_choice(response, relabeled)
        original_label = (
            to_original.get(relabeled_label)
            if relabeled_label is not None
            else None
        )
        if original_label is not None:
            votes.append(original_label)
        trace.append(
            {
                "offset": offset,
                "response": response,
                "relabeled_choice": relabeled_label,
                "original_choice": original_label,
                "routing": route,
            }
        )

    selected = _counterbalanced_majority(votes, direct_label)
    inferencer.last_prompt_router_decision = {
        "task": "explicit_choice",
        "policy": "counterbalanced_cyclic_labels",
        "selected_choice": selected,
        "direct_choice": direct_label,
        "votes": votes,
        "num_passes": num_passes,
        "trace": trace,
    }
    return selected or direct or ""


def _understanding_choice_verifier(
    inferencer,
    images: list[Image.Image],
    prompt: str,
    params: dict[str, Any],
    choices: dict[str, str],
) -> str:
    """Resolve only disagreements between direct and early-exit passes."""

    policy = dict(params.get("choice_verifier") or {})
    direct_params = copy.deepcopy(params)
    direct_params["deliberation"] = {"enabled": False}
    direct_params["choice_verifier"] = {"enabled": False}
    direct = _understanding(inferencer, images, prompt, direct_params)
    direct_route = copy.deepcopy(
        getattr(inferencer, "last_prompt_router_decision", None)
    )

    refine_params = copy.deepcopy(params)
    refine_params["choice_verifier"] = {"enabled": False}
    refine_params["deliberation"] = {
        "enabled": True,
        "draft_depth": int(policy.get("draft_depth", 26)),
        "draft_tokens": int(policy.get("draft_tokens", 4)),
        "draft_style": "answer",
    }
    refined = _understanding(inferencer, images, prompt, refine_params)
    refine_route = copy.deepcopy(
        getattr(inferencer, "last_prompt_router_decision", None)
    )

    direct_label = _extract_explicit_choice(direct, choices)
    refined_label = _extract_explicit_choice(refined, choices)
    arbitration = None
    arbitration_route = None
    arbitration_risk = None
    selection_reason = "single_valid_candidate"
    selected = refined_label or direct_label
    if direct_label is not None and direct_label == refined_label:
        selection_reason = "direct_refinement_agreement"
    if (
        direct_label is not None
        and refined_label is not None
        and direct_label != refined_label
    ):
        arbitration_prompt = (
            f"{prompt}\n\n"
            "Two independent inference paths disagree. The direct path "
            f"selected {direct_label}; the early-representation refinement "
            f"selected {refined_label}. Re-evaluate the image and question "
            "and return only the correct option letter."
        )
        arbitration = _understanding(
            inferencer,
            images,
            arbitration_prompt,
            direct_params,
        )
        arbitration_route = copy.deepcopy(
            getattr(inferencer, "last_prompt_router_decision", None)
        )
        arbitration_label = _extract_explicit_choice(
            arbitration,
            choices,
        )
        arbitration_trace = (
            arbitration_route.get("trace", [])
            if isinstance(arbitration_route, dict)
            else []
        )
        if arbitration_trace:
            arbitration_risk = float(
                arbitration_trace[0].get("risk_upper_bound", 0.0)
            )
        moderate_risk = float(
            policy.get("moderate_risk_threshold", 0.10)
        )
        high_risk = float(policy.get("high_risk_threshold", 0.15))
        if not 0 <= moderate_risk < high_risk:
            raise ValueError(
                "choice verifier risk thresholds must satisfy "
                "0 <= moderate < high"
            )
        if arbitration_risk is not None and arbitration_risk >= high_risk:
            selected = direct_label
            selection_reason = "unstable_arbitration_fallback_to_direct"
        elif (
            arbitration_risk is not None
            and arbitration_risk >= moderate_risk
        ):
            selected = refined_label
            selection_reason = "moderate_risk_prefers_early_refinement"
        else:
            selected = arbitration_label or refined_label
            selection_reason = "low_risk_arbitration"

    inferencer.last_prompt_router_decision = {
        "task": "explicit_choice",
        "policy": "direct_early_representation_verifier",
        "selected_choice": selected,
        "selection_reason": selection_reason,
        "arbitration_risk": arbitration_risk,
        "direct": {
            "response": direct,
            "choice": direct_label,
            "routing": direct_route,
        },
        "refinement": {
            "response": refined,
            "choice": refined_label,
            "routing": refine_route,
        },
        "arbitration": (
            {
                "response": arbitration,
                "choice": _extract_explicit_choice(
                    arbitration or "",
                    choices,
                ),
                "routing": arbitration_route,
            }
            if arbitration is not None
            else None
        ),
    }
    return selected or refined or direct


def _generation(
    inferencer,
    prompt: str,
    params: dict[str, Any],
    images: list[Image.Image] | None = None,
) -> Image.Image:
    input_terms: list[str | Image.Image] = []
    if images:
        input_terms.extend(images)
    input_terms.append(prompt)
    route = (
        inferencer.model.depth_route(
            inferencer.depth_task,
            kind="generation",
        )
        if hasattr(inferencer.model, "depth_route")
        else nullcontext()
    )
    with route:
        output = inferencer.interleave_inference(
            input_terms,
            understanding_output=False,
            think=bool(params.get("think", False)),
            max_think_token_n=int(params.get("max_think_token_n", 1000)),
            do_sample=bool(params.get("do_sample", False)),
            text_temperature=float(params.get("text_temperature", 0.3)),
            text_mode=str(params.get("text_mode", "und")),
            cfg_text_scale=float(params.get("cfg_text_scale", 4.0)),
            cfg_img_scale=float(params.get("cfg_img_scale", 2.0)),
            cfg_interval=list(params.get("cfg_interval", [0.0, 1.0])),
            timestep_shift=float(params.get("timestep_shift", 3.0)),
            num_timesteps=int(params.get("num_timesteps", 50)),
            cfg_renorm_min=float(params.get("cfg_renorm_min", 0.0)),
            cfg_renorm_type=str(params.get("cfg_renorm_type", "text_channel")),
            image_shapes=_normalize_image_size(
                params.get("image_size"), default=(1024, 1024)
            ),
            enable_taylorseer=bool(params.get("enable_taylorseer", False)),
        )
    image = next((item for item in output if isinstance(item, Image.Image)), None)
    if image is None:
        raise RuntimeError("Generation did not return an image.")
    return image


def _run_generation_benchmark(cfg: dict[str, Any]) -> int:
    benchmark = str(cfg.get("benchmark", "")).lower()
    dataset_cfg = dict(cfg.get("dataset") or {})
    gen_cfg = dict(cfg.get("generation") or {})
    output_dir = _resolve_path(str(cfg.get("output_dir", f"eval_outputs/{benchmark}_bagel")))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = dataset_cfg.get("metadata_file") or dataset_cfg.get("prompt_file")
    if not isinstance(metadata_file, str):
        raise ValueError("Generation benchmark config needs dataset.metadata_file or dataset.prompt_file.")
    records = _load_records(_resolve_path(metadata_file, must_exist=True))

    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        records = records[: int(max_samples)]

    prompt_keys = list(dataset_cfg.get("prompt_keys") or [dataset_cfg.get("prompt_key", "prompt")])
    id_keys = list(dataset_cfg.get("id_keys") or [dataset_cfg.get("id_key", "id"), "index", "prompt_id"])
    num_images = int(gen_cfg.get("num_images", 4 if benchmark == "geneval" else 1))
    layout = str(gen_cfg.get("output_layout", "geneval" if benchmark == "geneval" else "flat"))
    skip_existing = bool(gen_cfg.get("skip_existing", True))

    inferencer = _build_dual_inferencer(cfg)
    manifest_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(tqdm(records, desc=f"{benchmark} generation")):
        prompt = str(_pick_first_present(record, prompt_keys))
        record_id = _pick_first_present(record, id_keys, required=False)
        if record_id is None:
            record_id = f"{record_index:05d}"
        record_id = str(record_id)

        if layout == "geneval":
            prompt_dir = output_dir / "images" / f"{record_index:05d}"
            sample_dir = prompt_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
            expected_paths = [sample_dir / f"{sample_idx:05d}.png" for sample_idx in range(num_images)]
        else:
            image_dir = output_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            expected_paths = [image_dir / f"{record_id}_{sample_idx:05d}.png" for sample_idx in range(num_images)]

        for sample_idx, image_path in enumerate(expected_paths):
            if skip_existing and image_path.exists():
                manifest_rows.append(
                    {
                        "id": record_id,
                        "index": record_index,
                        "sample_index": sample_idx,
                        "prompt": prompt,
                        "image_path": str(image_path),
                        "skipped": True,
                    }
                )
                continue

            image = _generation(inferencer, prompt, gen_cfg)
            image.save(image_path)
            manifest_rows.append(
                {
                    "id": record_id,
                    "index": record_index,
                    "sample_index": sample_idx,
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "skipped": False,
                }
            )

    _write_jsonl(output_dir / "predictions.jsonl", manifest_rows)
    print(f"[bagel eval] wrote generation manifest to {output_dir / 'predictions.jsonl'}")
    return 0


def _mme_postprocess(response: str) -> str:
    response = response.replace("\n", "").replace("不是", "No").replace("是", "Yes").replace("否", "No")
    response = response.lower().replace("true", "yes").replace("false", "no")
    return re.sub(re.compile(r"[\u4e00-\u9fa5]"), "", response)


MME_EVAL_TYPES = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


def _mme_parse_prediction(prediction: str) -> str:
    value = str(prediction).strip().lower()
    if value in {"yes", "no"}:
        return value
    prefix = value[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def _mme_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score MME rows with the official accuracy + paired-accuracy rule."""

    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        category_rows.setdefault(str(row["category"]), []).append(row)

    by_category: dict[str, dict[str, float | int]] = {}
    for category, items in category_rows.items():
        if len(items) % 2:
            raise ValueError(
                f"MME category {category!r} has an odd row count: "
                f"{len(items)}"
            )
        correct = 0
        paired_correct = 0
        other = 0
        for index in range(0, len(items), 2):
            pair_correct = 0
            pair_ids = {
                str(items[index].get("question_id", "")).strip(),
                str(items[index + 1].get("question_id", "")).strip(),
            }
            if len(pair_ids) != 1:
                raise ValueError(
                    f"MME pair mismatch in {category!r} at rows "
                    f"{index}/{index + 1}: {sorted(pair_ids)}"
                )
            for item in items[index : index + 2]:
                target = str(item["gt"]).strip().lower()
                if target not in {"yes", "no"}:
                    raise ValueError(
                        f"Invalid MME target {target!r} in {category!r}"
                    )
                prediction = _mme_parse_prediction(str(item["answer"]))
                other += int(prediction == "other")
                is_correct = prediction == target
                correct += int(is_correct)
                pair_correct += int(is_correct)
            paired_correct += int(pair_correct == 2)
        question_accuracy = correct / len(items)
        pair_count = len(items) // 2
        pair_accuracy = paired_correct / pair_count
        by_category[category] = {
            "score": 100.0 * (question_accuracy + pair_accuracy),
            "question_accuracy": 100.0 * question_accuracy,
            "pair_accuracy": 100.0 * pair_accuracy,
            "correct": correct,
            "questions": len(items),
            "paired_correct": paired_correct,
            "pairs": pair_count,
            "other": other,
        }

    totals: dict[str, float] = {}
    missing: dict[str, list[str]] = {}
    for eval_type, categories in MME_EVAL_TYPES.items():
        missing_categories = [
            category for category in categories
            if category not in by_category
        ]
        if missing_categories:
            missing[eval_type] = missing_categories
        totals[eval_type] = sum(
            float(by_category[category]["score"])
            for category in categories
            if category in by_category
        )
    return {
        "protocol": "MME accuracy plus paired-image accuracy",
        "totals": totals,
        "by_category": by_category,
        "missing_categories": missing,
        "rows": len(rows),
        "complete": not missing,
    }


def _write_mme_score(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    score = _mme_score_rows(rows)
    with (output_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    lines: list[str] = []
    for eval_type, categories in MME_EVAL_TYPES.items():
        lines.append(f"=========== {eval_type} ===========")
        lines.append(f"total score: {score['totals'][eval_type]}")
        lines.append("")
        for category in categories:
            if category in score["by_category"]:
                lines.append(
                    f"\t {category} score: "
                    f"{score['by_category'][category]['score']}"
                )
        lines.extend(["", ""])
    (output_dir / "results.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _mme_complete_pair_row_limit(raw_limit: Any) -> int | None:
    if raw_limit in (None, ""):
        return None
    limit = int(raw_limit)
    if limit < 2:
        raise ValueError(
            "MME max_samples_per_category must keep at least one "
            "two-question image pair."
        )
    return 2 * (limit // 2)


def _mme_select_hf_rows(
    dataset,
    max_samples_per_category: int | None,
) -> list[dict[str, Any]]:
    """Select complete MME question pairs, balanced by category."""

    category_rows: dict[str, list[dict[str, Any]]] = {}
    for item in dataset:
        row = dict(item)
        category_rows.setdefault(str(row["category"]), []).append(row)

    selected: list[dict[str, Any]] = []
    row_limit = _mme_complete_pair_row_limit(max_samples_per_category)
    for category in sorted(category_rows):
        rows = category_rows[category]
        if len(rows) % 2:
            raise ValueError(
                f"MME source category {category!r} has an odd row count: "
                f"{len(rows)}"
            )
        limit = len(rows)
        if row_limit is not None:
            limit = min(len(rows), row_limit)
        selected.extend(rows[:limit])
    return selected


MMBENCH_COLLECTIONS = {
    "mmbench_dev_20230712": {
        "root": "eval/vlm/data/mmbench/mmbench_dev_20230712.tsv",
        "type": "dev",
        "language": "en",
    },
    "mmbench_dev_cn_20231003": {
        "root": "eval/vlm/data/mmbench/mmbench_dev_cn_20231003.tsv",
        "type": "dev",
        "language": "cn",
    },
    "mmbench_dev_en_20231003": {
        "root": "eval/vlm/data/mmbench/mmbench_dev_en_20231003.tsv",
        "type": "dev",
        "language": "en",
    },
    "mmbench_test_cn_20231003": {
        "root": "eval/vlm/data/mmbench/mmbench_test_cn_20231003.tsv",
        "type": "test",
        "language": "cn",
    },
    "mmbench_test_en_20231003": {
        "root": "eval/vlm/data/mmbench/mmbench_test_en_20231003.tsv",
        "type": "test",
        "language": "en",
    },
    "MMBench_TEST_EN_V11": {
        "root": "eval/vlm/data/mmbench/MMBench_TEST_EN_V11.tsv",
        "type": "test",
        "language": "en",
    },
    "MMBench_TEST_CN_V11": {
        "root": "eval/vlm/data/mmbench/MMBench_TEST_CN_V11.tsv",
        "type": "test",
        "language": "cn",
    },
}


MMBENCH_ABBRS = {
    "coarse_perception": "CP",
    "finegrained_perception (instance-level)": "FP-S",
    "finegrained_perception (cross-instance)": "FP-C",
    "logic_reasoning": "LR",
    "relation_reasoning": "RR",
    "attribute_reasoning": "AR",
}


def _mmbench_options(row: pd.Series) -> dict[str, str]:
    options: dict[str, str] = {}
    for cand in string.ascii_uppercase:
        if cand in row and not pd.isna(row[cand]):
            options[cand] = str(row[cand])
    return options


def _mmbench_prompt(
    question: str,
    options: dict[str, str],
    hint: str | None,
    language: str = "en",
) -> str:
    lines: list[str] = []
    if hint:
        lines.append(str(hint))
    lines.append(question)
    if options:
        lines.extend(f"{key}. {value}" for key, value in options.items())
        lines.append(
            MMBENCH_ANSWER_INSTRUCTIONS.get(
                str(language).strip().lower(),
                MMBENCH_ANSWER_INSTRUCTIONS["en"],
            )
        )
    return "\n".join(lines)


def _mmbench_select_group_rows(
    df: pd.DataFrame,
    max_groups: int | None,
) -> tuple[pd.DataFrame, list[int]]:
    """Select complete circular groups, never a raw TSV-row prefix."""
    group_ids = (df["index"].astype(int) % 1_000_000).tolist()
    ordered_groups = list(dict.fromkeys(group_ids))
    if max_groups is None or int(max_groups) == 0:
        selected_groups = ordered_groups
    else:
        if int(max_groups) < 0:
            raise ValueError("MMBench max_samples must be non-negative.")
        selected_groups = ordered_groups[: int(max_groups)]
    selected_set = set(selected_groups)
    selected = df[
        (df["index"].astype(int) % 1_000_000).isin(selected_set)
    ].copy()
    return selected, selected_groups


def _mmbench_incomplete_groups(
    expected_df: pd.DataFrame,
    outputs: list[dict[str, Any]],
) -> dict[int, list[int]]:
    expected: dict[int, set[int]] = {}
    for index in expected_df["index"].astype(int).tolist():
        expected.setdefault(index % 1_000_000, set()).add(index)
    observed: dict[int, set[int]] = {}
    for item in outputs:
        index = int(item["index"])
        observed.setdefault(index % 1_000_000, set()).add(index)
    return {
        group_id: sorted(indices - observed.get(group_id, set()))
        for group_id, indices in expected.items()
        if indices - observed.get(group_id, set())
    }


def _mmbench_resolve_image_blob(image_map: dict[int, str], idx: int) -> str:
    value = image_map.get(idx, "")
    if not isinstance(value, str):
        return ""
    if len(value) <= 64:
        try:
            target = int(value)
        except (TypeError, ValueError):
            return value
        target_value = image_map.get(target, "")
        if isinstance(target_value, str) and len(target_value) > 64:
            return target_value
    return value


def _mmbench_decode_image(image_b64: str, image_dir: Path, row_index: int) -> Path:
    # Evaluation artifacts may store an absolute image path instead of an
    # inline base64 blob.  This is useful when reconstructing a benchmark
    # manifest from a prior official run without duplicating gigabytes of
    # image data into a TSV file.
    image_ref = str(image_b64).strip()
    if image_ref.startswith("file://"):
        image_ref = image_ref[7:]
    image_path = Path(image_ref).expanduser()
    try:
        if image_path.is_file():
            return image_path
    except OSError:
        # A base64 blob can begin with a slash (for example, JPEG blobs often
        # start with "/9j/").  Treat an overlong pseudo-path as inline data.
        pass

    staged_root = os.environ.get("MMBENCH_STAGED_IMAGE_ROOT", "").strip()
    if staged_root:
        staged_path = Path(staged_root).expanduser() / image_path.name
        if staged_path.is_file():
            return staged_path

    image_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(base64.b64decode(image_ref))).convert("RGB")
    image_path = image_dir / f"{row_index}.png"
    image.save(image_path, format="PNG")
    return image_path


_MMBENCH_REJECT_TO_ANSWER = [
    "Sorry, I can't help with images of people yet.",
    "I can't process this file.",
    "I'm sorry, but without the image provided",
    "Cannot determine the answer",
]
_MMBENCH_EXACT_CHARS_TO_SPACE = ".()[],:;!*#{}"
_MMBENCH_EXPLICIT_OPTION = re.compile(
    r"(?:option|choice|answer(?:\s+is)?|选项|答案(?:是|为)?)\s*[:：]?\s*([A-Z])\b",
    re.IGNORECASE,
)


def _mmbench_moe_post_process(
    prediction: Any,
    choices: dict[str, str],
) -> str:
    """Mirror the local MoE MMBench evaluator's prediction normalization."""

    answer = str(prediction).strip()
    if len(answer) == 1:
        return answer
    if not answer:
        return "C"
    if answer[0] in choices:
        return answer[0]
    for label, option_text in choices.items():
        if str(option_text) in answer:
            return label
    return answer


def _mmbench_can_infer_option(answer: str, choices: dict[str, str]) -> str | bool:
    if "Failed to obtain answer via API" in answer:
        return False
    for err in _MMBENCH_REJECT_TO_ANSWER:
        if err in answer:
            return "Z"

    # Match eval/vlm/eval/mmbench/evaluate_mmbench.py::post_process: direct
    # decodes such as "A.", "A。" and "A because ..." resolve to their first
    # option character before circular scoring.
    stripped = answer.strip()
    if stripped and stripped[0] in choices:
        return stripped[0]

    answer_mod = copy.copy(answer)
    for char in _MMBENCH_EXACT_CHARS_TO_SPACE:
        answer_mod = answer_mod.replace(char, " ")
    splits = [item.strip() for item in answer_mod.split()]

    count = sum(1 for cand in choices if cand in splits)
    if count == 1:
        for cand in choices:
            if cand in splits and splits.index(cand) > len(splits) - 5:
                return cand
    elif count == 0 and "Z" in splits:
        return "Z"
    return False


def _mmbench_can_infer_text(answer: str, choices: dict[str, str]) -> str | bool:
    answer_low = answer.lower()
    total_len = sum(len(str(value)) for value in choices.values())
    if len(answer_low) > 2 * total_len:
        return False
    lowered = {key: str(value).lower() for key, value in choices.items()}
    cands = [key for key, value in lowered.items() if value in answer_low]
    if len(cands) == 1:
        return cands[0]
    return False


def _mmbench_can_infer(answer: Any, choices: dict[str, str]) -> str | bool:
    answer = str(answer)
    by_option = _mmbench_can_infer_option(answer, choices)
    if by_option:
        return by_option
    explicit = _MMBENCH_EXPLICIT_OPTION.findall(answer)
    explicit = [candidate.upper() for candidate in explicit if candidate.upper() in choices]
    if explicit and len(set(explicit)) == 1:
        return explicit[-1]
    return _mmbench_can_infer_text(answer, choices)


def _mmbench_build_option_str(options: dict[str, str]) -> str:
    option_str = "There are several options: \n"
    for cand, content in options.items():
        if content is None:
            continue
        option_str += f"{cand}. {content}\n"
    return option_str


def _mmbench_cn_string(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


_MMBENCH_JUDGE_PROMPT_EN = (
    "You are an AI assistant who will help me to match an answer with several options "
    "of a single-choice question. You are provided with a question, several options, "
    "and an answer, and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
    "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)

_MMBENCH_JUDGE_PROMPT_CN = (
    "你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。"
    "你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。"
    "如果所有选项的意义都与答案显著不同，则输出 Z。"
    "你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。"
    "问题: {}?\n选项: {}\n答案: {}\n输出: "
)


class _MMBenchJudge:
    def __init__(self, model_path: str, max_new_tokens: int = 32) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[mmbench] loading judge LLM: {model_path}", flush=True)
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        raw = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        return raw

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _mmbench_extract_answer(item: dict[str, Any], judge: _MMBenchJudge | None) -> dict[str, str]:
    choices = dict(item.get("choices") or item.get("options") or {})
    prediction = str(item.get("prediction", ""))
    processed = _mmbench_moe_post_process(prediction, choices)
    if processed in choices:
        return {"opt": processed, "log": prediction}
    if judge is None:
        return {
            "opt": "Z",
            "log": "MoE post-processing found no exact option and no judge LLM is configured.",
        }

    ret = _mmbench_can_infer(prediction, choices)
    if ret:
        return {"opt": str(ret), "log": prediction}

    option_str = _mmbench_build_option_str(choices)
    language = str(item.get("language", "en"))
    template = _MMBENCH_JUDGE_PROMPT_CN if language == "cn" or _mmbench_cn_string(str(item.get("question", ""))) else _MMBENCH_JUDGE_PROMPT_EN
    prompt = template.format(item.get("question", ""), option_str, prediction)
    for _ in range(3):
        judged = judge.generate(prompt)
        ret = _mmbench_can_infer(judged, choices)
        if ret:
            return {"opt": str(ret), "log": judged}
    options_pool = list(choices) + (["Z"] if "Z" not in choices else [])
    return {"opt": random.choice(options_pool), "log": "Judge failed; chose a fallback option."}


def _mmbench_prefetch_group(sub_items: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[Any], list[Any]]:
    gts: list[Any] = []
    preds: list[Any] = []
    for idx, item in enumerate(sub_items):
        gts.append(item.get("gt_answer"))
        choices = dict(item.get("choices", {}))
        processed = _mmbench_moe_post_process(
            item.get("prediction", ""),
            choices,
        )
        preds.append(processed if processed in choices else False)
        if preds[-1] and gts[-1] != preds[-1]:
            return (
                {
                    "hit": 0,
                    "log": f"Failed in prefetch rotation {idx}: gt={gts[-1]} pred={item.get('prediction', '')} inferred={preds[-1]}",
                },
                gts,
                preds,
            )
    if all(g == p for g, p in zip(gts, preds)):
        return {"hit": 1, "log": "Succeed during prefetch"}, gts, preds
    return None, gts, preds


def _mmbench_eval_group(sub_items: list[dict[str, Any]], judge: _MMBenchJudge | None) -> dict[str, Any]:
    result, gts, preds = _mmbench_prefetch_group(sub_items)
    if result is not None:
        return result
    log = ""
    for idx, item in enumerate(sub_items):
        if preds[idx]:
            log += f"Rotation {idx} matched by prefetch.\n"
            continue
        extracted = _mmbench_extract_answer(item, judge)
        item["extraction"] = extracted["opt"]
        preds[idx] = extracted["opt"]
        if preds[idx] != gts[idx]:
            log += f"Failed rotation {idx}: gt={gts[idx]} inferred={preds[idx]} log={extracted['log']}\n"
            return {"hit": 0, "log": log}
    return {"hit": 1, "log": log}


def _mmbench_report_acc(df: pd.DataFrame) -> dict[str, float]:
    if "split" not in df.columns:
        df = df.copy()
        df["split"] = "none"
    splits = sorted(df["split"].dropna().unique().tolist())
    metrics: dict[str, float] = {}
    for group in (None, "l2-category", "category"):
        if group is None:
            for split in splits:
                sub = df[df["split"] == split]["hit"]
                metrics[f"split={split}|Overall"] = float(sub.mean()) if len(sub) else float("nan")
        elif group in df.columns:
            for ability in sorted(df[group].dropna().unique().tolist()):
                ability_name = MMBENCH_ABBRS.get(ability, ability)
                sub_df = df[df[group] == ability]
                for split in splits:
                    sub = sub_df[sub_df["split"] == split]["hit"]
                    metrics[f"split={split}|{ability_name}"] = float(sub.mean()) if len(sub) else float("nan")
    return metrics


def _mathvista_quick_extract(response: str, problem: dict[str, Any]) -> str | None:
    if response == "":
        return ""
    question_type = problem.get("question_type", "")
    answer_type = problem.get("answer_type", "")
    choices = problem.get("choices") or []

    if question_type == "multi_choice":
        stripped_response = str(response).strip()
        choice_texts = [str(choice).strip() for choice in choices]
        if stripped_response in choice_texts:
            return str(choices[choice_texts.index(stripped_response)])
        lowered_choices = {
            choice.casefold(): index
            for index, choice in enumerate(choice_texts)
        }
        if stripped_response.casefold() in lowered_choices:
            return str(choices[lowered_choices[stripped_response.casefold()]])
        option_match = re.search(
            r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|：)?\s*\(?([A-Z])\)?",
            response,
            re.IGNORECASE,
        )
        if option_match:
            index = ord(option_match.group(1).upper()) - ord("A")
            if 0 <= index < len(choices):
                return str(choices[index])
        lowered = response.lower()
        matching_choices = [
            str(choice) for choice in choices if str(choice).lower() in lowered
        ]
        if len(matching_choices) == 1:
            return matching_choices[0]
    if answer_type == "integer":
        parsed = _mathvista_parse_single_numeric_value(response)
        if parsed is not None:
            return str(int(float(parsed)))
        try:
            return str(int(response))
        except (TypeError, ValueError):
            pass
    if answer_type == "float":
        parsed = _mathvista_parse_single_numeric_value(response)
        if parsed is not None:
            return str(float(parsed))
        try:
            return str(float(response))
        except (TypeError, ValueError):
            pass
        numeric_matches = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            response.replace(",", ""),
        )
        if numeric_matches:
            return str(float(numeric_matches[-1]))
    match = re.search(
        r"(?:final\s+answer|answer)\s*(?:is|:|：)\s*(.*)",
        response,
        re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return None


_MATHVISTA_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _mathvista_parse_single_numeric_value(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, (int, float)) and not isinstance(response, bool):
        return str(response)
    text = str(response).strip()
    lowered = text.casefold().strip().strip(".")
    if lowered in _MATHVISTA_NUMBER_WORDS:
        return str(_MATHVISTA_NUMBER_WORDS[lowered])
    match = re.match(
        r"^[\s\$€£¥]*"
        r"(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)"
        r"\s*(?:%|percent|percentage)?"
        r"\s*(?:[a-zA-Z°/% ._\-]+)?"
        r"[\s.。,:;!?)\]]*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    numeric_tokens = re.findall(
        r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+",
        text,
    )
    if len(numeric_tokens) != 1:
        return None
    return match.group("num").replace(",", "")


def _mathvista_run_local_extraction(
    results: dict[str, Any],
    model_path: str,
    max_new_tokens: int,
    use_quick_extract: bool = True,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval.vlm.eval.mathvista.prompts.ext_ans import demo_prompt

    need_llm: list[str] = []
    for pid, problem in results.items():
        if problem.get("extraction"):
            continue
        if use_quick_extract:
            quick = _mathvista_quick_extract(str(problem.get("response", "")), problem)
            if quick is not None:
                problem["extraction"] = quick
                continue
        need_llm.append(pid)

    if not need_llm:
        return results

    print(f"[mathvista] loading extraction LLM: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for pid in tqdm(need_llm, desc="MathVista extraction"):
        problem = results[pid]
        test_prompt = f"{problem.get('query', '')}\n\n{problem.get('response', '')}"
        prompt = f"{demo_prompt.strip()}\n\n{test_prompt}\n\nExtracted answer: "
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        extraction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in extraction:
            extraction = extraction.split("</think>", 1)[1].strip()
        problem["extraction"] = extraction.split("\n", 1)[0].strip()

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _mmstar_prompt(question: str) -> str:
    style = os.environ.get("FLEXUMM_MMSTAR_PROMPT_STYLE", "").strip().lower()
    if style in {"careful", "direct_careful"}:
        suffix = (
            "Look carefully at the image and all options. "
            "Answer with only the option letter."
        )
    elif style in {"cot", "reasoning", "brief_reasoning"}:
        suffix = (
            "Think briefly about the image and options, then end with "
            "\"Final answer: <option letter>\"."
        )
    else:
        suffix = "Answer with the option's letter from the given choices directly."
    return question if suffix.lower() in question.lower() else f"{question.strip()}\n{suffix}"


def _mmstar_options(question: str) -> dict[str, str]:
    delimiter = re.search(
        r"\b(?:Options|Choices)\s*:",
        question,
        flags=re.IGNORECASE,
    )
    if delimiter is not None:
        option_text = question[delimiter.end():]
    else:
        first_option = re.search(
            r"(?:^|[\n\r])\s*(?:\([A-H]\)|[A-H]\s*[:：.)）])\s+",
            question,
            flags=re.IGNORECASE,
        )
        if first_option is None:
            return {}
        option_text = question[first_option.start():]
    matches = list(
        re.finditer(
            r"(?:^|[\n\r]|(?<=\s))\s*(?:\(([A-H])\)|([A-H])\s*[:：.)）])\s*",
            option_text,
            flags=re.IGNORECASE,
        )
    )
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(option_text)
        label = (match.group(1) or match.group(2)).upper()
        options[label] = option_text[start:end].strip().rstrip(",")
    return options


def _mmstar_extract_option(response: str, options: dict[str, str]) -> str:
    candidates = set(options) if options else set("ABCDEFGH")
    text = str(response or "")
    for pattern in (
        r"(?:final answer|answer|option|choice)\s*(?:is|:|：)?\s*\(?([A-H])\)?",
        r"^\s*\(?([A-H])\)?\s*(?:[:：.)]|$)",
        r"\b([A-H])\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in candidates:
            return match.group(1).upper()
    lowered = text.lower()
    for letter, value in options.items():
        if str(value).strip().lower() in lowered:
            return letter
    return ""


def _mmstar_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        name = str(row.get(key) or "unknown")
        bucket = buckets.setdefault(name, {"correct": 0, "total": 0})
        bucket["correct"] += int(bool(row.get("answer_ok")))
        bucket["total"] += 1
    return {
        name: {
            "accuracy": round(100.0 * values["correct"] / max(values["total"], 1), 2),
            "correct": values["correct"],
            "total": values["total"],
        }
        for name, values in sorted(buckets.items())
    }


def _write_mmstar_score(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    root: str,
    split: str,
) -> None:
    scored_rows = [row for row in rows if row.get("gt_answer")]
    correct = sum(1 for row in scored_rows if row["answer_ok"])
    total = len(scored_rows)
    accuracy = 100.0 * correct / max(total, 1)
    score = {
        "benchmark": "mmstar",
        "dataset": root,
        "split": split,
        "overall": {
            "accuracy": round(accuracy, 2),
            "correct": int(correct),
            "total": int(total),
        },
        "by_category": _mmstar_breakdown(scored_rows, "category"),
        "by_l2_category": _mmstar_breakdown(scored_rows, "l2_category"),
        "results_jsonl": str(output_dir / "results.jsonl"),
    }
    with (output_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    with (output_dir / "results.txt").open("w", encoding="utf-8") as f:
        f.write(f"Overall: acc={accuracy:.4f}\n")
        f.write(f"Correct: {correct}/{total}\n")


def _run_mme(cfg: dict[str, Any]) -> int:
    from datasets import load_dataset

    dataset_cfg = dict(cfg.get("dataset") or {})
    params = dict(cfg.get("understanding") or {})
    output_dir = _resolve_path(str(cfg.get("output_dir", "eval_outputs/mme_bagel")))
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_suffix = str(dataset_cfg.get("prompt_suffix", "Answer the question using a single word or phrase."))

    inferencer = _build_dual_inferencer(cfg)
    manifest_rows: list[dict[str, Any]] = []
    source = str(dataset_cfg.get("source", "files")).strip().lower()
    output_handles: dict[str, Any] = {}
    try:
        raw_limit = dataset_cfg.get("max_samples_per_category")
        if raw_limit in (None, ""):
            raw_limit = cfg.get("max_samples")
        row_limit = _mme_complete_pair_row_limit(raw_limit)
        if source == "hf":
            root = str(dataset_cfg.get("root", "lmms-lab/MME"))
            split = str(dataset_cfg.get("split", "test"))
            cache_dir = _resolve_path(
                str(
                    dataset_cfg.get(
                        "cache_dir",
                        "eval/vlm/data/mme_hf",
                    )
                )
            )
            dataset = load_dataset(
                root,
                split=split,
                cache_dir=str(cache_dir),
            )
            selected_rows = _mme_select_hf_rows(dataset, row_limit)
            for source_order, row in enumerate(selected_rows):
                row["_source_order"] = source_order
            selected_pairs = [
                selected_rows[index : index + 2]
                for index in range(0, len(selected_rows), 2)
            ]
            selected_rows = [
                row
                for pair in _shard_sequence(selected_pairs, cfg)
                for row in pair
            ]
            iterator = tqdm(selected_rows, desc="MME inference")
            for item in iterator:
                category = str(item["category"])
                question_id = str(item["question_id"])
                image_name = Path(question_id).name
                question = str(item["question"])
                gt = str(item["answer"])
                image = item["image"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                prompt = f"{question} {prompt_suffix}".strip()
                pred = _mme_postprocess(
                    _understanding(inferencer, [image], prompt, params)
                )
                handle = output_handles.get(category)
                if handle is None:
                    handle = (output_dir / f"{category}.txt").open(
                        "w",
                        encoding="utf-8",
                    )
                    output_handles[category] = handle
                print(image_name, question, gt, pred, sep="\t", file=handle)
                manifest_rows.append(
                    {
                        "category": category,
                        "question_id": question_id,
                        "image": image_name,
                        "question": question,
                        "gt": gt,
                        "answer": pred,
                        "source_order": item.get("_source_order"),
                        "routing": copy.deepcopy(
                            getattr(
                                inferencer,
                                "last_prompt_router_decision",
                                None,
                            )
                        ),
                    }
                )
        elif source == "files":
            question_root = _resolve_path(
                str(
                    dataset_cfg.get(
                        "question_root",
                        "eval/vlm/eval/mme/Your_Results",
                    )
                ),
                must_exist=True,
            )
            image_root = _resolve_path(
                str(
                    dataset_cfg.get(
                        "image_root",
                        "eval/vlm/data/mme/MME_Benchmark_release_version",
                    )
                )
            )
            for question_file in sorted(question_root.glob("*.txt")):
                category = question_file.stem
                out_file = output_dir / question_file.name
                with (
                    question_file.open("r", encoding="utf-8") as fin,
                    out_file.open("w", encoding="utf-8") as fout,
                ):
                    lines = [line.strip() for line in fin if line.strip()]
                    if len(lines) % 2:
                        raise ValueError(
                            f"MME source category {category!r} has an "
                            f"odd row count: {len(lines)}"
                        )
                    if row_limit is not None:
                        lines = lines[: min(len(lines), row_limit)]
                    indexed_lines = list(enumerate(lines))
                    line_pairs = [
                        indexed_lines[index : index + 2]
                        for index in range(0, len(indexed_lines), 2)
                    ]
                    sharded_lines = [
                        item
                        for pair in _shard_sequence(line_pairs, cfg)
                        for item in pair
                    ]
                    for category_row_index, line in tqdm(
                        sharded_lines,
                        desc=f"MME {category}",
                    ):
                        image_name, question, gt = line.split("\t")[:3]
                        image_path = image_root / category / image_name
                        if not image_path.exists():
                            image_path = (
                                image_root
                                / category
                                / "images"
                                / image_name
                            )
                        if not image_path.exists():
                            raise FileNotFoundError(image_path)
                        prompt = f"{question} {prompt_suffix}".strip()
                        pred = _mme_postprocess(
                            _understanding(
                                inferencer,
                                [_load_image(image_path)],
                                prompt,
                                params,
                            )
                        )
                        print(
                            image_name,
                            question,
                            gt,
                            pred,
                            sep="\t",
                            file=fout,
                        )
                        manifest_rows.append(
                            {
                                "category": category,
                                "question_id": (
                                    f"{category}/{image_name}"
                                ),
                                "image": image_name,
                                "question": question,
                                "gt": gt,
                                "answer": pred,
                                "category_row_index": category_row_index,
                                "routing": copy.deepcopy(
                                    getattr(
                                        inferencer,
                                        "last_prompt_router_decision",
                                        None,
                                    )
                                ),
                            }
                        )
        else:
            raise ValueError(
                f"Unsupported MME dataset source: {source!r}"
            )
    finally:
        for handle in output_handles.values():
            handle.close()

    _write_jsonl(output_dir / "results.jsonl", manifest_rows)
    if bool(cfg.get("run_scoring", False)):
        _write_mme_score(output_dir, manifest_rows)
    return 0


def _run_mmstar(cfg: dict[str, Any]) -> int:
    import torch
    from datasets import load_dataset

    dataset_cfg = dict(cfg.get("dataset") or {})
    params = dict(cfg.get("understanding") or {})
    root = str(dataset_cfg.get("root", "Lin-Chen/MMStar"))
    split = str(dataset_cfg.get("split", "val"))
    cache_dir = _resolve_path(str(dataset_cfg.get("cache_dir", "eval/vlm/data/MMStar")))
    output_dir = _resolve_path(str(cfg.get("output_dir", "eval_outputs/mmstar_bagel")))
    output_dir.mkdir(parents=True, exist_ok=True)
    _mode, run_gen, run_score = _dataset_mode(
        dataset_cfg,
        cfg,
        default_score=True,
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}

    if run_gen:
        local_parquet = cache_dir / "MMStar" / "mmstar.parquet"
        if local_parquet.is_file():
            # The dataset volume stores the Hub export as a parquet file.  Use
            # it directly when Hub access is disabled, preserving its row
            # order and converting the binary JPEG column below.
            dataset = load_dataset(
                "parquet", data_files=str(local_parquet), split="train"
            )
        else:
            try:
                dataset = load_dataset(root, split=split, cache_dir=str(cache_dir))
            except ConnectionError as error:
                # Support an offline Hugging Face Arrow cache whose dataset
                # identifier is not discoverable by the installed datasets
                # version.  The stream contains the same real MMStar rows.
                if "Couldn't reach" not in str(error):
                    raise
                import pyarrow as pa

                arrow_candidates = sorted(cache_dir.rglob(f"*{split}.arrow"))
                if len(arrow_candidates) != 1:
                    raise RuntimeError(
                        "Could not identify a unique cached MMStar Arrow "
                        f"stream for split={split!r}: {arrow_candidates}"
                    ) from error
                with pa.ipc.open_stream(str(arrow_candidates[0])) as reader:
                    dataset = reader.read_all().to_pylist()
        max_samples = cfg.get("max_samples")
        if max_samples is not None:
            dataset = dataset.select(range(min(int(max_samples), len(dataset))))
        dataset = _shard_hf_dataset(dataset, cfg)

        inferencer = _build_dual_inferencer(cfg)
        try:
            for item in tqdm(dataset, desc="MMStar inference"):
                data = dict(item)
                image = data.pop("image")
                if isinstance(image, dict):
                    image_bytes = image.get("bytes")
                    if not image_bytes:
                        raise ValueError(
                            "Cached MMStar image has no embedded image bytes."
                        )
                    with Image.open(BytesIO(image_bytes)) as decoded:
                        image = decoded.convert("RGB")
                if isinstance(image, (bytes, bytearray)):
                    with Image.open(BytesIO(image)) as decoded:
                        image = decoded.convert("RGB")
                if image.mode != "RGB":
                    image = image.convert("RGB")
                index = int(data.get("index", len(rows)))
                question = str(data["question"])
                options = _mmstar_options(question)
                depth_task = _mmstar_router_task(data)
                inferencer.depth_task = depth_task
                raw_answer = _understanding_choice(
                    inferencer,
                    [image],
                    _mmstar_prompt(question),
                    params,
                    options,
                )
                answer = _mmstar_extract_option(raw_answer, options)
                gt_answer = str(data.get("answer", "")).strip().upper()
                predictions[str(index)] = answer
                routing = copy.deepcopy(
                    getattr(inferencer, "last_prompt_router_decision", None)
                )
                rows.append(
                    {
                        "index": index,
                        "question": question,
                        "prediction": answer,
                        "raw_prediction": raw_answer,
                        "gt_answer": gt_answer,
                        "answer_ok": bool(gt_answer) and answer == gt_answer,
                        "category": data.get("category"),
                        "l2_category": data.get("l2_category"),
                        "meta_info": data.get("meta_info"),
                        "depth_task": depth_task,
                        "routing": routing,
                    }
                )
        finally:
            del inferencer

        with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        _write_jsonl(output_dir / "results.jsonl", rows)
    elif (output_dir / "results.jsonl").exists():
        rows = _read_jsonl(output_dir / "results.jsonl")
    else:
        raise FileNotFoundError(output_dir / "results.jsonl")

    if run_score or bool(cfg.get("run_scoring", False)):
        _write_mmstar_score(output_dir, rows, root=root, split=split)

    torch.cuda.empty_cache()
    return 0


def _run_mmmu(cfg: dict[str, Any]) -> int:
    import torch
    from datasets import Dataset, concatenate_datasets, load_dataset
    from eval.vlm.eval.mmmu.data_utils import CAT_SHORT2LONG, process_single_sample
    from eval.vlm.eval.mmmu.evaluate_mmmu import post_process

    dataset_cfg = dict(cfg.get("dataset") or {})
    params = dict(cfg.get("understanding") or {})
    root = str(dataset_cfg.get("root", "MMMU/MMMU"))
    split = str(dataset_cfg.get("split", "validation"))
    output_dir = _resolve_path(str(cfg.get("output_dir", "eval_outputs/mmmu_bagel")))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve_path(str(dataset_cfg.get("cache_dir", "eval/vlm/data/MMMU")))
    _mode, run_gen, run_score = _dataset_mode(dataset_cfg, cfg)

    prompt_map = dict(
        dataset_cfg.get("prompt")
        or {
            "multiple-choice": "Answer with the option's letter from the given choices directly.",
            "open": "Answer the question using a single word or phrase.",
        }
    )

    rows: list[dict[str, Any]] = []
    predictions: dict[str, str] = {}

    if run_gen:
        subject_names = dataset_cfg.get("subjects") or list(CAT_SHORT2LONG.values())
        sub_datasets = []
        for subject in tqdm(subject_names, desc="Loading MMMU subjects"):
            try:
                dataset = load_dataset(
                    root,
                    subject,
                    split=split,
                    cache_dir=str(cache_dir),
                )
            except ConnectionError as error:
                # Some cluster wheels cannot discover a dataset cached by a
                # nearby `datasets` release while offline.  The cache still
                # contains the canonical Arrow stream, so load that exact
                # subject/split without contacting the Hub.
                subject_cache = cache_dir / "MMMU___mmmu" / subject
                arrow_paths = sorted(
                    subject_cache.rglob(f"mmmu-{split}.arrow")
                )
                if len(arrow_paths) != 1:
                    raise RuntimeError(
                        "Could not identify a unique cached MMMU Arrow "
                        f"stream for subject={subject!r}, split={split!r}: "
                        f"{arrow_paths}"
                    ) from error
                dataset = Dataset.from_file(str(arrow_paths[0]))
            sub_datasets.append(dataset)
        dataset = concatenate_datasets(sub_datasets)
        max_samples = cfg.get("max_samples")
        if max_samples is not None:
            dataset = dataset.select(range(min(int(max_samples), len(dataset))))
        dataset = _shard_hf_dataset(dataset, cfg)

        inferencer = _build_dual_inferencer(cfg)
        try:
            for item in tqdm(dataset, desc="MMMU inference"):
                data = process_single_sample(item)
                question = data["question"].strip()
                choices = ast.literal_eval(data["options"])
                options = {}
                letters = list("ABCDEFGHIJKLM")
                if choices:
                    choice_lines = []
                    for idx, choice in enumerate(choices):
                        letter = letters[idx]
                        choice_lines.append(f"{letter}. {choice.strip()}")
                        options[letter] = choice.strip()
                    question += "\n" + "\n".join(choice_lines)
                question += "\n" + str(prompt_map[data["question_type"]])
                images = [
                    image.convert("RGB") if image.mode != "RGB" else image
                    for image in data["image"]
                    if image is not None
                ]
                if images:
                    images[0] = images[0].resize(
                        (images[0].width * 2, images[0].height * 2),
                        Image.BILINEAR,
                    )

                depth_task = _mmmu_router_task(data["id"])
                inferencer.depth_task = depth_task
                pred = (
                    _understanding_choice(
                        inferencer,
                        images,
                        question.strip(),
                        params,
                        options,
                    )
                    if options
                    else _understanding(inferencer, images, question.strip(), params)
                )
                answer = post_process(pred, options) if options else pred
                predictions[data["id"]] = answer
                routing = copy.deepcopy(
                    getattr(inferencer, "last_prompt_router_decision", None)
                )
                rows.append(
                    {
                        "question": question,
                        "answer": answer,
                        "raw_answer": pred,
                        "gt_answers": data.get("answer"),
                        "data_id": data["id"],
                        "depth_task": depth_task,
                        "routing": routing,
                    }
                )
        finally:
            del inferencer

        with (output_dir / "prediction.json").open("w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        _write_jsonl(output_dir / "results.jsonl", rows)

    if split == "validation" and (run_score or bool(cfg.get("run_scoring", False))):
        if not (output_dir / "prediction.json").exists():
            raise FileNotFoundError(output_dir / "prediction.json")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "eval.vlm.eval.mmmu.main_eval_only",
                "--output_path",
                str(output_dir / "prediction.json"),
                "--answer_path",
                "eval/vlm/eval/mmmu/answer_dict_val.json",
                "--out-dir",
                str(output_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    torch.cuda.empty_cache()
    return 0


def _run_mmbench(cfg: dict[str, Any]) -> int:
    dataset_cfg = dict(cfg.get("dataset") or cfg.get("mmbench") or {})
    params = dict(cfg.get("understanding") or {})
    output_dir = _resolve_path(str(cfg.get("output_dir", "eval_outputs/mmbench_bagel")))
    image_dir = _resolve_path(str(dataset_cfg.get("image_dir", output_dir / "images")))
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_value = dataset_cfg.get("datasets", dataset_cfg.get("name", "mmbench_dev_20230712"))
    if isinstance(datasets_value, str):
        datasets = [item.strip() for item in datasets_value.split(",") if item.strip()]
    else:
        datasets = [str(item).strip() for item in datasets_value if str(item).strip()]
    if not datasets:
        raise ValueError("MMBench config needs at least one dataset.")

    dataset_paths_cfg = dataset_cfg.get("dataset_paths", {})
    if not isinstance(dataset_paths_cfg, dict):
        dataset_paths_cfg = {}
    mode = str(dataset_cfg.get("mode", "full")).strip().lower()
    if mode not in {"full", "generate", "score"}:
        raise ValueError("MMBench dataset.mode must be one of: full, generate, score.")
    run_gen = mode in {"full", "generate"}
    run_score = mode in {"full", "score"}
    resume = bool(dataset_cfg.get("resume", True))

    judge_cfg = dict(dataset_cfg.get("llm_extract") or {})
    judge_model_path = str(judge_cfg.get("model_path", "")).strip()
    judge_max_new_tokens = int(judge_cfg.get("max_new_tokens", 32))
    score_summary: dict[str, Any] = {
        "benchmark": "mmbench",
        "datasets": datasets,
        "mode": mode,
        "out_dir": str(output_dir),
        "protocol": {
            "scoring": "MMBench v1.1 circular; all rotations required",
            "max_samples_unit": "base_question_groups",
            "prompt": "official language-specific direct-letter instruction",
            "inference_api": str(
                params.get("inference_api", "interleave")
            ).strip().lower(),
            "vit_transform": {
                "max_image_size": int(
                    (cfg.get("backbone_cfg") or {}).get(
                        "vit_max_image_size",
                        VLM_EVAL_VIT_TRANSFORM["max_image_size"],
                    )
                ),
                "min_image_size": int(
                    (cfg.get("backbone_cfg") or {}).get(
                        "vit_min_image_size",
                        VLM_EVAL_VIT_TRANSFORM["min_image_size"],
                    )
                ),
                "image_stride": int(
                    (cfg.get("backbone_cfg") or {}).get(
                        "vit_image_stride",
                        VLM_EVAL_VIT_TRANSFORM["image_stride"],
                    )
                ),
                "max_pixels": int(
                    (cfg.get("backbone_cfg") or {}).get(
                        "vit_max_pixels",
                        VLM_EVAL_VIT_TRANSFORM["max_pixels"],
                    )
                ),
            },
        },
    }

    inferencer = _build_dual_inferencer(cfg) if run_gen else None
    try:
        for ds_name in datasets:
            entry = MMBENCH_COLLECTIONS.get(ds_name, {})
            dataset_path = _resolve_path(str(dataset_paths_cfg.get(ds_name, entry.get("root", ds_name))), must_exist=True)
            language = str(dataset_cfg.get("language", entry.get("language", "en")))
            if inferencer is not None:
                inferencer.depth_task = f"mmbench_{language.lower()}"
            df = pd.read_csv(dataset_path, sep="\t")
            df["index"] = df["index"].astype(int)
            max_groups = cfg.get("max_samples")
            eval_df, selected_group_ids = _mmbench_select_group_rows(
                df,
                int(max_groups) if max_groups is not None else None,
            )
            selected_group_ids = _shard_sequence(selected_group_ids, cfg)
            if selected_group_ids:
                selected_group_set = set(selected_group_ids)
                eval_df = eval_df[
                    eval_df["index"]
                    .astype(int)
                    .mod(1_000_000)
                    .isin(selected_group_set)
                ].copy()
            else:
                eval_df = eval_df.iloc[0:0].copy()
            target_indices = set(eval_df["index"].astype(int).tolist())
            score_summary[f"{ds_name}_selected_circular_groups"] = len(
                selected_group_ids
            )
            score_summary[f"{ds_name}_selected_tsv_rows"] = len(eval_df)

            jsonl_path = output_dir / f"{ds_name}_results.jsonl"
            checkpoint_path = output_dir / f"{ds_name}_checkpoint.jsonl"
            outputs: list[dict[str, Any]] = []
            done: set[int] = set()

            if resume:
                for candidate in (checkpoint_path, jsonl_path):
                    if candidate.exists():
                        with candidate.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                item = json.loads(line)
                                if int(item["index"]) not in target_indices:
                                    continue
                                outputs.append(item)
                                done.add(int(item["index"]))
                        print(f"[mmbench] resume {ds_name}: {len(done)} done from {candidate}", flush=True)
                        break

            if run_gen:
                if inferencer is None:
                    raise RuntimeError("MMBench generation requested without an inferencer.")
                image_map: dict[int, str] = {}
                if "image" in df.columns:
                    for _, row in df.iterrows():
                        image_map[int(row["index"])] = str(row["image"]) if not pd.isna(row["image"]) else ""

                with checkpoint_path.open("a", encoding="utf-8") as writer:
                    for _, row in tqdm(
                        eval_df.iterrows(),
                        total=len(eval_df),
                        desc=f"MMBench {ds_name} inference",
                    ):
                        row_index = int(row["index"])
                        if row_index in done:
                            continue
                        image_b64 = _mmbench_resolve_image_blob(image_map, row_index)
                        if not image_b64:
                            continue
                        options = _mmbench_options(row)
                        hint = str(row["hint"]) if "hint" in row and not pd.isna(row["hint"]) else None
                        question = str(row["question"])
                        prompt = _mmbench_prompt(
                            question,
                            options,
                            hint,
                            language,
                        )
                        image_path = _mmbench_decode_image(image_b64, image_dir, row_index)
                        prediction = _understanding_choice(
                            inferencer,
                            [_load_image(image_path)],
                            prompt,
                            params,
                            options,
                        )
                        gt_answer = None
                        if "answer" in row and not pd.isna(row["answer"]):
                            gt_answer = str(row["answer"]).strip().upper()
                        item = {
                            "index": row_index,
                            "question": question,
                            "options": options,
                            "choices": options,
                            "prediction": prediction,
                            "gt_answer": gt_answer,
                            "language": language,
                            "routing": getattr(
                                inferencer,
                                "last_prompt_router_decision",
                                None,
                            ),
                        }
                        outputs.append(item)
                        done.add(row_index)
                        writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                        writer.flush()
                        os.fsync(writer.fileno())

                with jsonl_path.open("w", encoding="utf-8") as writer:
                    for item in outputs:
                        writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                score_summary[f"{ds_name}_output_jsonl"] = str(jsonl_path)
            elif not jsonl_path.exists():
                raise FileNotFoundError(f"No MMBench generation output found: {jsonl_path}")

            if run_score:
                if not outputs:
                    with jsonl_path.open("r", encoding="utf-8") as reader:
                        outputs = [
                            item
                            for line in reader
                            if line.strip()
                            for item in [json.loads(line)]
                            if int(item["index"]) in target_indices
                        ]
                incomplete_groups = _mmbench_incomplete_groups(eval_df, outputs)
                if incomplete_groups:
                    preview = dict(list(incomplete_groups.items())[:5])
                    raise RuntimeError(
                        "MMBench circular scoring requires every option "
                        f"rotation; {len(incomplete_groups)} groups are "
                        f"incomplete (missing-index preview: {preview})."
                    )
                for item in outputs:
                    item["choices"] = dict(item.get("choices") or item.get("options") or {})

                groups: dict[int, list[dict[str, Any]]] = {}
                for item in outputs:
                    groups.setdefault(int(item["index"]) % 1_000_000, []).append(item)
                for group_items in groups.values():
                    group_items.sort(key=lambda item: int(item["index"]))

                group_results: dict[int, dict[str, Any]] = {}
                pending: list[tuple[int, list[dict[str, Any]]]] = []
                for group_id, group_items in groups.items():
                    prefetch, _gts, _preds = _mmbench_prefetch_group(group_items)
                    if prefetch is None:
                        pending.append((group_id, group_items))
                    else:
                        group_results[group_id] = prefetch

                judge: _MMBenchJudge | None = None
                try:
                    if pending and judge_model_path:
                        if inferencer is not None:
                            del inferencer
                            inferencer = None
                        judge = _MMBenchJudge(judge_model_path, max_new_tokens=judge_max_new_tokens)
                    for group_id, group_items in tqdm(pending, desc=f"MMBench {ds_name} scoring"):
                        group_results[group_id] = _mmbench_eval_group(group_items, judge)
                finally:
                    if judge is not None:
                        judge.close()

                df_work = eval_df.copy()
                df_work["g_index"] = df_work["index"].astype(int) % 1_000_000
                generated_groups = set(group_results)
                df_scored = df_work[df_work["g_index"].isin(generated_groups)]
                df_scored = df_scored[df_scored["index"].astype(int) == df_scored["g_index"].astype(int)].copy()
                df_scored["hit"] = df_scored["g_index"].map(lambda group_id: int(group_results[int(group_id)]["hit"]))
                df_scored["log"] = df_scored["g_index"].map(lambda group_id: group_results[int(group_id)]["log"])

                metrics = _mmbench_report_acc(df_scored)
                overall_vals = [value for key, value in metrics.items() if key.endswith("|Overall")]
                overall_acc = 100.0 * (overall_vals[0] if overall_vals else 0.0)
                hit_count = int(df_scored["hit"].sum()) if "hit" in df_scored else 0
                question_count = int(len(df_scored))

                for item in outputs:
                    group_id = int(item["index"]) % 1_000_000
                    result = group_results.get(group_id, {"hit": 0, "log": ""})
                    item["group_hit"] = int(result.get("hit", 0))
                with jsonl_path.open("w", encoding="utf-8") as writer:
                    for item in outputs:
                        writer.write(json.dumps(item, ensure_ascii=False) + "\n")

                score_payload = {
                    "overall": {
                        "accuracy": round(overall_acc, 2),
                        "hit_count": hit_count,
                        "question_count": question_count,
                        "mode": "circular",
                    },
                    "metrics": {key: round(100.0 * value, 2) for key, value in metrics.items()},
                    "results_jsonl": str(jsonl_path),
                }
                score_path = output_dir / f"{ds_name}_score.json"
                with score_path.open("w", encoding="utf-8") as f:
                    json.dump(score_payload, f, indent=2, ensure_ascii=False)
                score_summary[f"{ds_name}_score_file"] = str(score_path)
                score_summary[f"{ds_name}_accuracy"] = round(overall_acc, 2)

        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(score_summary, f, indent=2, ensure_ascii=False)
    finally:
        if inferencer is not None:
            del inferencer

    return 0


def _run_mathvista(cfg: dict[str, Any]) -> int:
    from datasets import Dataset, load_dataset

    dataset_cfg = dict(cfg.get("dataset") or {})
    params = dict(cfg.get("understanding") or {})
    root = str(dataset_cfg.get("root", "AI4Math/MathVista"))
    split = str(dataset_cfg.get("split", "testmini"))
    output_dir = _resolve_path(str(cfg.get("output_dir", "eval_outputs/mathvista_bagel")))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve_path(str(dataset_cfg.get("cache_dir", "eval/vlm/data/MathVista")))
    cot = bool(dataset_cfg.get("cot", False))
    mode = str(dataset_cfg.get("mode", "full" if bool(cfg.get("run_scoring", False)) else "generate")).lower()
    if mode not in {"full", "generate", "score"}:
        raise ValueError("MathVista dataset.mode must be one of: full, generate, score.")
    run_gen = mode in {"full", "generate"}
    run_score = mode in {"full", "score"} or bool(cfg.get("run_scoring", False))
    results_path = output_dir / "results.json"

    if run_gen:
        try:
            data = load_dataset(root, cache_dir=str(cache_dir))[split]
        except (ConnectionError, TypeError) as error:
            # Some cached MathVista exports use the legacy ``{"_type":
            # "Image"}`` feature metadata, which newer datasets versions
            # reject while the Arrow stream itself remains valid.  Read the
            # local cached stream directly and preserve its decoded image
            # bytes so this benchmark remains offline and deterministic.
            if isinstance(error, TypeError) and "dataclass type" not in str(error):
                raise
            arrow_candidates = sorted(
                cache_dir.rglob(f"*math_vista-{split}.arrow")
            )
            if len(arrow_candidates) != 1:
                raise RuntimeError(
                    "Could not identify a unique cached MathVista Arrow "
                    f"stream for split={split!r}: {arrow_candidates}"
                ) from error
            data = Dataset.from_file(str(arrow_candidates[0]))
        max_samples = cfg.get("max_samples")
        if max_samples is not None:
            limit = min(int(max_samples), len(data))
            data = (
                data.select(range(limit))
                if hasattr(data, "select")
                else data[:limit]
            )
        data = _shard_hf_dataset(data, cfg)

        inferencer = _build_dual_inferencer(cfg)
        outputs: dict[str, dict[str, Any]] = {}
        cot_instruction = (
            "Your task is to answer the question below. Give step by step reasoning before you answer, "
            'and when you are ready to answer, please use the format "Final answer: .."\n\nQuestion:\n\n{question}'
        )

        for item in tqdm(data, desc="MathVista inference"):
            data_item = dict(item)
            decoded_image = data_item.pop("decoded_image")
            if isinstance(decoded_image, dict):
                decoded_bytes = decoded_image.get("bytes")
                if not decoded_bytes:
                    raise ValueError(
                        "Cached MathVista decoded_image has no image bytes."
                    )
                with Image.open(BytesIO(decoded_bytes)) as opened:
                    image = opened.convert("RGB")
            else:
                image = decoded_image.convert("RGB")
            question = data_item["query"]
            if cot:
                question = cot_instruction.format(question=question)
            raw_choices = data_item.get("choices") or []
            mathvista_choices = (
                {
                    string.ascii_uppercase[index]: str(choice)
                    for index, choice in enumerate(raw_choices)
                }
                if isinstance(raw_choices, list)
                else {}
            )
            depth_task = _mathvista_router_task(data_item)
            inferencer.depth_task = depth_task
            pred = (
                _understanding_choice(
                    inferencer,
                    [image],
                    question,
                    params,
                    mathvista_choices,
                )
                if mathvista_choices
                else _understanding(inferencer, [image], question, params)
            )
            if mathvista_choices and pred in mathvista_choices:
                pred = mathvista_choices[pred]
            data_item["response"] = pred
            data_item["depth_task"] = depth_task
            data_item["routing"] = copy.deepcopy(
                getattr(inferencer, "last_prompt_router_decision", None)
            )
            outputs[str(data_item["pid"])] = data_item

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    if run_score:
        if not results_path.exists():
            raise FileNotFoundError(f"MathVista results file not found for scoring: {results_path}")
        llm_extract = dict(dataset_cfg.get("llm_extract") or {})
        llm_model_path = str(llm_extract.get("model_path", "")).strip()
        if llm_model_path:
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
            results = _mathvista_run_local_extraction(
                results,
                model_path=llm_model_path,
                max_new_tokens=int(llm_extract.get("max_new_tokens", 2048)),
                use_quick_extract=bool(dataset_cfg.get("quick_extract", True)),
            )
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
            unresolved = []
            if bool(dataset_cfg.get("quick_extract", True)):
                for pid, problem in results.items():
                    extraction = _mathvista_quick_extract(
                        str(problem.get("response", "")), problem
                    )
                    if extraction is None:
                        unresolved.append(pid)
                    else:
                        problem["extraction"] = extraction
                with results_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                unresolved = list(results)

        if (
            not llm_model_path
            and unresolved
            and bool(dataset_cfg.get("allow_external_extract", False))
        ):
            extract_cmd = [
                sys.executable,
                "eval/vlm/eval/mathvista/extract_answer_mp.py",
                "--output_file",
                "results.json",
                "--output_dir",
                str(output_dir),
            ]
            if bool(dataset_cfg.get("quick_extract", True)):
                extract_cmd.append("--quick_extract")
            subprocess.run(extract_cmd, cwd=REPO_ROOT, check=True)
        elif not llm_model_path and unresolved:
            # Keep evaluation fully local and deterministic when no judge is
            # configured. The official normalizer can directly handle exact
            # choices and simple scalar responses; unresolved prose is scored
            # as incorrect rather than silently invoking an external API.
            for pid in unresolved:
                results[pid]["extraction"] = str(
                    results[pid].get("response", "")
                ).strip()
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        subprocess.run(
            [
                sys.executable,
                "eval/vlm/eval/mathvista/calculate_score.py",
                "--output_file",
                "results.json",
                "--output_dir",
                str(output_dir),
                "--score_file",
                "score.json",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    return 0


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return cfg
    cfg = dict(cfg)
    if getattr(args, "model_path", None):
        backbone_cfg = dict(cfg.get("backbone_cfg") or {})
        backbone_cfg["model_path"] = args.model_path
        cfg["backbone_cfg"] = backbone_cfg
    if getattr(args, "checkpoint_path", None):
        backbone_cfg = dict(cfg.get("backbone_cfg") or {})
        backbone_cfg["checkpoint_path"] = args.checkpoint_path
        cfg["backbone_cfg"] = backbone_cfg
    if getattr(args, "offload_folder", None):
        backbone_cfg = dict(cfg.get("backbone_cfg") or {})
        backbone_cfg["offload_folder"] = args.offload_folder
        cfg["backbone_cfg"] = backbone_cfg
    if getattr(args, "output_dir", None):
        cfg["output_dir"] = args.output_dir
    if getattr(args, "max_samples", None) is not None:
        cfg["max_samples"] = args.max_samples
    if getattr(args, "num_shards", None) is not None:
        num_shards = int(args.num_shards)
        if num_shards < 1:
            raise ValueError("--num-shards must be at least 1.")
        cfg["num_shards"] = num_shards
    if getattr(args, "shard_index", None) is not None:
        shard_index = int(args.shard_index)
        if shard_index < 0:
            raise ValueError("--shard-index must be non-negative.")
        cfg["shard_index"] = shard_index
    if getattr(args, "dataset_mode", None):
        dataset_cfg = dict(cfg.get("dataset") or {})
        dataset_cfg["mode"] = str(args.dataset_mode)
        cfg["dataset"] = dataset_cfg
    if getattr(args, "run_scoring", False):
        cfg["run_scoring"] = True
    if getattr(args, "no_run_scoring", False):
        cfg["run_scoring"] = False
        dataset_cfg = dict(cfg.get("dataset") or {})
        if dataset_cfg.get("mode") == "full":
            dataset_cfg["mode"] = "generate"
            cfg["dataset"] = dataset_cfg
    understanding_cfg = dict(cfg.get("understanding") or {})
    if getattr(args, "think", None) is not None:
        understanding_cfg["think"] = bool(args.think)
    if getattr(args, "max_new_tokens", None) is not None:
        max_new_tokens = int(args.max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive.")
        understanding_cfg["max_new_tokens"] = max_new_tokens
    if getattr(args, "deliberate", None) is not None:
        deliberation = dict(understanding_cfg.get("deliberation") or {})
        deliberation["enabled"] = bool(args.deliberate)
        understanding_cfg["deliberation"] = deliberation
    if getattr(args, "draft_tokens", None) is not None:
        draft_tokens = int(args.draft_tokens)
        if draft_tokens <= 0:
            raise ValueError("draft_tokens must be positive")
        deliberation = dict(understanding_cfg.get("deliberation") or {})
        deliberation["draft_tokens"] = draft_tokens
        understanding_cfg["deliberation"] = deliberation
    if getattr(args, "draft_depth", None) is not None:
        draft_depth = int(args.draft_depth)
        if draft_depth <= 0:
            raise ValueError("draft_depth must be positive")
        deliberation = dict(understanding_cfg.get("deliberation") or {})
        deliberation["draft_depth"] = draft_depth
        understanding_cfg["deliberation"] = deliberation
    if getattr(args, "draft_style", None) is not None:
        deliberation = dict(understanding_cfg.get("deliberation") or {})
        deliberation["draft_style"] = str(args.draft_style)
        understanding_cfg["deliberation"] = deliberation
    if getattr(args, "choice_consensus", None) is not None:
        choice_consensus = dict(
            understanding_cfg.get("choice_consensus") or {}
        )
        choice_consensus["enabled"] = bool(args.choice_consensus)
        understanding_cfg["choice_consensus"] = choice_consensus
    if getattr(args, "choice_text_scoring", None) is not None:
        choice_text_scoring = dict(
            understanding_cfg.get("choice_text_scoring") or {}
        )
        choice_text_scoring["enabled"] = bool(args.choice_text_scoring)
        understanding_cfg["choice_text_scoring"] = choice_text_scoring
    if getattr(args, "choice_verifier", None) is not None:
        choice_verifier = dict(
            understanding_cfg.get("choice_verifier") or {}
        )
        choice_verifier["enabled"] = bool(args.choice_verifier)
        understanding_cfg["choice_verifier"] = choice_verifier
    if getattr(args, "choice_permutation", None) is not None:
        choice_permutation = dict(
            understanding_cfg.get("choice_permutation") or {}
        )
        choice_permutation["enabled"] = bool(args.choice_permutation)
        understanding_cfg["choice_permutation"] = choice_permutation
    if getattr(args, "choice_permutation_max_passes", None) is not None:
        max_passes = int(args.choice_permutation_max_passes)
        if max_passes < 2:
            raise ValueError("--choice-permutation-max-passes must be at least 2")
        choice_permutation = dict(
            understanding_cfg.get("choice_permutation") or {}
        )
        choice_permutation["max_passes"] = max_passes
        understanding_cfg["choice_permutation"] = choice_permutation
    cfg["understanding"] = understanding_cfg
    return cfg


def run_eval(config_path: str, args: argparse.Namespace | None = None) -> int:
    cfg = _load_yaml(config_path)
    cfg = _apply_cli_overrides(cfg, args)
    benchmark = str(cfg.get("benchmark", "")).lower()
    task = str(cfg.get("task", "")).lower()

    if task == "generation" or benchmark in {"geneval", "dpg_bench", "dpg"}:
        result = _run_generation_benchmark(cfg)
    elif benchmark == "mme":
        result = _run_mme(cfg)
    elif benchmark == "mmmu":
        result = _run_mmmu(cfg)
    elif benchmark == "mmbench":
        result = _run_mmbench(cfg)
    elif benchmark == "mathvista":
        result = _run_mathvista(cfg)
    elif benchmark == "mmstar":
        result = _run_mmstar(cfg)
    else:
        raise ValueError(
            f"Unsupported benchmark/task: benchmark={benchmark!r}, task={task!r}"
        )
    _write_routing_summary(cfg)
    return result


def run_eval_command(args: argparse.Namespace) -> int:
    return run_eval(args.config, args)
