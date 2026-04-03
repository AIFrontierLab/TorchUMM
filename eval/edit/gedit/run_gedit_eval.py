#!/usr/bin/env python3
"""GEdit-Bench runner — editing + VIEScore evaluation + statistics.

Orchestrates the full GEdit pipeline:
  Phase 1 (edit):      Use UMM InferencePipeline to edit images per GEdit-Bench instructions
  Phase 2 (score):     Subprocess call to run_gedit_score.py (VIEScore via Qwen2.5-VL)
  Phase 3 (calculate): Subprocess call to calculate_statistics.py (score aggregation)

Follows the same architecture pattern as eval/generation/wise/run_wise_eval.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from umm.core.config import load_config
from umm.inference import InferencePipeline

# ---------------------------------------------------------------------------
# GEdit task types
# ---------------------------------------------------------------------------

_GEDIT_TASK_TYPES = [
    "background_change", "color_alter", "material_alter", "motion_change",
    "ps_human", "style_change", "subject-add", "subject-remove",
    "subject-replace", "text_change", "tone_transfer",
]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2",
        "showo": "show_o2",
        "janus": "janus_pro",
    }
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_gedit_dataset(data_root: Optional[Path]):
    """Load GEdit-Bench dataset from local disk or HuggingFace."""
    from datasets import load_dataset, load_from_disk

    if data_root and data_root.is_dir():
        print(f"[gedit] loading dataset from disk: {data_root}", flush=True)
        dataset = load_from_disk(str(data_root))
    else:
        print("[gedit] downloading dataset from HuggingFace: stepfun-ai/GEdit-Bench", flush=True)
        dataset = load_dataset("stepfun-ai/GEdit-Bench")

    # Handle both Dataset and DatasetDict
    if hasattr(dataset, 'keys') and 'train' in dataset:
        return dataset['train']
    return dataset


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------

def _extract_saved_path(result: Any, fallback_dir: Path) -> str:
    """Return the path of the edited image from a pipeline result."""
    if isinstance(result, dict):
        for key in ("saved_paths", "output_path", "image_path", "image_paths"):
            val = result.get(key)
            if isinstance(val, list) and val and isinstance(val[0], str) and val[0]:
                p = Path(val[0])
                if p.is_file():
                    return str(p)
            if isinstance(val, str) and val:
                p = Path(val)
                if p.is_file():
                    return str(p)
        if fallback_dir.is_dir():
            img_exts = {".png", ".jpg", ".jpeg", ".webp"}
            candidates = sorted(
                [f for f in fallback_dir.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0])
    return ""


def _clear_workspace(workspace: Path) -> None:
    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    for f in workspace.iterdir():
        if f.is_file() and f.suffix.lower() in img_exts:
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Image editing
# ---------------------------------------------------------------------------

def run_editing(
    pipeline: InferencePipeline,
    backbone: str,
    dataset,
    images_dir: Path,
    model_name: str,
    request_params: Dict[str, Any],
    resume: bool,
    instruction_language: str = "all",
    task_type: str = "all",
) -> Dict[str, Any]:
    """Edit images according to GEdit-Bench instructions."""
    from collections import defaultdict

    # Group by task type
    dataset_by_group = defaultdict(list)
    for item in dataset:
        if instruction_language == "all" or item['instruction_language'] == instruction_language:
            dataset_by_group[item['task_type']].append(item)

    if task_type != "all":
        groups = [task_type]
    else:
        groups = _GEDIT_TASK_TYPES

    total = sum(len(dataset_by_group[g]) for g in groups)
    n_ok = n_skip = n_err = 0

    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    for group_name in groups:
        group_items = dataset_by_group.get(group_name, [])
        if not group_items:
            continue

        for item in tqdm(group_items, desc=f"[gedit edit] {group_name}"):
            key = item['key']
            lang = item['instruction_language']
            instruction = item['instruction']

            # Output path follows GEdit-Bench directory structure
            out_dir = images_dir / model_name / "fullset" / group_name / lang
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{key}.png"

            if resume and out_file.is_file():
                n_skip += 1
                continue

            _clear_workspace(workspace)
            expected_path = workspace / f"{key}.png"

            # Get the input image from dataset
            input_image = item['input_image_raw']

            payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "editing",
                "prompt": instruction,
                "images": [input_image],
                "output_path": str(expected_path),
                "params": request_params,
            }
            try:
                result = pipeline.run(payload)
            except Exception as exc:
                print(f"[gedit] key={key} editing error: {exc}", flush=True)
                n_err += 1
                continue

            if expected_path.is_file():
                saved = str(expected_path)
            else:
                saved = _extract_saved_path(result, workspace)

            if saved and Path(saved).is_file():
                shutil.copy2(saved, str(out_file))
                n_ok += 1
            else:
                print(f"[gedit] key={key}: no edited image produced", flush=True)
                n_err += 1

    return {"total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


# ---------------------------------------------------------------------------
# Phase 2: VIEScore evaluation (subprocess)
# ---------------------------------------------------------------------------

def run_scoring(
    images_dir: Path,
    save_dir: Path,
    model_name: str,
    vlm_cfg: Dict[str, Any],
    instruction_language: str,
    task_type: str,
    dataset_path: Optional[Path],
    repo_root: Path,
) -> int:
    """Call run_gedit_score.py as subprocess."""
    script = Path(__file__).parent / "run_gedit_score.py"
    if not script.exists():
        raise FileNotFoundError(f"run_gedit_score.py not found: {script}")

    backbone = str(vlm_cfg.get("backbone", "qwen25vl"))
    model_path = str(vlm_cfg.get("model_path", ""))

    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--edited_images_dir", str(images_dir),
        "--save_dir", str(save_dir),
        "--backbone", backbone,
        "--instruction_language", instruction_language,
        "--task_type", task_type,
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])
    if dataset_path:
        cmd.extend(["--dataset_path", str(dataset_path)])

    env = os.environ.copy()
    # Add eval/edit/gedit to PYTHONPATH so viescore imports work
    gedit_dir = str(Path(__file__).parent)
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    new_paths = f"{gedit_dir}:{src_dir}"
    env["PYTHONPATH"] = new_paths if not existing_pp else f"{new_paths}:{existing_pp}"

    print(f"[gedit] running VIEScore evaluation: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Phase 3: Statistics calculation (subprocess)
# ---------------------------------------------------------------------------

def run_calculate(
    save_dir: Path,
    model_name: str,
    vlm_cfg: Dict[str, Any],
    instruction_language: str,
    repo_root: Path,
) -> int:
    """Call calculate_statistics.py as subprocess."""
    script = Path(__file__).parent / "calculate_statistics.py"
    if not script.exists():
        raise FileNotFoundError(f"calculate_statistics.py not found: {script}")

    backbone = str(vlm_cfg.get("backbone", "qwen25vl"))

    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--backbone", backbone,
        "--save_path", str(save_dir),
        "--language", instruction_language,
    ]

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[gedit] running statistics calculation: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEdit-Bench Runner")
    parser.add_argument("--config", required=True, help="YAML config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    raw_cfg = load_config(str(config_path))

    # Parse config blocks
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    gedit_cfg = raw_cfg.get("gedit", {}) if isinstance(raw_cfg.get("gedit"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}

    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "gedit":
        raise ValueError(f"Expected eval.benchmark: gedit, got: {benchmark or '<empty>'}")

    # Backbone
    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required.")
    backbone = _normalize_backbone_name(backbone_raw)
    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        backbone_cfg = {}

    request_cfg = inference_cfg.get("request", {})
    request_params: Dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    # GEdit config
    data_root_val = gedit_cfg.get("data_root")
    data_root = _resolve_path(str(data_root_val), repo_root) if data_root_val else None

    out_dir = _resolve_path(
        str(gedit_cfg.get("out_dir", f"output/gedit/{backbone}")), repo_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    resume = bool(gedit_cfg.get("resume", True))
    instruction_language = str(gedit_cfg.get("instruction_language", "all"))
    task_type = str(gedit_cfg.get("task_type", "all"))
    model_name = str(gedit_cfg.get("model_name", backbone))

    # Mode
    mode = str(gedit_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[gedit] unknown mode '{mode}', defaulting to 'full'", flush=True)
        mode = "full"

    run_edit = mode in ("full", "generate")
    run_score = mode in ("full", "score")

    vlm_cfg = gedit_cfg.get("vlm_eval", {})
    if not isinstance(vlm_cfg, dict):
        vlm_cfg = {}
    run_calc = bool(gedit_cfg.get("run_calculate", True))

    print(
        f"[gedit] backbone={backbone}, data_root={data_root}, "
        f"out_dir={out_dir}, mode={mode}, resume={resume}, "
        f"language={instruction_language}, task_type={task_type}",
        flush=True,
    )

    # Load dataset
    dataset = _load_gedit_dataset(data_root)
    print(f"[gedit] loaded {len(dataset)} samples", flush=True)

    edit_summary: Optional[Dict[str, Any]] = None

    # Phase 1: Editing
    if run_edit:
        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)
        edit_summary = run_editing(
            pipeline=pipeline,
            backbone=backbone,
            dataset=dataset,
            images_dir=images_dir,
            model_name=model_name,
            request_params=request_params,
            resume=resume,
            instruction_language=instruction_language,
            task_type=task_type,
        )
        print(
            f"[gedit] editing done — "
            f"ok={edit_summary['ok']}, skipped={edit_summary['skipped']}, error={edit_summary['error']}",
            flush=True,
        )

    # Phase 2: VIEScore evaluation
    if run_score:
        rc = run_scoring(
            images_dir=images_dir,
            save_dir=scores_dir,
            model_name=model_name,
            vlm_cfg=vlm_cfg,
            instruction_language=instruction_language,
            task_type=task_type,
            dataset_path=data_root,
            repo_root=repo_root,
        )
        if rc != 0:
            print(f"[gedit] VIEScore evaluation failed with rc={rc}", flush=True)

    # Phase 3: Statistics
    if run_score and run_calc:
        rc = run_calculate(
            save_dir=scores_dir,
            model_name=model_name,
            vlm_cfg=vlm_cfg,
            instruction_language=instruction_language,
            repo_root=repo_root,
        )
        if rc != 0:
            print(f"[gedit] calculate_statistics.py failed with rc={rc}", flush=True)

    # Save summary
    summary: Dict[str, Any] = {
        "benchmark": "gedit",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "mode": mode,
    }
    if edit_summary:
        summary["editing"] = edit_summary

    score_output_path = gedit_cfg.get("score_output_path")
    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[gedit] wrote summary to {score_path}", flush=True)

    print(f"[gedit] completed. backbone={backbone}, mode={mode}", flush=True)


if __name__ == "__main__":
    main()
