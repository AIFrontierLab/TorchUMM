#!/usr/bin/env python3
"""ImgEdit-Bench runner — editing + Qwen2.5-VL scoring + statistics.

Supports three benchmark suites:
  - singleturn: 737 items, 9 edit types, type-specific scoring prompts
  - uge:        47 items, harder edits, unified scoring prompt
  - multiturn:  30 items (3 categories × 10), sequential multi-turn editing

Config key `suite` selects which suite(s) to run: singleturn | uge | multiturn | all.

Follows the same architecture pattern as eval/edit/gedit/run_gedit_eval.py.
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
# Constants
# ---------------------------------------------------------------------------

_IMGEDIT_EDIT_TYPES = [
    "adjust", "add", "remove", "replace", "background",
    "style", "extract", "action", "compose",
]

_MULTITURN_CATEGORIES = ["content_memory", "content_understand", "version_backtrace"]

_ALL_SUITES = ["singleturn", "uge", "multiturn"]

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
    aliases = {"showo2": "show_o2", "showo": "show_o2", "janus": "janus_pro"}
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dict_json(path: Path) -> Dict[str, Any]:
    """Load a JSON dict file (e.g. singleturn basic_edit.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file (one JSON object per line)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------

def _extract_saved_path(result: Any, fallback_dir: Path) -> str:
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


def _run_single_edit(
    pipeline: InferencePipeline,
    backbone: str,
    input_image,  # PIL Image
    prompt: str,
    output_path: Path,
    workspace: Path,
    request_params: Dict[str, Any],
) -> bool:
    """Run a single editing call and save result to output_path. Returns True on success."""
    _clear_workspace(workspace)
    expected_path = workspace / output_path.name

    payload: Dict[str, Any] = {
        "backbone": backbone,
        "task": "editing",
        "prompt": prompt,
        "images": [input_image],
        "output_path": str(expected_path),
        "params": request_params,
    }
    try:
        result = pipeline.run(payload)
    except Exception as exc:
        print(f"[imgedit] editing error: {exc}", flush=True)
        return False

    saved = str(expected_path) if expected_path.is_file() else _extract_saved_path(result, workspace)

    if saved and Path(saved).is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, str(output_path))
        return True
    return False


# ---------------------------------------------------------------------------
# Phase 1a: Singleturn editing
# ---------------------------------------------------------------------------

def run_singleturn_editing(
    pipeline: InferencePipeline,
    backbone: str,
    benchmark_data: Path,
    images_dir: Path,
    origin_img_root: Path,
    model_name: str,
    request_params: Dict[str, Any],
    resume: bool,
    edit_type: str = "all",
) -> Dict[str, Any]:
    from collections import defaultdict
    from PIL import Image

    dataset = _load_dict_json(benchmark_data / "basic_edit.json")
    dataset_by_type = defaultdict(list)
    for key, item in dataset.items():
        dataset_by_type[item.get("edit_type", "unknown")].append((key, item))

    groups = [edit_type] if edit_type != "all" else _IMGEDIT_EDIT_TYPES
    total = sum(len(dataset_by_type[g]) for g in groups)
    n_ok = n_skip = n_err = 0

    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    out_model_dir = images_dir / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)

    for group_name in groups:
        for key, item in tqdm(dataset_by_type.get(group_name, []), desc=f"[singleturn] {group_name}"):
            out_file = out_model_dir / f"{key}.png"
            if resume and out_file.is_file():
                n_skip += 1
                continue
            origin_path = origin_img_root / "singleturn" / item["id"]
            if not origin_path.is_file():
                print(f"[singleturn] key={key}: original not found: {origin_path}", flush=True)
                n_err += 1
                continue
            input_image = Image.open(str(origin_path)).convert("RGB")
            if _run_single_edit(pipeline, backbone, input_image, item["prompt"], out_file, workspace, request_params):
                n_ok += 1
            else:
                print(f"[singleturn] key={key}: no edited image produced", flush=True)
                n_err += 1

    return {"suite": "singleturn", "total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


# ---------------------------------------------------------------------------
# Phase 1b: UGE editing
# ---------------------------------------------------------------------------

def run_uge_editing(
    pipeline: InferencePipeline,
    backbone: str,
    images_dir: Path,
    origin_img_root: Path,
    model_name: str,
    request_params: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    from PIL import Image

    # Load UGE annotations directly from Benchmark/hard/annotation.jsonl (ground truth)
    uge_img_root = origin_img_root / "hard"
    ann_path = uge_img_root / "annotation.jsonl"
    if not ann_path.is_file():
        raise FileNotFoundError(f"UGE annotation not found: {ann_path}")
    items = _load_jsonl(ann_path)

    total = len(items)
    n_ok = n_skip = n_err = 0

    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    out_model_dir = images_dir / model_name
    out_model_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(tqdm(items, desc="[uge]")):
        img_id = item["id"]
        key = Path(img_id).stem  # e.g. "000029784"
        out_file = out_model_dir / f"uge_{key}.png"
        if resume and out_file.is_file():
            n_skip += 1
            continue
        origin_path = uge_img_root / img_id
        if not origin_path.is_file():
            print(f"[uge] {img_id}: original not found: {origin_path}", flush=True)
            n_err += 1
            continue
        input_image = Image.open(str(origin_path)).convert("RGB")
        if _run_single_edit(pipeline, backbone, input_image, item["prompt"], out_file, workspace, request_params):
            n_ok += 1
        else:
            print(f"[uge] {img_id}: no edited image produced", flush=True)
            n_err += 1

    return {"suite": "uge", "total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


# ---------------------------------------------------------------------------
# Phase 1c: Multiturn editing
# ---------------------------------------------------------------------------

def run_multiturn_editing(
    pipeline: InferencePipeline,
    backbone: str,
    images_dir: Path,
    origin_img_root: Path,
    model_name: str,
    request_params: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    from PIL import Image

    # Multiturn images are in Benchmark/multiturn/{category}/
    multiturn_root = origin_img_root / "multiturn"
    total = 0
    n_ok = n_skip = n_err = 0

    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    for category in _MULTITURN_CATEGORIES:
        cat_dir = multiturn_root / category
        ann_path = cat_dir / "annotation.json"
        if not ann_path.is_file():
            print(f"[multiturn] annotation.json not found: {ann_path}", flush=True)
            continue

        items = _load_jsonl(ann_path)
        out_model_dir = images_dir / model_name / f"multiturn_{category}"
        out_model_dir.mkdir(parents=True, exist_ok=True)

        for item in tqdm(items, desc=f"[multiturn] {category}"):
            img_id = item["id"]
            base_name = Path(img_id).stem

            # Collect turns
            turns = []
            for t in range(1, 10):
                tk = f"turn{t}"
                if tk in item:
                    turns.append(item[tk])
                else:
                    break

            total += len(turns)

            # Check if all turns already done (resume)
            last_turn_file = out_model_dir / f"{base_name}_turn{len(turns)}.png"
            if resume and last_turn_file.is_file():
                n_skip += len(turns)
                continue

            # Load original image
            origin_path = cat_dir / img_id
            if not origin_path.is_file():
                print(f"[multiturn] {category}/{img_id}: original not found", flush=True)
                n_err += len(turns)
                continue

            current_image = Image.open(str(origin_path)).convert("RGB")

            for turn_idx, turn_prompt in enumerate(turns, start=1):
                out_file = out_model_dir / f"{base_name}_turn{turn_idx}.png"
                if resume and out_file.is_file():
                    # Load this turn's output as input for next turn
                    current_image = Image.open(str(out_file)).convert("RGB")
                    n_skip += 1
                    continue

                if _run_single_edit(pipeline, backbone, current_image, turn_prompt, out_file, workspace, request_params):
                    n_ok += 1
                    current_image = Image.open(str(out_file)).convert("RGB")
                else:
                    print(f"[multiturn] {category}/{img_id} turn{turn_idx}: failed", flush=True)
                    n_err += 1
                    break  # Cannot continue sequential turns if one fails

    return {"suite": "multiturn", "total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


# ---------------------------------------------------------------------------
# Phase 2: Qwen2.5-VL scoring (subprocess)
# ---------------------------------------------------------------------------

def run_scoring(
    images_dir: Path,
    scores_dir: Path,
    model_name: str,
    vlm_cfg: Dict[str, Any],
    origin_img_root: Path,
    benchmark_data: Path,
    suite: str,
    edit_type: str,
    repo_root: Path,
) -> int:
    script = Path(__file__).parent / "run_imgedit_score.py"
    if not script.exists():
        raise FileNotFoundError(f"run_imgedit_score.py not found: {script}")

    model_path = str(vlm_cfg.get("model_path", ""))
    cmd = [
        sys.executable, str(script),
        "--model_name", model_name,
        "--edited_images_dir", str(images_dir / model_name),
        "--save_dir", str(scores_dir),
        "--origin_img_root", str(origin_img_root),
        "--benchmark_data", str(benchmark_data),
        "--suite", suite,
        "--edit_type", edit_type,
    ]
    if model_path:
        cmd.extend(["--model_path", model_path])

    env = os.environ.copy()
    imgedit_dir = str(Path(__file__).parent)
    gedit_dir = str(repo_root / "eval" / "edit" / "gedit")
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    new_paths = f"{imgedit_dir}:{gedit_dir}:{src_dir}"
    env["PYTHONPATH"] = new_paths if not existing_pp else f"{new_paths}:{existing_pp}"

    print(f"[imgedit] running scoring (suite={suite}): {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Phase 3: Statistics calculation (subprocess)
# ---------------------------------------------------------------------------

def run_calculate(scores_dir: Path, benchmark_data: Path, suite: str, repo_root: Path) -> int:
    script = Path(__file__).parent / "calculate_statistics.py"
    if not script.exists():
        raise FileNotFoundError(f"calculate_statistics.py not found: {script}")

    cmd = [
        sys.executable, str(script),
        "--scores_dir", str(scores_dir),
        "--benchmark_data", str(benchmark_data),
        "--suite", suite,
    ]

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[imgedit] running statistics (suite={suite}): {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImgEdit-Bench Runner")
    parser.add_argument("--config", required=True, help="YAML config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    raw_cfg = load_config(str(config_path))

    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    imgedit_cfg = raw_cfg.get("imgedit", {}) if isinstance(raw_cfg.get("imgedit"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}

    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "imgedit":
        raise ValueError(f"Expected eval.benchmark: imgedit, got: {benchmark or '<empty>'}")

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

    # Paths
    origin_img_root_val = imgedit_cfg.get("origin_img_root")
    if not origin_img_root_val:
        raise ValueError("`imgedit.origin_img_root` is required.")
    origin_img_root = _resolve_path(str(origin_img_root_val), repo_root)

    benchmark_data_val = imgedit_cfg.get("benchmark_data")
    benchmark_data = _resolve_path(str(benchmark_data_val), repo_root) if benchmark_data_val else Path(__file__).parent

    out_dir = _resolve_path(str(imgedit_cfg.get("out_dir", f"output/imgedit/{backbone}")), repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume = bool(imgedit_cfg.get("resume", True))
    edit_type = str(imgedit_cfg.get("edit_type", "all"))
    model_name = str(imgedit_cfg.get("model_name", backbone))

    # Suite
    suite = str(imgedit_cfg.get("suite", "all")).strip().lower()
    suites_to_run = _ALL_SUITES if suite == "all" else [suite]

    # Mode
    mode = str(imgedit_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[imgedit] unknown mode '{mode}', defaulting to 'full'", flush=True)
        mode = "full"
    run_edit = mode in ("full", "generate")
    run_score = mode in ("full", "score")

    vlm_cfg = imgedit_cfg.get("vlm_eval", {})
    if not isinstance(vlm_cfg, dict):
        vlm_cfg = {}
    run_calc = bool(imgedit_cfg.get("run_calculate", True))

    print(
        f"[imgedit] backbone={backbone}, suites={suites_to_run}, mode={mode}, "
        f"origin_img_root={origin_img_root}, out_dir={out_dir}, resume={resume}",
        flush=True,
    )

    # Run each suite
    pipeline = None
    all_summaries: List[Dict[str, Any]] = []

    for s in suites_to_run:
        print(f"\n{'='*60}\n[imgedit] === Suite: {s} ===\n{'='*60}", flush=True)

        suite_images_dir = out_dir / "images" / s
        suite_images_dir.mkdir(parents=True, exist_ok=True)
        suite_scores_dir = out_dir / "scores" / s
        suite_scores_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Editing
        if run_edit:
            if pipeline is None:
                pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

            if s == "singleturn":
                summary = run_singleturn_editing(
                    pipeline, backbone, benchmark_data, suite_images_dir,
                    origin_img_root, model_name, request_params, resume, edit_type,
                )
            elif s == "uge":
                summary = run_uge_editing(
                    pipeline, backbone, suite_images_dir,
                    origin_img_root, model_name, request_params, resume,
                )
            elif s == "multiturn":
                summary = run_multiturn_editing(
                    pipeline, backbone, suite_images_dir,
                    origin_img_root, model_name, request_params, resume,
                )
            else:
                print(f"[imgedit] unknown suite '{s}', skipping", flush=True)
                continue

            all_summaries.append(summary)
            print(
                f"[imgedit] {s} editing done — ok={summary['ok']}, "
                f"skipped={summary['skipped']}, error={summary['error']}",
                flush=True,
            )

        # Phase 2: Scoring
        if run_score:
            rc = run_scoring(
                suite_images_dir, suite_scores_dir, model_name, vlm_cfg,
                origin_img_root, benchmark_data, s, edit_type, repo_root,
            )
            if rc != 0:
                print(f"[imgedit] scoring failed for suite={s}, rc={rc}", flush=True)

        # Phase 3: Statistics
        if run_score and run_calc:
            rc = run_calculate(suite_scores_dir, benchmark_data, s, repo_root)
            if rc != 0:
                print(f"[imgedit] statistics failed for suite={s}, rc={rc}", flush=True)

    # Save overall summary
    overall: Dict[str, Any] = {
        "benchmark": "imgedit",
        "backbone": backbone,
        "suites": suites_to_run,
        "out_dir": str(out_dir),
        "mode": mode,
        "editing": all_summaries,
    }
    score_output_path = imgedit_cfg.get("score_output_path")
    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[imgedit] wrote summary to {score_path}", flush=True)

    print(f"\n[imgedit] all done. backbone={backbone}, suites={suites_to_run}, mode={mode}", flush=True)


if __name__ == "__main__":
    main()
