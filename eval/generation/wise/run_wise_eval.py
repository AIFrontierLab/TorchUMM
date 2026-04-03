#!/usr/bin/env python3
"""WISE benchmark runner — generation + VLM evaluation + WiScore calculation.

Orchestrates the full WISE pipeline:
  Phase 1 (generate): Use UMM InferencePipeline to generate {prompt_id}.png images
  Phase 2 (eval):     Subprocess call to vlm_eval.py (Qwen2.5-VL scoring)
  Phase 3 (calculate): Subprocess call to Calculate.py (WiScore aggregation)

Follows the same architecture pattern as eval/generation/dpg_bench/run_generation_and_eval.py.
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
# WISE data file definitions
# ---------------------------------------------------------------------------

_WISE_DATA_FILES = [
    "cultural_common_sense.json",
    "spatio-temporal_reasoning.json",
    "natural_science.json",
]

_BATCH_CAPABLE_BACKBONES = {"emu3"}

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


def _load_prompts(data_root: Path) -> List[Dict[str, Any]]:
    """Load all WISE prompts from the three category JSON files."""
    prompts: List[Dict[str, Any]] = []
    for fname in _WISE_DATA_FILES:
        fpath = data_root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"WISE data file not found: {fpath}")
        items = json.loads(fpath.read_text(encoding="utf-8"))
        prompts.extend(items)
    prompts.sort(key=lambda x: x["prompt_id"])
    return prompts


# ---------------------------------------------------------------------------
# Image extraction helpers (from generation results)
# ---------------------------------------------------------------------------

def _extract_saved_path(result: Any, fallback_dir: Path, prompt_id: int) -> str:
    """Return the path of the generated image from a pipeline generation result."""
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
        rc = result.get("returncode")
        if rc not in (None, 0):
            stderr_tail = (result.get("stderr") or "")[-600:]
            print(f"[wise] gen subprocess rc={rc}: {stderr_tail}", flush=True)
    return ""


def _clear_workspace(workspace: Path) -> None:
    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    for f in workspace.iterdir():
        if f.is_file() and f.suffix.lower() in img_exts:
            f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Phase 1: Image generation
# ---------------------------------------------------------------------------

def run_generation(
    pipeline: InferencePipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """Generate one image per prompt and save as {prompt_id}.png."""
    if max_samples > 0:
        prompts = prompts[:max_samples]

    if backbone in _BATCH_CAPABLE_BACKBONES and hasattr(pipeline.backbone, "generate_batch"):
        return _run_generation_batch(pipeline, backbone, prompts, images_dir, request_params, resume)
    return _run_generation_sequential(pipeline, backbone, prompts, images_dir, request_params, resume)


def _run_generation_batch(
    pipeline: InferencePipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    images_dir.mkdir(parents=True, exist_ok=True)
    total = len(prompts)
    n_skip = 0

    pending: List[Dict[str, Any]] = []
    for item in prompts:
        pid = item["prompt_id"]
        out_file = images_dir / f"{pid}.png"
        if resume and out_file.is_file():
            n_skip += 1
            continue
        pending.append({"pid": pid, "prompt_text": item["Prompt"], "out_file": out_file})

    if not pending:
        print(f"[wise] all {total} prompts already completed (resume=True)", flush=True)
        return {"total": total, "ok": 0, "skipped": n_skip, "error": 0}

    print(f"[wise] batch mode: {len(pending)} images to generate ({n_skip} skipped)", flush=True)
    prompt_items = [{"prompt": p["prompt_text"], "output_path": str(p["out_file"])} for p in pending]

    try:
        results = pipeline.backbone.generate_batch(prompt_items, gen_cfg=request_params)
    except Exception as exc:
        print(f"[wise] batch generation error: {exc}", flush=True)
        return {"total": total, "ok": 0, "skipped": n_skip, "error": len(pending)}

    n_ok = n_err = 0
    for i, p in enumerate(pending):
        ok = i < len(results) and results[i].get("ok", False) and p["out_file"].is_file()
        if ok:
            n_ok += 1
        else:
            print(f"[wise] prompt_id={p['pid']}: no image produced", flush=True)
            n_err += 1
    return {"total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


def _run_generation_sequential(
    pipeline: InferencePipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
) -> Dict[str, Any]:
    images_dir.mkdir(parents=True, exist_ok=True)
    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    n_ok = n_skip = n_err = 0

    for item in tqdm(prompts, total=total, desc="[wise gen]"):
        pid = item["prompt_id"]
        out_file = images_dir / f"{pid}.png"

        if resume and out_file.is_file():
            n_skip += 1
            continue

        _clear_workspace(workspace)
        expected_path = workspace / f"{pid}.png"
        payload: Dict[str, Any] = {
            "backbone": backbone,
            "task": "generation",
            "prompt": item["Prompt"],
            "output_path": str(expected_path),
            "params": request_params,
        }
        try:
            result = pipeline.run(payload)
        except Exception as exc:
            print(f"[wise] prompt_id={pid} generation error: {exc}", flush=True)
            n_err += 1
            continue

        if expected_path.is_file():
            saved = str(expected_path)
        else:
            saved = _extract_saved_path(result, workspace, pid)

        if saved and Path(saved).is_file():
            shutil.copy2(saved, str(out_file))
            n_ok += 1
        else:
            print(f"[wise] prompt_id={pid}: no image produced", flush=True)
            n_err += 1

    return {"total": total, "ok": n_ok, "skipped": n_skip, "error": n_err}


# ---------------------------------------------------------------------------
# Phase 2: VLM evaluation (subprocess)
# ---------------------------------------------------------------------------

def run_vlm_eval(
    data_root: Path,
    image_dir: Path,
    output_dir: Path,
    vlm_cfg: Dict[str, Any],
    repo_root: Path,
) -> int:
    """Call vlm_eval.py as subprocess."""
    script = Path(__file__).parent / "vlm_eval.py"
    if not script.exists():
        raise FileNotFoundError(f"vlm_eval.py not found: {script}")

    model_name = str(vlm_cfg.get("model_name", "Qwen/Qwen2.5-VL-72B-Instruct"))
    max_new_tokens = str(vlm_cfg.get("max_new_tokens", 512))
    max_retries = str(vlm_cfg.get("max_retries", 2))

    cmd = [
        sys.executable, str(script),
        "--data_root", str(data_root),
        "--image_dir", str(image_dir),
        "--output_dir", str(output_dir),
        "--model_name", model_name,
        "--max_new_tokens", max_new_tokens,
        "--max_retries", max_retries,
    ]
    attn_impl = vlm_cfg.get("attn_implementation")
    if attn_impl:
        cmd.extend(["--attn_implementation", str(attn_impl)])

    env = os.environ.copy()
    src_dir = str(repo_root / "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing_pp else f"{src_dir}:{existing_pp}"

    print(f"[wise] running VLM evaluation: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# Phase 3: WiScore calculation (subprocess)
# ---------------------------------------------------------------------------

def run_calculate(output_dir: Path) -> int:
    """Call Calculate.py as subprocess."""
    script = Path(__file__).parent / "Calculate.py"
    if not script.exists():
        raise FileNotFoundError(f"Calculate.py not found: {script}")

    score_files = [
        str(output_dir / "cultural_common_sense_scores.jsonl"),
        str(output_dir / "spatio-temporal_reasoning_scores.jsonl"),
        str(output_dir / "natural_science_scores.jsonl"),
    ]

    # Check all score files exist
    missing = [f for f in score_files if not Path(f).is_file()]
    if missing:
        print(f"[wise] Warning: missing score files: {missing}", flush=True)
        # Run with whatever exists
        score_files = [f for f in score_files if Path(f).is_file()]
        if not score_files:
            print("[wise] No score files found, skipping Calculate.py", flush=True)
            return 1

    cmd = [sys.executable, str(script)] + score_files + ["--category", "all"]

    print(f"[wise] running WiScore calculation: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WISE Benchmark Runner")
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
    wise_cfg = raw_cfg.get("wise", {}) if isinstance(raw_cfg.get("wise"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}

    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "wise":
        raise ValueError(f"Expected eval.benchmark: wise, got: {benchmark or '<empty>'}")

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

    # WISE config
    data_root_val = wise_cfg.get("data_root")
    if not data_root_val:
        data_root = Path(__file__).parent / "data"
    else:
        data_root = _resolve_path(str(data_root_val), repo_root)

    # Auto-detect data/ subdirectory (e.g. /workspace/model/WISE -> /workspace/model/WISE/data)
    if (data_root / "data").is_dir() and not (data_root / _WISE_DATA_FILES[0]).is_file():
        data_root = data_root / "data"

    out_dir = _resolve_path(
        str(wise_cfg.get("out_dir", f"output/wise/{backbone}")), repo_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    results_dir = out_dir / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(wise_cfg.get("max_samples", 0) or 0)
    resume = bool(wise_cfg.get("resume", True))
    score_output_path = wise_cfg.get("score_output_path")

    # Mode
    mode = str(wise_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[wise] unknown mode '{mode}', defaulting to 'full'", flush=True)
        mode = "full"

    run_gen = mode in ("full", "generate")
    run_eval = mode in ("full", "score")

    vlm_cfg = wise_cfg.get("vlm_eval", {})
    if not isinstance(vlm_cfg, dict):
        vlm_cfg = {}
    run_vlm = bool(vlm_cfg.get("enabled", True))
    run_calc = bool(wise_cfg.get("run_calculate", run_vlm))

    print(
        f"[wise] backbone={backbone}, data_root={data_root}, "
        f"out_dir={out_dir}, mode={mode}, max_samples={max_samples or 'all'}, resume={resume}",
        flush=True,
    )

    # Load prompts
    prompts = _load_prompts(data_root)
    print(f"[wise] loaded {len(prompts)} prompts", flush=True)

    gen_summary: Optional[Dict[str, Any]] = None

    # Phase 1: Generation
    if run_gen:
        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)
        gen_summary = run_generation(
            pipeline=pipeline,
            backbone=backbone,
            prompts=prompts,
            images_dir=images_dir,
            request_params=request_params,
            resume=resume,
            max_samples=max_samples,
        )
        print(
            f"[wise] generation done — "
            f"ok={gen_summary['ok']}, skipped={gen_summary['skipped']}, error={gen_summary['error']}",
            flush=True,
        )

    # Phase 2: VLM evaluation
    if run_eval and run_vlm:
        rc = run_vlm_eval(
            data_root=data_root,
            image_dir=images_dir,
            output_dir=results_dir,
            vlm_cfg=vlm_cfg,
            repo_root=repo_root,
        )
        if rc != 0:
            print(f"[wise] VLM evaluation failed with rc={rc}", flush=True)

    # Phase 3: WiScore calculation
    if run_eval and run_calc:
        rc = run_calculate(output_dir=results_dir)
        if rc != 0:
            print(f"[wise] Calculate.py failed with rc={rc}", flush=True)

    # Save summary
    summary: Dict[str, Any] = {
        "benchmark": "wise",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "mode": mode,
    }
    if gen_summary:
        summary["generation"] = gen_summary

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[wise] wrote summary to {score_path}", flush=True)

    print(f"[wise] completed. backbone={backbone}, mode={mode}", flush=True)


if __name__ == "__main__":
    main()
