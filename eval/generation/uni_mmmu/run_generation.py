#!/usr/bin/env python3
"""Uni-MMMU generation-only evaluation script.

Usage:
    python eval/generation/uni_mmmu/run_generation.py --config configs/eval/uni_mmmu/emu3.yaml

This script handles ONLY the generation phase of Uni-MMMU evaluation.
It loads a UMM YAML config, creates an InferencePipeline, dispatches to
the 6 task runners, and writes per-task and overall summaries.
"""
from __future__ import annotations

# fmt: off
import argparse
import contextlib
import csv
import glob as glob_module
import json
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config
from umm.inference import InferencePipeline

# fmt: on


# =============================================================================
# Context API -- mirrors the sample_code_example GPT interface
# =============================================================================

@dataclass
class CtxImagePath:
    path: str
    mime: str = "image/png"


@dataclass
class ContextItem:
    kind: Literal["text", "image"]
    payload: Union[str, CtxImagePath]


def add_text(ctx: List[ContextItem], text: str) -> None:
    ctx.append(ContextItem("text", text))


def add_image_path(
    ctx: List[ContextItem],
    path: Union[str, Path],
    mime: str = "image/png",
) -> None:
    ctx.append(ContextItem("image", CtxImagePath(str(path), mime)))


# =============================================================================
# Text-extraction helpers
# =============================================================================

def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
        # Handle adapters that return {"understandings": [{"response": "..."}]}
        for list_key in ("understandings",):
            container = output.get(list_key)
            if isinstance(container, list):
                for item in container:
                    text = _extract_text(item)
                    if text:
                        return text
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _extract_saved_path(result: Any, out_path: Path) -> str:
    """Return the path of the first image written by a generation/editing call."""
    if isinstance(result, dict):
        saved = result.get("saved_paths") or result.get("output_path")
        if isinstance(saved, list) and saved:
            return str(saved[0])
        if isinstance(saved, str) and saved:
            return saved
        img = result.get("image")
        if isinstance(img, Image.Image):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            return str(out_path)
        imgs = result.get("images")
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(str(out_path), format="PNG")
            return str(out_path)
    if isinstance(result, Image.Image):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), format="PNG")
        return str(out_path)
    # Backbone may have already written the file to out_path directly
    if out_path.is_file() and out_path.stat().st_size > 0:
        return str(out_path)
    return ""


# =============================================================================
# Pipeline-backed generators
# =============================================================================

def _ctx_to_text_and_images(
    ctx: List[ContextItem],
    prompt_suffix: str = "",
) -> Tuple[str, List[str]]:
    text_parts: List[str] = []
    images: List[str] = []
    for item in ctx:
        if item.kind == "text":
            text_parts.append(str(item.payload))
        elif item.kind == "image":
            images.append(item.payload.path)  # type: ignore[union-attr]
    if prompt_suffix:
        text_parts.append(prompt_suffix)
    return "\n".join(text_parts), images


def _smart_resize_for_understand(
    height: int, width: int, max_pixels: int, factor: int = 8,
) -> Tuple[int, int]:
    """Compute target (h, w) that satisfies *max_pixels* and factor alignment.

    Mirrors the ``smart_resize`` logic used by Emu3's VisionTokenizer
    image processor so the downstream processor will not re-adjust.
    """
    import math
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def _maybe_resize_images(
    images: List[str],
    max_pixels: int,
) -> Tuple[List[str], List[Path]]:
    """Resize images exceeding *max_pixels* and return new paths.

    Returns ``(image_paths, temp_files)`` where *temp_files* are Path
    objects the caller should keep alive until the images are consumed.
    """
    out: List[str] = []
    temps: List[Path] = []
    for img_path in images:
        img = Image.open(img_path)
        w, h = img.size
        if w * h <= max_pixels:
            out.append(img_path)
            continue
        new_h, new_w = _smart_resize_for_understand(h, w, max_pixels)
        img = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
        tmp = Path(img_path).parent / f"_tmp_resized_{Path(img_path).stem}.png"
        img.save(str(tmp), format="PNG")
        out.append(str(tmp))
        temps.append(tmp)
    return out, temps


def generate_text_from_context(
    pipeline: InferencePipeline,
    backbone: str,
    ctx: List[ContextItem],
    prompt_suffix: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> str:
    prompt, images = _ctx_to_text_and_images(ctx, prompt_suffix)

    max_understand_pixels = (params or {}).get("max_understand_pixels")
    temp_files: List[Path] = []
    if max_understand_pixels and images:
        images, temp_files = _maybe_resize_images(images, int(max_understand_pixels))

    payload: Dict[str, Any] = {
        "backbone": backbone,
        "task": "understanding",
        "prompt": prompt or "Describe what you see.",
        "images": images,
        "params": params or {},
    }
    try:
        result = pipeline.run(payload)
    except (ValueError, NotImplementedError) as exc:
        if "image" in str(exc).lower() or isinstance(exc, NotImplementedError):
            return ""
        raise
    finally:
        for tmp in temp_files:
            tmp.unlink(missing_ok=True)
    return _extract_text(result)


def generate_image_from_context(
    pipeline: InferencePipeline,
    backbone: str,
    ctx: List[ContextItem],
    out_path: Union[str, Path],
    prompt_suffix: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Any]:
    """Generate an image conditioned on the accumulated context.

    Tries ``task="editing"`` when context contains images (image-conditioned
    generation), falling back to ``task="generation"`` (text-only) if the
    backbone does not implement editing.

    Returns ``(saved_image_path, raw_result)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a workspace dir so adapters that treat output_path as a directory
    # don't turn out_path itself into a directory.
    workspace = out_path.parent / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    prompt, images = _ctx_to_text_and_images(ctx, prompt_suffix)
    result: Any = None

    workspace_file = str(workspace / out_path.name)

    if images:
        try:
            payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "editing",
                "prompt": prompt or "Generate an image.",
                "images": images,
                "output_path": workspace_file,
                "params": params or {},
            }
            result = pipeline.run(payload)
        except (NotImplementedError, ValueError, RuntimeError, TypeError):
            result = None

    if result is None:
        # Fall back to pure text-conditioned generation
        payload = {
            "backbone": backbone,
            "task": "generation",
            "prompt": prompt or "Generate an image.",
            "output_path": workspace_file,
            "params": params or {},
        }
        result = pipeline.run(payload)

    # Log subprocess failures so errors (e.g. CUDA OOM) are visible.
    # Some adapters return {"error": ..., "stderr": ...} without "returncode",
    # so also treat a top-level "error" key (with no images) as a failure signal.
    subprocess_failed = isinstance(result, dict) and (
        result.get("returncode") not in (None, 0)
        or (result.get("error") and not result.get("images"))
    )
    if subprocess_failed:
        stderr_tail = (result.get("stderr") or "")[-800:]
        rc = result.get("returncode", "?")
        err_msg = result.get("error", "")
        print(f"[gen] backbone failed (rc={rc}): {err_msg}\n{stderr_tail}")

    # Find the generated image in result metadata or by scanning the workspace
    saved = _extract_saved_path(result, workspace / out_path.name)
    if not saved:
        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = sorted(
            [f for f in workspace.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            saved = str(candidates[0])

    # Copy the result to the final out_path so callers always get a plain file
    if saved and Path(saved).is_file():
        shutil.copy2(saved, str(out_path))
        saved = str(out_path)

    if not saved and subprocess_failed:
        rc = result.get("returncode", "?")
        stderr_tail = (result.get("stderr") or "").strip().splitlines()
        hint = stderr_tail[-1] if stderr_tail else "unknown error"
        raise RuntimeError(
            f"Image generation subprocess failed (rc={rc}) and produced no image: {hint}"
        )

    return saved, result


# =============================================================================
# Config helpers
# =============================================================================

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
        "januspro": "janus_pro",
        "omnigen": "omnigen2",
        "blip3": "blip3o",
        "blip3_o": "blip3o",
        "token_flow": "tokenflow",
    }
    return aliases.get(normalized, normalized)


def _load_eval_cfg(
    config_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    uni_mmmu_cfg = (
        raw_cfg.get("uni_mmmu", {})
        if isinstance(raw_cfg.get("uni_mmmu"), dict)
        else {}
    )
    inference_cfg = (
        raw_cfg.get("inference", {})
        if isinstance(raw_cfg.get("inference"), dict)
        else {}
    )
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, uni_mmmu_cfg, inference_cfg


# =============================================================================
# Shared resume / IO helpers
# =============================================================================

def _previous_images_missing(
    case_dir: Path,
    images_key: str = "images_saved",
) -> bool:
    rj = case_dir / "result.json"
    if not rj.exists():
        # No record of what was generated — conservative: treat as missing.
        return True
    try:
        rec = json.loads(rj.read_text(encoding="utf-8"))
        # If no valid image paths were saved (empty list, list of empty
        # strings, or missing key), treat as missing so resume will re-try.
        imgs = [p for p in (rec.get(images_key) or []) if p]
        if not imgs:
            return True
        # If we recorded an expected count, verify we have enough images.
        expected = rec.get("expected_image_count")
        if expected is not None and len(imgs) < expected:
            return True
        return any(not Path(p).is_file() for p in imgs)
    except Exception:
        return True


def _write_result(case_dir: Path, record: Dict[str, Any]) -> None:
    (case_dir / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _print_task_summary(task_name: str, summary: Dict[str, Any]) -> None:
    print(
        f"[uni_mmmu | {task_name}] "
        f"Total={summary['count_total']}, "
        f"Processed={summary['count_processed']}, "
        f"Success={summary['count_success']}, "
        f"Errors={summary['count_error']}, "
        f"Skipped={summary['count_skipped']}"
    )


def _sanitize_name(s: str, maxlen: int = 120) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (s or "").strip())[:maxlen] or "item"


# =============================================================================
# TASK: Science (Physics / Chemistry / Biology)
# =============================================================================

_SCIENCE_PROMPT = """\
You are a unified vision-language model. You will be given:

(1) one initial image, and
(2) a textual condition describing an operation/environmental change.

Your job:
- Infer the UNIQUE final state using real-world knowledge and deterministic reasoning.
- Do NOT restate the condition as the result; derive the result causally.
- Do NOT introduce new persistent objects unless they follow necessarily from the condition.
- Keep the scene consistent: objects present initially should remain unless the condition implies removal.
- Output EXACTLY:
<OUTPUT_PROMPT> a concise, deterministic explanation (<=120 words) ending with a precise visual description of the final state. No hedging, no multiple possibilities. </OUTPUT_PROMPT>
And generate EXACTLY ONE image depicting the final state (no extra text).

Hard constraints:
- Deterministic, single outcome.
- No meta talk about prompts, models, or pipelines.
- Do not copy the condition as the result; reason from it.
"""


def _load_science_cases(
    data_root: Path,
    max_samples: int,
) -> List[Dict[str, Any]]:
    data_json = data_root / "science" / "dim_all.json"
    if not data_json.exists():
        raise FileNotFoundError(f"Science data not found: {data_json}")
    obj = json.loads(data_json.read_text(encoding="utf-8"))
    pool: List[Dict[str, Any]] = []
    for block in obj:
        for s in block.get("samples", []):
            imgs = s.get("input_image_file_path_list") or []
            cond = s.get("input_prompt")
            if imgs and cond and isinstance(imgs, list) and isinstance(cond, str):
                img_path = Path(imgs[0])
                if not img_path.is_absolute():
                    # JSON paths are relative to the Uni-MMMU repo root, e.g.
                    # "./data/science/imgs/foo.png" where "data/" == data_root.
                    # Strip the leading "./data/" or "data/" prefix so the path
                    # resolves correctly against data_root directly.
                    rel = img_path.as_posix()
                    for prefix in ("./data/", "data/"):
                        if rel.startswith(prefix):
                            rel = rel[len(prefix):]
                            break
                    img_path = data_root / rel
                pool.append({
                    "initial_image": str(img_path),
                    "condition": cond.strip(),
                    "meta": {
                        "level_1": block.get("level_1_category"),
                        "level_2": block.get("level_2_category"),
                    },
                })
    return pool if max_samples <= 0 else pool[:max_samples]


def run_task_science(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "science"
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_science_cases(data_root, max_samples)
    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(cases),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for idx, case in tqdm(enumerate(cases, 1), total=len(cases), desc=f"[{task_name}]"):
        case_dir = out_dir / f"case_{idx:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_marker = case_dir / "_done.ok"

        if resume and done_marker.exists() and not _previous_images_missing(case_dir):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": f"case_{idx:02d}", "status": "skipped_resume"})
            manifest_rows.append([idx, case["initial_image"], case["condition"][:140], "", "[SKIPPED]"])
            continue

        summary["count_processed"] += 1
        record: Dict[str, Any] = {
            "id": f"case_{idx:02d}",
            "status": "unknown",
            "case_dir": str(case_dir),
            "initial_image": case["initial_image"],
            "condition": case["condition"],
            "meta": case["meta"],
            "expected_image_count": 1,
            "text_file": None,
            "images_saved": [],
            "errors": [],
        }

        try:
            ctx: List[ContextItem] = []
            add_text(ctx, _SCIENCE_PROMPT)
            add_text(ctx, "Initial image:")
            add_image_path(ctx, case["initial_image"])
            add_text(ctx, f"Condition: {case['condition']}")

            # Step 1: text explanation
            full_text = generate_text_from_context(
                pipeline, backbone, ctx, params=request_params
            )
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)

            add_text(ctx, full_text)

            # Step 2: final-state image
            img_out = case_dir / "model_image_01.png"
            img_path, _ = generate_image_from_context(
                pipeline, backbone, ctx,
                out_path=img_out,
                prompt_suffix="Generate EXACTLY ONE image depicting the final state.",
                params=request_params,
            )
            record["images_saved"] = [img_path]
            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            manifest_rows.append([idx, case["initial_image"], case["condition"][:140], img_path, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([idx, case["initial_image"], case["condition"][:140], "", f"[ERROR] {exc}"])

        _write_result(case_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["case_idx", "initial_image", "condition_preview", "generated_image", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# TASK: Math Geometry
# =============================================================================

_MATH_GEO_PROMPT_TMPL = """\
You are a geometry diagram editor and solver.

TASK ORDER:
1) OVERLAY: On the attached base figure, overlay the auxiliary lines EXACTLY as specified below.
   - Add overlays only; do not move/erase the original objects or labels.
   - Keep labels (A, B, C, ...) unchanged and clearly visible.
   - Draw clean, visible lines.

2) REASONING: Give a concise, logically ordered solution or proof (<=150 words), using the constructed auxiliary lines.
   - Keep math tokens (triangle, angle, sqrt, pi, degree) unchanged.
   - Reference elements by their labels.

3) FINISHING:
   - For calculation problems, end with:  **Final answer: <VALUE>**.
   - For proving problems, end with:     **Conclusion: <STATEMENT>**.

PROBLEM:
{PROBLEM_TEXT}

CHOICES (if any):
{CHOICES_TEXT}

AUXILIARY LINES TO DRAW (English; follow exactly and draw these first):
{AUX_EN}
"""


def run_task_math_geo(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "math_geo"
    out_dir = out_root / "math"  # match original Uni-MMMU directory name
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_json = data_root / "math_data" / "filtered.json"
    if not filtered_json.exists():
        raise FileNotFoundError(f"Math geometry data not found: {filtered_json}")
    data: Dict[str, Dict[str, Any]] = json.loads(filtered_json.read_text(encoding="utf-8"))
    root_dir_r = filtered_json.parent.resolve()

    items: List[Tuple[str, str, Dict[str, Any]]] = [
        (big_k, small_k, item)
        for big_k, group in data.items()
        for small_k, item in group.items()
    ]
    if max_samples > 0:
        items = items[:max_samples]

    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(items),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for big_k, small_k, item in tqdm(items, desc=f"[{task_name}]"):
        dir_name = f"{_sanitize_name(big_k)}__{_sanitize_name(small_k)}"
        ex_dir = out_dir / dir_name
        ex_dir.mkdir(parents=True, exist_ok=True)
        done_marker = ex_dir / "_done.ok"

        if resume and done_marker.exists() and not _previous_images_missing(ex_dir):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": dir_name, "status": "skipped_resume"})
            manifest_rows.append([big_k, small_k, "", "", "[SKIPPED]"])
            continue

        summary["count_processed"] += 1
        record: Dict[str, Any] = {
            "id": dir_name,
            "big_key": big_k,
            "small_key": small_k,
            "status": "unknown",
            "ex_dir": str(ex_dir),
            "type": item.get("type"),
            "original_image": item.get("original_image"),
            "expected_image_count": 1,
            "text_file": None,
            "images_saved": [],
            "errors": [],
        }

        orig_rel = item.get("original_image")
        if not orig_rel:
            record["status"] = "error"
            record["errors"].append("Missing 'original_image' field in JSON")
            _write_result(ex_dir, record)
            summary["count_error"] += 1
            manifest_rows.append([big_k, small_k, "", "", "[MISS-FIELD original_image]"])
            summary["per_item"].append(record)
            continue

        orig_path = Path(orig_rel)
        orig_abs = orig_path if orig_path.is_absolute() else (root_dir_r / orig_path).resolve()
        if not orig_abs.exists():
            record["status"] = "error"
            record["errors"].append(f"Image not found: {orig_abs}")
            _write_result(ex_dir, record)
            summary["count_error"] += 1
            manifest_rows.append([big_k, small_k, str(orig_rel), "", "[MISS-FILE]"])
            summary["per_item"].append(record)
            continue

        problem_text = item.get("problem_text_en") or item.get("problem_text") or "(no problem text)"
        choices = item.get("choices_en")
        choices_text = "\n".join(choices) if isinstance(choices, list) and choices else "(no choices)"
        aux_en = (item.get("auxiliary_text_en") or "").strip()
        prompt = _MATH_GEO_PROMPT_TMPL.format(
            PROBLEM_TEXT=problem_text,
            CHOICES_TEXT=choices_text,
            AUX_EN=aux_en,
        )

        try:
            ctx: List[ContextItem] = []
            add_text(ctx, prompt)
            add_text(ctx, "BASE FIGURE: The following image is the original diagram.")
            add_image_path(ctx, str(orig_abs))

            # Step 1: overlay image
            overlay_path = ex_dir / "model_image_01.png"
            img_path, _ = generate_image_from_context(
                pipeline, backbone, ctx,
                out_path=overlay_path,
                prompt_suffix=(
                    "STEP 1 - OVERLAY now: Output EXACTLY ONE image of the base figure "
                    "with the auxiliary lines overlaid as specified. "
                    "Do not change existing objects/labels. No text, no captions."
                ),
                params=request_params,
            )
            if not Path(img_path).exists():
                raise RuntimeError(f"Overlay image not written: {img_path}")
            record["images_saved"] = [img_path]

            add_text(ctx, "OVERLAY RESULT (reference for reasoning):")
            add_image_path(ctx, str(img_path))

            # Step 2: text reasoning
            text_out = generate_text_from_context(
                pipeline, backbone, ctx,
                prompt_suffix=(
                    "STEP 2 - REASONING now: Provide ONLY the concise solution/proof (<=150 words), "
                    "using the auxiliary lines. "
                    "For calculation problems, end with '**Final answer: <VALUE>**'. "
                    "For proving problems, end with '**Conclusion: <STATEMENT>**'. "
                    "Output TEXT ONLY (no images)."
                ),
                params=request_params,
            )
            text_fn = ex_dir / "model_text.txt"
            text_fn.write_text(text_out or "", encoding="utf-8")
            record["text_file"] = str(text_fn)

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            manifest_rows.append([big_k, small_k, str(orig_rel), img_path, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([big_k, small_k, str(orig_rel), "", f"[ERROR] {exc}"])

        _write_result(ex_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["big_key", "small_key", "original_image_rel", "generated_overlay", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# TASK: Jigsaw
# =============================================================================

_JIGSAW_PROMPT = """\
You are a unified vision-language model. You will be given:
(1) a 2x2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images ("Candidate 0" and "Candidate 1").

Your job:
- For each candidate, synthesize a completed 2x2 image by placing that candidate EXACTLY into the
  bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering,
  no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate,
  mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell

3) Analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "<=30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.

Inputs:
"""


def run_task_jigsaw(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "jigsaw"
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_root / "jigsaw_dataset_2x2ref"
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Jigsaw metadata not found: {meta_path}")
    ds = json.loads(meta_path.read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = ds.get("items", [])
    if max_samples > 0:
        items = items[:max_samples]

    def _resolve(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        pth = Path(p)
        if pth.is_absolute():
            return pth
        # Strip leading "./data/" or "data/" prefix since dataset_dir
        # already points inside the data tree.
        rel = pth.as_posix()
        for prefix in ("./data/jigsaw_dataset_2x2ref/", "data/jigsaw_dataset_2x2ref/",
                        "./data/", "data/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        return dataset_dir / rel

    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(items),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for k, it in tqdm(enumerate(items, 1), total=len(items), desc=f"[{task_name}]"):
        ex_id = it.get("id", f"ex_{k:05d}")
        ex_dir = out_dir / _sanitize_name(ex_id)
        ex_dir.mkdir(parents=True, exist_ok=True)
        done_marker = ex_dir / "_done.ok"

        if resume and done_marker.exists() and not _previous_images_missing(ex_dir):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": ex_id, "status": "skipped_resume"})
            continue

        summary["count_processed"] += 1
        cand_paths = it.get("candidate_paths") or ["", ""]
        record: Dict[str, Any] = {
            "id": ex_id,
            "status": "unknown",
            "ex_dir": str(ex_dir),
            "ref_2x2": it.get("ref_panel", {}).get("ref_image_path"),
            "cand0": cand_paths[0] if len(cand_paths) > 0 else "",
            "cand1": cand_paths[1] if len(cand_paths) > 1 else "",
            "expected_image_count": 2,
            "text_file": None,
            "images_saved": [],
            "errors": [],
        }

        try:
            ref_path = _resolve(record["ref_2x2"])
            cand0_path = _resolve(record["cand0"])
            cand1_path = _resolve(record["cand1"])
            for lbl, p in [("ref_2x2", ref_path), ("cand0", cand0_path), ("cand1", cand1_path)]:
                if not p or not p.exists():
                    raise FileNotFoundError(f"Missing input '{lbl}': {p}")

            base_ctx: List[ContextItem] = []
            add_text(base_ctx, _JIGSAW_PROMPT)
            add_text(base_ctx, "REFERENCE_2x2:")
            add_image_path(base_ctx, ref_path)  # type: ignore[arg-type]
            add_text(base_ctx, "CANDIDATE_0:")
            add_image_path(base_ctx, cand0_path)  # type: ignore[arg-type]
            add_text(base_ctx, "CANDIDATE_1:")
            add_image_path(base_ctx, cand1_path)  # type: ignore[arg-type]

            # Step 1: image with Candidate 0
            out0 = ex_dir / "model_image_01.png"
            img0_path, _ = generate_image_from_context(
                pipeline, backbone, list(base_ctx),
                out_path=out0,
                prompt_suffix=(
                    "Output ONLY item (1): a single image with Candidate 0 placed in the "
                    "bottom-right cell. No text."
                ),
                params=request_params,
            )
            if not Path(img0_path).exists():
                raise RuntimeError(f"Candidate-0 image not written: {img0_path}")

            add_text(base_ctx, "COMPLETED WITH CANDIDATE 0:")
            add_image_path(base_ctx, str(img0_path))

            # Step 2: image with Candidate 1
            out1 = ex_dir / "model_image_02.png"
            img1_path, _ = generate_image_from_context(
                pipeline, backbone, list(base_ctx),
                out_path=out1,
                prompt_suffix=(
                    "Output ONLY item (2): a single image with Candidate 1 placed in the "
                    "bottom-right cell. No text."
                ),
                params=request_params,
            )
            if not Path(img1_path).exists():
                raise RuntimeError(f"Candidate-1 image not written: {img1_path}")

            record["images_saved"] = [img0_path, img1_path]

            add_text(base_ctx, "COMPLETED WITH CANDIDATE 1:")
            add_image_path(base_ctx, str(img1_path))

            # Step 3: text reasoning + decision
            text_out = generate_text_from_context(
                pipeline, backbone, list(base_ctx),
                prompt_suffix=(
                    'Now output EXACTLY ONE '
                    '<FINAL_ANSWER_JSON>{"choice": 0 or 1, "rationale": "<=30 words"}'
                    '</FINAL_ANSWER_JSON>\n'
                    "Do not output any additional images."
                ),
                params=request_params,
            )
            text_fn = ex_dir / "model_text.txt"
            text_fn.write_text(text_out or "", encoding="utf-8")
            record["text_file"] = str(text_fn)

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            manifest_rows.append([ex_id, img0_path, img1_path, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([ex_id, "", "", f"[ERROR] {exc}"])

        _write_result(ex_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["id", "generated_image_cand0", "generated_image_cand1", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# TASK: Maze
# =============================================================================

_MAZE_PROMPT = """\
You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE: generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer.
   - FORBIDDEN: collage/montage/spritesheet/grid/multi-panel/stacked images; no arrows, captions.

2) After all step images, emit EXACTLY ONE LINE:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>

NO EXTRAS
- No tools, no OCR, no explanations, no text except the single <ANSWER_JSON> line.
"""

_RE_STEPS_DIR = re.compile(
    r"^(?P<prefix>maze_(?P<h>\d+)x(?P<w>\d+))_(?P<id>\d{5})_steps$"
)
_MAZE_STEP0_NAME = "maze_step_0000.png"


def _extract_maze_id(steps_dir: Path) -> Optional[Tuple[str, str]]:
    m = _RE_STEPS_DIR.match(steps_dir.name)
    if not m:
        return None
    return m.group("id"), m.group("prefix")


def _load_maze_gt_moves(step0: Path, maze_root: Path) -> Tuple[List[str], int]:
    steps_dir = step0.parent
    parsed = _extract_maze_id(steps_dir)
    if parsed:
        id_str, prefix = parsed
        candidate = maze_root / f"{prefix}_steps_{id_str}.json"
        if candidate.exists():
            try:
                obj = json.loads(candidate.read_text(encoding="utf-8"))
                moves = [
                    str(x).lower()
                    for x in (obj.get("steps_long") or obj.get("steps") or [])
                ]
                return moves, len(moves)
            except Exception:
                pass
    pngs = sorted(glob_module.glob(str(steps_dir / "maze_step_*.png")))
    k = max(0, len(pngs) - 1)
    return [], k


def run_task_maze(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "maze"
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    maze_root = data_root / "maze"
    if not maze_root.exists():
        raise FileNotFoundError(f"Maze data directory not found: {maze_root}")

    step0_list = sorted(maze_root.rglob(f"*/{_MAZE_STEP0_NAME}"))
    if max_samples > 0:
        step0_list = step0_list[:max_samples]

    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(step0_list),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for step0 in tqdm(step0_list, desc=f"[{task_name}]"):
        steps_dir = step0.parent
        parsed = _extract_maze_id(steps_dir)
        mid = parsed[0] if parsed else uuid.uuid4().hex[:8]
        case_dir = out_dir / f"case_{mid}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_marker = case_dir / "_done.ok"

        if resume and done_marker.exists() and not _previous_images_missing(
            case_dir, "images_saved_flatten"
        ):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": mid, "status": "skipped_resume"})
            manifest_rows.append([mid, str(step0), "", "[SKIPPED]"])
            continue

        summary["count_processed"] += 1

        try:
            _, k = _load_maze_gt_moves(step0, maze_root)
        except Exception as exc:
            record_err: Dict[str, Any] = {
                "id": mid, "status": "error",
                "case_dir": str(case_dir), "step0": str(step0),
                "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)],
            }
            _write_result(case_dir, record_err)
            summary["count_error"] += 1
            summary["per_item"].append(record_err)
            manifest_rows.append([mid, str(step0), "", f"[ERROR] {exc}"])
            continue

        record: Dict[str, Any] = {
            "id": mid,
            "status": "unknown",
            "case_dir": str(case_dir),
            "step0": str(step0),
            "expected_image_count": k,
            "text_file": None,
            "images_saved_flatten": [],
            "errors": [],
        }

        try:

            cand_dir = case_dir / "cand_01"
            cand_dir.mkdir(parents=True, exist_ok=True)
            images_flat: List[str] = []

            ctx: List[ContextItem] = []
            add_text(ctx, _MAZE_PROMPT)
            add_image_path(ctx, str(step0))

            stem = step0.stem  # "maze_step_0000"
            for i in range(1, k + 1):
                # Plan the next move (text)
                step_text = generate_text_from_context(
                    pipeline, backbone, ctx,
                    prompt_suffix=(
                        f'Planning step {i}. '
                        'Output ONE sentence: "Next, move one step up/down/left/right."'
                    ),
                    params=request_params,
                )
                add_text(ctx, step_text)

                # Generate the visual state after the move
                out_path = cand_dir / f"{stem}_step_{i:04d}.png"
                img_path, _ = generate_image_from_context(
                    pipeline, backbone, ctx,
                    out_path=out_path,
                    prompt_suffix=f"Generate the maze image for step {i} (one legal move applied).",
                    params=request_params,
                )
                images_flat.append(img_path)
                add_image_path(ctx, str(img_path))

            # Final move list as JSON
            final_text = generate_text_from_context(
                pipeline, backbone, ctx,
                prompt_suffix=(
                    "Emit EXACTLY ONE LINE: "
                    "<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>. No other text."
                ),
                params=request_params,
            )
            text_path = case_dir / "model_text.txt"
            text_path.write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(text_path)
            record["images_saved_flatten"] = images_flat

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            first_img = images_flat[0] if images_flat else ""
            manifest_rows.append([mid, str(step0), first_img, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([mid, str(step0), "", f"[ERROR] {exc}"])

        _write_result(case_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["maze_id", "step0_path", "generated_first_image", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# TASK: Sliding Puzzle
# =============================================================================

_SLIDING_PROMPT = """\
You are a precise sliding puzzle solver.

TASK
- You will be given two images: an INITIAL state and a FINAL state of a 3x3 sliding puzzle.
- Find the sequence of moves to transform the INITIAL state into the FINAL state.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" slides an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the empty (red) space moves.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE: generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each image depicts the puzzle state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep the visual style identical to the inputs; only tile positions change.
   - FORBIDDEN: collage/montage/grid/stacked images; no arrows, captions, overlays.

2) After all step images, emit EXACTLY ONE LINE:
   <ANSWER_JSON>["down","right","up"]</ANSWER_JSON>

NO EXTRAS
- No tools, no explanations, no text except the single <ANSWER_JSON> line.
"""

_SLIDING_STEP_GLOB = "demo_step_*.png"


def _load_sliding_gt_and_k(
    gt_json_path: Optional[str],
    steps_dir: Optional[Union[str, Path]],
) -> Tuple[List[str], int]:
    if gt_json_path and Path(gt_json_path).exists():
        try:
            obj = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
            moves = [str(x).lower() for x in obj.get("steps_words", [])]
            return moves, len(moves)
        except Exception:
            pass
    if steps_dir and Path(steps_dir).exists():
        pngs = sorted(Path(steps_dir).glob(_SLIDING_STEP_GLOB))
        k = max(0, len(pngs) - 1)
        return [], k
    return [], 0


def _record_to_paths_sliding(
    data_dir: Path,
    rec: Dict[str, Any],
) -> Tuple[str, str, Optional[str], str]:
    """Resolve init/final image paths and GT JSON from a sliding puzzle record."""
    def _abs(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        pth = Path(p)
        if pth.is_absolute():
            return pth
        # JSON paths may carry a "./data/sliding/" or "data/sliding/" prefix
        # relative to the Uni-MMMU repo root.  Strip it so the remainder
        # resolves correctly against *data_dir* (which is already data_root/sliding).
        rel = pth.as_posix()
        for prefix in ("./data/sliding/", "data/sliding/", "./data/", "data/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        return data_dir / rel

    steps_dir_rel = rec.get("steps_dir") or rec.get("step_dir") or rec.get("steps_path")
    if steps_dir_rel:
        steps_dir_abs = _abs(str(steps_dir_rel))
        if steps_dir_abs and steps_dir_abs.exists():
            frames = sorted(steps_dir_abs.glob(_SLIDING_STEP_GLOB))
            if len(frames) >= 2:
                init_png = str(frames[0])
                final_png = str(frames[-1])
                gt_rel = rec.get("steps_words_json") or rec.get("gt_json")
                gt_json = str(_abs(str(gt_rel))) if gt_rel and _abs(str(gt_rel)) and _abs(str(gt_rel)).exists() else None  # type: ignore[union-attr]
                case_name = steps_dir_abs.name
                return init_png, final_png, gt_json, case_name

    init_png = _abs(rec.get("initial_png") or rec.get("init_png") or rec.get("init"))
    final_png = _abs(rec.get("final_png") or rec.get("goal_png") or rec.get("final"))
    gt_rel = rec.get("steps_words_json")
    gt_json_p = _abs(str(gt_rel)) if gt_rel else None
    case_name = str(rec.get("case_name") or rec.get("id") or uuid.uuid4().hex[:8])

    if not init_png or not init_png.exists() or not final_png or not final_png.exists():
        raise FileNotFoundError(f"Record missing valid initial/final images: {rec}")

    gt_json = str(gt_json_p) if gt_json_p and gt_json_p.exists() else None
    return str(init_png), str(final_png), gt_json, case_name


def run_task_sliding(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "sliding"
    out_dir = out_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sliding_dir = data_root / "sliding"
    summary_json_path = sliding_dir / "summary_steps_le_8.json"
    if not summary_json_path.exists():
        raise FileNotFoundError(f"Sliding puzzle summary not found: {summary_json_path}")
    records_full: List[Dict[str, Any]] = json.loads(
        summary_json_path.read_text(encoding="utf-8")
    ).get("items", [])
    if max_samples > 0:
        records_full = records_full[:max_samples]

    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(records_full),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for rec in tqdm(records_full, desc=f"[{task_name}]"):
        try:
            init_png, final_png, gt_json_path, case_name = _record_to_paths_sliding(
                sliding_dir, rec
            )
        except Exception as exc:
            manifest_rows.append(["(parse-failed)", "", "", "", f"[SKIP] {exc}"])
            continue

        case_dir = out_dir / f"case_{case_name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_marker = case_dir / "_done.ok"

        if resume and done_marker.exists() and not _previous_images_missing(
            case_dir, "images_saved_flatten"
        ):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": case_name, "status": "skipped_resume"})
            manifest_rows.append([case_name, init_png, final_png, "", "[SKIPPED]"])
            continue

        summary["count_processed"] += 1

        try:
            _, k = _load_sliding_gt_and_k(gt_json_path, None)
        except Exception as exc:
            record_err: Dict[str, Any] = {
                "id": case_name, "status": "error",
                "case_dir": str(case_dir),
                "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)],
            }
            _write_result(case_dir, record_err)
            summary["count_error"] += 1
            summary["per_item"].append(record_err)
            manifest_rows.append([case_name, init_png, final_png, "", f"[ERROR] {exc}"])
            continue

        record: Dict[str, Any] = {
            "id": case_name,
            "status": "unknown",
            "case_dir": str(case_dir),
            "init_png": init_png,
            "final_png": final_png,
            "gt_json": gt_json_path,
            "expected_image_count": k,
            "text_file": None,
            "images_saved_flatten": [],
            "errors": [],
        }

        try:
            cand_dir = case_dir / "cand_01"
            cand_dir.mkdir(parents=True, exist_ok=True)
            images_flat: List[str] = []

            ctx: List[ContextItem] = []
            add_text(ctx, _SLIDING_PROMPT)
            add_text(ctx, "NEW TASK: Initial state.")
            add_image_path(ctx, init_png)
            add_text(ctx, "NEW TASK: Final state (target).")
            add_image_path(ctx, final_png)

            stem = Path(init_png).stem
            for i in range(1, k + 1):
                step_text = generate_text_from_context(
                    pipeline, backbone, ctx,
                    prompt_suffix=(
                        f'Planning step {i}. '
                        'Output ONE sentence: "Next, move one step up/down/left/right."'
                    ),
                    params=request_params,
                )
                add_text(ctx, step_text)

                step_out = cand_dir / f"{stem}_step_{i:04d}.png"
                img_path, _ = generate_image_from_context(
                    pipeline, backbone, ctx,
                    out_path=step_out,
                    prompt_suffix=f"Generate the puzzle image for step {i} (one legal move applied).",
                    params=request_params,
                )
                images_flat.append(img_path)
                add_image_path(ctx, str(img_path))

            final_text = generate_text_from_context(
                pipeline, backbone, ctx,
                prompt_suffix=(
                    "Emit EXACTLY ONE LINE: "
                    "<ANSWER_JSON>[\"move1\",\"move2\",...]</ANSWER_JSON>. No other text."
                ),
                params=request_params,
            )
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)
            record["images_saved_flatten"] = images_flat

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            first_img = images_flat[0] if images_flat else ""
            manifest_rows.append([case_name, init_png, final_png, first_img, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([case_name, init_png, final_png, "", f"[ERROR] {exc}"])

        _write_result(case_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["case_name", "initial_png", "final_png", "generated_first_image", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# TASK: Code Rendering (SVG)
# =============================================================================

_CODE_RENDERING_PROMPT = """\
You will be given SVG source code. Internally parse and render it without tools, then output:
(1) <RENDER_SUMMARY>...</RENDER_SUMMARY> (<=60 words, objective, deterministic description of the final image)
(2) One final rendered image.

Rendering rules (strict):
- Canvas size: determined by <svg> width/height and viewBox.
- Background: only as explicitly drawn (e.g. a <rect>); do not add defaults.
- Coordinates: respect viewBox; (x,y,r,cx,cy, etc.) in user space.
- Stacking: later elements overlay earlier ones.
- Styles: fill, stroke, stroke-width, opacity, fill-rule, stroke-linecap/join, etc.
- Defaults follow SVG spec (e.g. fill=black, stroke=none).
- Transforms: apply from right to left; all path/text positions affected.

Never output reasoning steps or explanations.
"""


def run_task_code_rendering(
    pipeline: InferencePipeline,
    backbone: str,
    data_root: Path,
    out_root: Path,
    max_samples: int,
    resume: bool,
    request_params: Dict[str, Any],
) -> Dict[str, Any]:
    task_name = "code_rendering"
    out_dir = out_root / "code"  # match original Uni-MMMU directory name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_root / "svg"
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Code rendering metadata not found: {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = data.get("samples", [])
    if max_samples > 0:
        cases = cases[:max_samples]

    summary: Dict[str, Any] = {
        "task": task_name,
        "count_total": len(cases),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": [],
    }
    manifest_rows: List[List[Any]] = []

    for idx, s in tqdm(enumerate(cases, 1), total=len(cases), desc=f"[{task_name}]"):
        sid = s.get("id", f"noid_{idx:02d}")
        diff = s.get("difficulty", "")
        svg_rel = s.get("svg")

        case_dir = out_dir / f"case_{idx:02d}_{_sanitize_name(sid)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_marker = case_dir / "_done.ok"

        if not svg_rel:
            record: Dict[str, Any] = {
                "id": f"case_{idx:02d}_{sid}",
                "status": "error",
                "errors": ["Missing 'svg' field in metadata"],
            }
            _write_result(case_dir, record)
            summary["count_error"] += 1
            summary["per_item"].append(record)
            manifest_rows.append([idx, sid, diff, "", "", "", "[MISS-FIELD svg]"])
            continue

        if resume and done_marker.exists() and not _previous_images_missing(case_dir):
            summary["count_skipped"] += 1
            summary["per_item"].append({"id": f"case_{idx:02d}_{sid}", "status": "skipped_resume"})
            manifest_rows.append([idx, sid, diff, str(svg_rel), "", "", "[SKIPPED]"])
            continue

        svg_path = dataset_dir / svg_rel
        if not svg_path.exists():
            record = {
                "id": f"case_{idx:02d}_{sid}",
                "status": "error",
                "errors": [f"SVG file not found: {svg_path}"],
            }
            _write_result(case_dir, record)
            summary["count_error"] += 1
            summary["per_item"].append(record)
            manifest_rows.append([idx, sid, diff, str(svg_rel), "", "", "[MISS-FILE svg]"])
            continue

        summary["count_processed"] += 1
        record = {
            "id": f"case_{idx:02d}_{sid}",
            "status": "unknown",
            "case_dir": str(case_dir),
            "svg": str(svg_rel),
            "png": s.get("png", ""),
            "difficulty": diff,
            "expected_image_count": 1,
            "text_file": None,
            "images_saved": [],
            "errors": [],
        }

        try:
            svg_text = svg_path.read_text(encoding="utf-8")

            # Step 1: think / describe (text only)
            ctx: List[ContextItem] = []
            add_text(ctx, _CODE_RENDERING_PROMPT)
            add_text(ctx, svg_text)
            full_text = generate_text_from_context(
                pipeline, backbone, ctx,
                prompt_suffix="Think step by step. Do NOT generate an image yet.",
                params=request_params,
            )
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)

            # Step 2: generate rendered image
            ctx2: List[ContextItem] = []
            add_text(ctx2, _CODE_RENDERING_PROMPT)
            add_text(ctx2, svg_text)
            add_text(ctx2, full_text)
            img_path, _ = generate_image_from_context(
                pipeline, backbone, ctx2,
                out_path=case_dir / "model_image_01.png",
                prompt_suffix="Generate EXACTLY ONE final rendered image of the SVG (no extra text).",
                params=request_params,
            )
            record["images_saved"] = [img_path]

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
            manifest_rows.append([idx, sid, diff, str(svg_rel), s.get("png", ""), img_path, ""])

        except Exception as exc:
            record["status"] = "error"
            record["errors"].extend([f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=3)])
            summary["count_error"] += 1
            manifest_rows.append([idx, sid, diff, str(svg_rel), "", "", f"[ERROR] {exc}"])

        _write_result(case_dir, record)
        summary["per_item"].append(record)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        out_dir / "manifest.csv",
        ["case_idx", "id", "difficulty", "svg_rel", "gt_png_rel", "generated_image", "notes"],
        manifest_rows,
    )
    _print_task_summary(task_name, summary)
    return summary


# =============================================================================
# Task registry
# =============================================================================

VALID_TASKS = ["science", "math_geo", "jigsaw", "maze", "sliding", "code_rendering"]

_TASK_RUNNERS = {
    "science": run_task_science,
    "math_geo": run_task_math_geo,
    "jigsaw": run_task_jigsaw,
    "maze": run_task_maze,
    "sliding": run_task_sliding,
    "code_rendering": run_task_code_rendering,
}


# =============================================================================
# Python-executable override (for subprocess-based adapters)
# =============================================================================

@contextlib.contextmanager
def _override_python_env(python_env: Optional[str]) -> Generator[None, None, None]:
    """Temporarily prepend a venv *bin* directory to ``PATH`` and set
    ``sys.executable`` so that adapters calling
    ``subprocess.run(["python", ...])`` pick up the correct interpreter.

    *python_env* should be the absolute path to a virtualenv's ``bin/``
    directory (e.g. ``/path/to/model/.venv/bin``).  It is read from the
    ``inference.python_env`` field of the eval YAML config.
    """
    if not python_env:
        yield
        return

    venv_bin = str(Path(python_env).expanduser())
    python_exe = str(Path(venv_bin) / "python3")

    original_exe = sys.executable
    original_path = os.environ.get("PATH", "")

    sys.executable = python_exe
    os.environ["PATH"] = f"{venv_bin}:{original_path}"
    try:
        yield
    finally:
        sys.executable = original_exe
        os.environ["PATH"] = original_path


# =============================================================================
# Main entry point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uni-MMMU generation phase (inference only).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a UMM YAML evaluation config file.",
    )
    args = parser.parse_args()

    config_path = str(args.config)
    eval_cfg, uni_mmmu_cfg, inference_cfg = _load_eval_cfg(config_path)

    # Determine repo root (3 levels up from the original cli location,
    # but for the standalone script we derive it from the config or CWD).
    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for Uni-MMMU eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: Dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    # data_root points to the extracted Uni-MMMU data/ directory
    data_root_value = uni_mmmu_cfg.get("data_root")
    if not data_root_value:
        raise ValueError(
            "`uni_mmmu.data_root` is required "
            "(path to the extracted Uni-MMMU data/ directory)."
        )
    data_root = _resolve_path(str(data_root_value), repo_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Uni-MMMU data root not found: {data_root}")

    # Determine which tasks to run
    tasks_value = uni_mmmu_cfg.get("tasks", "all")
    if tasks_value in ("all", None):
        tasks = list(VALID_TASKS)
    elif isinstance(tasks_value, str):
        tasks = [t.strip() for t in tasks_value.split(",") if t.strip()]
    elif isinstance(tasks_value, list):
        tasks = [str(t).strip() for t in tasks_value if str(t).strip()]
    else:
        tasks = list(VALID_TASKS)

    invalid = [t for t in tasks if t not in VALID_TASKS]
    if invalid:
        raise ValueError(
            f"Unknown Uni-MMMU tasks: {invalid}. Valid tasks: {VALID_TASKS}"
        )

    out_dir = _resolve_path(
        str(uni_mmmu_cfg.get("out_dir", f"output/uni_mmmu/{backbone}")),
        repo_root,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = int(uni_mmmu_cfg.get("max_samples", 0) or 0)
    resume = bool(uni_mmmu_cfg.get("resume", True))

    print(
        f"[uni_mmmu generation] backbone={backbone}, tasks={tasks}, "
        f"data_root={data_root}, out_dir={out_dir}"
    )

    overall_summary: Dict[str, Any] = {
        "benchmark": "uni_mmmu",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "tasks": tasks,
        "mode": "generate",
        "task_summaries": {},
    }

    # ── Generation phase ──
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    python_env = inference_cfg.get("python_env")
    if python_env:
        print(f"[uni_mmmu generation] using python_env: {python_env}")

    with _override_python_env(python_env):
        for task_name in tasks:
            runner = _TASK_RUNNERS[task_name]
            try:
                task_summary = runner(
                    pipeline=pipeline,
                    backbone=backbone,
                    data_root=data_root,
                    out_root=out_dir,
                    max_samples=max_samples,
                    resume=resume,
                    request_params=request_params,
                )
                overall_summary["task_summaries"][task_name] = {
                    "total": task_summary["count_total"],
                    "success": task_summary["count_success"],
                    "error": task_summary["count_error"],
                    "skipped": task_summary["count_skipped"],
                }
            except Exception as exc:
                print(f"[uni_mmmu generation] Task '{task_name}' failed: {exc}")
                traceback.print_exc()
                overall_summary["task_summaries"][task_name] = {"error": str(exc)}

    # Free pipeline GPU memory
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write overall summary
    summary_path = out_dir / "overall_summary.json"
    summary_path.write_text(
        json.dumps(overall_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[uni_mmmu generation] wrote overall summary to {summary_path}")
    print(f"[uni_mmmu generation] completed. backbone={backbone}, outputs={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
