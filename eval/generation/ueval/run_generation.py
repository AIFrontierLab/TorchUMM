#!/usr/bin/env python3
"""
UEval generation script — generates text + image outputs for each UEval prompt.

Called via subprocess from the cli/ueval_eval.py wrapper. Uses umm's InferencePipeline
to run various backbones (bagel, emu3, janus_pro, etc.).

Usage:
    python eval/generation/ueval/run_generation.py --config configs/eval/ueval_bagel.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config


# ---------------------------------------------------------------------------
# Helpers
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
        "januspro": "janus_pro",
        "omnigen": "omnigen2",
        "blip3": "blip3o",
        "blip3_o": "blip3o",
        "token_flow": "tokenflow",
    }
    return aliases.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Dataset loading (HuggingFace or local JSON)
# ---------------------------------------------------------------------------

def _load_prompts_hf(
    hf_dataset: str,
    domains: Optional[List[str]],
    max_samples: int,
    local_cache: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load UEval prompts from a local cache dir or HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required for UEval. "
            "Install with: pip install datasets"
        ) from exc

    # Try local cache first (e.g. /datasets/ueval/UEval from Modal volume)
    if local_cache and Path(local_cache).is_dir():
        print(f"[ueval] loading dataset from local cache: {local_cache}")
        dataset = load_dataset(local_cache, split="test")
    else:
        print(f"[ueval] loading dataset from HuggingFace: {hf_dataset}")
        dataset = load_dataset(hf_dataset, split="test")
    data: List[Dict[str, Any]] = [dict(item) for item in dataset]
    print(f"[ueval] loaded {len(data)} items")

    if domains:
        domain_set = {d.lower() for d in domains}
        data = [
            item for item in data
            if (item.get("task") or item.get("task_type", "")).lower() in domain_set
        ]
        print(f"[ueval] filtered to {len(data)} items for domains: {domains}")

    if max_samples > 0:
        data = data[:max_samples]
    return data


def _load_prompts_local(json_path: Path, max_samples: int) -> List[Dict[str, Any]]:
    """Load UEval prompts from a local JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"UEval local data not found: {json_path}")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        for key in ("data", "items", "entries", "examples", "prompts"):
            if isinstance(raw.get(key), list):
                data = raw[key]
                break
        else:
            data = [raw]
    else:
        data = []
    if max_samples > 0:
        data = data[:max_samples]
    return data


# ---------------------------------------------------------------------------
# Text / image extraction helpers
# ---------------------------------------------------------------------------

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


def _extract_saved_path(result: Any, fallback_dir: Path) -> str:
    if isinstance(result, dict):
        for key in ("saved_paths", "output_path", "image_path"):
            val = result.get(key)
            if isinstance(val, list) and val and isinstance(val[0], str) and val[0]:
                p = Path(val[0])
                if p.is_file():
                    return str(p)
            if isinstance(val, str) and val:
                p = Path(val)
                if p.is_file():
                    return str(p)
        # Try PIL Image objects
        img = result.get("image")
        if isinstance(img, Image.Image):
            out_path = fallback_dir / "generated.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            return str(out_path)
        imgs = result.get("images")
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], Image.Image):
            out_path = fallback_dir / "generated.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(str(out_path), format="PNG")
            return str(out_path)
    if isinstance(result, Image.Image):
        out_path = fallback_dir / "generated.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), format="PNG")
        return str(out_path)
    # Scan fallback_dir for recently created images
    if fallback_dir.is_dir():
        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = sorted(
            [f for f in fallback_dir.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return str(candidates[-1])
    return ""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_BATCH_CAPABLE_BACKBONES: set[str] = {"show_o2"}


def _run_generation(
    pipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    out_dir: Path,
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate text + image for each UEval prompt."""
    if backbone in _BATCH_CAPABLE_BACKBONES and hasattr(pipeline.backbone, "generate_batch"):
        return _run_generation_batch(
            pipeline, backbone, prompts, out_dir, images_dir, request_params, resume,
        )
    return _run_generation_sequential(
        pipeline, backbone, prompts, out_dir, images_dir, request_params, resume,
    )


def _run_generation_batch(
    pipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    out_dir: Path,
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Two-phase batch: batch text understanding, then batch image generation."""
    images_dir.mkdir(parents=True, exist_ok=True)

    model_outputs: List[Dict[str, Any]] = []
    results_path = out_dir / "model_outputs.json"

    existing_lookup: Dict[str, Dict[str, Any]] = {}
    if resume and results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            for item in existing:
                item_id = str(item.get("id", ""))
                if item_id:
                    existing_lookup[item_id] = item
        except Exception:
            pass

    n_ok = n_skip = n_err = 0

    # Collect items that need processing (skip resumed)
    pending: List[Dict[str, Any]] = []
    for item in prompts:
        item_id = str(item.get("id", ""))
        prompt_text = item.get("prompt") or item.get("question") or ""

        if resume and item_id in existing_lookup:
            prev = existing_lookup[item_id]
            has_text = bool(prev.get("text_answer"))
            has_images = bool(prev.get("image_answer"))
            if has_text or has_images:
                model_outputs.append(prev)
                n_skip += 1
                continue

        pending.append({
            "item_id": item_id,
            "prompt_text": prompt_text,
            "task_type": item.get("task") or item.get("task_type", ""),
            "question_type": item.get("question_type", ""),
        })

    if not pending:
        print(f"[ueval] all {len(prompts)} items already completed (resume=True)")
        results_path.write_text(
            json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return model_outputs, {"total": len(prompts), "ok": 0, "skipped": n_skip, "error": 0}

    bb = pipeline.backbone

    # Prefer unified path (single model load) if available
    if hasattr(bb, "run_unified_batch"):
        print(f"[ueval] using unified batch (single model load) for {len(pending)} items ...")
        try:
            text_results, gen_results = bb.run_unified_batch(
                items=pending,
                images_dir=images_dir,
                understanding_params=request_params,
                gen_params=request_params,
            )
        except Exception as exc:
            print(f"[ueval] unified batch error: {exc}")
            text_results = [{"text": ""}] * len(pending)
            gen_results = [{"ok": False}] * len(pending)

        for i, p in enumerate(pending):
            text_answer = text_results[i].get("text", "") if i < len(text_results) else ""
            final_img = images_dir / f"{p['item_id']}.png"
            image_paths: List[str] = []
            ok = i < len(gen_results) and gen_results[i].get("ok", False)
            if ok and final_img.is_file():
                image_paths.append(str(final_img))

            if text_answer or image_paths:
                n_ok += 1
            else:
                n_err += 1

            model_outputs.append({
                "id": p["item_id"],
                "prompt": p["prompt_text"],
                "task_type": p["task_type"],
                "question_type": p["question_type"],
                "text_answer": text_answer,
                "image_answer": image_paths,
            })

        results_path.write_text(
            json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {"total": len(prompts), "ok": n_ok, "skipped": n_skip, "error": n_err}
        return model_outputs, summary

    # Fallback: two-phase batch (separate model loads)
    # Phase 1: batch text understanding
    print(f"[ueval] batch understanding {len(pending)} items ...")
    understand_items = [
        {"prompt": p["prompt_text"], "images": []}
        for p in pending
    ]
    try:
        text_results = bb.understand_batch(understand_items, understanding_cfg=request_params)
    except Exception as exc:
        print(f"[ueval] batch understanding error: {exc}")
        text_results = [{"text": ""}] * len(pending)

    n_text_ok = sum(1 for r in text_results if r.get("text"))
    print(f"[ueval] understanding done: {n_text_ok}/{len(pending)} have text answers")

    # Build gen prompts from text answers
    for i, p in enumerate(pending):
        text_answer = text_results[i].get("text", "") if i < len(text_results) else ""
        p["text_answer"] = text_answer
        if text_answer:
            p["gen_prompt"] = (
                f"{p['prompt_text']}\n\n"
                f"Based on the following description, generate an image:\n"
                f"{text_answer}"
            )
        else:
            p["gen_prompt"] = p["prompt_text"]
        p["final_img"] = images_dir / f"{p['item_id']}.png"

    # Save intermediate results after Phase 1 (text understanding)
    for p in pending:
        model_outputs.append({
            "id": p["item_id"],
            "prompt": p["prompt_text"],
            "task_type": p["task_type"],
            "question_type": p["question_type"],
            "text_answer": p["text_answer"],
            "image_answer": [],
        })
    results_path.write_text(
        json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ueval] saved intermediate results after Phase 1")

    # Phase 2: batch image generation
    print(f"[ueval] batch generating {len(pending)} images ...")
    prompt_items = [
        {"prompt": p["gen_prompt"], "output_path": str(p["final_img"])}
        for p in pending
    ]
    try:
        gen_results = bb.generate_batch(prompt_items, gen_cfg=request_params)
    except Exception as exc:
        print(f"[ueval] batch generation error: {exc}")
        gen_results = [{"ok": False}] * len(pending)

    # Update model_outputs with image results (pending items start at index n_skip)
    for i, p in enumerate(pending):
        text_answer = p["text_answer"]
        image_paths_fb: List[str] = []
        ok = i < len(gen_results) and gen_results[i].get("ok", False)
        if ok and p["final_img"].is_file():
            image_paths_fb.append(str(p["final_img"]))

        if text_answer or image_paths_fb:
            n_ok += 1
        else:
            n_err += 1

        # Update the already-appended entry
        model_outputs[n_skip + i]["image_answer"] = image_paths_fb

    # Save final results
    results_path.write_text(
        json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {"total": len(prompts), "ok": n_ok, "skipped": n_skip, "error": n_err}
    return model_outputs, summary


def _run_generation_sequential(
    pipeline,
    backbone: str,
    prompts: List[Dict[str, Any]],
    out_dir: Path,
    images_dir: Path,
    request_params: Dict[str, Any],
    resume: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Original sequential generation path for non-batch backbones."""
    images_dir.mkdir(parents=True, exist_ok=True)
    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    model_outputs: List[Dict[str, Any]] = []
    results_path = out_dir / "model_outputs.json"

    existing_lookup: Dict[str, Dict[str, Any]] = {}
    if resume and results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
            for item in existing:
                item_id = str(item.get("id", ""))
                if item_id:
                    existing_lookup[item_id] = item
        except Exception:
            pass

    n_ok = n_skip = n_err = 0

    for item in tqdm(prompts, desc="[ueval gen]"):
        item_id = str(item.get("id", ""))
        prompt_text = item.get("prompt") or item.get("question") or ""

        if resume and item_id in existing_lookup:
            prev = existing_lookup[item_id]
            has_text = bool(prev.get("text_answer"))
            has_images = bool(prev.get("image_answer"))
            if has_text or has_images:
                model_outputs.append(prev)
                n_skip += 1
                continue

        text_answer = ""
        image_paths: List[str] = []

        # Step 1: generate text (understanding)
        try:
            text_payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "understanding",
                "prompt": prompt_text,
                "images": [],
                "params": request_params,
            }
            text_result = pipeline.run(text_payload)
            text_answer = _extract_text(text_result)
            # Debug: show what the backbone returned for text understanding
            if isinstance(text_result, dict):
                rc = text_result.get("returncode")
                err_flag = text_result.get("error", "")
                stderr_tail = (text_result.get("stderr") or "")[-500:]
                print(
                    f"[ueval] id={item_id} text_result: rc={rc}, "
                    f"text_len={len(text_answer)}, error={err_flag!r}",
                    flush=True,
                )
                if not text_answer:
                    print(
                        f"[ueval] id={item_id} WARNING: empty text_answer. "
                        f"stderr_tail={stderr_tail!r}",
                        flush=True,
                    )
            else:
                print(
                    f"[ueval] id={item_id} text_result type={type(text_result).__name__}, "
                    f"text_len={len(text_answer)}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ueval] id={item_id} text generation error: {exc}")
            traceback.print_exc()

        # Step 2: generate image, conditioned on the text answer
        if text_answer:
            gen_prompt = (
                f"{prompt_text}\n\n"
                f"Based on the following description, generate an image:\n"
                f"{text_answer}"
            )
        else:
            gen_prompt = prompt_text
        try:
            gen_payload: Dict[str, Any] = {
                "backbone": backbone,
                "task": "generation",
                "prompt": gen_prompt,
                "params": request_params,
            }
            gen_result = pipeline.run(gen_payload)
            # Debug: inspect what the backbone returned
            if isinstance(gen_result, dict):
                img_val = gen_result.get("image")
                print(
                    f"[ueval] id={item_id} gen_result keys={list(gen_result.keys())}, "
                    f"image type={type(img_val).__name__}, "
                    f"is_pil={isinstance(img_val, Image.Image)}",
                    flush=True,
                )
            else:
                print(
                    f"[ueval] id={item_id} gen_result type={type(gen_result).__name__}",
                    flush=True,
                )
            saved = _extract_saved_path(gen_result, workspace)
            print(f"[ueval] id={item_id} saved_path={saved!r}", flush=True)
            if saved and Path(saved).is_file():
                final_img = images_dir / f"{item_id}.png"
                shutil.copy2(saved, str(final_img))
                image_paths.append(str(final_img))
                print(f"[ueval] id={item_id} image copied to {final_img}", flush=True)
        except Exception as exc:
            print(f"[ueval] id={item_id} image generation error: {exc}")
            traceback.print_exc()

        if text_answer or image_paths:
            n_ok += 1
        else:
            n_err += 1

        output_item: Dict[str, Any] = {
            "id": item_id,
            "prompt": prompt_text,
            "task_type": item.get("task") or item.get("task_type", ""),
            "question_type": item.get("question_type", ""),
            "text_answer": text_answer,
            "image_answer": image_paths,
        }
        model_outputs.append(output_item)

        # Checkpoint every item
        results_path.write_text(
            json.dumps(model_outputs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {"total": len(prompts), "ok": n_ok, "skipped": n_skip, "error": n_err}
    return model_outputs, summary


# ---------------------------------------------------------------------------
# Python env override (for backbones that need a specific venv)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _override_python_env(python_env: Optional[str]) -> Generator[None, None, None]:
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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_eval_cfg(
    config_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    ueval_cfg = (
        raw_cfg.get("ueval", {})
        if isinstance(raw_cfg.get("ueval"), dict)
        else {}
    )
    inference_cfg = (
        raw_cfg.get("inference", {})
        if isinstance(raw_cfg.get("inference"), dict)
        else {}
    )
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, ueval_cfg, inference_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="UEval generation (text + image)")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    eval_cfg, ueval_cfg, inference_cfg = _load_eval_cfg(args.config)
    repo_root = Path(__file__).resolve().parents[3]

    # --- Backbone ---
    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for UEval generation.")
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

    # --- UEval config ---
    hf_dataset = str(ueval_cfg.get("hf_dataset", "zlab-princeton/UEval"))
    local_data = ueval_cfg.get("local_data")
    local_cache = ueval_cfg.get("local_cache")

    domains_value = ueval_cfg.get("domains")
    domains: Optional[List[str]] = None
    if isinstance(domains_value, list) and domains_value:
        domains = [str(d) for d in domains_value]
    elif isinstance(domains_value, str) and domains_value.strip() and domains_value.strip() != "all":
        domains = [d.strip() for d in domains_value.split(",") if d.strip()]

    out_dir = _resolve_path(
        str(ueval_cfg.get("out_dir", f"output/ueval/{backbone}")), repo_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = out_dir / "images"
    max_samples = int(ueval_cfg.get("max_samples", 0) or 0)
    resume = bool(ueval_cfg.get("resume", True))

    python_env = inference_cfg.get("python_env")
    if python_env:
        print(f"[ueval] using python_env: {python_env}")

    print(
        f"[ueval] backbone={backbone}, out_dir={out_dir}, "
        f"max_samples={max_samples or 'all'}, resume={resume}"
    )

    # --- Load prompts ---
    if local_data:
        local_path = _resolve_path(str(local_data), repo_root)
        prompts = _load_prompts_local(local_path, max_samples)
    else:
        prompts = _load_prompts_hf(hf_dataset, domains, max_samples, local_cache=local_cache)
    print(f"[ueval] loaded {len(prompts)} prompts")

    # --- Run generation ---
    from umm.inference import InferencePipeline
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    with _override_python_env(python_env):
        model_outputs, gen_summary = _run_generation(
            pipeline=pipeline,
            backbone=backbone,
            prompts=prompts,
            out_dir=out_dir,
            images_dir=images_dir,
            request_params=request_params,
            resume=resume,
        )

    print(
        f"[ueval] generation done — "
        f"ok={gen_summary['ok']}, skipped={gen_summary['skipped']}, error={gen_summary['error']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
