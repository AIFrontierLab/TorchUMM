#!/usr/bin/env python3
"""
Unified-Bench generation — round-trip I2T + T2I for each reference image.

Flow (matches UAE paper methodology):
  for each reference image in ref_dir:
    1. caption = backbone.understand(ref_image, instruction=CAPTIONING_INSTRUCTION)
    2. gen_image = backbone.generate(caption, seed=seed)
    3. save gen_image -> out_dir/images/{idx}.png
    4. append (idx, ref_path, caption) -> out_dir/captions.jsonl

The reference images have no prompts of their own — the whole point of the
benchmark is to measure how well a unified model round-trips an image through
its own understanding + generation pipeline.

Called via subprocess from cli/unified_bench.py.

Usage:
    python eval/generation/unified_bench/run_generation.py --config <config.yaml>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config


# Default captioning instruction verbatim from UAE demo.py:232-234
# (model/UAE/demo.py). Kept here so the benchmark runs out of the box with
# paper-faithful behavior; override via config `unified_bench.captioning_instruction`.
DEFAULT_CAPTIONING_INSTRUCTION = (
    "You are an expert vision-language model.\n"
    "Your task is: Given an input image, generate a **textual description** of "
    "the image. If there is text in the image, transcribe it inside double quotes \"\".\n"
    "Now, carefully analyze the input image and output the full description."
)


_BACKBONE_ALIASES = {
    "showo2": "show_o2",
    "showo": "show_o2",
    "janus": "janus_pro",
    "januspro": "janus_pro",
    "emu35": "emu3_5",
    "emu3.5": "emu3_5",
    "omnigen": "omnigen2",
}

# Backbones that are unified (support both I2T and T2I). Others (e.g. blip3o,
# tokenflow) cannot run unified-bench round-trip and should be rejected before
# loading the pipeline.
_UNIFIED_BACKBONES = {
    "bagel", "janus_pro", "emu3", "emu3_5",
    "show_o", "show_o2", "omnigen2",
    "janus_flow", "mmada",
}


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return _BACKBONE_ALIASES.get(normalized, normalized)


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else repo_root / path


def _natural_key(path: Path) -> Tuple[int, str]:
    """Sort '10.jpg' after '9.jpg' instead of lexicographically."""
    stem = path.stem
    m = re.match(r"^(\d+)$", stem)
    if m:
        return (0, f"{int(m.group(1)):012d}")
    return (1, stem)


def _list_reference_images(ref_dir: Path) -> List[Path]:
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference image dir not found: {ref_dir}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=_natural_key)
    if not files:
        raise RuntimeError(f"No reference images found in {ref_dir}")
    return files


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
        # Omnigen2-style: {"understandings": [{"response": "...", ...}]}
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
        img = result.get("image")
        if isinstance(img, Image.Image):
            out_path = fallback_dir / "generated.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            return str(out_path)
        imgs = result.get("images")
        if isinstance(imgs, list) and imgs:
            if isinstance(imgs[0], str) and imgs[0]:
                p = Path(imgs[0])
                if p.is_file():
                    return str(p)
            elif isinstance(imgs[0], Image.Image):
                out_path = fallback_dir / "generated.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imgs[0].save(str(out_path), format="PNG")
                return str(out_path)
    if isinstance(result, Image.Image):
        out_path = fallback_dir / "generated.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(out_path), format="PNG")
        return str(out_path)
    if fallback_dir.is_dir():
        img_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = sorted(
            [f for f in fallback_dir.rglob("*") if f.is_file() and f.suffix.lower() in img_exts],
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return str(candidates[-1])
    return ""


def _run_generation(
    pipeline,
    backbone: str,
    ref_images: List[Path],
    out_dir: Path,
    images_dir: Path,
    captioning_instruction: str,
    request_params: Dict[str, Any],
    resume: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    workspace = images_dir / "_gen_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    captions_path = out_dir / "captions.jsonl"
    existing_captions: Dict[str, Dict[str, Any]] = {}
    if resume and captions_path.is_file():
        try:
            for line in captions_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    existing_captions[str(entry.get("id", ""))] = entry
        except Exception:
            existing_captions = {}

    results: List[Dict[str, Any]] = []
    n_ok = n_skip = n_err = 0

    pbar = tqdm(ref_images, desc="[unified_bench gen]")
    for ref_path in pbar:
        item_id = ref_path.stem
        final_img = images_dir / f"{item_id}.png"

        if resume and final_img.is_file() and item_id in existing_captions:
            results.append(existing_captions[item_id])
            n_skip += 1
            continue

        caption = ""
        saved_path = ""

        # Step 1: I2T — ask the backbone's understanding module to caption the ref image.
        try:
            text_payload = {
                "backbone": backbone,
                "task": "understanding",
                "prompt": captioning_instruction,
                "images": [str(ref_path)],
                "params": request_params,
            }
            text_result = pipeline.run(text_payload)
            caption = _extract_text(text_result)
            if not caption:
                print(
                    f"[unified_bench] id={item_id} WARNING: empty caption from I2T",
                    flush=True,
                )
        except Exception as exc:
            print(f"[unified_bench] id={item_id} I2T error: {exc}")
            traceback.print_exc()

        # Step 2: T2I — regenerate image from the model's own caption.
        if caption:
            try:
                gen_payload = {
                    "backbone": backbone,
                    "task": "generation",
                    "prompt": caption,
                    "params": request_params,
                }
                gen_result = pipeline.run(gen_payload)
                saved = _extract_saved_path(gen_result, workspace)
                if saved and Path(saved).is_file():
                    shutil.copy2(saved, str(final_img))
                    saved_path = str(final_img)
            except Exception as exc:
                print(f"[unified_bench] id={item_id} T2I error: {exc}")
                traceback.print_exc()

        if saved_path:
            n_ok += 1
        else:
            n_err += 1

        entry = {
            "id": item_id,
            "ref_path": str(ref_path),
            "caption": caption,
            "gen_path": saved_path,
        }
        results.append(entry)

        # Checkpoint after every item so resume works even if we crash mid-run.
        with captions_path.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return results, {"total": len(ref_images), "ok": n_ok, "skipped": n_skip, "error": n_err}


def _load_eval_cfg(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    bench_cfg = raw_cfg.get("unified_bench", {}) if isinstance(raw_cfg.get("unified_bench"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    return eval_cfg, bench_cfg, inference_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified-Bench round-trip generation")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    _, bench_cfg, inference_cfg = _load_eval_cfg(args.config)
    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for Unified-Bench.")
    backbone = _normalize_backbone_name(backbone_raw)
    if backbone not in _UNIFIED_BACKBONES:
        raise ValueError(
            f"Backbone '{backbone}' is not a unified (I2T + T2I) model. "
            f"Unified-Bench requires one of: {sorted(_UNIFIED_BACKBONES)}."
        )

    backbone_cfg = inference_cfg.get("backbone_cfg", {}) or {}
    if not isinstance(backbone_cfg, dict):
        backbone_cfg = {}

    request_cfg = inference_cfg.get("request", {}) or {}
    request_params: Dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    ref_dir = _resolve_path(
        str(bench_cfg.get("ref_dir", "/datasets/unified_bench/Image")), repo_root
    )
    out_dir = _resolve_path(
        str(bench_cfg.get("out_dir", f"output/unified_bench/{backbone}")), repo_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"

    resume = bool(bench_cfg.get("resume", True))
    max_samples = int(bench_cfg.get("max_samples", 0) or 0)

    captioning_instruction = str(
        bench_cfg.get("captioning_instruction", DEFAULT_CAPTIONING_INSTRUCTION)
    ).strip()
    if not captioning_instruction:
        captioning_instruction = DEFAULT_CAPTIONING_INSTRUCTION

    ref_images = _list_reference_images(ref_dir)
    if max_samples > 0:
        ref_images = ref_images[:max_samples]

    print(
        f"[unified_bench] backbone={backbone}, ref_dir={ref_dir}, out_dir={out_dir}, "
        f"count={len(ref_images)}, resume={resume}"
    )
    print(f"[unified_bench] captioning_instruction={captioning_instruction[:120]}...")

    from umm.inference import InferencePipeline
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    _, summary = _run_generation(
        pipeline=pipeline,
        backbone=backbone,
        ref_images=ref_images,
        out_dir=out_dir,
        images_dir=images_dir,
        captioning_instruction=captioning_instruction,
        request_params=request_params,
        resume=resume,
    )

    print(
        f"[unified_bench] generation done — "
        f"ok={summary['ok']}, skipped={summary['skipped']}, error={summary['error']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
