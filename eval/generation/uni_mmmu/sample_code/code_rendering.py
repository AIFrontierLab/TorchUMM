# -*- coding: utf-8 -*-
import sys, os
import os, json, base64, uuid, csv, time, random, traceback, re, shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable

from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal, Union
from gradio_client import Client, handle_file
from tqdm import tqdm



DATASET_DIR = Path("./data/svg")
META_PATH   = DATASET_DIR / "metadata.json"

OUT_ROOT    = Path("./outputs/gpt/code")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES_TOTAL = None
SHUFFLE_ITEMS     = False
RANDOM_SEED       = 123

RESUME_SKIP_DONE        = True
FORCE_RERUN             = False
REGENERATE_IF_NO_IMAGE  = False 

SLEEP_SEC         = 0.1

PROMPT_TEMPLATE = """You will be given SVG source code. Internally parse and render it without tools, then output:
(1) <RENDER_SUMMARY>…</RENDER_SUMMARY> (≤60 words, objective, deterministic description of the final image)
(2) One final rendered image.
Rendering rules (strict):
Canvas size: determined by <svg> width/height and viewBox.
Background: only as explicitly drawn (e.g. a <rect>); do not add defaults.
Coordinates: respect viewBox; (x,y,r,cx,cy, etc.) in user space.
Stacking: later elements overlay earlier ones.
Styles: fill, stroke, stroke-width, opacity, fill-rule, stroke-linecap/join, etc.
Defaults follow SVG spec (e.g. fill=black, stroke=none).
Transforms: apply from right to left; all path/text positions affected.
On success, output summary + image.
Never output reasoning steps or explanations.
"""


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_cases(meta_json: Path, total: Optional[int], shuffle: bool) -> List[Dict[str, Any]]:
    data = json.loads(meta_json.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if shuffle:
        random.Random(RANDOM_SEED).shuffle(samples)
    return samples if total is None else samples[:total]

def preview_text(s: str, n: int = 200) -> str:
    return (s or "").replace("\n", " ").strip()[:n]

def previous_images_missing(case_dir: Path) -> bool:
    rj = case_dir / "result.json"
    if not rj.exists():
        return False
    try:
        rec = json.loads(rj.read_text(encoding="utf-8"))
        imgs = rec.get("images_saved") or []
        if not imgs:
            return True
        for p in imgs:
            if not Path(p).exists():
                return True
        return False
    except Exception:
        return False

# ============== Main ==============
def main():
    assert META_PATH.exists(), f"metadata.json not found: {META_PATH}"
    cases = load_cases(META_PATH, NUM_SAMPLES_TOTAL, SHUFFLE_ITEMS)

    summary = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "meta_path": str(META_PATH),
        "out_root": str(OUT_ROOT),
        "count_total": len(cases),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "count_regenerated": 0,
        "per_item": []
    }
    manifest_rows = []  # [idx, id, difficulty, svg_rel, gt_png_rel, gen_first_rel, text_preview, notes]

    print(f"[INFO] Loaded {len(cases)} cases from {META_PATH}")
    print(f"[INFO] Output root: {OUT_ROOT}")

    for idx, s in tqdm(enumerate(cases, 1)):
        sid     = s.get("id", f"noid_{idx:02d}")
        diff    = s.get("difficulty", "")
        svg_rel = s.get("svg")
        png_rel = s.get("png", "")

        if not svg_rel:
            case_dir = OUT_ROOT / f"case_{idx:02d}_{sid}_missing"
            ensure_dir(case_dir)
            record = {"id": f"case_{idx:02d}_{sid}", "status": "error", "errors": ["Missing 'svg' field in metadata"]}
            (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["count_error"] += 1
            summary["per_item"].append(record)
            manifest_rows.append([idx, sid, diff, str(svg_rel), str(png_rel), "", "", "[MISS-FIELD svg]"])
            continue

        svg_path = DATASET_DIR / svg_rel
        png_path = DATASET_DIR / png_rel if png_rel else None

        case_dir = OUT_ROOT / f"case_{idx:02d}_{sid}"
        ensure_dir(case_dir)
        done_marker = case_dir / "_done.ok"

        should_skip = False
        regen_reason = ""
        if not FORCE_RERUN and RESUME_SKIP_DONE and done_marker.exists():
            if REGENERATE_IF_NO_IMAGE and previous_images_missing(case_dir):
                regen_reason = "previous_run_no_image"
                should_skip = False
            else:
                should_skip = True

        if should_skip:
            summary["count_skipped"] += 1
            summary["per_item"].append({
                "id": f"case_{idx:02d}_{sid}",
                "status": "skipped_resume",
                "case_dir": str(case_dir),
                "svg": str(svg_rel),
                "png": str(png_rel)
            })
            manifest_rows.append([idx, sid, diff, str(svg_rel), str(png_rel), "", "", "[SKIPPED]"])
            continue

        is_regen = bool(regen_reason)
        if is_regen:
            summary["count_regenerated"] += 1
        summary["count_processed"] += 1

        record = {
            "id": f"case_{idx:02d}_{sid}",
            "status": "unknown",
            "case_dir": str(case_dir),
            "svg": str(svg_rel),
            "png": str(png_rel),
            "difficulty": diff,
            "text_file": None,
            "images_saved": [],
            "errors": [],
            "regen": is_regen,
            "regen_reason": regen_reason or ""
        }

        if not svg_path.exists():
            record["status"] = "error"
            record["errors"].append(f"SVG not found: {svg_path}")
            (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["count_error"] += 1
            manifest_rows.append([idx, sid, diff, str(svg_rel), str(png_rel), "", "", "[MISS-FILE svg]"])
            continue

        svg_text = svg_path.read_text(encoding="utf-8")

        try:

            ordered_ctx: List[ContextItem] = []
            add_text(ordered_ctx, PROMPT_TEMPLATE)
            add_text(ordered_ctx, svg_text)
            
            
            full_text = generate_text_from_context(
                ordered_ctx,
                prompt_suffix="now thinking, do not generate image now",  
            )
            
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)
            ordered_ctx: List[ContextItem] = []
            add_text(ordered_ctx, PROMPT_TEMPLATE)
            add_text(ordered_ctx, svg_text)
            add_text(ordered_ctx, full_text)
            
            img_path, _ = generate_image_from_context(
                ordered_ctx,
                prompt_suffix="Generate EXACTLY ONE final rendered image of the SVG (no extra text).",
                out_path=str(case_dir / "model_image_01.png"),
            )
            record["images_saved"] = [img_path]



            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1

            first_img_rel = ""
            if img_path:
                try:
                    first_img_rel = str(Path(img_path).relative_to(OUT_ROOT))
                except Exception:
                    first_img_rel = img_path

            note = "[REGEN]" if is_regen else ""
            manifest_rows.append([
                idx, sid, diff,
                str(svg_rel),
                str(png_rel) if png_rel else "",
                first_img_rel,
                preview_text(full_text, 200),
                note
            ])

        except Exception as e:
            raise e
            record["status"] = "error"
            record["errors"].append(f"API/Runner exception: {e}")
            record["errors"].append(traceback.format_exc(limit=2))
            summary["count_error"] += 1

            note = "[REGEN][ERROR]" if is_regen else "[ERROR]"
            manifest_rows.append([
                idx, sid, diff,
                str(svg_rel), str(png_rel) if png_rel else "",
                "", "", f"{note} {type(e).__name__}: {e}"
            ])

        (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["per_item"].append(record)
        time.sleep(SLEEP_SEC)

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mfp = OUT_ROOT / "manifest.csv"
    with mfp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "case_idx", "id", "difficulty",
            "svg_rel", "gt_png_rel",
            "generated_first_image_rel",
            "model_text_preview",
            "notes"
        ])
        w.writerows(manifest_rows)

    print("\n=== SUMMARY ===")
    print(f"Total: {summary['count_total']}, "
          f"Processed: {summary['count_processed']}, "
          f"Success: {summary['count_success']}, "
          f"Errors: {summary['count_error']}, "
          f"Skipped: {summary['count_skipped']}, "
          f"Regenerated(no-image): {summary['count_regenerated']}")
    print(f"Saved → {OUT_ROOT/'summary.json'} and {OUT_ROOT/'manifest.csv'}")

if __name__ == "__main__":
    main()
