# -*- coding: utf-8 -*-

import sys, os

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from string import Template
from tqdm import tqdm
import json, csv, re, time, random, traceback


# ===================== Config =====================
FILTERED_JSON = Path("./data/math_data/filtered.json")
ROOT_DIR      = FILTERED_JSON.parent
ROOT_DIR_R    = ROOT_DIR.resolve()

OUT_DIR       = Path("./outputs/gpt/math")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Batch controls
TEST_LIMIT       = None      # None=run all
SHUFFLE_ITEMS    = False
RANDOM_SEED      = 123
SLEEP_SEC        = 0.2       # politeness delay

# Resume controls
RESUME_SKIP_DONE = True      # skip if _done.ok exists
FORCE_RERUN      = False     # ignore resume marker and rerun

# ===================== Prompt =====================
PROMPT_TMPL = Template("""You are a geometry diagram editor and solver.

TASK ORDER:
1) OVERLAY: On the attached base figure, overlay the auxiliary lines EXACTLY as specified below.
   - Add overlays only; do not move/erase the original objects or labels.
   - Keep labels (A, B, C, …) unchanged and clearly visible.
   - Draw clean, visible lines.

2) REASONING: Give a concise, logically ordered solution or proof (≤150 words), using the constructed auxiliary lines.
   - Keep math tokens (△, ∠, √, π, °) unchanged.
   - Reference elements by their labels.

3) FINISHING:
   - For calculation problems, end with:  **Final answer: <VALUE>**.
   - For proving problems, end with:     **Conclusion: <STATEMENT>**.

PROBLEM:
$PROBLEM_TEXT

CHOICES (if any):
$CHOICES_TEXT

AUXILIARY LINES TO DRAW (English; follow exactly and draw these first):
$AUX_EN
""")

# ===================== Helpers =====================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (s or "").strip())[:120] or "item"

def preview_line(s: Optional[str], n: int = 140) -> str:
    return (s or "").strip().replace("\n", " ")[:n]

# ===================== Main =====================
def main():
    data: Dict[str, Dict[str, Any]] = json.loads(FILTERED_JSON.read_text(encoding="utf-8"))

    # Flatten items
    items: List[Tuple[str, str, Dict[str, Any]]] = []
    for big_k, group in data.items():
        for small_k, item in group.items():
            items.append((big_k, small_k, item))

    if SHUFFLE_ITEMS:
        random.Random(RANDOM_SEED).shuffle(items)
    if TEST_LIMIT:
        items = items[:TEST_LIMIT]

    # Summary + manifest
    summary = {
        "runner": "ovis-u1 two-stage (image→text)",
        "filtered_json": str(FILTERED_JSON),
        "out_dir": str(OUT_DIR),
        "count_total": len(items),
        "count_processed": 0,   # attempted (not skipped)
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "per_item": []
    }
    manifest_rows: List[List[str]] = []

    with tqdm(total=len(items), desc="Ovis-U1 sampling (geometry)") as pbar:
        for big_k, small_k, item in items:
            dir_name = f"{sanitize_name(big_k)}__{sanitize_name(small_k)}"
            ex_dir = OUT_DIR / dir_name
            ensure_dir(ex_dir)
            done_marker = ex_dir / "_done.ok"

            # Resume
            if RESUME_SKIP_DONE and not FORCE_RERUN and done_marker.exists():
                summary["count_skipped"] += 1
                summary["per_item"].append({
                    "id": dir_name, "big_key": big_k, "small_key": small_k,
                    "status": "skipped_resume", "ex_dir": str(ex_dir)
                })
                manifest_rows.append([
                    big_k, small_k, item.get("type"), item.get("original_image"),
                    "", "", "[SKIPPED]"
                ])
                pbar.update(1)
                continue

            summary["count_processed"] += 1
            record = {
                "id": dir_name,
                "big_key": big_k,
                "small_key": small_k,
                "status": "unknown",
                "ex_dir": str(ex_dir),
                "type": item.get("type"),
                "problem_text_en": item.get("problem_text_en"),
                "problem_text": item.get("problem_text"),
                "choices_en": item.get("choices_en"),
                "auxiliary_text_en": item.get("auxiliary_text_en"),
                "original_image": item.get("original_image"),
                "text_file": None,
                "images_saved": [],
                "errors": []
            }

            # Resolve original image (no guessing)
            orig_rel = item.get("original_image")
            if not orig_rel:
                record["status"] = "error"
                record["errors"].append("Missing 'original_image' in JSON")
                (ex_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                summary["count_error"] += 1
                manifest_rows.append([
                    big_k, small_k, item.get("type"), str(orig_rel), "", "", "[MISS-FIELD original_image]"
                ])
                pbar.update(1)
                continue

            orig_path = Path(orig_rel)
            orig_abs = orig_path if orig_path.is_absolute() else (ROOT_DIR_R / orig_path).resolve()
            if not orig_abs.exists():
                record["status"] = "error"
                record["errors"].append(f"Original image not found: {orig_abs}")
                (ex_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                summary["count_error"] += 1
                manifest_rows.append([
                    big_k, small_k, item.get("type"), str(orig_rel), "", "", "[MISS-FILE]"
                ])
                pbar.update(1)
                continue

            # Build prompt text
            problem_text = record["problem_text_en"] or record["problem_text"] or "(no problem text)"
            choices = record["choices_en"]
            choices_text = "\n".join(choices) if isinstance(choices, list) and choices else "(no choices)"
            aux_en = (record["auxiliary_text_en"] or "").strip()

            prompt = PROMPT_TMPL.safe_substitute(
                PROBLEM_TEXT=problem_text,
                CHOICES_TEXT=choices_text,
                AUX_EN=aux_en
            )

            try:
                ctx: List[ContextItem] = []
                add_text(ctx, prompt)
                add_text(ctx, "BASE FIGURE: The following image is the original diagram.")
                add_image_path(ctx, str(orig_abs))

                overlay_path = ex_dir / "model_image_01.png"
                img_path, _ = generate_image_from_context(
                    ctx,
                    out_path=str(overlay_path),
                    prompt_suffix=(
                        "STEP 1 — OVERLAY now: Output EXACTLY ONE image of the base figure "
                        "with the auxiliary lines overlaid as specified. "
                        "Do not change existing objects/labels, do not add text, no captions."
                    ),
                )
                if not Path(img_path).exists() or Path(img_path).stat().st_size == 0:
                    raise RuntimeError(f"Overlay image not written: {img_path}")

                record["images_saved"] = [str(img_path)]

                add_text(ctx, "OVERLAY RESULT (reference for reasoning):")
                add_image_path(ctx, str(img_path))

                text_out = generate_text_from_context(
                    ctx,
                    prompt_suffix=(
                        "STEP 2 — REASONING now: Provide ONLY the concise solution/proof (≤150 words), "
                        "using the auxiliary lines. Keep math tokens (△, ∠, √, π, °) unchanged. "
                        "For calculation problems, end with '**Final answer: <VALUE>**'. "
                        "For proving problems, end with '**Conclusion: <STATEMENT>**'. "
                        "Output TEXT ONLY (no images, no markdown images)."
                    ),
                )

                text_fn = ex_dir / "model_text.txt"
                text_fn.write_text(text_out or "", encoding="utf-8")
                record["text_file"] = str(text_fn)

                # Done marker (for resume)
                done_marker.write_text("ok", encoding="utf-8")
                record["status"] = "ok"
                summary["count_success"] += 1

                # Manifest row
                gen_rel = ""
                if img_path:
                    try:
                        gen_rel = str(Path(img_path).relative_to(OUT_DIR))
                    except Exception:
                        gen_rel = str(img_path)
                manifest_rows.append([
                    big_k, small_k, item.get("type"), str(orig_rel),
                    item.get("auxiliary_image") or "",
                    gen_rel,
                    preview_line(problem_text),
                ])

            except Exception as e:
                record["status"] = "error"
                record["errors"].append(f"API/Runner exception: {e}")
                record["errors"].append(traceback.format_exc(limit=2))
                summary["count_error"] += 1
                manifest_rows.append([
                    big_k, small_k, item.get("type"), str(orig_rel),
                    item.get("auxiliary_image") or "",
                    "",
                    f"[ERROR] {type(e).__name__}: {e}"
                ])

            # Persist per-item record
            (ex_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            summary["per_item"].append(record)
            time.sleep(SLEEP_SEC)
            pbar.update(1)

    # Write summary.json
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write manifest.csv
    with (OUT_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "big_key", "small_key", "type",
            "original_image_rel", "aux_image_rel(ref)", "generated_overlay_rel_first",
            "problem_preview",
        ])
        w.writerows(manifest_rows)

    print("\n=== SUMMARY ===")
    print(f"Total: {summary['count_total']}, "
          f"Processed: {summary['count_processed']}, "
          f"Success: {summary['count_success']}, "
          f"Errors: {summary['count_error']}, "
          f"Skipped: {summary['count_skipped']}")
    print(f"Saved → {OUT_DIR/'summary.json'} and {OUT_DIR/'manifest.csv'}")

if __name__ == "__main__":
    main()
