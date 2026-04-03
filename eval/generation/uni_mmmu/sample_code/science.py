# -*- coding: utf-8 -*-

import os, json, random, base64, uuid, csv, re, time, traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm


DATA_JSON   = Path("./data/science/dim_all.json")

RUN_ROOT    = Path("./outputs/gpt/science")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES     = None
SHUFFLE_ITEMS   = False
RANDOM_SEED     = 123

RESUME_SKIP_DONE       = False
FORCE_RERUN            = False
REGENERATE_IF_NO_IMAGE = True

SLEEP_SEC       = 0.1

# ---------------- Prompt Template (UNCHANGED) ----------------
PROMPT_TEMPLATE = """You are a unified vision-language model. You will be given:

(1) one initial image, and 
(2) a textual condition describing an operation/environmental change.

Your job:
- Infer the UNIQUE final state using real-world knowledge and deterministic reasoning.
- Do NOT restate the condition as the result; derive the result causally.
- Do NOT introduce new persistent objects unless they follow necessarily from the condition (e.g., foam from gas, puddle from melting).
- Keep the scene consistent: objects present initially should remain unless the condition implies their removal.
- Output EXACTLY:
<OUTPUT_PROMPT> a concise, deterministic explanation (≤120 words) ending with a precise visual description of the final state. No hedging, no multiple possibilities. </OUTPUT_PROMPT>
And generate EXACTLY ONE image depicting the final state (no extra text).

Hard constraints:
- Deterministic, single outcome.
- No meta talk about prompts, models, or pipelines.
- Do not copy the condition as the result; reason from it.
"""

def load_cases(data_json: Path, num: Optional[int]) -> List[Dict[str, Any]]:
    obj = json.loads(data_json.read_text(encoding="utf-8"))
    pool = []
    for block in obj:
        for s in block.get("samples", []):
            imgs = s.get("input_image_file_path_list") or []
            cond = s.get("input_prompt")
            if imgs and cond and isinstance(imgs, list) and isinstance(cond, str):
                pool.append({
                    "initial_image": imgs[0],
                    "condition": cond.strip(),
                    "meta": {
                        "level_1": block.get("level_1_category"),
                        "level_2": block.get("level_2_category"),
                        "raw": s
                    }
                })
    if SHUFFLE_ITEMS:
        random.Random(RANDOM_SEED).shuffle(pool)
    return pool if (num is None) else pool[:num]

def save_all_images_blobs(images: List[Tuple[str, bytes]], out_dir: Path) -> list:
    """Kept for parity; not used by Ovis path (we save a single PNG path directly)."""
    saved = []
    for i, (mime, blob) in enumerate(images, 1):
        m = (mime or "").lower()
        ext = ".png" if "png" in m else (".jpg" if ("jpeg" in m or "jpg" in m) else (".webp" if "webp" in m else ".bin"))
        fp  = out_dir / f"model_image_{i:02d}{ext}"
        fp.write_bytes(blob)
        saved.append(str(fp))
    return saved

def preview_line(s: str, n: int = 180) -> str:
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



# =============================== Main ===============================
def main():
    assert DATA_JSON.exists(), f"Data json not found: {DATA_JSON}"
    cases = load_cases(DATA_JSON, NUM_SAMPLES)

    summary = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "data_json": str(DATA_JSON),
        "run_root": str(RUN_ROOT),
        "count_total": len(cases),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "count_regenerated": 0,
        "per_item": []
    }
    manifest_rows = []

    print(f"[INFO] Loaded {len(cases)} cases from {DATA_JSON}")
    print(f"[INFO] Output root: {RUN_ROOT}")


    for idx, case in tqdm(enumerate(cases, 1)):
        initial_image = case["initial_image"]
        condition     = case["condition"]
        meta          = case.get("meta", {})

        case_dir = RUN_ROOT / f"case_{idx:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_marker = case_dir / "_done.ok"

        should_skip = False
        is_regen = False
        regen_reason = ""
        if not FORCE_RERUN and RESUME_SKIP_DONE and done_marker.exists():
            if REGENERATE_IF_NO_IMAGE and previous_images_missing(case_dir):
                is_regen = True
                regen_reason = "previous_run_no_image"
            else:
                should_skip = True

        if should_skip:
            summary["count_skipped"] += 1
            summary["per_item"].append({
                "id": f"case_{idx:02d}",
                "status": "skipped_resume",
                "case_dir": str(case_dir),
                "initial_image": initial_image,
                "condition_preview": preview_line(condition)
            })
            manifest_rows.append([idx, initial_image, preview_line(condition), "", "[SKIPPED]"])
            continue

        if is_regen:
            summary["count_regenerated"] += 1

        summary["count_processed"] += 1

        record = {
            "id": f"case_{idx:02d}",
            "status": "unknown",
            "case_dir": str(case_dir),
            "initial_image": initial_image,
            "condition": condition,
            "meta": meta,
            "text_file": None,
            "images_saved": [],
            "errors": [],
            "regen": is_regen,
            "regen_reason": regen_reason
        }

        try:
            ctx: List[ContextItem] = []
            add_text(ctx, PROMPT_TEMPLATE)
            add_text(ctx, "Initial image:")
            add_image_path(ctx, initial_image)
            add_text(ctx, f"Condition: {condition}")

            # 1) TEXT — explanation (saved as model_text.txt)
            full_text = generate_text_from_context(ctx, prompt_suffix="")
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)


            add_text(ctx, full_text)
            # 2) IMAGE — exactly one final-state image (saved as model_image_01.png)
            img_out = case_dir / "model_image_01.png"
            img_path = generate_image_from_context(ctx, out_path=img_out)
            record["images_saved"] = [str(img_path)]

            # Done
            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1

            # manifest: first image relative to RUN_ROOT
            try:
                first_img_rel = str(img_path.relative_to(RUN_ROOT))
            except Exception:
                first_img_rel = str(img_path)
            note = "[REGEN]" if is_regen else ""
            manifest_rows.append([idx, initial_image, preview_line(condition), first_img_rel, note])

        except Exception as e:
            record["status"] = "error"
            record["errors"].append(f"API/Runner exception: {e}")
            record["errors"].append(traceback.format_exc(limit=2))
            summary["count_error"] += 1
            note = "[REGEN][ERROR]" if is_regen else "[ERROR]"
            manifest_rows.append([idx, initial_image, preview_line(condition), "", f"{note} {type(e).__name__}: {e}"])

        (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["per_item"].append(record)
        time.sleep(SLEEP_SEC)

    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RUN_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_idx", "initial_image", "condition_text_preview", "generated_image_first_rel", "notes"])
        w.writerows(manifest_rows)

    print("\n=== SUMMARY ===")
    print(f"Total: {summary['count_total']}, "
          f"Processed: {summary['count_processed']}, "
          f"Success: {summary['count_success']}, "
          f"Errors: {summary['count_error']}, "
          f"Skipped: {summary['count_skipped']}, "
          f"Regenerated(no-image): {summary['count_regenerated']}")
    print(f"Saved → {RUN_ROOT/'summary.json'} and {RUN_ROOT/'manifest.csv'}")

if __name__ == "__main__":
    main()
