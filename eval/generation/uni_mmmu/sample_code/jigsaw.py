# -*- coding: utf-8 -*-
import sys, os
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json, time, random, traceback, re

from PIL import Image


# --------------------
# Config
# --------------------
DATASET_DIR = Path("./data/jigsaw_dataset_2x2ref")
OUT_ROOT    = Path("./outputs/gpt/jigsaw")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_FN  = OUT_ROOT / "summary.json"


MAX_TESTS     = None 
START_INDEX   = 0
SLEEP_SEC     = 0.2
RANDOM_SEED   = 42
SHUFFLE_ITEMS = False

RESUME_SKIP_DONE = True
FORCE_RERUN      = False


PROMPT = """
You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images (“Candidate 0” and “Candidate 1”).

Your job:
- For each candidate, synthesize a completed 2×2 image by placing that candidate EXACTLY into the bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering, no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate, mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell


3) analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "≤30 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.
- The only edits allowed are pasting the candidate into the bottom-right cell and necessary size matching for that cell. All other pixels must remain identical to the reference.

Inputs :
"""

# --------------------
# Helpers
# --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_dataset(meta_path: Path) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (s or "").strip())[:120] or "item"

def resolve_under_dataset(p: Optional[str]) -> Optional[Path]:
    if not p: return None
    pth = Path(p)
    return pth if pth.is_absolute() else (DATASET_DIR / pth)


def preview_line(s: Optional[str], n: int = 160) -> str:
    return (s or "").strip().replace("\n", " ")[:n]

# --------------------
# Main
# --------------------
def main():
    meta_path = DATASET_DIR / "metadata.json"
    ds = load_dataset(meta_path)
    items = ds.get("items", [])
    total = len(items)

    order = list(range(total))
    if SHUFFLE_ITEMS:
        random.Random(RANDOM_SEED).shuffle(order)

    start = max(0, START_INDEX)
    end = total if MAX_TESTS is None else min(total, start + MAX_TESTS)
    run_idx = order[start:end]

    summary = {
        "runner": "ovis-u1 jigsaw (triptych input; image0→image1→text)",
        "dataset_dir": str(DATASET_DIR),
        "out_root": str(OUT_ROOT),
        "start_index": START_INDEX,
        "max_tests": MAX_TESTS,
        "ran_count": 0,
        "success_count": 0,
        "error_count": 0,
        "skipped_count": 0,
        "per_item": []
    }

    for k, idx in enumerate(tqdm(run_idx, desc="Ovis Jigsaw (triptych)"), 1):
        it = items[idx]
        ex_id = it.get("id", f"ex_{idx:05d}")
        ex_dir = OUT_ROOT / sanitize_name(ex_id)
        ensure_dir(ex_dir)

        done_marker = ex_dir / "_done.ok"
        if RESUME_SKIP_DONE and not FORCE_RERUN and done_marker.exists():
            summary["skipped_count"] += 1
            summary["per_item"].append({
                "id": ex_id, "dataset_index": idx,
                "status": "skipped_resume", "ex_dir": str(ex_dir)
            })
            continue

        summary["ran_count"] += 1
        record: Dict[str, Any] = {
            "id": ex_id,
            "dataset_index": idx,
            "status": "unknown",
            "ex_dir": str(ex_dir),
            "ref_2x2": it.get("ref_panel", {}).get("ref_image_path"),
            "cand0": (it.get("candidate_paths") or ["",""])[0],
            "cand1": (it.get("candidate_paths") or ["",""])[1],
            "text_file": None,
            "images_saved": [],
            "errors": []
        }

        try:
            ref_path   = resolve_under_dataset(record["ref_2x2"])
            cand0_path = resolve_under_dataset(record["cand0"])
            cand1_path = resolve_under_dataset(record["cand1"])
            for lbl, p in [("ref_2x2", ref_path), ("cand0", cand0_path), ("cand1", cand1_path)]:
                if not p or not p.exists():
                    raise FileNotFoundError(f"Missing input {lbl}: {p}")


            base_ctx: List[ContextItem] = []
            add_text(base_ctx, PROMPT)
            add_text(base_ctx, "REFERENCE_2x2:")
            add_image_path(base_ctx, ref_path)
            add_text(base_ctx, "CANDIDATE_0:")
            add_image_path(base_ctx, cand0_path)
            add_text(base_ctx, "CANDIDATE_1:")
            add_image_path(base_ctx, cand1_path)

            ctx0 = list(base_ctx)
            out0 = ex_dir / "model_image_01.png"
            img0_path, _ = generate_image_from_context(
                ctx0,
                out_path=str(out0),
                prompt_suffix=(
                    "Output ONLY item (1): a single image with Candidate 0 placed in the bottom-right cell. No text."
                ),
            )
            if not Path(img0_path).exists() or Path(img0_path).stat().st_size == 0:
                raise RuntimeError(f"Candidate 0 image not written: {img0_path}")

            add_text(base_ctx, "COMPLETED WITH CANDIDATE 0:")
            add_image_path(base_ctx, str(img0_path))
            ctx1 = list(base_ctx)
            out1 = ex_dir / "model_image_02.png"
            img1_path, _ = generate_image_from_context(
                ctx1,
                out_path=str(out1),
                prompt_suffix=(
                    "Output ONLY item (2): a single image with Candidate 1 placed in the bottom-right cell. No text."
                ),
            )
            if not Path(img1_path).exists() or Path(img1_path).stat().st_size == 0:
                raise RuntimeError(f"Candidate 1 image not written: {img1_path}")

            record["images_saved"] = [str(img0_path), str(img1_path)]

            add_text(base_ctx, "COMPLETED WITH CANDIDATE 1:")
            add_image_path(base_ctx, str(img1_path))
            ctx_text = list(base_ctx)


            text_out = generate_text_from_context(
                ctx_text,
                prompt_suffix=(
                    "Now output EXACTLY ONE <FINAL_ANSWER_JSON>{\"choice\": 0 or 1, \"rationale\": \"≤30 words\"}</FINAL_ANSWER_JSON>\n"
                    "Do not output any additional images."
                ),
            )

            text_fn = ex_dir / "model_text.txt"
            text_fn.write_text(text_out or "", encoding="utf-8")
            record["text_file"] = str(text_fn)

            (ex_dir / "_done.ok").write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["success_count"] += 1

        except Exception as e:
            # raise e
            record["status"] = "error"
            record["errors"].append(f"Runner exception: {e}")
            record["errors"].append(traceback.format_exc(limit=2))
            summary["error_count"] += 1

        with open(ex_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        summary["per_item"].append(record)
        time.sleep(SLEEP_SEC)

        if k % 10 == 0:
            print(f"[{k}/{len(run_idx)}] {ex_id} → {record['status']}")

    with open(SUMMARY_FN, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Processed: {summary['ran_count']}, Success: {summary['success_count']}, "
          f"Errors: {summary['error_count']}, Skipped: {summary['skipped_count']}")
    print(f"Saved: {SUMMARY_FN}")

if __name__ == "__main__":
    main()
