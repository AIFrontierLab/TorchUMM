# -*- coding: utf-8 -*-
import sys, os
import os, re, json, uuid, csv, time, random, traceback, base64, glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


SUMMARY_JSON = Path("./data/sliding/summary_steps_le_8.json")

data_dir = "path_to_ummmu"

RUN_ROOT = Path("./outputs/gpt/sliding")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

EXAMPLE_PROBLEM_IMAGES = [
    "./data/sliding/demo_3x3_00001_steps/demo_step_0000.png",  # Initial
    "./data/sliding/demo_3x3_00001_steps/demo_step_0003.png",  # Final
]
EXAMPLE_SOLUTION_IMAGES = [
    "./data/sliding/demo_3x3_00001_steps/demo_step_0001.png",
    "./data/sliding/demo_3x3_00001_steps/demo_step_0002.png",
]
EXAMPLE_ANS_JSON_PATH = Path("./data/sliding/demo_3x3_steps_words_00001.json")

NUM_SAMPLES             = None    
SHUFFLE_ITEMS           = False
RANDOM_SEED             = 123
RESUME_SKIP_DONE        = True    
FORCE_RERUN             = False  
REGENERATE_IF_NO_IMAGE  = False  
SLEEP_SEC               = 2       

PROMPT_TEMPLATE = """You are a precise sliding puzzle solver.

TASK
- You will be given two images: an INITIAL state and a FINAL state of a 3x3 sliding puzzle.
- The goal is to find the sequence of moves to transform the INITIAL state into the FINAL state.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" consists of sliding an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the empty (red) space moves. For example, if the blue tile is directly above the red space, moving the blue tile down into the red space's position is a "up" move.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the puzzle state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep the visual style identical to the inputs; only tile positions change.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: collage/montage/grid/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["down","right","up"]</ANSWER_JSON>


NO EXTRAS
- No tools, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions.

REMINDERS
- First decide the full path, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched.

After the single <ANSWER_JSON>…</ANSWER_JSON> line, output nothing else.
"""

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def read_demo_moves_json() -> str | None:
    try:
        obj = json.loads(EXAMPLE_ANS_JSON_PATH.read_text(encoding="utf-8"))
        seq = [str(s).lower() for s in obj.get("steps_words", [])]
        return json.dumps(seq)
    except Exception:
        return None

def previous_images_missing(case_dir: Path) -> bool:
    rj = case_dir / "result.json"
    if not rj.exists(): return False
    try:
        rec = json.loads(rj.read_text(encoding="utf-8"))
        imgs = rec.get("images_saved_flatten", [])
        if not imgs: return True
        for p in imgs:
            if not Path(p).exists(): return True
        return False
    except Exception:
        return False

STEP_GLOB = "demo_step_*.png"
def record_to_paths(data_dir: str, rec: Dict[str, Any]) -> Tuple[str, str, Optional[str], str]:
    steps_dir = rec.get("steps_dir") or rec.get("step_dir") or rec.get("steps_path")
    steps_dir = os.path.join(data_dir, steps_dir)
    if steps_dir and Path(steps_dir).exists():
        frames = sorted(Path(steps_dir).glob(STEP_GLOB))
        frames = [str(p) for p in frames]
        if len(frames) >= 2:
            init_png = frames[0]
            final_png = frames[-1]
            gt_json = rec.get("steps_words_json") or rec.get("gt_json") or rec.get("json")
            gt_json = os.path.join(data_dir, gt_json)
            if gt_json and not Path(gt_json).exists():
                gt_json = None
            return init_png, final_png, gt_json, Path(steps_dir).name

    init_png = rec.get("initial_png") or rec.get("init_png") or rec.get("init")
    final_png = rec.get("final_png")   or rec.get("goal_png") or rec.get("final")
    gt_json   = rec.get("steps_words_json")
    case_name = rec.get("case_name") or rec.get("id") or str(uuid.uuid4())[:8]
    init_png = os.path.join(data_dir, init_png)
    final_png = os.path.join(data_dir, final_png)
    gt_json = os.path.join(data_dir, gt_json)
    

    if not (init_png and final_png and Path(init_png).exists() and Path(final_png).exists()):
        raise FileNotFoundError(f"Record missing initial/final images: {rec}")
    if gt_json and not Path(gt_json).exists():
        gt_json = None
    return str(init_png), str(final_png), (str(gt_json) if gt_json else None), str(case_name)

def load_gt_moves_and_k(gt_json_path: Optional[str], steps_dir_maybe: Optional[str]=None) -> Tuple[List[str], int]:
    """优先读 GT JSON（steps_words）；否则用 steps 目录 PNG 数量 - 1 估 k。"""
    if gt_json_path:
        try:
            obj = json.loads(Path(gt_json_path).read_text(encoding="utf-8"))
            moves = [str(x).lower() for x in obj.get("steps_words", [])]
            return moves, len(moves)
        except Exception:
            pass
    if steps_dir_maybe and Path(steps_dir_maybe).exists():
        pngs = sorted(Path(steps_dir_maybe).glob(STEP_GLOB))
        k = max(0, len(pngs) - 1)  # 去掉 step_0000
        return [], k
    return [], 0

def build_fewshot_ctx() -> List[ContextItem]:
    ctx: List[ContextItem] = []
    add_text(ctx, "--- DEMONSTRATION START ---")
    if Path(EXAMPLE_PROBLEM_IMAGES[0]).exists():
        add_text(ctx, "DEMONSTRATION: The initial state.")
        add_image_path(ctx, EXAMPLE_PROBLEM_IMAGES[0])
    if Path(EXAMPLE_PROBLEM_IMAGES[1]).exists():
        add_text(ctx, "DEMONSTRATION: The final state to reach.")
        add_image_path(ctx, EXAMPLE_PROBLEM_IMAGES[1])
    if EXAMPLE_SOLUTION_IMAGES:
        add_text(ctx, "DEMONSTRATION: The sequence of moves to solve the puzzle (one image per move).")
        for pth in EXAMPLE_SOLUTION_IMAGES:
            if Path(pth).exists():
                add_image_path(ctx, pth)
    demo_moves = read_demo_moves_json()
    if demo_moves is not None:
        add_text(ctx, "DEMONSTRATION: The final moves list in JSON format.")
        add_text(ctx, f"<ANSWER_JSON>{demo_moves}</ANSWER_JSON>")
    add_text(ctx, "--- DEMONSTRATION END ---")
    return ctx

def main():
    assert SUMMARY_JSON.exists(), f"Missing {SUMMARY_JSON}"
    records_full = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")).get("items", [])
    if SHUFFLE_ITEMS:
        random.Random(RANDOM_SEED).shuffle(records_full)
    records = records_full if (NUM_SAMPLES is None) else records_full[:NUM_SAMPLES]

    summary = {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "summary_json": str(SUMMARY_JSON),
        "run_root": str(RUN_ROOT),
        "count_total": len(records),
        "count_processed": 0,
        "count_success": 0,
        "count_error": 0,
        "count_skipped": 0,
        "count_regenerated": 0,
        "per_item": []
    }
    manifest_rows = []  # [case_name, init_rel, final_rel, first_img_rel, notes]

    print(f"[INFO] Loaded {len(records)} records from {SUMMARY_JSON}")
    print(f"[INFO] Output root: {RUN_ROOT}")

    fewshot_ctx = build_fewshot_ctx()

    for rec in records:
        try:
            init_png, final_png, gt_json_path, case_name = record_to_paths(data_dir, rec)
        except Exception as e:
            manifest_rows.append(["(parse-failed)", "", "", "", f"[SKIP] {type(e).__name__}: {e}"])
            continue

        case_dir = ensure_dir(RUN_ROOT / f"case_{case_name}")
        done_marker = case_dir / "_done.ok"

        should_skip = False
        is_regen = False
        if not FORCE_RERUN and RESUME_SKIP_DONE and done_marker.exists():
            if REGENERATE_IF_NO_IMAGE and previous_images_missing(case_dir):
                is_regen = True
            else:
                should_skip = True

        if should_skip:
            summary["count_skipped"] += 1
            summary["per_item"].append({
                "id": case_name, "status": "skipped_resume",
                "case_dir": str(case_dir),
                "init": init_png, "final": final_png
            })
            manifest_rows.append([case_name, init_png, final_png, "", "[SKIPPED]"])
            continue
        if is_regen:
            summary["count_regenerated"] += 1

        summary["count_processed"] += 1

        record: Dict[str, Any] = {
            "id": case_name,
            "status": "unknown",
            "case_dir": str(case_dir),
            "init_png": init_png,
            "final_png": final_png,
            "gt_json": gt_json_path,
            "text_file": None,
            "images_saved_flatten": [],
            "errors": [],
            "regen": is_regen
        }

        try:
            moves_long, k = load_gt_moves_and_k(gt_json_path)

            ctx: List[ContextItem] = []
            ctx.extend(fewshot_ctx)
            add_text(ctx, PROMPT_TEMPLATE)
            add_text(ctx, "Now solve the NEW TASK below. Emit ONE separate image per move, then a single <ANSWER_JSON> line.")
            add_text(ctx, "NEW TASK: Initial state.")
            add_image_path(ctx, init_png)
            add_text(ctx, "NEW TASK: Final state is exactly the same as example")

            cand_dir = ensure_dir(case_dir / "cand_01")
            images_flat: List[str] = []
            stem = Path(init_png).stem
            for i in range(1, k+1):
                step_out = cand_dir / f"{stem}_step_{i:04d}.png"
                step_text = generate_text_from_context(
                    ctx,
                    prompt_suffix=f'Now planing for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right.'
                )
                add_text(ctx, step_text)
                suffix = (f"Now, generate the image for step {i}. ")

                img_path, _ = generate_image_from_context(
                    ctx,
                    out_path=str(step_out),
                    prompt_suffix=suffix
                )
                images_flat.append(str(img_path))
                add_image_path(ctx, str(img_path))
                if SLEEP_SEC: time.sleep(SLEEP_SEC)

            final_text = generate_text_from_context(
                ctx,
                prompt_suffix="Now, emit EXACTLY ONE LINE containing ONLY the final move list "
                              "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            txt_path = case_dir / "model_text.txt"
            txt_path.write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(txt_path)

            record["images_saved_flatten"] = images_flat
            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1

            first_rel = ""
            if images_flat:
                try:
                    first_rel = str(Path(images_flat[0]).relative_to(RUN_ROOT))
                except Exception:
                    first_rel = images_flat[0]
            note = "[REGEN]" if is_regen else ""
            manifest_rows.append([case_name, init_png, final_png, first_rel, note])

        except Exception as e:
            record["status"] = "error"
            record["errors"].append(f"API/Runner exception: {e}")
            record["errors"].append(traceback.format_exc(limit=2))
            summary["count_error"] += 1
            note = "[REGEN][ERROR]" if is_regen else "[ERROR]"
            manifest_rows.append([case_name, init_png, final_png, "", f"{note} {type(e).__name__}: {e}"])

        (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["per_item"].append(record)
        if SLEEP_SEC: time.sleep(SLEEP_SEC)

    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    mfp = RUN_ROOT / "manifest.csv"
    with mfp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_name", "initial_png", "final_png", "generated_first_image_rel", "notes"])
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
