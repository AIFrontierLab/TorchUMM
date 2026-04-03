# -*- coding: utf-8 -*-
"""
Maze Few-shot → New Case — jigsaw-style batch caller (Ovis-U1; k 次生图 + 末尾生文)
- 与原 Gemini 版保持：Prompt/IO/目录/断点续跑/统计 不变
- 仅替换为 Ovis-U1，并且：根据 GT 步数 k → 调 k 次 generate_image_from_context → 最后 generate_text_from_context
- 逐样本目录：model_text.txt、cand_01/maze_step_*.png、result.json、_done.ok
- 全局：summary.json + manifest.csv
"""

import os, re, json, uuid, csv, time, random, traceback, glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, Literal
from dataclasses import dataclass
from tqdm import tqdm


EXAMPLE_PROBLEM_PATH = Path("./data/maze/maze_6x6_00015_steps/maze_step_0000.png")
EXAMPLE_STEPS = [
    Path("./data/maze/maze_6x6_00015_steps/maze_step_0001.png"),
    Path("./data/maze/maze_6x6_00015_steps/maze_step_0002.png"),
]
EXAMPLE_MOVES_JSON_PATH = Path("./data/maze/maze_6x6_steps_00015.json")

MAZE_ROOT = Path("./data/maze")

RUN_ROOT  = Path("./outputs/gpt/maze")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES   = None
EXCLUDE_EXAMPLE_ID = True
SHUFFLE_ITEMS = False
RANDOM_SEED   = 2025

RESUME_SKIP_DONE       = True
FORCE_RERUN            = False
REGENERATE_IF_NO_IMAGE = False

SLEEP_SEC = 0.2

PROMPT_TEMPLATE = """You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>…</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""

STEP0_NAME = "maze_step_0000.png"
RE_STEPS_DIR = re.compile(r"^(?P<prefix>maze_(?P<h>\d+)x(?P<w>\d+))_(?P<id>\d{5})_steps$")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def find_all_step0(root: Path) -> List[Path]:
    return sorted(root.rglob(f"*/{STEP0_NAME}"))

def extract_id_dims_from_steps_dir(steps_dir: Path) -> Optional[Tuple[str, str]]:
    m = RE_STEPS_DIR.match(steps_dir.name)
    if not m: return None
    return m.group("id"), m.group("prefix")  # ("00015","maze_6x6")

def example_id() -> Optional[str]:
    if not EXAMPLE_PROBLEM_PATH.exists(): return None
    out = extract_id_dims_from_steps_dir(EXAMPLE_PROBLEM_PATH.parent)
    return out[0] if out else None

def derive_steps_json_from_step0(step0: Path) -> Optional[Path]:
    steps_dir = step0.parent
    parsed = extract_id_dims_from_steps_dir(steps_dir)
    if not parsed: return None
    id_str, prefix = parsed
    candidate = MAZE_ROOT / f"{prefix}_steps_{id_str}.json"
    return candidate if candidate.exists() else None

def load_gt_moves_and_k(step0: Path) -> Tuple[List[str], int]:
    """优先读 steps_long；否则用 steps 目录 PNG 数量-1 估计 k。"""
    js = derive_steps_json_from_step0(step0)
    if js and js.exists():
        try:
            obj = json.loads(js.read_text(encoding="utf-8"))
            moves = obj.get("steps_long") or obj.get("steps") or []
            moves = [str(x).lower() for x in moves]
            return moves, len(moves)
        except Exception:
            pass
    pngs = sorted(glob.glob(str(step0.parent / "maze_step_*.png")))
    k = max(0, len(pngs) - 1) 
    return [], k

def example_moves_json_str() -> Optional[str]:
    try:
        obj = json.loads(EXAMPLE_MOVES_JSON_PATH.read_text(encoding="utf-8"))
        seq = [s.lower() for s in obj.get("steps_long", [])]
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


import torch
from PIL import Image
from transformers import AutoModelForCausalLM

@dataclass
class CtxImagePath:
    path: str
    mime: str = "image/png"

@dataclass
class ContextItem:
    kind: Literal["text","image"]
    payload: Union[str, CtxImagePath, Image.Image]

def add_text(ctx: List[ContextItem], text: str):
    ctx.append(ContextItem("text", text))

def add_image_path(ctx: List[ContextItem], path: str, mime: str = "image/png"):
    ctx.append(ContextItem("image", CtxImagePath(path, mime)))



def build_fewshot_ctx() -> List[ContextItem]:
    ctx: List[ContextItem] = []
    if EXAMPLE_PROBLEM_PATH.exists():
        add_image_path(ctx, str(EXAMPLE_PROBLEM_PATH))
        add_text(ctx, "DEMONSTRATION: Example problem image above.")
    for i, p in enumerate(EXAMPLE_STEPS, 1):
        if Path(p).exists():
            add_image_path(ctx, str(p))
            add_text(ctx, f"DEMONSTRATION: example step image #{i}.")
    mv = example_moves_json_str()
    if mv is not None:
        add_text(ctx, f"DEMONSTRATION: final moves\n<ANSWER_JSON>{mv}</ANSWER_JSON>")
    return ctx

# ---------------- Main ----------------
def main():
    step0_list = find_all_step0(MAZE_ROOT)
    if not step0_list:
        raise FileNotFoundError(f"No step0 images found under {MAZE_ROOT}")

    ex_id = example_id() if EXCLUDE_EXAMPLE_ID else None
    if ex_id:
        step0_list = [p for p in step0_list if (extract_id_dims_from_steps_dir(p.parent) or [""])[0] != ex_id]

    if SHUFFLE_ITEMS:
        random.Random(RANDOM_SEED).shuffle(step0_list)
    if NUM_SAMPLES is not None:
        step0_list = step0_list[:NUM_SAMPLES]

    summary = {
        "project_id": PROJECT_ID, "location": LOCATION,
        "run_root": str(RUN_ROOT), "maze_root": str(MAZE_ROOT),
        "count_total": len(step0_list), "count_processed": 0, "count_success": 0,
        "count_error": 0, "count_skipped": 0, "count_regenerated": 0, "per_item": []
    }
    manifest_rows = []


    fewshot_ctx = build_fewshot_ctx()

    for step0 in tqdm(step0_list):
        steps_dir = Path(step0).parent
        parsed = extract_id_dims_from_steps_dir(steps_dir)
        mid = (parsed[0] if parsed else uuid.uuid4().hex[:8])
        case_dir = ensure_dir(RUN_ROOT / f"case_{mid}")
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
            summary["per_item"].append({"id": mid, "status": "skipped_resume", "case_dir": str(case_dir), "step0": str(step0)})
            manifest_rows.append([mid, str(step0), "", "[SKIPPED]"])
            continue
        if is_regen:
            summary["count_regenerated"] += 1

        summary["count_processed"] += 1
        record: Dict[str, Any] = {
            "id": mid, "status": "unknown", "case_dir": str(case_dir), "step0": str(step0),
            "text_file": None, "images_saved_flatten": [], "errors": [], "regen": is_regen
        }

        try:
            gt_moves_long, k = load_gt_moves_and_k(step0)

            cand_dir = ensure_dir(case_dir / "cand_01")
            images_flat: List[str] = []

            ctx: List[ContextItem] = []
            ctx.extend(fewshot_ctx)
            add_text(ctx, PROMPT_TEMPLATE)
            add_image_path(ctx, str(step0)) 

            stem = Path(step0).stem  # maze_step_0000
            for i in range(1, k+1):
                step_text = generate_text_from_context(
                    ctx,
                    prompt_suffix=f'Now planing for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right.'
                )
                add_text(ctx, step_text)
                print(step_text)
                out_path = cand_dir / f"{stem}_step_{i:04d}.png"
                img_path, _ = generate_image_from_context(
                    ctx, out_path=str(out_path),
                    prompt_suffix=f"Now, generate the image for step {i}. "
                )
                images_flat.append(str(img_path))
                add_image_path(ctx, str(img_path))
                if SLEEP_SEC: time.sleep(SLEEP_SEC)

            final_text = generate_text_from_context(
                ctx,
                prompt_suffix="After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                              "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text."
            )
            text_path = case_dir / "model_text.txt"
            text_path.write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(text_path)

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
            manifest_rows.append([mid, str(step0), first_rel, note])

        except Exception as e:
            raise e
            record["status"] = "error"
            record["errors"].append(f"API/Runner exception: {e}")
            record["errors"].append(traceback.format_exc(limit=2))
            summary["count_error"] += 1
            note = "[REGEN][ERROR]" if is_regen else "[ERROR]"
            manifest_rows.append([mid, str(step0), "", f"{note} {type(e).__name__}: {e}"])

        (case_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["per_item"].append(record)
        if SLEEP_SEC: time.sleep(SLEEP_SEC)

    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RUN_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["maze_id", "step0_path", "generated_first_image_rel", "notes"]); w.writerows(manifest_rows)

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
