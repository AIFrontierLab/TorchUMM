# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import base64
import csv
import glob
import html
import json
import math
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration, AutoProcessor
)
from qwen_vl_utils import process_vision_info

try:
    from dreamsim import dreamsim as dreamsim_fn
except ImportError:
    dreamsim_fn = None
try:
    import cairosvg
except ImportError:
    cairosvg = None
try:
    from IPython.display import display, HTML
except ImportError:
    display, HTML = None, None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None



def summarize_all_tasks(configs: Configurations):
    """
    Collects all individual task evaluation summaries and compiles them into a single Excel file.
    Skips tasks where the summary.json file is not found.
    """
    print("\n" + "="*20 + " Aggregating All Task Summaries " + "="*20)
    
    eval_path = Path(configs.eval_path)
    summary_data = {}

    # Column names as specified
    columns = [
        "Jigsaw image score (0-1)", "Jigsaw text Acc",
        "Maze img Acc (step|sample)", "Maze text Acc (step|sample)",
        "Sliding img acc (step|sample)", "Sliding text Acc (step|sample)",
        "math_overlay_acc", "math_text_acc",
        "Science text reasoning acc", "Science text result acc", "Science img acc",
        "Svg text acc", "Svg img shape color score [0-5]", "Svg img pos score [0-5]"
    ]
    
    # Helper to read JSON safely
    read_json_safe = UtilityHelpers.read_json
    
    def get_task_summary(task_name: str, file_path: Path) -> dict:
        """Checks for summary file, prints a note if missing, and then reads it."""
        if not file_path.is_file():
            print(f"[NOTE] Skipping summary for '{task_name}': File not found at {file_path}")
            return {}
        return read_json_safe(file_path)

    # 1. Jigsaw
    jigsaw_summary = get_task_summary("Jigsaw", eval_path / "jigsaw" / "summary.json")
    summary_data["Jigsaw image score (0-1)"] = jigsaw_summary.get("metrics", {}).get("image_score_penalized")
    summary_data["Jigsaw text Acc"] = jigsaw_summary.get("metrics", {}).get("text_accuracy")

    # 2. Maze
    maze_summary = get_task_summary("Maze", eval_path / "maze" / "summary.json")
    summary_data["Maze img Acc (step|sample)"] = maze_summary.get("img_accuracy_frame_macro")
    summary_data["Maze text Acc (step|sample)"] = maze_summary.get("text_accuracy_frame_macro")

    # 3. Sliding Puzzle
    sliding_summary = get_task_summary("Sliding Puzzle", eval_path / "sliding" / "summary.json")
    summary_data["Sliding img acc (step|sample)"] = sliding_summary.get("img_accuracy_frame_macro")
    summary_data["Sliding text Acc (step|sample)"] = sliding_summary.get("text_accuracy_frame_macro")

    # 4. Math (Geometry)
    math_summary = get_task_summary("Geometry", eval_path / "math" / "eval_summary.json")
    summary_data["math_overlay_acc"] = math_summary.get("overlay_acc")
    summary_data["math_text_acc"] = math_summary.get("text_acc")

    # 5. Science
    science_summary = get_task_summary("Science", eval_path / "science" / "eval_summary.json")
    summary_data["Science text reasoning acc"] = science_summary.get("text_reasoning_acc")
    summary_data["Science text result acc"] = science_summary.get("text_result_acc")
    summary_data["Science img acc"] = science_summary.get("image_acc")

    # 6. SVG (Code)
    svg_summary = get_task_summary("SVG", eval_path / "code" / "eval_summary.json")
    summary_data["Svg text acc"] = svg_summary.get("text_semantic_match_rate")
    summary_data["Svg img shape color score [0-5]"] = svg_summary.get("avg_shape_color_accuracy")
    summary_data["Svg img pos score [0-5]"] = svg_summary.get("avg_position_accuracy")

    # Create DataFrame and save to Excel
    df = pd.DataFrame([summary_data], columns=columns)
    output_file = eval_path / f"all_tasks_summary_{configs.model_name}.xlsx"
    
    try:
        df.to_excel(output_file, index=False)
        print(f"\nSuccessfully generated summary Excel file at: {output_file}")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")

class UtilityHelpers:
    """A collection of static utility methods for common tasks like file I/O and data parsing."""

    @staticmethod
    def read_json(path: Union[str, Path]) -> Any:
        try:
            with Path(path).open("r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def write_json(obj: Any, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def sanitize_filename(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (name or "").strip())[:120] or "item"

    @staticmethod
    def image_to_b64_data_url(
        image_path: Union[str, Path], 
        max_w: Optional[int] = 2048
    ) -> str:
        """
        Reads an image file, optionally resizes it while preserving aspect ratio,
        and returns it as a Base64 encoded data URL.

        This function mimics the logic from the second script, including resizing.

        Args:
            image_path: The path to the image file.
            max_w: The maximum width for the image. If the image is wider, it will
                be resized. If None, no resizing is performed.

        Returns:
            A Base64 data URL string (e.g., "data:image/jpeg;base64,..."),
            or an empty string if the file doesn't exist.
        """
        path = Path(image_path)
        if not path.is_file():
            print(f"Warning: Image file not found at {path}")
            return ""

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            ext = path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif ext == ".png":
                mime_type = "image/png"
            elif ext == ".webp":
                mime_type = "image/webp"
            else:
                mime_type = "application/octet-stream" 

        try:
            with Image.open(path) as im:
                if max_w and im.width > max_w:
                    ratio = max_w / float(im.width)
                    new_height = max(1, int(im.height * ratio))
                    im = im.resize((max_w, new_height), Image.Resampling.LANCZOS)

                buffer = BytesIO()
                save_format = "PNG" if mime_type == "image/png" else "JPEG"
                
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and 'transparency' in im.info):
                    save_format = "PNG"

                if save_format == "JPEG" and im.mode != "RGB":
                    im = im.convert("RGB")

                im.save(buffer, format=save_format)
                image_bytes = buffer.getvalue()
                
                mime_type = f"image/{save_format.lower()}"

        except Exception as e:
            print(f"Warning: Pillow failed to process {path} ({e}). Using raw file bytes.")
            image_bytes = path.read_bytes()

        b64_string = base64.b64encode(image_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{b64_string}"

    @staticmethod
    def find_first_json_substring(text: str) -> Optional[str]:
        if not text:
            return None
        start_index = text.find('{')
        if start_index == -1:
            return None
        brace_depth, in_string, is_escaped = 0, False, False
        for i in range(start_index, len(text)):
            char = text[i]
            if char == '"' and not is_escaped:
                in_string = not in_string
            if not in_string:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        return text[start_index: i + 1]
            is_escaped = char == '\\' and not is_escaped
        return None

    @staticmethod
    def parse_maze_image(img_path: str, **kwargs) -> Dict[str, Any]:
        def _hex_to_rgb01(h:str)->np.ndarray:
            h=h.lstrip('#')
            return np.array([int(h[i:i+2],16) for i in (0,2,4)], dtype=np.float32)/255.0
        PALETTE = {
            "floor": _hex_to_rgb01("#f4efe6"), "wall": _hex_to_rgb01("#1f2937"),
            "start": _hex_to_rgb01("#2563eb"), "goal": _hex_to_rgb01("#22c55e"),
            "white": _hex_to_rgb01("#ffffff"), "path": _hex_to_rgb01("#ef4444"),
        }
        def _rgb_img_to01(img: Image.Image) -> np.ndarray:
            if img.mode != "RGB": img = img.convert("RGB")
            return np.asarray(img).astype(np.float32)/255.0
        def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.linalg.norm(a - b, axis=-1)
        def _closest_label(rgb01: np.ndarray, labels=("floor","wall","start","goal")):
            ds = np.stack([_dist(rgb01, PALETTE[lab]) for lab in labels], axis=-1)
            return np.argmin(ds, axis=-1), ds.min(axis=-1)
        def _find_board_square(rgb01: np.ndarray, content_labels=("wall","start","goal","path"),
                            whiteish_labels=("white","floor"), row_frac_thresh: float = 0.003,
                            margin_pixels: int = 2):
            H, W, _ = rgb01.shape
            d_content = np.stack([_dist(rgb01, PALETTE[k]) for k in content_labels], axis=-1).min(axis=-1)
            d_white   = np.stack([_dist(rgb01, PALETTE[k]) for k in whiteish_labels], axis=-1).min(axis=-1)
            content_mask = (d_content + 0.01) < d_white
            def fr():
                for i in range(H):
                    if content_mask[i].mean() > row_frac_thresh: return i
                return 0
            def lr():
                for i in range(H-1, -1, -1):
                    if content_mask[i].mean() > row_frac_thresh: return i
                return H-1
            def fc():
                for j in range(W):
                    if content_mask[:,j].mean() > row_frac_thresh: return j
                return 0
            def lc():
                for j in range(W-1, -1, -1):
                    if content_mask[:,j].mean() > row_frac_thresh: return j
                return W-1
            top, bottom = fr(), lr()
            left, right = fc(), lc()
            top    = max(0, top - margin_pixels)
            left   = max(0, left - margin_pixels)
            bottom = min(H-1, bottom + margin_pixels)
            right  = min(W-1, right + margin_pixels)
            h = bottom - top + 1; w = right - left + 1
            side = max(h, w)
            cy = (top + bottom) // 2; cx = (left + right) // 2
            half = side // 2
            t = max(0, cy - half); b = min(W, t + side); t = b - side
            l = max(0, cx - half); r = min(H, l + side); l = r - side
            return t, l, b, r

        grid_h = kwargs.get("grid_h", 6); grid_w = kwargs.get("grid_w", 6)
        tolerance = kwargs.get("tolerance", 0.60); sg_factor = kwargs.get("sg_factor", 0.5)
        heuristic_min_start = kwargs.get("heuristic_min_start", 0.08)
        heuristic_min_goal  = kwargs.get("heuristic_min_goal", 0.25)
        
        img = Image.open(img_path); rgb = _rgb_img_to01(img)
        t,l,b,r = _find_board_square(rgb); board = rgb[t:b, l:r, :]
        H, W = board.shape[:2]; cell_h = H / grid_h; cell_w = W / grid_w
        label_names = ("floor","wall","start","goal")
        L_FLOOR, L_WALL, L_START, L_GOAL = 0,1,2,3
        s_thresh = tolerance * 0.25
        g_thresh = tolerance * 0.75
        grid_labels = []; start_pos=None; goal_pos=None
        frac_map = np.zeros((grid_h, grid_w, len(label_names)), dtype=np.float32)
        for gr in range(grid_h):
            row_syms = []
            y0 = int(round(gr * cell_h)); y1 = int(round((gr+1) * cell_h))
            for gc in range(grid_w):
                x0 = int(round(gc * cell_w)); x1 = int(round((gc+1) * cell_w))
                tile = board[y0:y1, x0:x1, :]
                if tile.size == 0: row_syms.append("?"); continue
                idx_map, _ = _closest_label(tile, labels=label_names)
                fracs = np.array([(idx_map == k).mean() for k in range(len(label_names))], dtype=np.float32)
                frac_map[gr,gc,:] = fracs
                f_floor, f_wall, f_start, f_goal = fracs[L_FLOOR],fracs[L_WALL],fracs[L_START],fracs[L_GOAL]
                if f_start >= s_thresh and f_goal >= g_thresh:
                    start_pos=(gr,gc); goal_pos=(gr,gc); sym="SG"
                elif f_start >= s_thresh:
                    start_pos=(gr,gc); sym="S"
                elif f_goal >= g_thresh:
                    goal_pos=(gr,gc); sym="G"
                else:
                    if f_wall >= tolerance: sym="#"
                    elif f_floor >= tolerance: sym=" "
                    else: sym="?"
                row_syms.append(sym)
            grid_labels.append(row_syms)
        if start_pos is None:
            best = np.unravel_index(np.argmax(frac_map[:,:,L_START]), (grid_h,grid_w))
            if frac_map[best[0],best[1],L_START] >= heuristic_min_start:
                start_pos = (int(best[0]), int(best[1])); grid_labels[start_pos[0]][start_pos[1]]="S"
        if goal_pos is None:
            best = np.unravel_index(np.argmax(frac_map[:,:,L_GOAL]), (grid_h,grid_w))
            if frac_map[best[0],best[1],L_GOAL] >= heuristic_min_goal:
                goal_pos = (int(best[0]), int(best[1]))
                grid_labels[goal_pos[0]][goal_pos[1]] = ("SG" if grid_labels[goal_pos[0]][goal_pos[1]]=="S" else "G")
        if start_pos is not None and start_pos == goal_pos:
            r0,c0 = start_pos; grid_labels[r0][c0] = "SG"
        ascii_rows = ["".join(row) for row in grid_labels]
        return {"ascii":"\n".join(ascii_rows),"grid":grid_labels,"start":start_pos,"goal":goal_pos}

    @staticmethod
    def parse_sliding_puzzle_image(img_path: str, **kwargs) -> Dict[str, Any]:
        grid_h = kwargs.get("grid_h", 3)
        grid_w = kwargs.get("grid_w", 3)
        num_categories = kwargs.get("num_categories", 9)
        tolerance = kwargs.get("tolerance", 0.80)
        
        def _hex_to_rgb01(h: str) -> np.ndarray:
            h = h.lstrip('#')
            return np.array([int(h[i:i+2],16) for i in (0,2,4)], dtype=np.float32)/255.0
        
        SET1_9 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]

        def _distinct_palette_k(k: int) -> List[np.ndarray]:
            if k <= 9: return [ _hex_to_rgb01(SET1_9[i % 9]) for i in range(k)]
            return [np.array(colorsys.hsv_to_rgb(i/float(k), 0.85, 0.95), dtype=np.float32) for i in range(k)]

        def _rgb_img_to01(img: Image.Image) -> np.ndarray:
            if img.mode != "RGB": img = img.convert("RGB")
            return np.asarray(img).astype(np.float32) / 255.0
        
        def _edist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.linalg.norm(a - b, axis=-1)

        def _find_square_board(rgb: np.ndarray, content_colors: List[np.ndarray], row_frac_thresh: float = 0.002, margin_px: int = 2) -> Tuple[int,int,int,int]:
            H,W,_ = rgb.shape
            d_white = _edist(rgb, np.array([1.0,1.0,1.0]))
            d_content = np.stack([_edist(rgb, c) for c in content_colors], axis=-1).min(axis=-1)
            content_mask = (d_content + 0.01) < d_white

            def fr(): return next((i for i in range(H) if content_mask[i].mean() > row_frac_thresh), 0)
            def lr(): return next((i for i in range(H-1,-1,-1) if content_mask[i].mean() > row_frac_thresh), H-1)
            def fc(): return next((j for j in range(W) if content_mask[:,j].mean() > row_frac_thresh), 0)
            def lc(): return next((j for j in range(W-1,-1,-1) if content_mask[:,j].mean() > row_frac_thresh), W-1)

            top, bottom = fr(), lr(); left, right = fc(), lc()
            top, left = max(0, top - margin_px), max(0, left - margin_px)
            bottom, right = min(H-1, bottom + margin_px), min(W-1, right + margin_px)

            h, w = bottom - top + 1, right - left + 1; side = max(h, w)
            cy, cx = (top + bottom)//2, (left + right)//2; half = side//2
            t = max(0, cy - half); b = min(H, t + side); t = b - side
            l = max(0, cx - half); r = min(W, l + side); l = r - side
            return t, l, b, r

        img = Image.open(img_path)
        rgb = _rgb_img_to01(img)
        palette = _distinct_palette_k(num_categories)
        t,l,b,r = _find_square_board(rgb, content_colors=palette)
        board = rgb[t:b, l:r, :]

        H,W,_ = board.shape
        cell_h, cell_w = H / grid_h, W / grid_w
        labels_grid: List[List[str]] = []

        for gr in range(grid_h):
            y0, y1 = int(round(gr*cell_h)), int(round((gr+1)*cell_h))
            row_syms: List[str] = []
            for gc in range(grid_w):
                x0, x1 = int(round(gc*cell_w)), int(round((gc+1)*cell_w))
                tile = board[y0:y1, x0:x1, :]
                if tile.size == 0: row_syms.append("?"); continue
                
                dstack = np.stack([_edist(tile, c) for c in palette], axis=-1)
                argmin = np.argmin(dstack, axis=-1)
                counts = np.array([(argmin == k).mean() for k in range(len(palette))], dtype=np.float32)
                k_star = int(counts.argmax())
                
                row_syms.append(str(k_star+1) if float(counts[k_star]) >= tolerance else "?")
            labels_grid.append(row_syms)

        ascii_str = "\n".join([" ".join(row) for row in labels_grid])
        return {"ascii": ascii_str, "grid": labels_grid}


class LocalTextLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    @torch.no_grad()
    def generate_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out_ids[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def generate_json(self, messages: List[Dict[str, str]], max_new_tokens: int = 1024) -> Dict[str, Any]:
        raw = self.generate_text(messages, max_new_tokens=max_new_tokens)
        #print("[DEBUG] Raw LM Output:", raw)
        json_str = UtilityHelpers.find_first_json_substring(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except Exception:
                pass
        return {"_raw": raw}


class LocalVL:
    def __init__(self, model_name: str, attn_implementation: Optional[str] = None):
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def generate_text(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> str:
        norm_msgs = []
        for m in messages:
            if m.get("role") != "user":
                norm_msgs.append(m)
                continue
            content = []
            for part in m.get("content", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    content.append({"type": "image", "image": url})
                elif part.get("type") == "image":
                    content.append(part)
                elif part.get("type") == "text":
                    content.append({"type": "text", "text": part["text"]})
            norm_msgs.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(norm_msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(norm_msgs)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text.strip()

    def generate_json(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> Dict[str, Any]:
        raw = self.generate_text(messages, max_new_tokens=max_new_tokens)
        # print("[DEBUG] Raw VL Output:", raw)
        json_str = UtilityHelpers.find_first_json_substring(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except Exception:
                pass
        return {"_raw": raw}


class GeometryEvaluator:
    """Evaluates geometry problems by judging generated auxiliary lines and reasoning text."""
    def __init__(self, config: Dict[str, Any], lm: LocalTextLM, vl: LocalVL):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['out_eval_dir']).mkdir(parents=True, exist_ok=True)
        self.results = []
        self.lm = lm
        self.vl = vl

    def _call_overlay_judge(self, base_img: Path, gt_overlay_img: Path, pred_img: Path, aux_en: str) -> Dict[str, Any]:
        user_parts = [
            {"type": "text", "text": self.config['overlay_user_tmpl'].format(aux_en=aux_en or "(none)")},
            {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(base_img)}},
            {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(gt_overlay_img)}},
            {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(pred_img)}},
        ]
        last_error = None
        for _ in range(self.config['max_retries']):
            try:
                data = self.vl.generate_json(
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": self.config['system_overlay']}]},
                        {"role": "user", "content": user_parts},
                    ],
                    max_new_tokens=512
                )
                return {
                    "overlay_ok": 1 if int(data.get("overlay_ok", 0)) == 1 else 0,
                    "overlay_reason": str(data.get("overlay_reason", "")).strip() or "No rationale.",
                    "raw_judgment": json.dumps(data, ensure_ascii=False)
                }
            except Exception as e:
                last_error = e
        return {"overlay_ok": 0, "overlay_reason": f"API Error: {last_error}", "raw_judgment": ""}

    def _call_text_judge(self, task_type: str, problem: str, gt_text: str, cand_text: str) -> Dict[str, Any]:
        user_text = self.config['text_user_tmpl'].format(
            task_type=task_type, problem=problem, gt_text=gt_text, cand_text=cand_text
        )
        last_error = None
        for _ in range(self.config['max_retries']):
            try:
                data = self.lm.generate_json(
                    messages=[
                        {"role": "system", "content": self.config['system_text']},
                        {"role": "user", "content": user_text},
                    ],
                    max_new_tokens=32767
                )
                return {
                    "reasoning_rigorous": 1 if int(data.get("reasoning_rigorous", 0)) == 1 else 0,
                    "conclusion_correct": 1 if int(data.get("conclusion_correct", 0)) == 1 else 0,
                    "text_ok": 1 if int(data.get("text_ok", 0)) == 1 else 0,
                    "text_reason": str(data.get("text_reason", "")).strip() or "No rationale.",
                    "raw_judgment": json.dumps(data, ensure_ascii=False)
                }
            except Exception as e:
                last_error = e
        return {"text_ok": 0, "text_reason": f"API Error: {last_error}", "raw_judgment": ""}

    def evaluate(self):
        data_path = Path(self.config['filtered_json'])
        data = self.utils.read_json(data_path)
        items = [
            (big_k, small_k, item)
            for big_k, group in data.items()
            for small_k, item in group.items()
        ]
        if self.config.get('max_retries'):
            pass
        if self.config.get('max_items'):
            items = items[:self.config['max_items']]
            
        
        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} geometry items.")
            items = items[:max_items]

        for big_k, small_k, item in tqdm(items, desc="Evaluating Geometry"):
            case_id = f"{self.utils.sanitize_filename(big_k)}__{self.utils.sanitize_filename(small_k)}"
            case_dir = Path(self.config['out_dir']) / case_id
            record = {"id": case_id, "type": item.get("type"), "status": "init"}

            if not case_dir.is_dir():
                record["status"] = "error_missing_case_dir"
                self.results.append(record)
                continue

            orig_img = (data_path.parent / item["original_image"]).resolve()
            aux_img = (data_path.parent / item["auxiliary_image"]).resolve()

            if not (orig_img.is_file() and aux_img.is_file()):
                record["status"] = "error_missing_gt_image"
                self.results.append(record)
                continue

            pred_imgs = sorted(case_dir.glob("model_image_*.*"))
            cand_text = (case_dir / "model_text.txt").read_text(encoding="utf-8") if (case_dir / "model_text.txt").exists() else ""

            if pred_imgs:
                overlay_eval = self._call_overlay_judge(orig_img, aux_img, pred_imgs[0], item.get("auxiliary_text_en", ""))
                record.update(overlay_eval)
            else:
                record["overlay_ok"] = 0
                record["overlay_reason"] = "No predicted image found."

            problem_text = item.get("problem_text_en") or item.get("problem_text", "")
            gt_text = item.get("solution_en") or item.get("solution", "")
            task_type = "CALCULATION" if (item.get("type") or "").lower().startswith("calc") else "PROVING"
            text_eval = self._call_text_judge(task_type, problem_text, gt_text, cand_text)

            record.update(text_eval)
            record["status"] = "ok"
            self.results.append(record)

    def summarize(self):
        if not self.results:
            print("No results to summarize for Geometry.")
            return
        summary = {
            "total_items": len(self.results),
            "scored_items": sum(1 for r in self.results if r.get("status") == "ok"),
            "ok_overlay": sum(r.get("overlay_ok", 0) for r in self.results),
            "ok_text": sum(r.get("text_ok", 0) for r in self.results),
        }
        summary["overlay_acc"] = summary["ok_overlay"] / summary["scored_items"] if summary["scored_items"] > 0 else 0
        summary["text_acc"] = summary["ok_text"] / summary["scored_items"] if summary["scored_items"] > 0 else 0

        out_eval_dir = Path(self.config['out_eval_dir'])
        self.utils.write_json(summary, out_eval_dir / "eval_summary.json")
        self.utils.write_json(self.results, out_eval_dir / "eval_details.json")

        print("\n=== Geometry Evaluation Summary ===")
        print(json.dumps(summary, indent=2))


class JigsawEvaluator:
    """Evaluates jigsaw puzzle solutions using DreamSim for image comparison and text parsing for choice correctness."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['eval_dir']).mkdir(parents=True, exist_ok=True)
        self.results = []

        if dreamsim_fn is None:
            raise ImportError("DreamSim is not installed. Please run 'pip install dreamsim'.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dreamsim_model, self.dreamsim_pre = dreamsim_fn(pretrained=True, device=self.device, cache_dir=self.config['dreamsim_cache'])

    def _resolve_dataset_path(self, raw_path: str) -> Path:
        """Resolve a possibly-relative path from metadata.json against dataset_dir.

        Metadata stores paths like './data/jigsaw_dataset_2x2ref/ex_00000/gt_completed_2x2.png'.
        Strip the leading './data/jigsaw_dataset_2x2ref/' (or similar) prefix and resolve
        against the configured dataset_dir so the path works in any environment.
        """
        p = Path(raw_path)
        if p.is_absolute():
            return p
        # Strip relative prefixes that mirror the repo layout
        rel = p.as_posix()
        dataset_dir = Path(self.config['dataset_dir'])
        for prefix in ("./data/jigsaw_dataset_2x2ref/", "data/jigsaw_dataset_2x2ref/"):
            if rel.startswith(prefix):
                return dataset_dir / rel[len(prefix):]
        # Fallback: try resolving against dataset_dir parent tree
        return dataset_dir / p.name if not p.exists() else p

    def _dreamsim_distance(self, img_a: Image.Image, img_b: Image.Image) -> float:
        img_a_rgb = img_a.convert("RGB")
        img_b_rgb = img_b.convert("RGB")
        tensor_a = self.dreamsim_pre(img_a_rgb).to(self.device)
        tensor_b = self.dreamsim_pre(img_b_rgb).to(self.device)
        with torch.no_grad():
            dist = self.dreamsim_model(tensor_a, tensor_b)
        return float(dist.item())

    def _parse_choice_from_text(self, text_path: Path) -> Tuple[Optional[int], str]:
        if not text_path.exists():
            return None, "missing_text"
        text = text_path.read_text(encoding="utf-8")
        match = re.search(r"<FINAL_ANSWER_JSON>\s*({.*?})\s*</FINAL_ANSWER_JSON>", text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None, "no_final_json_block"
        try:
            data = json.loads(match.group(1))
            choice = data.get("choice")
            if choice in (0, 1):
                return int(choice), "ok"
            return None, "invalid_choice_value"
        except json.JSONDecodeError:
            return None, "bad_json"

    def evaluate(self):
        meta = self.utils.read_json(Path(self.config['dataset_dir']) / "metadata.json")
        item_index = {item['id']: item for item in meta.get("items", [])}
        run_summary = self.utils.read_json(Path(self.config['out_root']) / "summary.json")
        run_ids = [item['id'] for item in run_summary.get("per_item", [])]
        items_to_eval = [item_index[i] for i in run_ids if i in item_index]
        
        
        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} jigsaw items.")
            items_to_eval = items_to_eval[:max_items]

        for item in tqdm(items_to_eval, desc="Evaluating Jigsaw"):
            item_id = item['id']
            case_dir = Path(self.config['out_root']) / item_id
            record = {"id": item_id, "label": item["label"]}

            choice, text_status = self._parse_choice_from_text(case_dir / "model_text.txt")
            record["text_status"] = text_status
            record["choice"] = choice
            record["text_correct"] = int(choice == item["label"]) if choice is not None else 0

            gt_ok_path = self._resolve_dataset_path(item["gt_completed_2x2_path"])
            gt_bad_path = self._resolve_dataset_path(item["gt_wrong_2x2_path"])
            gt_c0_path, gt_c1_path = (gt_ok_path, gt_bad_path) if item["label"] == 0 else (gt_bad_path, gt_ok_path)

            pred_imgs = sorted(glob.glob(str(case_dir / "model_image_*.*")))
            if len(pred_imgs) == 2:
                try:
                    im0 = Image.open(pred_imgs[0])
                    im1 = Image.open(pred_imgs[1])
                    gt0 = Image.open(gt_c0_path)
                    gt1 = Image.open(gt_c1_path)
                    d0 = self._dreamsim_distance(im0, gt0)
                    d1 = self._dreamsim_distance(im1, gt1)
                    record.update({"image_status": "ok", "d0": d0, "d1": d1, "d_mean": (d0 + d1) / 2.0})
                except Exception as e:
                    record.update({"image_status": f"read_error: {e}", "d_mean": self.config['penalty_distance']})
            else:
                record.update({"image_status": f"invalid_count_{len(pred_imgs)}", "d_mean": self.config['penalty_distance']})
            self.results.append(record)

    def summarize(self):
        if not self.results:
            print("No results to summarize for Jigsaw.")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(Path(self.config['eval_dir']) / "per_item.csv", index=False)

        valid_distances = df[df['image_status'] == 'ok']['d_mean'].tolist()
        penalized_distances = df['d_mean'].tolist()

        summary = {
            "counts": {
                "total_items": len(df),
                "image_ok": int((df['image_status'] == 'ok').sum()), # <-- CONVERTED TO INT
                "text_parsed": int((df['text_status'] == 'ok').sum())   # <-- CONVERTED TO INT
            },
            "metrics": {
                "mean_distance_penalized": float(np.mean(penalized_distances)) if penalized_distances else None,
                "image_score_penalized": float(1.0 - np.mean(penalized_distances)) if penalized_distances else None,
                "text_accuracy": float(df['text_correct'].mean()) if len(df) else 0.0
            }
        }

        self.utils.write_json(summary, Path(self.config['eval_dir']) / "summary.json")
        print("\n=== Jigsaw Evaluation Summary ===")
        print(json.dumps(summary, indent=2))


class ScienceEvaluator:
    """Evaluates science-based image editing tasks, judging both text reasoning and image correctness."""
    def __init__(self, config: Dict[str, Any], vl: LocalVL):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['eval_root']).mkdir(parents=True, exist_ok=True)
        self.results = []
        self.vl = vl

    def _evaluate_text(self, scene: str, condition: str, gt_text: str, pred_text: str) -> Dict[str, Any]:
        user_prompt = self.config['text_user_tmpl'].format(
            scene=(scene or '(none)'), 
            condition=(condition or '(none)'), 
            gt_text=gt_text, 
            pred_text=pred_text
        )
        last_error = None
        for _ in range(self.config['max_retries']):
            try:
                data = self.vl.generate_json(
                    messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.config['text_system']}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}]
                    }
                    ],
                    max_new_tokens=1024
                )
                return {
                    "reasoning_correct": 1 if data.get("reasoning_correct") else 0,
                    "result_correct": 1 if data.get("result_correct") else 0,
                }
            except Exception as e:
                last_error = e
        return {"reasoning_correct": 0, "result_correct": 0, "error": str(last_error)}

    def _evaluate_image(self, pred_image_path: Path, initial_image_path: Path, condition: str, gt_text: str) -> Dict[str, Any]:
        user_prompt = self.config['image_user_tmpl'].format(condition=condition, gt_text=gt_text)
        last_error = None
        for _ in range(self.config['max_retries']):
            try:
                data = self.vl.generate_json(
                    messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.config['image_system']}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(initial_image_path)}},
                            {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(pred_image_path)}},
                        ],
                    }],
                    max_new_tokens=512
                )
                return {"image_correct": 1 if data.get("image_correct") else 0}
            except Exception as e:
                last_error = e
        return {"image_correct": 0, "error": str(last_error)}

    def evaluate(self):
        data_json_path = Path(self.config['data_json'])
        cases = self.utils.read_json(data_json_path)
        flat_cases = [sample for block in cases for sample in block.get("samples", [])]
        if self.config.get('num_samples'):
            flat_cases = flat_cases[:self.config['num_samples']]

        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            #print(f"[INFO] Limiting to the first {max_items} science items.")
            flat_cases = flat_cases[:max_items]

        # Image paths in JSON are relative (e.g. "data/science/imgs/xxx.png")
        # and need to be resolved against the dataset root.
        # data_json is at {dataset_root}/data/science/dim_all.json
        data_anchor = data_json_path.resolve().parent.parent.parent

        for idx, case in enumerate(tqdm(flat_cases, desc="Evaluating Science"), 1):
            case_dir = Path(self.config['run_root']) / f"case_{idx:02d}"
            record = {"case_id": f"case_{idx:02d}"}

            initial_image = (case.get("input_image_file_path_list") or [None])[0]
            condition = case.get("input_prompt")
            gt_text = case.get("output_prompt")
            pred_text_path = case_dir / "model_text.txt"
            pred_imgs = sorted(case_dir.glob("model_image_*.*"))

            if pred_text_path.exists() and gt_text:
                pred_text = pred_text_path.read_text("utf-8")
                record["text_eval"] = self._evaluate_text("", condition, gt_text, pred_text)
            else:
                record["text_eval"] = {"reasoning_correct": 0, "result_correct": 0}

            if pred_imgs and initial_image:
                img_path = Path(initial_image)
                if not img_path.is_absolute():
                    img_path = data_anchor / img_path
                record["image_eval"] = self._evaluate_image(pred_imgs[0], img_path, condition, gt_text)
            else:
                record["image_eval"] = {"image_correct": 0}

            self.results.append(record)

    def summarize(self):
        if not self.results:
            print("No results to summarize for Science.")
            return

        total = len(self.results)
        summary = {
            "total_cases": total,
            "text_reasoning_correct": sum(r["text_eval"].get("reasoning_correct", 0) for r in self.results),
            "text_result_correct": sum(r["text_eval"].get("result_correct", 0) for r in self.results),
            "image_correct": sum(r["image_eval"].get("image_correct", 0) for r in self.results),
        }
        summary["text_reasoning_acc"] = summary["text_reasoning_correct"] / total if total > 0 else 0
        summary["text_result_acc"] = summary["text_result_correct"] / total if total > 0 else 0
        summary["image_acc"] = summary["image_correct"] / total if total > 0 else 0

        eval_root = Path(self.config['eval_root'])
        self.utils.write_json(summary, eval_root / "eval_summary.json")
        self.utils.write_json(self.results, eval_root / "eval_details.json")

        print("\n=== Science Evaluation Summary ===")
        print(json.dumps(summary, indent=2))


class CodeSVGEvaluator:
    """Evaluates SVG code generation by scoring generated images and checking semantic descriptions."""
    def __init__(self, config: Dict[str, Any], vl: LocalVL):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['eval_out_dir']).mkdir(parents=True, exist_ok=True)
        self.results = []
        self.vl = vl
        if cairosvg is None:
            print("Warning: 'cairosvg' not found. SVG to PNG conversion will not be available.")

    def _rasterize_svg(self, svg_path: Path, case_dir: Path) -> Optional[Path]:
        if not cairosvg or not svg_path.exists():
            return None
        try:
            png_bytes = cairosvg.svg2png(bytestring=svg_path.read_text(encoding="utf-8").encode("utf-8"))
            out_path = case_dir / "_ref_from_svg.png"
            out_path.write_bytes(png_bytes)
            return out_path
        except Exception:
            return None

    def _get_reference_image(self, case_dir: Path) -> Optional[Path]:
        result_json = self.utils.read_json(case_dir / "result.json")
        dataset_dir = Path(self.config['dataset_dir'])
        if result_json.get("png"):
            gt_path = dataset_dir / result_json["png"]
            if gt_path.exists():
                return gt_path
        if result_json.get("svg"):
            return self._rasterize_svg(dataset_dir / result_json["svg"], case_dir)
        return None

    def _eval_image(self, candidate_img: Path, ref_img: Path) -> Dict[str, Any]:
        try:
            data = self.vl.generate_json(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config['image_prompt']},
                        {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(ref_img)}},
                        {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(candidate_img)}},
                    ]
                }],
                max_new_tokens=512
            )
            data["image_match_binary"] = 1 if data.get("verdict") == "match" else 0
            data.setdefault("shape_color_accuracy", float(data.get("shape_color_accuracy", 0)))
            data.setdefault("position_accuracy", float(data.get("position_accuracy", 0)))
            return data
        except Exception as e:
            return {"verdict": "mismatch", "explanation": f"API Error: {e}", "image_match_binary": 0}

    def _eval_text(self, render_summary: str, ref_img: Path) -> Dict[str, Any]:
        if not render_summary:
            return {"text_semantic_match": 0, "explanation": "No RENDER_SUMMARY found."}
        try:
            data = self.vl.generate_json(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.config['text_prompt']},
                        {"type": "image_url", "image_url": {"url": self.utils.image_to_b64_data_url(ref_img)}},
                        {"type": "text", "text": f"Candidate description:\n{render_summary}"}
                    ]
                }],
                max_new_tokens=512
            )
            data["text_semantic_match"] = 1 if data.get("text_semantic_match") else 0
            return data
        except Exception as e:
            return {"text_semantic_match": 0, "explanation": f"API Error: {e}"}

    def evaluate(self):
        case_dirs = sorted(Path(self.config['sample_root']).glob("case_*_*"))
            
        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} SVG items.")
            case_dirs = case_dirs[:max_items]

        for case_dir in tqdm(case_dirs, desc="Evaluating SVG"):
            ref_img = self._get_reference_image(case_dir)
            if not ref_img:
                self.results.append({"case_dir": str(case_dir), "error": "no_reference_image"})
                continue

            text_path = case_dir / "model_text.txt"
            full_text = text_path.read_text('utf-8') if text_path.exists() else ''
            render_summary_match = re.search(r"<RENDER_SUMMARY>\s*(.*?)\s*</RENDER_SUMMARY>", full_text, re.I | re.S)
            render_summary = render_summary_match.group(1).strip() if render_summary_match else ""

            image_evals = []
            for img_path in sorted(case_dir.glob("model_image_*.*")):
                image_evals.append(self._eval_image(img_path, ref_img))

            text_eval = self._eval_text(render_summary, ref_img)

            self.results.append({
                "case_dir": str(case_dir),
                "image_evals": image_evals,
                "text_eval": text_eval,
            })

    def summarize(self):
        if not self.results:
            print("No results to summarize for SVG.")
            return

        image_binary_hits = []
        best_shape_scores, best_pos_scores = [], []

        for item in self.results:
            evals = item.get("image_evals", [])
            if not evals:
                image_binary_hits.append(0)
                best_shape_scores.append(0)
                best_pos_scores.append(0)
                continue

            any_match = any(e.get("image_match_binary") == 1 for e in evals)
            image_binary_hits.append(1 if any_match else 0)
            best_eval = max(
                evals,
                key=lambda e: float(e.get("shape_color_accuracy", 0)) + float(e.get("position_accuracy", 0)),
                default={}
            )
            best_shape_scores.append(float(best_eval.get("shape_color_accuracy", 0)))
            best_pos_scores.append(float(best_eval.get("position_accuracy", 0)))

        text_hits = [int(bool(item.get("text_eval", {}).get("text_semantic_match", 0))) for item in self.results]

        summary = {
            "total_cases": len(self.results),
            "image_match_binary_rate": float(np.mean(image_binary_hits)) if image_binary_hits else 0.0,
            "avg_shape_color_accuracy": float(np.mean(best_shape_scores)) if best_shape_scores else 0.0,
            "avg_position_accuracy": float(np.mean(best_pos_scores)) if best_pos_scores else 0.0,
            "text_semantic_match_rate": float(np.mean(text_hits)) if text_hits else 0.0,
        }

        self.utils.write_json(summary, Path(self.config['eval_out_dir']) / "eval_summary.json")
        self.utils.write_json(self.results, Path(self.config['eval_out_dir']) / "eval_details.json")

        print("\n=== SVG Generation Evaluation Summary ===")
        print(json.dumps(summary, indent=2))




class MazeEvaluator:

    _DIR_VECT = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['out_root']).mkdir(parents=True, exist_ok=True)
        self.results = []

    def _grids_exact_equal(self, a: List[List[str]], b: List[List[str]]) -> bool:
        if a is None or b is None: return False
        if len(a) != len(b): return False
        if len(a) == 0: return len(b) == 0
        if len(a[0]) != len(b[0]): return False
        
        return all(r1 == r2 for r1, r2 in zip(a, b))

    def evaluate(self):
        case_dirs = sorted(Path(self.config['run_root']).glob("case_*"))
        
        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} maze items.")
            case_dirs = case_dirs[:max_items]
            
        for case_dir in tqdm(case_dirs, desc="Evaluating Maze"):
            rec = self.utils.read_json(case_dir / "result.json")
            step0_path = rec.get("step0")
            if not step0_path:
                continue

            gt_json_path = self._derive_gt_json_path(step0_path)
            gt_moves = [str(m).lower() for m in self.utils.read_json(gt_json_path).get("steps_long", [])] if gt_json_path else []

            step0_parsed = self.utils.parse_maze_image(step0_path, **self.config.get('parser_params', {}))
            if step0_parsed.get("grid") is None or step0_parsed.get("start") is None:
                print(f"[WARN] Failed to parse step0 image for {case_dir.name}. Skipping.")
                continue
            
            walls = [[c == '#' for c in row] for row in step0_parsed["grid"]]
            start_pos = step0_parsed.get("start")
            goal_pos = step0_parsed.get("goal")

            exp_grids = self._simulate_expected_grids(start_pos, goal_pos, walls, gt_moves)

            cand_texts = (case_dir / "model_text.txt").read_text('utf-8').split("\n\n\n-----\n\n\n") if (case_dir / "model_text.txt").exists() else []
            cand_image_dirs = sorted(case_dir.glob("cand_*"))

            num_cands = max(len(cand_texts), len(cand_image_dirs) if self.config['evaluate_images'] else 0)

            for i in range(num_cands):
                record = {"case_id": case_dir.name, "cand_idx": i + 1}
                
                raw_text = cand_texts[i] if i < len(cand_texts) else ""
                pred_moves = self._parse_moves(raw_text)
                record.update(self._evaluate_text_moves(pred_moves, gt_moves))

                if self.config['evaluate_images']:
                    img_list = sorted(cand_image_dirs[i].glob("*.*")) if i < len(cand_image_dirs) else []
                    pred_grids = [self.utils.parse_maze_image(p, **self.config.get('parser_params', {})).get("grid") for p in img_list]
                    record.update(self._evaluate_image_grids(pred_grids, exp_grids, walls))
                else:
                    record.update({"img_exact": 0, "img_frame_acc": 0, "parse_frame_success": 0.0, "parse_all_ok": 0})

                self.results.append(record)

    def _derive_gt_json_path(self, step0_path_str: str) -> Optional[Path]:
        p = Path(step0_path_str)
        m = re.match(r"^(maze_\d+x\d+?)_(\d{5})_steps$", p.parent.name)
        if m:
            path = p.parent.parent / f"{m.group(1)}_steps_{m.group(2)}.json"
            return path if path.exists() else None
        return None

    def _simulate_expected_grids(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int], walls: List[List[bool]], moves: List[str]) -> List[List[List[str]]]:
        if not start_pos or not walls:
            return []
            
        positions = self._simulate_positions_from_moves(start_pos, walls, moves)
        return [self._make_grid_from_state(walls, goal_pos, pos) for pos in positions]

    def _simulate_positions_from_moves(self, start_rc: Tuple[int, int], walls: List[List[bool]], moves: List[str]) -> List[Tuple[int, int]]:

        R, C = len(walls), len(walls[0])
        r, c = start_rc
        trajectory = []
        for mv in moves:
            dr, dc = self._DIR_VECT.get(mv, (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not walls[nr][nc]:
                r, c = nr, nc
            trajectory.append((r, c))
        return trajectory

    def _make_grid_from_state(self, walls: List[List[bool]], goal_rc: Optional[Tuple[int, int]], cur_rc: Optional[Tuple[int, int]]) -> List[List[str]]:

        R = len(walls); C = len(walls[0]) if R > 0 else 0
        grid = [[' ' for _ in range(C)] for _ in range(R)]
        for r in range(R):
            for c in range(C):
                if walls[r][c]:
                    grid[r][c] = '#'
        
        if goal_rc:
            gr, gc = goal_rc
            grid[gr][gc] = 'G'
            
        if cur_rc:
            r, c = cur_rc
            if cur_rc == goal_rc:
                grid[r][c] = 'SG'
            else:
                grid[r][c] = 'S'
                
        return grid

    def _parse_moves(self, text: str) -> List[str]:
        """
        Parse the last <ANSWER_JSON> block from the model text.
        Handles both flat lists ["up","down"] and nested lists [["up","down"]].
        """
        matches = list(re.finditer(r"<ANSWER_JSON>\s*(\[.*?\])\s*</ANSWER_JSON>", text, re.IGNORECASE | re.DOTALL))
        if not matches:
            return []

        last_match = matches[-1].group(1)
        try:
            arr = json.loads(last_match)
            # Flatten one level if the model wrapped moves in an extra list
            if arr and isinstance(arr[0], list):
                arr = arr[0]
            return [str(x).strip().lower() for x in arr]
        except json.JSONDecodeError:
            return []

    def _evaluate_text_moves(self, pred: List[str], gt: List[str]) -> Dict[str, float]:
        """
        评估文本移动的准确性。
        """
        if not gt:
            return {"text_exact": 0, "text_frame_acc": 0.0}
        
        match_count = sum(1 for p, g in zip(pred, gt) if p == g)
        return {
            "text_exact": 1 if pred == gt else 0,
            "text_frame_acc": match_count / len(gt),
        }

    def _evaluate_image_grids(self, pred: List[List[List[str]]], gt: List[List[List[str]]], walls: List[List[bool]]) -> Dict[str, float]:
        """
        （已更新）评估图像网格的准确性和解析成功率。
        """
        gt_len = len(gt)
        pred_len = len(pred)

        if gt_len == 0:
            return {"img_exact": 0, "img_frame_acc": 0.0, "parse_frame_success": 0.0, "parse_all_ok": 0}

        match_count = sum(1 for i in range(gt_len) if i < pred_len and self._grids_exact_equal(pred[i], gt[i]))
        img_frame_acc = match_count / gt_len
        img_exact = 1 if pred_len == gt_len and match_count == gt_len else 0
        
        if pred_len == 0:
            parse_frame_success = 0.0
            parse_all_ok = 0
        else:
            parse_ok_count = sum(1 for p in pred if self._is_parse_ok(p, walls))
            parse_frame_success = parse_ok_count / pred_len
            parse_all_ok = 1 if parse_ok_count == pred_len else 0

        return {
            "img_exact": img_exact,
            "img_frame_acc": img_frame_acc,
            "parse_frame_success": parse_frame_success,
            "parse_all_ok": parse_all_ok,
        }

    def _is_parse_ok(self, grid: Optional[List[List[str]]], walls: List[List[bool]]) -> bool:

        if grid is None: return False
        

        if len(grid) != len(walls) or (len(grid) > 0 and len(grid[0]) != len(walls[0])):
            return False

        R, C = len(walls), len(walls[0])
        cnt_S = cnt_G = cnt_SG = 0
        
        for r in range(R):
            for c in range(C):
                cell = grid[r][c]
                if cell == '?': return False
                if (cell == '#') != walls[r][c]: return False
                if cell == 'S': cnt_S += 1
                elif cell == 'G': cnt_G += 1
                elif cell == 'SG': cnt_SG += 1
        
        has_valid_sg = (cnt_SG == 1 and cnt_S == 0 and cnt_G == 0) or \
                       (cnt_SG == 0 and cnt_S == 1 and cnt_G == 1)
        
        return has_valid_sg

    def summarize(self):
        if not self.results:
            print("No results to summarize for Maze.")
            return

        df = pd.DataFrame(self.results)
        summary = {
            "total_candidates": len(df),
            "text_accuracy_exact": float(df['text_exact'].mean()),
            "text_accuracy_frame_macro": float(df['text_frame_acc'].mean()),
        }
        
        if self.config['evaluate_images']:
            summary.update({
                "img_accuracy_exact": float(df['img_exact'].mean()),
                "img_accuracy_frame_macro": float(df['img_frame_acc'].mean()),
                "parse_accuracy_all_ok": float(df['parse_all_ok'].mean()),
                "parse_accuracy_frame_macro": float(df['parse_frame_success'].mean()),
            })

        self.utils.write_json(summary, Path(self.config['out_root']) / "summary.json")
        df.to_csv(Path(self.config['out_root']) / "manifest_eval.csv", index=False)

        print("\n=== Maze Evaluation Summary ===")
        print(json.dumps(summary, indent=2))


class SlidingPuzzleEvaluator(MazeEvaluator):
    """Evaluates sliding puzzle tasks, with its own summary logic."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.utils = UtilityHelpers()
        Path(self.config['out_root']).mkdir(parents=True, exist_ok=True)
        self.results = []

    def evaluate(self):
        """Runs the complete evaluation for the Sliding Puzzle dataset."""
        case_dirs = sorted(Path(self.config['run_root']).glob("case_*"))
        
        max_items = self.config.get('max_items_per_task')
        if max_items is not None and max_items > 0:
            print(f"[INFO] Limiting to the first {max_items} sliding puzzle items.")
            case_dirs = case_dirs[:max_items]
            
            
        for case_dir in tqdm(case_dirs, desc="Evaluating Sliding Puzzle"):
            rec = self.utils.read_json(case_dir / "result.json")
            step0_path = rec.get("init_png")
            if not step0_path:
                continue

            gt_json_path, gt_img_paths = self._derive_gt_files(step0_path)
            gt_moves = self.utils.read_json(gt_json_path).get("steps_words", []) if gt_json_path else []
            
            gt_ascii_seq = []
            if self.config['evaluate_images'] and gt_img_paths:
                gt_ascii_seq = [self.utils.parse_sliding_puzzle_image(p, **self.config.get('parser_params', {}))['ascii'] for p in gt_img_paths]

            cand_texts_path = case_dir / "model_text.txt"
            cand_texts = cand_texts_path.read_text('utf-8').split("\n\n\n-----\n\n\n") if cand_texts_path.exists() else []
            cand_image_dirs = sorted(case_dir.glob("cand_*"))

            for i in range(max(len(cand_texts), len(cand_image_dirs))):
                record = {"case_id": case_dir.name, "cand_idx": i + 1}
                pred_moves = self._parse_moves(cand_texts[i]) if i < len(cand_texts) else []
                record.update(self._evaluate_text_moves(pred_moves, gt_moves))
                
                if self.config['evaluate_images']:
                    img_list = sorted(cand_image_dirs[i].glob("*.*")) if i < len(cand_image_dirs) else []
                    pred_ascii_seq = [self.utils.parse_sliding_puzzle_image(p, **self.config.get('parser_params', {}))['ascii'] for p in img_list]
                    record.update(self._evaluate_image_ascii(pred_ascii_seq, gt_ascii_seq))
                
                
                self.results.append(record)

    def _derive_gt_files(self, step0_path_str: str) -> Tuple[Optional[Path], Optional[List[Path]]]:
        """Derives paths for ground truth JSON and step images."""
        p = Path(step0_path_str)
        m = re.match(r"^(.*?_3x3)_(\d{5})_steps$", p.parent.name)
        if m:
            json_path = p.parent.parent / f"{m.group(1)}_steps_words_{m.group(2)}.json"
            img_paths = sorted([f for f in p.parent.glob("*.png") if "0000" not in f.name])
            return json_path if json_path.exists() else None, img_paths
        return None, None

    def _evaluate_image_ascii(self, pred: list, gt: list) -> dict:
        """Evaluates image sequences based on their ASCII representations."""
        if not gt:
            return {"img_exact": 1 if not pred else 0, "img_frame_acc": 1.0 if not pred else 0.0}
        
        match_count = sum(1 for p, g in zip(pred, gt) if p == g)
        return {
            "img_exact": 1 if pred == gt else 0,
            "img_frame_acc": match_count / len(gt),
        }

    def summarize(self):
        """
        A dedicated summary method for the Sliding Puzzle task.
        This avoids the KeyError by not calling the parent's summarize method.
        """
        if not self.results:
            print("No results to summarize for Sliding Puzzle.")
            return

        df = pd.DataFrame(self.results)
        summary = {
            "total_candidates": len(df),
            "text_accuracy_exact": float(df['text_exact'].mean()) if len(df) else 0.0,
            "text_accuracy_frame_macro": float(df['text_frame_acc'].mean()) if len(df) else 0.0,
        }
        
        if self.config['evaluate_images'] and 'img_exact' in df.columns:
            summary.update({
                "img_accuracy_exact": float(df['img_exact'].mean()),
                "img_accuracy_frame_macro": float(df['img_frame_acc'].mean()),
            })

        self.utils.write_json(summary, Path(self.config['out_root']) / "summary.json")
        df.to_csv(Path(self.config['out_root']) / "manifest_eval.csv", index=False)

        print("\n=== Sliding Puzzle Evaluation Summary ===")
        print(json.dumps(summary, indent=2))


@dataclass
class Configurations:
    """Centralized configuration for all evaluators."""
    model_name: str = "gpt"
    base_path: str = "path-to-UMMMU-home"

    qwen3_model_name: str = "Qwen/Qwen3-32B"
    qwen2_5_vl_model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    vl_attn_impl: Optional[str] = None 
    
    max_items_per_task: Optional[int] = None
    dreamsim_cache: str = "/model_cache/dreamsim"

    def __post_init__(self):
        self.outputs_path = f"{self.base_path}/outputs/{self.model_name}"
        self.eval_path = f"{self.base_path}/eval/{self.model_name}"

        self.geometry = {
            "filtered_json": f"{self.base_path}/data/math_data/filtered.json",
            "out_dir": f"{self.outputs_path}/math",
            "out_eval_dir": f"{self.eval_path}/math",
            "max_retries": 2,
            "max_items_per_task": self.max_items_per_task,
            "system_overlay": (
                "You are a strict yet fair judge for geometry diagram overlays.\n"
                "Your job: compare images to decide if the candidate overlay correctly draws the REQUIRED auxiliary lines.\n"
                "Input images:\n"
                "  (1) base: original (no overlays), the clean figure.\n"
                "  (2) gt  : ground-truth overlay (what SHOULD be drawn).\n"
                "  (3) pred: ONLY ONE candidate overlay to judge.\n"
                "Also given the English text listing auxiliary lines to draw.\n\n"
                "Rules:\n"
                "  - Focus on whether the REQUIRED lines exist (correct endpoints/relations) in the candidate.\n"
                "  - Ignore extra small ticks/labels/harmless marks if they do not distort the meaning.\n"
                "  - Tolerate minor pixel/position noise if geometry intent is clear (parallel/perpendicular/through-point).\n"
                "  - ALL specified lines must be present; otherwise mark 0.\n"
                '  - Output MUST be a compact JSON: {"overlay_ok":0|1,"overlay_reason":"<short>"}\n'
            ),
            "system_text": (
                "You are a rigorous grader for geometry reasoning.\n"
                "Given: problem statement (text), ground-truth solution text (reference), and a candidate solution text.\n"
                "Decide two things: (i) is the reasoning rigorous (no major logical gaps or false claims), "
                "(ii) is the final conclusion correct.\n"
                "For CALCULATION problems: conclusion correctness means the final NUMERIC result matches the ground-truth "
                "(ignore formatting/units; radicals/π are okay if numerically equivalent), even if choices differ.\n"
                "For PROVING problems: conclusion correctness means the claim is indeed established (may differ in steps but must be valid).\n"
                "Only if (rigorous AND correct) → text_ok=1, else 0.\n"
                'Output MUST be a compact JSON: {"reasoning_rigorous":0|1,"conclusion_correct":0|1,'
                '"calc_numeric_match":0|1|null,"text_ok":0|1,"text_reason":"<short>"}\n'
            ),
            "overlay_user_tmpl": """
Task: Check if the candidate overlay draws the REQUIRED auxiliary lines.

Auxiliary lines to draw (English; draw ALL): 
{aux_en}

Important:
- Evaluate ONLY ONE candidate overlay image (pred_1).
- Allow extra harmless marks/letters if they don't distort meaning.
- Focus on presence/placement/direction of REQUIRED lines.

Images in order:
- base (original, no overlays)
- gt (ground-truth overlay)
- pred_1 (the ONLY candidate to judge)

Return strictly-formatted JSON only. Do Not Use any markers like ```json ...  ```
""",
"text_user_tmpl": """Task: Grade the candidate solution text for rigor and correctness.

Type: {task_type}
Problem (may have EN/CN):
{problem}

Ground-truth solution text (reference, not necessarily the only path):
{gt_text}

Candidate solution text (to grade):
{cand_text}

Rules:
- Only accept text_ok=1 if (reasoning_rigorous=1 AND conclusion_correct=1).
- For calculation: calc_numeric_match=1 if numeric answer equals reference (choices ignored).
- Return strictly-formatted JSON only. Do Not Use any markers like ```json ...  ```
            """
        }

        self.jigsaw = {
            "dataset_dir": f"{self.base_path}/data/jigsaw_dataset_2x2ref",
            "out_root": f"{self.outputs_path}/jigsaw",
            "eval_dir": f"{self.eval_path}/jigsaw",
            "dreamsim_cache": self.dreamsim_cache,
            "penalty_distance": 1.0,
            "max_items_per_task": self.max_items_per_task,
        }

        self.science = {
            "data_json": f"{self.base_path}/data/science/dim_all.json",
            "run_root": f"{self.outputs_path}/science",
            "eval_root": f"{self.eval_path}/science",
            "model_name": "qwen2_5_vl_local",
            "max_retries": 3,
            "max_items_per_task": self.max_items_per_task,
            "num_samples": None,
            "text_system": (
                "You are a strict but science-aware evaluator.\n"
                'Output ONLY a compact JSON: {"reasoning_correct":0|1, "result_correct":0|1}.\n'
                "- reasoning_correct = 1 if the explanation identifies and applies appropriate causal/world mechanisms "
                "linking the condition to the final state (no incorrect laws). Otherwise 0.\n"
                "- result_correct = 1 if the claimed final state is reasonable/plausible under the scene and condition, "
                "consistent with real-world science/commonsense. Prefer agreement with the provided GT text, but if it "
                "differs from GT, still set 1 when it remains scientifically sound and does not contradict the condition/scene. "
                "Otherwise 0.\n"
                "No meta talk."
            ),
            "text_user_tmpl": """
Decide two binary labels for the candidate explanation:
1) reasoning_correct: 1 if it identifies & applies appropriate causal/world knowledge connecting the condition to the final state; 0 otherwise.
2) result_correct: 1 if the final state described is reasonable/plausible under the scene and condition and consistent with real-world science/commonsense.
- Use the Ground Truth (GT) text as a preferred reference.
- If the candidate differs from GT, still mark result_correct=1 when it is scientifically sound and not contradictory to the scene/condition.
- Penalize wrong mechanisms and contradictions.

Scene: {scene}
Condition: {condition}
Ground truth final state (text, reference):
{gt_text}
Candidate explanation (text to evaluate):
{pred_text}
Return ONLY JSON.
""",
            "image_system": (
                "You are a strict visual editor-evaluator.\n"
                "You will be given: (A) an INITIAL image, (B) a TARGET text that describes the desired final state, "
                "and (C) a CANDIDATE image supposedly obtained by editing the initial image to satisfy the target.\n"
                "Judge with two criteria only:\n"
                "1) Background consistency: the candidate should preserve the initial scene's persistent layout/identity "
                "(e.g., camera viewpoint, major objects, textures) unless the target requires changes.\n"
                "2) Semantic edit correctness: the candidate should realize the target text's change(s) without contradictions.\n"
                'Ignore style or minor rendering differences. Output ONLY a compact JSON: {"image_correct":0|1}.'
            ),
            "image_user_tmpl": """
Condition (context): {condition}
TARGET text (final state to achieve):
{gt_text}
"""
        }

        self.code_svg = {
            "dataset_dir": f"{self.base_path}/data/svg",
            "sample_root": f"{self.outputs_path}/code",
            "eval_out_dir": f"{self.eval_path}/code",
            "model_name": "qwen2_5_vl_local",
            "max_cases": None,
            "max_items_per_task": self.max_items_per_task,
            "image_prompt": """\
You are a precise visual evaluator. Compare the **Reference (expected)** drawing against the **Candidate** drawing. The reference originates from SVG code that may include circles, polygons (3–6 sides), lines, and curves, sometimes generated with loops and at three difficulty tiers:

* Tier A: single shape
* Tier B: multiple shapes (no overlap)
* Tier C: multiple shapes (overlap allowed; layering matters)

Your job is to judge two abilities:

1. Inferring unit sizes and colors implied by the code/spec.
2. Planning absolute and relative positions in the layout.

### Color Palette (qualitative matching)

Colors appear only from this set (fills/strokes):
PALETTE = \['#E74C3C','#3498DB','#2ECC71','#F1C40F','#9B59B6','#34495E','#1ABC9C','#E67E22']
Treat color matching *qualitatively*: map the Candidate’s visible color to the **nearest palette color by appearance** (light/dark variations OK). A palette-neighbor match counts as correct; off-palette hues or clearly wrong family count as errors.

### What to check (high-level)

* **Shapes & counts (strict):** shape type and **polygon side count must match exactly** (triangle/4/5/6-gon). Circles vs polygons must not be confused. Lines/curves should preserve topology (straight vs curved).
* **Colors (qualitative):** per object, check if fill/stroke maps to the expected palette member (tint/shade OK).
* **Positions (qualitative):** judge relative layout (left/center/right; top/middle/bottom), ordering, spacing relations (e.g., equal spacing), alignments, rotation/angles, scale **relative to canvas**. Use human-style categories, e.g., “three circles evenly spaced horizontally; overall group center-left”.
* **Overlaps & z-order (Tier C):** verify intended overlap patterns and layer order. Partial occlusion is acceptable if consistent with the reference.
* **Tolerances:** minor antialiasing/compression and ±1 px stroke differences are negligible. Ignore tiny noise <1% canvas area. Focus on whether relationships and categories remain correct.
* **Loops/patterns:** if the SVG implies repeated elements (via loops), verify **count**, **regularity (rhythm/spacing)**, and **pattern alignment**.

### Scoring (integers only)

Return **two scores in \[0–5]**, where:

* 0 = complete mismatch
* 1 = very poor (vast majority incorrect)
* 2 = poor (many major issues)
* 3 = fair/partial (mixed; roughly half right)
* 4 = good (most aspects correct; small issues)
* 5 = perfect/near-perfect

**Shape & Color Accuracy (0–5)** — weight shape/type/side-count highest; colors are qualitative to palette.

* **5**: all types & side counts exact; all intended objects present; colors map to correct palette items; only negligible rendering differences.
* **4**: one or two *minor* issues (e.g., small radius/size deviation, slight stroke variance, or 1 palette-neighbor ambiguity).
* **3**: multiple minor issues or one *major* (e.g., one missing/extra object *or* one wrong side count); color family mostly right.
* **2**: several majors (multiple wrong types/counts or many color families wrong).
* **1**: only a few elements resemble the reference.
* **0**: entirely different content.

If unexpected shapes or text symbols appear, the Shape&Color Accuracy score must also be deducted.

**Position Accuracy (0–5)** — qualitative categories for placement and relationships.

* **5**: all relative orders, alignments, equal-spacing patterns, global placement (e.g., “center-left”), rotations, and overlaps match.
* **4**: small drifts that keep the same categories (still left-to-right order; spacing looks equal by eye; alignment near-exact).
* **3**: noticeable deviations that break some relationships (e.g., spacing no longer equal, slight order change in a subgroup), but overall layout still recognizable.
* **2**: many relationships broken (order flips, obvious misalignment, wrong global region); overlap patterns largely wrong.
* **1**: only a few coarse relations match.
* **0**: placement bears no resemblance.



### Verdict

* **"match"** if both scores ≥ 4
* **"partial"** if both ≥ 2 **or** (one ≥ 4 and the other ≥ 1)
* **"mismatch"** otherwise

### Output format

Return a **strict JSON object** with these keys only:

* `"shape_color_accuracy"`: integer in \[0–5]
* `"position_accuracy"`: integer in \[0–5]
* `"verdict"`: one of \["match","partial","mismatch"]
* `"explanation"`: ≤50 words, concise, no extra formatting

**Output strictly JSON without markdown or commentary.**
""",
            "text_prompt": """\
You are a strict semantic checker. Decide if the candidate description **semantically matches** the **Reference** drawing (binary).

Criteria for a match (all must hold):
- Objects: types and counts are correct (e.g., circle vs polygon, triangle vs 4/5/6-gon).
- Colors: qualitative family matches (palette-neighbor acceptable; off-palette is wrong).
- Positions: relative layout/ordering/spacing/alignments/rotations/overlaps match at a human categorical level.

Tolerate tiny rendering noise; focus on semantics.

Return STRICT JSON only:
{"text_semantic_match": 0 or 1, "explanation": "≤30 words"}
"""
        }

        self.maze = {
            "run_root": f"{self.outputs_path}/maze",
            "out_root": f"{self.eval_path}/maze",
            "evaluate_images": True,
            "max_items_per_task": self.max_items_per_task,
            "parser_params": {
                "grid_h": 6, "grid_w": 6, "tolerance": 0.75, "sg_factor": 0.5
            }
        }

        self.sliding_puzzle = {
            "run_root": f"{self.outputs_path}/sliding",
            "out_root": f"{self.eval_path}/sliding",
            "evaluate_images": True,
            "max_items_per_task": self.max_items_per_task,
            "parser_params": {
                "grid_h": 3, "grid_w": 3, "num_categories": 9, "tolerance": 0.80
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations for various multimodal tasks.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt",
        help="A unique name for the model being evaluated, used for output paths (e.g., 'Bagel')."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="path-to-UMMMU-home",
        help="Base path containing outputs/ and eval/ directories.",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Limit the number of items to evaluate per task for a quick test run."
    )
    parser.add_argument(
        "--qwen3_model", type=str, default=None,
        help="Override Qwen3 text LM model name/path.",
    )
    parser.add_argument(
        "--qwen2_5_vl_model", type=str, default=None,
        help="Override Qwen2.5-VL model name/path.",
    )
    parser.add_argument(
        "--vl_attn_impl", type=str, default=None,
        help="Attention implementation for VL model (e.g., flash_attention_2).",
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Override data root path (default: {base_path}/data).",
    )
    parser.add_argument(
        "--outputs_path", type=str, default=None,
        help="Override outputs path (default: {base_path}/outputs/{model_name}).",
    )
    parser.add_argument(
        "--eval_path", type=str, default=None,
        help="Override eval output path (default: {base_path}/eval/{model_name}).",
    )
    parser.add_argument(
        "--dreamsim_cache", type=str, default=None,
        help="Override DreamSim model cache directory (default: /tmp/dreamsim_cache).",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="Skip tasks whose eval_summary.json already exists.",
    )
    args = parser.parse_args()

    print(f"Evaluating model: {args.model_name}")

    configs = Configurations(
        model_name=args.model_name,
        base_path=args.base_path,
        max_items_per_task=args.max_items,
    )
    if args.qwen3_model:
        configs.qwen3_model_name = args.qwen3_model
    if args.qwen2_5_vl_model:
        configs.qwen2_5_vl_model_name = args.qwen2_5_vl_model
    if args.vl_attn_impl:
        configs.vl_attn_impl = args.vl_attn_impl
    if args.dreamsim_cache:
        configs.dreamsim_cache = args.dreamsim_cache
        configs.jigsaw["dreamsim_cache"] = args.dreamsim_cache
    if args.data_root:
        # Override data paths in all task configs
        data_root = args.data_root
        configs.geometry["filtered_json"] = f"{data_root}/math_data/filtered.json"
        configs.jigsaw["dataset_dir"] = f"{data_root}/jigsaw_dataset_2x2ref"
        configs.science["data_json"] = f"{data_root}/science/dim_all.json"
        configs.code_svg["dataset_dir"] = f"{data_root}/svg"
    if args.outputs_path:
        # Override generation output paths in all task configs
        op = args.outputs_path
        configs.outputs_path = op
        configs.geometry["out_dir"] = f"{op}/math"
        configs.jigsaw["out_root"] = f"{op}/jigsaw"
        configs.science["run_root"] = f"{op}/science"
        configs.code_svg["sample_root"] = f"{op}/code"
        configs.maze["run_root"] = f"{op}/maze"
        configs.sliding_puzzle["run_root"] = f"{op}/sliding"
    if args.eval_path:
        # Override eval output paths in all task configs
        ep = args.eval_path
        configs.eval_path = ep
        configs.geometry["out_eval_dir"] = f"{ep}/math"
        configs.jigsaw["eval_dir"] = f"{ep}/jigsaw"
        configs.science["eval_root"] = f"{ep}/science"
        configs.code_svg["eval_out_dir"] = f"{ep}/code"
        configs.maze["out_root"] = f"{ep}/maze"
        configs.sliding_puzzle["out_root"] = f"{ep}/sliding"

    resume = args.resume

    def _task_done(eval_dir: str, summary_name: str = "eval_summary.json") -> bool:
        """Check if a task's scoring is already complete (for --resume)."""
        return resume and Path(eval_dir, summary_name).is_file()

    print("="*20 + " Loading Local Models " + "="*20)
    lm = LocalTextLM(configs.qwen3_model_name)
    vl = LocalVL(configs.qwen2_5_vl_model_name, attn_implementation=configs.vl_attn_impl)

    # 1) Geometry
    print("\n" + "="*20 + " Starting Geometry Evaluation " + "="*20)
    if _task_done(configs.geometry['out_eval_dir']):
        print("Resuming: Geometry eval_summary.json exists, skipping.")
    elif Path(configs.geometry['out_dir']).is_dir():
        geometry_evaluator = GeometryEvaluator(configs.geometry, lm=lm, vl=vl)
        geometry_evaluator.evaluate()
        geometry_evaluator.summarize()
    else:
        print(f"Skipping Geometry: Output directory not found at {configs.geometry['out_dir']}")

    # 2) Jigsaw
    print("\n" + "="*20 + " Starting Jigsaw Evaluation " + "="*20)
    if _task_done(configs.jigsaw['eval_dir'], "summary.json"):
        print("Resuming: Jigsaw summary.json exists, skipping.")
    elif Path(configs.jigsaw['out_root']).is_dir():
        jigsaw_evaluator = JigsawEvaluator(configs.jigsaw)
        jigsaw_evaluator.evaluate()
        jigsaw_evaluator.summarize()
    else:
        print(f"Skipping Jigsaw: Output directory not found at {configs.jigsaw['out_root']}")

    # 3) Science
    print("\n" + "="*20 + " Starting Science Evaluation " + "="*20)
    if _task_done(configs.science['eval_root']):
        print("Resuming: Science eval_summary.json exists, skipping.")
    elif Path(configs.science['run_root']).is_dir():
        science_evaluator = ScienceEvaluator(configs.science, vl=vl)
        science_evaluator.evaluate()
        science_evaluator.summarize()
    else:
        print(f"Skipping Science: Output directory not found at {configs.science['run_root']}")

    # 4) SVG
    print("\n" + "="*20 + " Starting SVG Code Evaluation " + "="*20)
    if _task_done(configs.code_svg['eval_out_dir']):
        print("Resuming: SVG eval_summary.json exists, skipping.")
    elif Path(configs.code_svg['sample_root']).is_dir():
        svg_evaluator = CodeSVGEvaluator(configs.code_svg, vl=vl)
        svg_evaluator.evaluate()
        svg_evaluator.summarize()
    else:
        print(f"Skipping SVG: Output directory not found at {configs.code_svg['sample_root']}")

    # 5) Maze
    print("\n" + "="*20 + " Starting Maze Evaluation " + "="*20)
    if _task_done(configs.maze['out_root']):
        print("Resuming: Maze eval_summary.json exists, skipping.")
    elif Path(configs.maze['run_root']).is_dir():
        maze_evaluator = MazeEvaluator(configs.maze)
        maze_evaluator.evaluate()
        maze_evaluator.summarize()
    else:
        print(f"Skipping Maze: Output directory not found at {configs.maze['run_root']}")

    # 6) Sliding
    print("\n" + "="*20 + " Starting Sliding Puzzle Evaluation " + "="*20)
    if _task_done(configs.sliding_puzzle['out_root']):
        print("Resuming: Sliding summary exists, skipping.")
    elif Path(configs.sliding_puzzle['run_root']).is_dir():
        sliding_puzzle_evaluator = SlidingPuzzleEvaluator(configs.sliding_puzzle)
        sliding_puzzle_evaluator.evaluate()
        sliding_puzzle_evaluator.summarize()
    else:
        print(f"Skipping Sliding Puzzle: Output directory not found at {sliding_output_path}")

    # Final Aggregation Step
    summarize_all_tasks(configs)

    print("\nAll available evaluations and final summary generation completed.")
