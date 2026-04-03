#!/usr/bin/env python3
"""
UEval Qwen-based scoring script — evaluates model outputs against rubrics.

Uses Qwen3-32B for text rubrics and Qwen2.5-VL-72B-Instruct for image rubrics.
Called via subprocess from the cli/ueval_eval.py wrapper.

Usage:
    python eval/generation/ueval/run_scoring.py --config configs/eval/modal_score_ueval_bagel.yaml
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config


# ---------------------------------------------------------------------------
# UEval task domains
# ---------------------------------------------------------------------------

_OPEN_TASKS = {"art", "life", "tech", "exercise"}
_CLOSED_TASKS = {"space", "textbook", "diagram", "paper"}


# ---------------------------------------------------------------------------
# Evaluation prompt templates
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are an expert AI evaluator. Your job is to look at a conversation, a rubric item and assess model outputs (text and images) against specific rubric criteria.
Return a JSON object with "criteria_met".
""".strip()

TEXT_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score how well the model's text answer satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Instructions
Return a JSON object with the field:  "criteria_met" (boolean or "not sure").
- Set "criteria_met" to true only if the rubric is fully satisfied. Use false if any requirement is missing or incorrect. If there is not enough information, return "not sure".
Return only the JSON object (no extra narration).
""".strip()

TEXT_RUBRIC_TEMPLATE = """
# Rubric Item
<<rubric_item>>
""".strip()

IMAGE_TEMPLATE_OPEN = """
You are evaluating whether the generated image (considered together with the accompanying text answer) satisfies the rubric.

# Conversation
Question: <<question>>
Text Answer: <<text_answer>>

# Instructions
You are given the question, the model's text answer, and the generated image(s). Judge whether the visual content (and its alignment with the text) meets the rubric.
Return a JSON object with "criteria_met".
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the requirement is that each image must include a visual depiction of the described action — it cannot rely solely on text rendered within the image as a substitute for visual content. For example, if the rubric says "Each image must directly correspond to a single, sequential step outlined in the text answer," then the image must visually represent the action described in the text (e.g., showing the motion, object, or scene), rather than merely displaying textual labels or written descriptions inside the image.
Return only the JSON object.
- One important exception to the above point is that when the criterion is used to evaluate the consistency between an image step and its corresponding text step, the image does not need to depict all actions or details mentioned in that step to meet the criterion.
For example, if the criterion states, "Each image must visually represent the primary action described in its corresponding numbered step in the text," then an image that clearly shows the main action—such as turning the oven dial to preheat—would still satisfy the criterion, even if the step also includes secondary actions (like preparing the baking tray or measuring ingredients).
The key point is that the image should accurately represent the primary action of the step, rather than all of its described details.
""".strip()

IMAGE_RUBRIC_TEMPLATE = """
# Rubric Item
<<rubric_item>>
""".strip()

IMAGE_TEMPLATE_CLOSED = """
You are evaluating whether the generated image satisfies an image-focused rubric.

# Question
<<question>>

# Instructions
You are given the question and the generated image(s). Judge whether the image meets the rubric. Return a JSON object with "criteria_met".
- Set "criteria_met" to true only if the rubric is completely satisfied; false otherwise. Use "not sure" if the evidence is insufficient.
- One important clarification regarding the image requirement is that each image must include a visual depiction of the described content — it cannot rely solely on text rendered within the image as a substitute for visual content.
Return only the JSON object. If any image consists purely of text with no visual content, it should be judged as false directly.
""".strip()


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _image_to_b64_data_url(
    image_path: Path,
    max_w: Optional[int] = 2048,
) -> str:
    """Read an image, optionally resize, and return as a base64 data URL."""
    path = Path(image_path)
    if not path.is_file():
        print(f"[ueval] WARNING: image file not found: {path}")
        return ""

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
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

            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                save_format = "PNG"

            if save_format == "JPEG" and im.mode != "RGB":
                im = im.convert("RGB")

            im.save(buffer, format=save_format)
            image_bytes = buffer.getvalue()
            mime_type = f"image/{save_format.lower()}"

    except Exception as e:
        print(f"[ueval] WARNING: Pillow failed to process {path} ({e}). Using raw bytes.")
        image_bytes = path.read_bytes()

    b64_string = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_string}"


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _find_first_json_substring(text: str) -> Optional[str]:
    """Extract the first valid JSON object from text using brace-depth tracking."""
    if not text:
        return None
    start_index = text.find("{")
    if start_index == -1:
        return None
    brace_depth, in_string, is_escaped = 0, False, False
    for i in range(start_index, len(text)):
        char = text[i]
        if char == '"' and not is_escaped:
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    return text[start_index : i + 1]
        is_escaped = char == "\\" and not is_escaped
    return None


# ---------------------------------------------------------------------------
# JSONL resume helpers
# ---------------------------------------------------------------------------

def _save_jsonl_append(record: dict, path: Path) -> None:
    """Append one JSON line to file (immediate persistence for resume)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load already-scored entries from JSONL for resume."""
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            records[str(obj["id"])] = obj
        except (json.JSONDecodeError, KeyError):
            pass
    return records


# ---------------------------------------------------------------------------
# Local evaluator model classes
# ---------------------------------------------------------------------------

class _LocalTextLM:
    """Text-only evaluator using Qwen3-32B."""

    def __init__(self, model_name: str, attn_implementation: Optional[str] = "flash_attention_2"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        print(f"[ueval] loading text model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: Dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    @torch.no_grad()
    def generate_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
        )
        gen = out_ids[0][len(inputs.input_ids[0]) :]
        result = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        del inputs, out_ids, gen
        torch.cuda.empty_cache()
        return result

    def generate_json(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> Dict[str, Any]:
        raw = self.generate_text(messages, max_new_tokens=max_new_tokens)
        json_str = _find_first_json_substring(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except Exception:
                pass
        return {"_raw": raw}


class _LocalVL:
    """Vision-language evaluator using Qwen2.5-VL-72B-Instruct."""

    def __init__(self, model_name: str, attn_implementation: Optional[str] = "flash_attention_2"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        kwargs: Dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        print(f"[ueval] loading vision-language model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def generate_text(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512) -> str:
        from qwen_vl_utils import process_vision_info

        norm_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") != "user":
                norm_msgs.append(m)
                continue
            content: List[Dict[str, Any]] = []
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
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        out_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
        )
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        result = out_text.strip()
        del inputs, out_ids, trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
        return result

    def generate_json(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512) -> Dict[str, Any]:
        raw = self.generate_text(messages, max_new_tokens=max_new_tokens)
        json_str = _find_first_json_substring(raw)
        if json_str:
            try:
                return json.loads(json_str)
            except Exception:
                pass
        return {"_raw": raw}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalize_criteria_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        lowered = cleaned.lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        return cleaned
    return value


def _criteria_value_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        if lowered in {"not sure", "unsure", "unknown"}:
            return False
    return False


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


def _get_question_type(task: str) -> str:
    task_lower = task.lower().strip()
    if task_lower in _OPEN_TASKS:
        return "open"
    elif task_lower in _CLOSED_TASKS:
        return "closed"
    return "open"


def _compute_score(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"met": 0, "total": 0, "rate": None}
    total = len(results)
    met = sum(1 for item in results if _criteria_value_to_bool(item.get("criteria_met")))
    rate = met / total if total else None
    return {"met": met, "total": total, "rate": rate}


_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Single item evaluation
# ---------------------------------------------------------------------------

def _evaluate_single_item(
    item: Dict[str, Any],
    model_text: str,
    model_images: List[str],
    text_lm: _LocalTextLM,
    vision_lm: _LocalVL,
    base_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single item's text and image rubrics using local models."""
    question = item.get("prompt", "")

    task = item.get("task_type") or item.get("task", "")
    question_type = _get_question_type(task) if task else "open"

    # --- Text rubrics ---
    text_results: List[Dict[str, Any]] = []
    if item.get("text_rubrics"):
        text_context_prompt = (
            TEXT_TEMPLATE.replace("<<question>>", question)
            .replace("<<text_answer>>", model_text or "")
        )

        for rubric in item.get("text_rubrics", []):
            # rubric can be a dict {"criterion": "..."} or a plain string
            criterion = rubric.get("criterion", "") if isinstance(rubric, dict) else str(rubric)
            rubric_prompt = TEXT_RUBRIC_TEMPLATE.replace("<<rubric_item>>", criterion)
            full_prompt = text_context_prompt + "\n\n" + rubric_prompt

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": full_prompt},
            ]

            parsed = None
            for attempt in range(_MAX_RETRIES):
                parsed = text_lm.generate_json(messages)
                if "criteria_met" in parsed:
                    break
                print(f"  [ueval] text rubric retry {attempt + 1}/{_MAX_RETRIES} for item {item.get('id')}")

            if parsed is None or "criteria_met" not in parsed:
                print(f"  [ueval] failed to parse text rubric response, defaulting to false")
                parsed = {"criteria_met": False, "_raw": str(parsed)}

            criteria_met = _normalize_criteria_value(parsed.get("criteria_met"))
            text_results.append({
                "criterion": criterion,
                "criteria_met": criteria_met,
                "explanation": parsed.get("explanation", ""),
                "raw_response": str(parsed),
            })

    # --- Image rubrics ---
    image_results: List[Dict[str, Any]] = []
    if item.get("image_rubrics"):
        template = IMAGE_TEMPLATE_OPEN if question_type == "open" else IMAGE_TEMPLATE_CLOSED
        context_prompt = (
            template.replace("<<question>>", question)
            .replace("<<text_answer>>", model_text or "")
        )

        # Encode images as base64 data URLs
        image_content_parts: List[Dict[str, Any]] = []
        for img_path in model_images:
            abs_path = Path(img_path) if Path(img_path).is_absolute() else base_dir / img_path
            b64_url = _image_to_b64_data_url(abs_path)
            if b64_url:
                image_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": b64_url},
                })

        for rubric in item.get("image_rubrics", []):
            # rubric can be a dict {"criterion": "..."} or a plain string
            criterion = rubric.get("criterion", "") if isinstance(rubric, dict) else str(rubric)
            rubric_prompt = IMAGE_RUBRIC_TEMPLATE.replace("<<rubric_item>>", criterion)
            full_prompt = context_prompt + "\n\n" + rubric_prompt

            content_parts = list(image_content_parts) + [
                {"type": "text", "text": full_prompt},
            ]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
                {"role": "user", "content": content_parts},
            ]

            parsed = None
            for attempt in range(_MAX_RETRIES):
                parsed = vision_lm.generate_json(messages)
                if "criteria_met" in parsed:
                    break
                print(f"  [ueval] image rubric retry {attempt + 1}/{_MAX_RETRIES} for item {item.get('id')}")

            if parsed is None or "criteria_met" not in parsed:
                print(f"  [ueval] failed to parse image rubric response, defaulting to false")
                parsed = {"criteria_met": False, "_raw": str(parsed)}

            criteria_met = _normalize_criteria_value(parsed.get("criteria_met"))
            image_results.append({
                "criterion": criterion,
                "criteria_met": criteria_met,
                "explanation": parsed.get("explanation", ""),
                "rubric_tags": rubric.get("tags", []) if isinstance(rubric, dict) else [],
                "type": rubric.get("type", "image") if isinstance(rubric, dict) else "image",
                "raw_response": str(parsed),
            })

    text_score = _compute_score(text_results)
    image_score = _compute_score(image_results)

    return {
        "id": item.get("id"),
        "task": task,
        "question": question,
        "question_type": question_type,
        "text_answer": model_text,
        "image_outputs": model_images,
        "text_results": text_results,
        "image_results": image_results,
        "text_score": text_score,
        "image_score": image_score,
    }


def _compute_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_rates = [
        itm["text_score"]["rate"]
        for itm in items
        if itm["text_score"]["rate"] is not None
    ]
    image_rates = [
        itm["image_score"]["rate"]
        for itm in items
        if itm["image_score"]["rate"] is not None
    ]
    all_rates = text_rates + image_rates

    # Per-task breakdown (8 UEval tasks)
    task_scores: Dict[str, List[float]] = {}
    for itm in items:
        task = itm.get("task", "")
        if not task:
            continue
        task_lower = task.lower().strip()
        t_rate = itm["text_score"]["rate"]
        i_rate = itm["image_score"]["rate"]
        rates = []
        if t_rate is not None:
            rates.append(t_rate)
        if i_rate is not None:
            rates.append(i_rate)
        if rates:
            task_scores.setdefault(task_lower, []).extend(rates)

    per_task = {}
    for task_name, rates in sorted(task_scores.items()):
        per_task[task_name] = {
            "avg_rate": sum(rates) / len(rates),
            "num_rubrics": len(rates),
        }

    return {
        "num_items": len(items),
        "num_items_with_text": len(text_rates),
        "num_items_with_image": len(image_rates),
        "text_avg_rate": sum(text_rates) / len(text_rates) if text_rates else None,
        "image_avg_rate": sum(image_rates) / len(image_rates) if image_rates else None,
        "overall_avg_rate": sum(all_rates) / len(all_rates) if all_rates else None,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# Dataset loading
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


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def _run_local_eval(
    model_output_path: Path,
    eval_output_path: Path,
    dataset: List[Dict[str, Any]],
    text_model: str,
    vl_model: str,
    text_field: str,
    image_field: str,
    limit: Optional[int],
) -> bool:
    """Run evaluation using local models (Qwen3-32B + Qwen2.5-VL-72B)."""
    start_time = time.time()
    base_dir = model_output_path.parent

    # Load model outputs
    model_outputs = json.loads(model_output_path.read_text(encoding="utf-8"))

    # Build lookup
    output_lookup: Dict[str, Tuple[str, List[str]]] = {}
    for entry in model_outputs:
        item_id = entry.get("id")
        if item_id is None:
            continue
        item_id_str = str(item_id)
        text_value = entry.get(text_field, "")
        image_value = entry.get(image_field)
        image_list = _ensure_list(image_value)
        output_lookup[item_id_str] = (text_value, image_list)

    # Load evaluator models
    text_lm = _LocalTextLM(text_model)
    vision_lm = _LocalVL(vl_model)

    items_to_process = dataset[:limit] if limit else dataset

    # Resume from JSONL (each scored item is appended immediately)
    results_jsonl_path = eval_output_path.with_suffix(".jsonl")
    existing = _load_existing_results(results_jsonl_path)
    processed_ids = set(existing.keys())
    results: List[Dict[str, Any]] = list(existing.values())

    remaining = [it for it in items_to_process if str(it.get("id", "")) not in processed_ids]
    if processed_ids:
        print(f"[ueval] resuming: {len(processed_ids)} items already scored, {len(remaining)} remaining")

    # Ensure output dir exists
    results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate remaining items
    for item in tqdm(remaining, desc="[ueval score]"):
        item_id = item.get("id")
        item_id_str = str(item_id) if item_id is not None else ""
        text_answer, image_paths = output_lookup.get(item_id_str, ("", []))

        try:
            evaluation = _evaluate_single_item(
                item=item,
                model_text=text_answer,
                model_images=image_paths,
                text_lm=text_lm,
                vision_lm=vision_lm,
                base_dir=base_dir,
            )
            results.append(evaluation)
            _save_jsonl_append(evaluation, results_jsonl_path)

        except Exception as e:
            print(f"[ueval] error evaluating ID {item_id}: {e}")
            traceback.print_exc()
            raise

    summary = _compute_summary(results)
    output_payload = {"results": results, "summary": summary}

    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"\n[ueval] ==== Evaluation Summary ====")
    print(f"[ueval] items evaluated: {summary['num_items']}")
    if summary["text_avg_rate"] is not None:
        print(f"[ueval] text avg score:  {summary['text_avg_rate']:.4f} ({summary['num_items_with_text']} items)")
    if summary["image_avg_rate"] is not None:
        print(f"[ueval] image avg score: {summary['image_avg_rate']:.4f} ({summary['num_items_with_image']} items)")
    if summary["overall_avg_rate"] is not None:
        print(f"[ueval] overall avg:     {summary['overall_avg_rate']:.4f}")
    if summary.get("per_task"):
        print(f"[ueval] ---- Per-Task Scores ----")
        for task_name, task_data in summary["per_task"].items():
            print(f"[ueval]   {task_name:12s}: {task_data['avg_rate']:.4f} ({task_data['num_rubrics']} rubrics)")
    print(f"[ueval] runtime: {hours:02d}:{minutes:02d}:{seconds:.2f}")

    return True


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="UEval Qwen-based scoring")
    parser.add_argument("--config", required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    ueval_cfg = raw_cfg.get("ueval", {})
    if not isinstance(ueval_cfg, dict):
        ueval_cfg = {}

    repo_root = Path(__file__).resolve().parents[3]

    # --- Paths ---
    out_dir = _resolve_path(str(ueval_cfg.get("out_dir", "output/ueval")), repo_root)
    model_output_path = out_dir / "model_outputs.json"
    eval_output_path = out_dir / "eval_results.json"

    if not model_output_path.exists():
        raise FileNotFoundError(
            f"[ueval] model_outputs.json not found at {model_output_path}. "
            f"Run generation first."
        )

    # --- Scoring config ---
    scoring_cfg = ueval_cfg.get("scoring", {})
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}
    text_model = str(scoring_cfg.get("text_model", "Qwen/Qwen3-32B"))
    vl_model = str(scoring_cfg.get("vl_model", "Qwen/Qwen2.5-VL-72B-Instruct"))
    text_field = str(scoring_cfg.get("text_field", "text_answer"))
    image_field = str(scoring_cfg.get("image_field", "image_answer"))
    score_limit = scoring_cfg.get("limit")
    if score_limit is not None:
        score_limit = int(score_limit)
    # --- Load dataset (for rubrics) ---
    hf_dataset = str(ueval_cfg.get("hf_dataset", "zlab-princeton/UEval"))
    local_cache = ueval_cfg.get("local_cache")

    domains_value = ueval_cfg.get("domains")
    domains: Optional[List[str]] = None
    if isinstance(domains_value, list) and domains_value:
        domains = [str(d) for d in domains_value]
    elif isinstance(domains_value, str) and domains_value.strip() and domains_value.strip() != "all":
        domains = [d.strip() for d in domains_value.split(",") if d.strip()]

    max_samples = int(ueval_cfg.get("max_samples", 0) or 0)
    prompts = _load_prompts_hf(hf_dataset, domains, max_samples, local_cache=local_cache)
    print(f"[ueval] loaded {len(prompts)} prompts")

    # --- Run scoring ---
    print(
        f"[ueval] scoring: text_model={text_model}, vl_model={vl_model}, "
        f"out_dir={out_dir}"
    )
    _run_local_eval(
        model_output_path=model_output_path,
        eval_output_path=eval_output_path,
        dataset=prompts,
        text_model=text_model,
        vl_model=vl_model,
        text_field=text_field,
        image_field=image_field,
        limit=score_limit,
    )

    # --- Write summary to separate file ---
    score_output_path = ueval_cfg.get("score_output_path")
    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        # Avoid overwriting the full results file
        if score_path.resolve() == eval_output_path.resolve():
            score_path = eval_output_path.parent / "eval_summary.json"
            print(f"[ueval] score_output_path same as eval_output_path, writing summary to {score_path}")
        if eval_output_path.exists():
            try:
                eval_data = json.loads(eval_output_path.read_text(encoding="utf-8"))
                score_path.parent.mkdir(parents=True, exist_ok=True)
                score_path.write_text(
                    json.dumps(eval_data.get("summary", {}), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[ueval] wrote summary to {score_path}")
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
