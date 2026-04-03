#!/usr/bin/env python3
"""WISE VLM evaluator — Qwen2.5-VL replacement for GPT-4o.

Mirrors the original model/WISE/gpt_eval.py interface:
  - Same evaluation prompt and system message
  - Same output format (per-category *_scores.jsonl and *_full.jsonl)
  - Same regex-based score extraction
  - Same 9.9 sentinel for failures
  - Resume support (skips already-evaluated prompt_ids)

Only difference: uses a local Qwen2.5-VL model instead of OpenAI GPT-4o API.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data category definitions (matches original WISE)
# ---------------------------------------------------------------------------

_WISE_DATA_FILES = [
    ("cultural_common_sense.json", "cultural_common_sense"),
    ("spatio-temporal_reasoning.json", "spatio-temporal_reasoning"),
    ("natural_science.json", "natural_science"),
]

# ---------------------------------------------------------------------------
# Score extraction — identical to model/WISE/gpt_eval.py
# ---------------------------------------------------------------------------

def extract_scores(evaluation_text: str) -> Dict[str, float]:
    """Extract scores via regex, matching original WISE gpt_eval.py logic."""
    score_pattern = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[:：]?\s*(\d)"
    matches = re.findall(score_pattern, evaluation_text, re.IGNORECASE)

    scores = {
        "consistency": 9.9,
        "realism": 9.9,
        "aesthetic_quality": 9.9,
    }

    for key, value in matches:
        key = key.lower().replace(" ", "_")
        if key in scores:
            scores[key] = float(value)

    return scores


# ---------------------------------------------------------------------------
# Image encoding — identical to model/WISE/gpt_eval.py
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Evaluation prompt — identical to model/WISE/gpt_eval.py
# ---------------------------------------------------------------------------

def build_evaluation_messages(prompt_data: Dict, image_base64: str) -> list:
    """Build evaluation messages — exact copy from model/WISE/gpt_eval.py."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a professional Vincennes image quality audit expert, please evaluate the image quality strictly according to the protocol."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt_data['Prompt']}"
EXPLANATION: "{prompt_data['Explanation']}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]


# ---------------------------------------------------------------------------
# Local VLM model wrapper
# ---------------------------------------------------------------------------

class LocalVLM:
    """Qwen2.5-VL model wrapper for evaluation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        attn_implementation: Optional[str] = None,
    ):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        kwargs: Dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        print(f"[vlm_eval] loading {model_name} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._torch = torch
        print(f"[vlm_eval] {model_name} loaded")

    @property
    def device(self):
        return self.model.device

    def evaluate(self, messages: list, max_new_tokens: int = 512) -> str:
        """Run VLM inference and return raw text output."""
        from qwen_vl_utils import process_vision_info

        # Normalize messages for Qwen2.5-VL format
        norm_msgs: list = []
        for m in messages:
            if m.get("role") != "user":
                # System message: flatten content list to string if needed
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                norm_msgs.append({"role": m["role"], "content": content})
                continue
            content: list = []
            for part in m.get("content", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    content.append({"type": "image", "image": url})
                elif part.get("type") == "image":
                    content.append(part)
                elif part.get("type") == "text":
                    content.append({"type": "text", "text": part["text"]})
            norm_msgs.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            norm_msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(norm_msgs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text.strip()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_prompts(json_path: str) -> Dict[int, Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["prompt_id"]: item for item in data}


def load_existing_scores(path: str) -> Dict[int, Dict]:
    """Load already-evaluated entries for resume support."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return {}
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records[obj["prompt_id"]] = obj
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return records


def save_jsonl_append(data: dict, path: str) -> None:
    """Append a single JSON line to file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_category(
    vlm: LocalVLM,
    json_path: str,
    image_dir: str,
    output_dir: str,
    max_new_tokens: int = 512,
    max_retries: int = 2,
) -> None:
    """Evaluate all images for a single category JSON file.

    Outputs:
      - {category_name}_scores.jsonl  (same format as original WISE)
      - {category_name}_full.jsonl    (same format as original WISE)
    """
    category_name = Path(json_path).stem  # e.g. "cultural_common_sense"
    scores_path = os.path.join(output_dir, f"{category_name}_scores.jsonl")
    full_path = os.path.join(output_dir, f"{category_name}_full.jsonl")

    prompts = load_prompts(json_path)
    existing_scores = load_existing_scores(scores_path)
    existing_full = load_existing_scores(full_path)

    done_ids = set(existing_scores.keys())
    tasks = []
    for pid, pdata in prompts.items():
        if pid in done_ids:
            continue
        img_path = os.path.join(image_dir, f"{pid}.png")
        if not os.path.exists(img_path):
            print(f"[vlm_eval] Warning: missing image {img_path}")
            continue
        tasks.append((pid, pdata, img_path))

    print(
        f"[vlm_eval] {category_name}: {len(prompts)} total, "
        f"{len(existing_scores)} done, {len(tasks)} remaining"
    )

    for pid, pdata, img_path in tasks:
        print(f"[vlm_eval] evaluating prompt_id={pid} ...")

        eval_text = ""
        scores = {"consistency": 9.9, "realism": 9.9, "aesthetic_quality": 9.9}

        for attempt in range(max_retries + 1):
            try:
                img_b64 = encode_image(img_path)
                messages = build_evaluation_messages(pdata, img_b64)
                eval_text = vlm.evaluate(messages, max_new_tokens=max_new_tokens)
                scores = extract_scores(eval_text)
                # If all three scores were extracted successfully, break
                if 9.9 not in scores.values():
                    break
            except Exception as e:
                print(f"[vlm_eval] attempt {attempt + 1} error for pid={pid}: {e}")

        full_record = {
            "prompt_id": pid,
            "prompt": pdata["Prompt"],
            "key": pdata["Explanation"],
            "image_path": img_path,
            "evaluation": eval_text,
        }
        score_record = {
            "prompt_id": pid,
            "Subcategory": pdata.get("Subcategory", ""),
            "consistency": scores["consistency"],
            "realism": scores["realism"],
            "aesthetic_quality": scores["aesthetic_quality"],
        }

        save_jsonl_append(full_record, full_path)
        save_jsonl_append(score_record, scores_path)

        print(
            f"  pid={pid}: consistency={scores['consistency']}, "
            f"realism={scores['realism']}, aesthetic_quality={scores['aesthetic_quality']}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="WISE VLM Evaluator (Qwen2.5-VL replacement for GPT-4o)"
    )
    parser.add_argument(
        "--data_root", required=True,
        help="Directory containing the 3 category JSON files"
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory containing generated images (named {prompt_id}.png)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write score and full JSONL files"
    )
    parser.add_argument(
        "--model_name", default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Qwen2.5-VL model name or path"
    )
    parser.add_argument(
        "--attn_implementation", default=None,
        help="Attention implementation (e.g. flash_attention_2)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512,
        help="Max new tokens for VLM generation"
    )
    parser.add_argument(
        "--max_retries", type=int, default=2,
        help="Max retries per image on evaluation failure"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect data/ subdirectory (e.g. /workspace/model/WISE -> .../WISE/data)
    data_root = args.data_root
    first_fname = _WISE_DATA_FILES[0][0]
    if not os.path.isfile(os.path.join(data_root, first_fname)):
        candidate = os.path.join(data_root, "data")
        if os.path.isfile(os.path.join(candidate, first_fname)):
            data_root = candidate

    # Load VLM once, evaluate all categories
    vlm = LocalVLM(
        model_name=args.model_name,
        attn_implementation=args.attn_implementation,
    )

    for fname, category_name in _WISE_DATA_FILES:
        json_path = os.path.join(data_root, fname)
        if not os.path.isfile(json_path):
            print(f"[vlm_eval] Warning: data file not found: {json_path}")
            continue

        evaluate_category(
            vlm=vlm,
            json_path=json_path,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
        )

    print(f"[vlm_eval] all categories evaluated. Results at {args.output_dir}")


if __name__ == "__main__":
    main()
