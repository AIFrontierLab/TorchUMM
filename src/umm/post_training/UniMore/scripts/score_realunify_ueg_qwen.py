#!/usr/bin/env python3
"""Score RealUnify Direct-UEG images with a local Qwen2.5-VL judge.

The question prompt, yes/no normalization, early exit, and all-questions-must-
match sample score follow RealUnify's ``eval/eval_generation.py``. The only
protocol substitution is the user-selected local judge model.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


OFFICIAL_PROMPT = (
    "Please answer the following question based on the image:\n"
    "Question: {question}"
    "\n\nYou should only reply yes or no, and do not provide any other extra content."
)


def is_primary() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen scorer for RealUnify UEG")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=15)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--invalid-question-policy",
        choices=("error", "skip-empty"),
        default="error",
        help="skip-empty ignores only entries whose question and answer are both empty",
    )
    parser.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="Use the Transformers native tensor-parallel plan under torchrun",
    )
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-image-pixels", type=int, default=0)
    return parser.parse_args()


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def item_key(item: dict[str, Any]) -> str:
    key = item.get("_realunify_key")
    if isinstance(key, str) and key:
        return key
    return f"UEG_direct:{item['task_type']}:{int(item['index']):04d}"


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[item_key(record)] = record
            except Exception as exc:
                raise ValueError(f"Invalid scorer JSONL at {path}:{line_number}") from exc
    return records


def normalize_official_response(response: str) -> str:
    normalized = response.lower()
    if "yes" in normalized:
        normalized = "yes"
    if "no" in normalized:
        normalized = "no"
    return normalized


class LocalQwenJudge:
    def __init__(
        self,
        model_name: str,
        attn_implementation: str | None,
        tensor_parallel: bool = False,
    ) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if tensor_parallel:
            kwargs["tp_plan"] = "auto"
        else:
            kwargs["device_map"] = "auto"
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if is_primary():
            mode = "tensor-parallel" if tensor_parallel else "device-map"
            print(f"[realunify-ueg] loading judge ({mode}): {model_name}", flush=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        if is_primary():
            print("[realunify-ueg] judge loaded", flush=True)

    @torch.no_grad()
    def answer(self, prompt: str, image: Image.Image, max_new_tokens: int) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        del inputs, output_ids, trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
        return response


class LocalVLLMQwenJudge:
    """Batched Qwen judge using the same model, prompts, and greedy decoding."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        batch_size: int,
        max_image_pixels: int,
    ) -> None:
        from umm.post_training.UniMore.qwen25_vl_vllm import Qwen25VLVLLM

        self.batch_size = batch_size
        if is_primary():
            print(
                f"[realunify-ueg] loading judge "
                f"(vLLM external TP={tensor_parallel_size}, batch={batch_size}): "
                f"{model_name}",
                flush=True,
            )
        self.engine = Qwen25VLVLLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=batch_size,
            max_image_pixels=max_image_pixels,
        )
        if is_primary():
            print("[realunify-ueg] judge loaded", flush=True)

    def answer_batch(
        self,
        prompts: list[str],
        images: list[Image.Image],
        max_new_tokens: int,
    ) -> list[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for prompt, image in zip(prompts, images)
        ]
        return self.engine.generate(messages, max_tokens=max_new_tokens)

    def answer(self, prompt: str, image: Image.Image, max_new_tokens: int) -> str:
        return self.answer_batch([prompt], [image], max_new_tokens)[0]


def usable_questions(
    item: dict[str, Any], invalid_question_policy: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    questions = item.get("question_list")
    if not isinstance(questions, list) or not questions:
        raise RuntimeError(f"Missing question_list for {item_key(item)}")
    valid: list[dict[str, str]] = []
    ignored: list[dict[str, Any]] = []
    for position, qa in enumerate(questions):
        if not isinstance(qa, dict):
            raise TypeError(f"Question {position} in {item_key(item)} is not an object")
        question = qa.get("question")
        answer = qa.get("answer")
        question_text = question.strip() if isinstance(question, str) else ""
        answer_text = answer.strip().lower() if isinstance(answer, str) else ""
        if question_text and answer_text in {"yes", "no"}:
            valid.append({"question": question_text, "answer": answer_text})
            continue
        if invalid_question_policy == "skip-empty" and not question_text and not answer_text:
            ignored.append({"position": position, "raw": qa})
            continue
        raise RuntimeError(
            f"Question {position} in {item_key(item)} must contain text and a yes/no answer"
        )
    if not valid:
        raise RuntimeError(f"No usable questions for {item_key(item)}")
    return valid, ignored


def validate_rows(
    rows: list[dict[str, Any]], expected_count: int, invalid_question_policy: str
) -> dict[str, Any]:
    if expected_count > 0 and len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} RealUnify rows, found {len(rows)}")
    keys = [item_key(item) for item in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("RealUnify input contains duplicate item keys")
    usable_count = 0
    ignored_count = 0
    warnings: list[str] = []
    for item in rows:
        image_path = Path(str(item.get("generated_image", "")))
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing generated image for {item_key(item)}: {image_path}")
        questions, ignored = usable_questions(item, invalid_question_policy)
        usable_count += len(questions)
        ignored_count += len(ignored)
        warnings.extend(
            f"{item_key(item)}: ignored empty question at position {entry['position']}"
            for entry in ignored
        )
    return {
        "sample_count": len(rows),
        "usable_question_count": usable_count,
        "ignored_question_count": ignored_count,
        "invalid_question_policy": invalid_question_policy,
        "warnings": warnings,
    }


def score_item(
    item: dict[str, Any],
    judge: Any,
    max_new_tokens: int,
    max_retries: int,
    invalid_question_policy: str,
) -> dict[str, Any]:
    scored = dict(item)
    details: list[dict[str, Any]] = []
    score = 1
    image_path = Path(str(item["generated_image"]))
    with Image.open(image_path) as source:
        image = source.convert("RGB").copy()

    questions, ignored_questions = usable_questions(item, invalid_question_policy)
    for question_index, qa in enumerate(questions):
        question = qa["question"]
        expected = qa["answer"]
        prompt = OFFICIAL_PROMPT.format(question=question)
        raw_response = ""
        for attempt in range(max_retries):
            raw_response = judge.answer(prompt, image, max_new_tokens)
            if raw_response:
                break
            if is_primary():
                print(
                    f"[realunify-ueg] empty response for {item_key(item)} "
                    f"question={question_index}, retry={attempt + 1}/{max_retries}",
                    flush=True,
                )
        normalized = normalize_official_response(raw_response)
        matches = normalized == expected
        details.append(
            {
                "question_index": question_index,
                "question": question,
                "expected": expected,
                "raw_response": raw_response,
                "normalized_response": normalized,
                "matches": matches,
            }
        )
        if not matches:
            score = 0
            break

    scored["score"] = score
    scored["judge_details"] = details
    scored["ignored_questions"] = ignored_questions
    scored["evaluated_question_count"] = len(details)
    scored["usable_question_count"] = len(questions)
    scored["raw_question_count"] = len(item["question_list"])
    return scored


def score_items_batched(
    items: list[dict[str, Any]],
    judge: LocalVLLMQwenJudge,
    max_new_tokens: int,
    max_retries: int,
    invalid_question_policy: str,
):
    queue: list[dict[str, Any]] = []
    for item in items:
        questions, ignored = usable_questions(item, invalid_question_policy)
        queue.append(
            {
                "item": item,
                "questions": questions,
                "ignored": ignored,
                "question_index": 0,
                "details": [],
            }
        )

    while queue:
        states = queue[: judge.batch_size]
        del queue[: judge.batch_size]
        prompts: list[str] = []
        images: list[Image.Image] = []
        for state in states:
            question = state["questions"][state["question_index"]]["question"]
            prompts.append(OFFICIAL_PROMPT.format(question=question))
            with Image.open(Path(str(state["item"]["generated_image"]))) as source:
                images.append(source.convert("RGB").copy())

        responses = [""] * len(states)
        pending_indices = list(range(len(states)))
        for _ in range(max_retries):
            if not pending_indices:
                break
            generated = judge.answer_batch(
                [prompts[index] for index in pending_indices],
                [images[index] for index in pending_indices],
                max_new_tokens,
            )
            retry: list[int] = []
            for index, response in zip(pending_indices, generated):
                responses[index] = response
                if not response:
                    retry.append(index)
            pending_indices = retry

        for state, raw_response in zip(states, responses):
            position = int(state["question_index"])
            qa = state["questions"][position]
            normalized = normalize_official_response(raw_response)
            matches = normalized == qa["answer"]
            state["details"].append(
                {
                    "question_index": position,
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "raw_response": raw_response,
                    "normalized_response": normalized,
                    "matches": matches,
                }
            )
            if matches and position + 1 < len(state["questions"]):
                state["question_index"] = position + 1
                queue.append(state)
                continue

            scored = dict(state["item"])
            scored["score"] = int(matches)
            scored["judge_details"] = state["details"]
            scored["ignored_questions"] = state["ignored"]
            scored["evaluated_question_count"] = len(state["details"])
            scored["usable_question_count"] = len(state["questions"])
            scored["raw_question_count"] = len(state["item"]["question_list"])
            yield scored


def build_summary(
    rows: list[dict[str, Any]],
    model_name: str,
    elapsed_seconds: float,
    validation: dict[str, Any],
    inference_backend: str = "transformers",
) -> dict[str, Any]:
    correct_by_task: Counter[str] = Counter()
    total_by_task: Counter[str] = Counter()
    for row in rows:
        task = str(row["task_type"])
        total_by_task[task] += 1
        if int(row.get("score", 0)) == 1:
            correct_by_task[task] += 1
    correct = sum(correct_by_task.values())
    total = sum(total_by_task.values())
    return {
        "benchmark": "RealUnify",
        "split": "UEG_direct",
        "judge_model": model_name,
        "judge_backend": "local_qwen2_5_vl",
        "inference_backend": inference_backend,
        "protocol": (
            "RealUnify eval/eval_generation.py prompt, response normalization, "
            "early exit, and all-usable-questions-must-match scoring"
        ),
        "invalid_question_policy": validation["invalid_question_policy"],
        "usable_question_count": validation["usable_question_count"],
        "ignored_question_count": validation["ignored_question_count"],
        "data_warnings": validation["warnings"],
        "correct": correct,
        "total": total,
        "official_score": correct / total if total else None,
        "accuracy_by_task": {
            task: {
                "correct": correct_by_task[task],
                "total": total_by_task[task],
                "accuracy": correct_by_task[task] / total_by_task[task],
            }
            for task in sorted(total_by_task)
        },
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> int:
    args = parse_args()
    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{args.input_json} must contain a JSON list")
    validation = validate_rows(rows, args.expected_count, args.invalid_question_policy)
    if is_primary():
        print(json.dumps({"dataset_validation": validation}, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "scored_results.jsonl"
    existing = load_existing(jsonl_path)
    pending = [item for item in rows if item_key(item) not in existing]
    if is_primary():
        print(
            f"[realunify-ueg] rows={len(rows)} resumed={len(existing)} pending={len(pending)}",
            flush=True,
        )

    started = time.monotonic()
    if pending:
        if args.backend == "vllm":
            judge = LocalVLLMQwenJudge(
                args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size,
                max_image_pixels=args.max_image_pixels,
            )
            scored_items = score_items_batched(
                pending,
                judge,
                max_new_tokens=args.max_new_tokens,
                max_retries=args.max_retries,
                invalid_question_policy=args.invalid_question_policy,
            )
        else:
            judge = LocalQwenJudge(
                args.model_name,
                args.attn_implementation or None,
                tensor_parallel=args.tensor_parallel,
            )
            scored_items = (
                score_item(
                    item,
                    judge,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                    invalid_question_policy=args.invalid_question_policy,
                )
                for item in pending
            )
        output = jsonl_path.open("a", encoding="utf-8") if is_primary() else None
        try:
            for scored in tqdm(
                scored_items,
                total=len(pending),
                desc="[realunify UEG score]",
                disable=not is_primary(),
            ):
                existing[item_key(scored)] = scored
                if output is not None:
                    output.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    output.flush()
                distributed_barrier()
        finally:
            if output is not None:
                output.close()

    ordered = [existing[item_key(item)] for item in rows]
    elapsed = time.monotonic() - started
    summary = build_summary(
        ordered,
        args.model_name,
        elapsed,
        validation,
        inference_backend=args.backend,
    )
    distributed_barrier()
    if is_primary():
        write_json_atomic(ordered, args.output_dir / "scored_results.json")
        write_json_atomic(summary, args.output_dir / "summary.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    distributed_barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
