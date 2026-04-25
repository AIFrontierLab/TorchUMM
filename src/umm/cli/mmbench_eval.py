from __future__ import annotations

import base64
import copy
import json
import os
import random
import string
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config


DS_COLLECTIONS = {
    "mmbench_dev_20230712": {
        "root": "data/mmbench/mmbench_dev_20230712.tsv",
        "type": "dev",
        "language": "en",
    },
    "mmbench_dev_cn_20231003": {
        "root": "data/mmbench/mmbench_dev_cn_20231003.tsv",
        "type": "dev",
        "language": "cn",
    },
    "mmbench_dev_en_20231003": {
        "root": "data/mmbench/mmbench_dev_en_20231003.tsv",
        "type": "dev",
        "language": "en",
    },
    "mmbench_test_cn_20231003": {
        "root": "data/mmbench/mmbench_test_cn_20231003.tsv",
        "type": "test",
        "language": "cn",
    },
    "mmbench_test_en_20231003": {
        "root": "data/mmbench/mmbench_test_en_20231003.tsv",
        "type": "test",
        "language": "en",
    },
    "ccbench_dev_cn": {
        "root": "data/mmbench/CCBench_legacy.tsv",
        "type": "dev",
        "language": "cn",
    },
    # V11 test splits — released with GT labels after the online eval service closed
    # on 2026-03-31 (see open-compass/MMBench#61). Same TSV schema as dev (four
    # rotations per question, index = base + k * 1_000_000 for k = 0..3).
    "MMBench_TEST_EN_V11": {
        "root": "/datasets/mmbench/MMBench_TEST_EN_V11.tsv",
        "type": "test",
        "language": "en",
    },
    "MMBench_TEST_CN_V11": {
        "root": "/datasets/mmbench/MMBench_TEST_CN_V11.tsv",
        "type": "test",
        "language": "cn",
    },
}


# Port of VLMEvalKit MMB_abbrs (vlmeval/dataset/utils/multiple_choice.py:17-24)
MMB_ABBRS = {
    "coarse_perception": "CP",
    "finegrained_perception (instance-level)": "FP-S",
    "finegrained_perception (cross-instance)": "FP-C",
    "logic_reasoning": "LR",
    "relation_reasoning": "RR",
    "attribute_reasoning": "AR",
}


# VLMEvalKit uses the same English scaffold for both EN and CN splits — only the
# question / option / hint fields themselves are localised inside the TSV.
# (vlmeval/dataset/image_mcq.py:210-245)
_ANSWER_INSTRUCTION = "Please select the correct answer from the options above. \n"


# ---------------------------------------------------------------------------
# Generation-time prompt (ports vlmeval/dataset/image_mcq.py:210-245)
# ---------------------------------------------------------------------------


def _build_prompt(question: str, options: "dict[str, str]", hint: str | None) -> str:
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"
    prompt = ""
    if hint:
        prompt += f"Hint: {hint}\n"
    prompt += f"Question: {question}\n"
    if options:
        prompt += options_prompt
        prompt += _ANSWER_INSTRUCTION
    return prompt


# ---------------------------------------------------------------------------
# Exact-matching extraction — ports vlmeval/utils/matching_util.py:12-116
# ---------------------------------------------------------------------------


_REJECT_TO_ANSWER = [
    "Sorry, I can't help with images of people yet.",
    "I can't process this file.",
    "I'm sorry, but without the image provided",
    "Cannot determine the answer",
]
_EXACT_CHARS_TO_SPACE = ".()[],:;!*#{}"


def _can_infer_option(answer: str, choices: "dict[str, str]") -> "str | bool":
    if "Failed to obtain answer via API" in answer:
        return False
    for err in _REJECT_TO_ANSWER:
        if err in answer:
            return "Z"

    answer_mod = copy.copy(answer)
    for c in _EXACT_CHARS_TO_SPACE:
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]

    def count_choice(tokens, cands, prefix: str = "", suffix: str = "") -> int:
        return sum(1 for c in cands if (prefix + c + suffix) in tokens)

    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > len(splits) - 5:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def _can_infer_text(answer: str, choices: "dict[str, str]") -> "str | bool":
    answer_low = answer.lower()
    total_len = sum(len(str(v)) for v in choices.values())
    if len(answer_low) > 2 * total_len:
        return False
    lowered = {k: str(v).lower() for k, v in choices.items()}
    cands = [k for k, v in lowered.items() if v in answer_low]
    if len(cands) == 1:
        return cands[0]
    return False


def _can_infer(answer: Any, choices: "dict[str, str]") -> "str | bool":
    answer = str(answer)
    copt = _can_infer_option(answer, choices)
    return copt if copt else _can_infer_text(answer, choices)


# ---------------------------------------------------------------------------
# LLM judge — ports vlmeval/dataset/utils/multiple_choice.py build_prompt(_cn)
# and build_option_str / cn_string.
# ---------------------------------------------------------------------------


_JUDGE_PROMPT_EN = (
    "You are an AI assistant who will help me to match "
    "an answer with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, "
    "and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
    "Example 1: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: a cute teddy bear\nYour output: A\n"
    "Example 2: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: Spider\nYour output: Z\n"
    "Example 3: \n"
    "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)


_JUDGE_PROMPT_CN = (
    "你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。"
    "你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。"
    "如果所有选项的意义都与答案显著不同，则输出 Z。"
    "你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。"
    "例 1:"
    "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n"
    "例 2: \n"
    "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n"
    "例 3: \n"
    "问题: {}?\n选项: {}\n答案: {}\n输出: "
)


def _build_option_str(options: "dict[str, str]") -> str:
    s = "There are several options: \n"
    for c, content in options.items():
        if content is None:
            continue
        if isinstance(content, float) and pd.isna(content):
            continue
        s += f"{c}. {content}\n"
    return s


def _cn_string(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def _build_judge_prompt(question: str, options: "dict[str, str]", prediction: str, language: str) -> str:
    option_str = _build_option_str(options)
    if language == "cn" or _cn_string(question):
        tmpl = _JUDGE_PROMPT_CN
    else:
        tmpl = _JUDGE_PROMPT_EN
    return tmpl.format(question, option_str, prediction)


# ---------------------------------------------------------------------------
# Judge model bundle — lazy loaded, shared across all extractions in a run.
# ---------------------------------------------------------------------------


class _JudgeBundle:
    def __init__(self, model_path: str, max_new_tokens: int = 32) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[mmbench] loading judge LLM: {model_path}", flush=True)
        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Strip Qwen3-style <think>...</think> if present so can_infer sees the letter
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        return raw

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# extract_answer_from_item — ports multiple_choice.py:359-406
# ---------------------------------------------------------------------------


def _extract_answer_from_item(item: "dict[str, Any]", judge: "_JudgeBundle | None") -> "dict[str, str]":
    """Return {'opt': letter, 'log': ...}, matching VLMEvalKit's extract_answer_from_item."""
    choices = item["choices"]
    prediction = str(item.get("prediction", ""))

    ret = _can_infer(prediction, choices)
    if ret:
        return {"opt": str(ret), "log": prediction}
    if judge is None:
        return {
            "opt": "Z",
            "log": "Failed in Prefetch, no LLM-based answer matching under `exact_matching` policy.",
        }

    prompt = _build_judge_prompt(
        question=str(item.get("question", "")),
        options=choices,
        prediction=prediction,
        language=item.get("language", "en"),
    )
    for _ in range(3):
        ans = judge.generate(prompt)
        if "Failed to obtain answer via API" in ans:
            continue
        ret = _can_infer(ans, choices)
        if ret:
            return {"opt": str(ret), "log": ans}
    # Random fallback (matches VLMEvalKit — the random choice ensures we still
    # produce a concrete letter rather than silently skipping the item).
    options_pool = list(choices) + (["Z"] if "Z" not in choices else [])
    return {"opt": random.choice(options_pool), "log": "Failed to predict, thus randomly generate one."}


# ---------------------------------------------------------------------------
# Circular evaluation — ports multiple_choice.py:409-471
# ---------------------------------------------------------------------------


def _prefetch_answer(item: "dict[str, Any]") -> "str | bool":
    return _can_infer(str(item.get("prediction", "")), item["choices"])


def _prefetch_circular_group(
    sub_items: "list[dict[str, Any]]",
) -> "tuple[dict | None, list, list]":
    """Returns (result_dict_or_None, gts, preds).

    If a definitive result can be decided via can_infer alone (all matched, or
    any rotation's prefetched answer conflicts with its GT), returns a dict with
    hit/log. Otherwise returns None (LLM fallback is needed).
    """
    gts: list = []
    preds: list = []
    for i, item in enumerate(sub_items):
        gts.append(item["gt_answer"])
        preds.append(_prefetch_answer(item))
        if preds[-1] and gts[-1] != preds[-1]:
            return (
                {
                    "hit": 0,
                    "log": (
                        f"Failed in Prefetching Rolling {i}: Answer is {gts[-1]}, "
                        f"Prediction is {item.get('prediction', '')}, "
                        f"Pre-fetched is {preds[-1]}."
                    ),
                },
                gts,
                preds,
            )
    if all(g == p for g, p in zip(gts, preds)):
        return {"hit": 1, "log": "Succeed During Pre-fetching"}, gts, preds
    return None, gts, preds


def _eval_circular_group(
    sub_items: "list[dict[str, Any]]",
    judge: "_JudgeBundle | None",
) -> "dict[str, Any]":
    """Evaluate one circular group (1-4 rotations of the same question).

    Returns {'hit': 0 or 1, 'log': ...}. hit=1 requires every rotation's
    extracted letter to match its own rotated GT.
    """
    result, gts, preds = _prefetch_circular_group(sub_items)
    if result is not None:
        return result

    log = ""
    for i, item in enumerate(sub_items):
        if preds[i]:
            log += f"Rolling {i} Matched.\n"
            continue
        res = _extract_answer_from_item(item, judge)
        opt, match_log = res["opt"], res["log"]
        preds[i] = opt
        if preds[i] != gts[i]:
            log += (
                f"Failed in Rolling {i}: Answer is {gts[i]}; "
                f"Prediction is {item.get('prediction', '')}; "
                f"Pre-fetched is {preds[i]}; Match Log is {match_log}.\n"
            )
            return {"hit": 0, "log": log}
        log += (
            f"Rolling {i}: Answer is {gts[i]}, "
            f"Prediction is {item.get('prediction', '')}, Pre-fetched is {preds[i]}.\n"
        )
    return {"hit": 1, "log": log}


# ---------------------------------------------------------------------------
# report_acc — ports multiple_choice.py:77-100
# ---------------------------------------------------------------------------


def _report_acc(df: pd.DataFrame) -> "dict[str, float]":
    """Return a flat dict keyed by 'split=<split>|<group>' → accuracy in [0,1].

    Empty (split, category) cells return NaN to match VLMEvalKit's
    np.mean([]) behaviour rather than 0.0 (which would distort
    average-of-averages aggregations downstream).
    """
    res: "dict[str, float]" = {}
    if "split" in df.columns:
        splits = sorted(df["split"].dropna().unique().tolist())
    else:
        df = df.copy()
        df["split"] = "none"
        splits = ["none"]

    for group in (None, "l2-category", "category"):
        if group is None:
            for sp in splits:
                sub = df[df["split"] == sp]["hit"]
                val = float(sub.mean()) if len(sub) else float("nan")
                res[f"split={sp}|Overall"] = val
        elif group not in df.columns:
            continue
        else:
            abilities = sorted(df[group].dropna().unique().tolist())
            for ab in abilities:
                ab_name = MMB_ABBRS.get(ab, ab)
                sub_df = df[df[group] == ab]
                for sp in splits:
                    cell = sub_df[sub_df["split"] == sp]["hit"]
                    val = float(cell.mean()) if len(cell) else float("nan")
                    res[f"split={sp}|{ab_name}"] = val
    return res


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _normalize_backbone_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "showo2": "show_o2",
        "showo": "show_o2",
        "janus": "janus_pro",
    }
    return aliases.get(normalized, normalized)


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        results = output.get("results")
        if isinstance(results, dict):
            for key in ("text", "answer", "response", "output"):
                value = results.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(results, list):
            for item in results:
                text = _extract_text(item)
                if text:
                    return text
        # Handle adapters that return {"understandings": [{"response": "..."}]}
        for list_key in ("understandings",):
            container = output.get(list_key)
            if isinstance(container, list):
                for item in container:
                    text = _extract_text(item)
                    if text:
                        return text
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _load_eval_cfg(config_path: str) -> "tuple[dict, dict, dict]":
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mmbench_cfg = raw_cfg.get("mmbench", {}) if isinstance(raw_cfg.get("mmbench"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmbench_cfg, inference_cfg


def _decode_image(image_b64: str, image_dir: Path, row_index: int) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    out_path = image_dir / f"{row_index}.png"
    image.save(out_path, format="PNG")
    return str(out_path)


def _get_dataset_paths(
    datasets: "list[str]",
    repo_root: Path,
    override_paths: "dict[str, Any]",
) -> "dict[str, Path]":
    resolved: dict[str, Path] = {}
    for name in datasets:
        if name in override_paths:
            resolved[name] = _resolve_path(str(override_paths[name]), repo_root)
            continue
        entry = DS_COLLECTIONS.get(name)
        if not entry:
            raise ValueError(f"Unknown MMBench dataset: {name}")
        resolved[name] = _resolve_path(str(entry["root"]), repo_root)
    return resolved


def _find_latest_jsonl(out_dir: Path, ds_name: str) -> "Path | None":
    candidates = sorted(out_dir.glob(f"{ds_name}_*.jsonl"))
    candidates = [c for c in candidates if "_checkpoint" not in c.name]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _parse_options(row: pd.Series) -> "dict[str, str]":
    """Collect A/B/C/D/E/... option letters whose cell is non-null, preserving TSV order."""
    options: dict[str, str] = {}
    for cand in string.ascii_uppercase:
        if cand in row and not pd.isna(row[cand]):
            options[cand] = row[cand]
    return options


def _resolve_image_blob(image_map: "dict[int, str]", idx: int) -> str:
    """Resolve MMBench's short-circuit image storage (a ≤64-char string that
    points to another row's index). Ports image_base.py:154-164."""
    value = image_map.get(idx, "")
    if not isinstance(value, str):
        return ""
    if len(value) <= 64:
        # It's a redirect to another row's full base64 blob
        try:
            target = int(value)
        except (TypeError, ValueError):
            return value
        target_val = image_map.get(target, "")
        if isinstance(target_val, str) and len(target_val) > 64:
            return target_val
    return value


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_mmbench_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mmbench_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mmbench":
        raise ValueError(f"Expected `eval.benchmark: mmbench`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MMBench eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: "dict[str, Any]" = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    datasets_value = mmbench_cfg.get("datasets", ["mmbench_dev_20230712"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["mmbench_dev_20230712"]
    if not datasets:
        raise ValueError("`mmbench.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mmbench_cfg.get("out_dir", f"output/mmbench/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mmbench_cfg.get("image_dir", out_dir / "images")), repo_root)
    score_output_path = mmbench_cfg.get("score_output_path")
    max_samples = int(mmbench_cfg.get("max_samples", 0) or 0)
    resume = bool(mmbench_cfg.get("resume", False))
    resume_jsonl = mmbench_cfg.get("resume_jsonl")

    mode = str(mmbench_cfg.get("mode", "generate")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[mmbench] unknown mode '{mode}', defaulting to 'generate'", flush=True)
        mode = "generate"
    run_gen = mode in ("full", "generate")
    run_score = mode in ("full", "score")

    llm_extract_cfg = mmbench_cfg.get("llm_extract", {})
    if not isinstance(llm_extract_cfg, dict):
        llm_extract_cfg = {}
    llm_model_path = str(llm_extract_cfg.get("model_path", "")).strip()
    llm_max_new_tokens = int(llm_extract_cfg.get("max_new_tokens", 32))

    dataset_paths = _get_dataset_paths(
        datasets=datasets,
        repo_root=repo_root,
        override_paths=mmbench_cfg.get("dataset_paths", {}) if isinstance(mmbench_cfg.get("dataset_paths"), dict) else {},
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: "dict[str, Any]" = {
        "benchmark": "mmbench",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
        "mode": mode,
    }

    # ── Phase 1: Generation ──
    if run_gen:
        from umm.inference import InferencePipeline

        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

        for ds_name in datasets:
            dataset_path = dataset_paths[ds_name]
            if not dataset_path.exists():
                raise FileNotFoundError(f"MMBench dataset not found: {dataset_path}")

            entry = DS_COLLECTIONS.get(ds_name, {})
            language = str(entry.get("language", "en"))
            df = pd.read_csv(dataset_path, sep="\t")
            df["index"] = df["index"].astype(int)

            # Resolve MMBench's image short-circuits so every row has its own blob.
            image_map: "dict[int, str]" = {}
            if "image" in df.columns:
                for _, row in df.iterrows():
                    image_map[int(row["index"])] = str(row["image"]) if not pd.isna(row["image"]) else ""

            checkpoint_jsonl = out_dir / f"{ds_name}_checkpoint.jsonl"
            outputs: "list[dict[str, Any]]" = []
            done_indices: "set[int]" = set()

            if resume:
                if checkpoint_jsonl.exists():
                    with checkpoint_jsonl.open("r", encoding="utf-8") as reader:
                        for line in reader:
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            outputs.append(item)
                            done_indices.add(int(item["index"]))
                    print(f"[mmbench] resume from checkpoint: {len(outputs)} done", flush=True)
                else:
                    jsonl_path: "Path | None" = None
                    if isinstance(resume_jsonl, str) and resume_jsonl:
                        jsonl_path = _resolve_path(resume_jsonl, repo_root)
                    else:
                        jsonl_path = _find_latest_jsonl(out_dir, ds_name)
                    if jsonl_path and jsonl_path.exists():
                        with jsonl_path.open("r", encoding="utf-8") as reader:
                            for line in reader:
                                line = line.strip()
                                if not line:
                                    continue
                                item = json.loads(line)
                                outputs.append(item)
                                done_indices.add(int(item["index"]))
                        print(f"[mmbench] resume from {jsonl_path}: {len(outputs)} done", flush=True)

            print(
                f"[mmbench] {ds_name}: {len(df)} total, {len(done_indices)} done, "
                f"{len(df) - len(done_indices)} remaining",
                flush=True,
            )

            with checkpoint_jsonl.open("a", encoding="utf-8") as ckpt_writer:
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"mmbench/{ds_name}", file=sys.stdout):
                    row_index = int(row["index"])
                    if row_index in done_indices:
                        continue

                    image_b64 = _resolve_image_blob(image_map, row_index)
                    if not image_b64:
                        # Skip rows without a usable image (rare).
                        continue
                    image_path = _decode_image(image_b64, image_dir, row_index=row_index)
                    options = _parse_options(row)
                    hint = None
                    if "hint" in row and not pd.isna(row["hint"]):
                        hint = str(row["hint"])

                    question_text = str(row["question"])
                    prompt = _build_prompt(question_text, options, hint)
                    payload = {
                        "backbone": backbone,
                        "task": "understanding",
                        "prompt": prompt,
                        "images": [image_path],
                        "params": request_params,
                        "metadata": {"index": row_index, "dataset": ds_name},
                    }
                    prediction = _extract_text(pipeline.run(payload))
                    gt = None
                    if "answer" in row and not pd.isna(row["answer"]):
                        gt = str(row["answer"]).strip().upper()
                    item = {
                        "index": row_index,
                        "question": question_text,
                        "options": options,
                        "prediction": prediction,
                        "gt_answer": gt,
                        "language": language,
                    }
                    outputs.append(item)
                    done_indices.add(row_index)
                    ckpt_writer.write(json.dumps(item) + "\n")
                    ckpt_writer.flush()
                    os.fsync(ckpt_writer.fileno())

                    if max_samples > 0 and len(outputs) >= max_samples:
                        break

            time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
            jsonl_path_out = out_dir / f"{ds_name}_{time_prefix}.jsonl"
            with jsonl_path_out.open("w", encoding="utf-8") as writer:
                for item in outputs:
                    writer.write(json.dumps(item) + "\n")

            if checkpoint_jsonl.exists():
                checkpoint_jsonl.unlink()

            summary[f"{ds_name}_output_jsonl"] = str(jsonl_path_out)

        if mode == "generate":
            print(f"[mmbench] generation phase done, outputs={out_dir}", flush=True)

        # Release generation GPU memory before loading the judge LLM.
        del pipeline
        import gc
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Phase 2: Circular scoring (VLMEvalKit-compatible) ──
    if run_score:
        judge: "_JudgeBundle | None" = None
        try:
            for ds_name in datasets:
                jsonl_path = None
                if f"{ds_name}_output_jsonl" in summary:
                    jsonl_path = Path(summary[f"{ds_name}_output_jsonl"])
                else:
                    jsonl_path = _find_latest_jsonl(out_dir, ds_name)
                if jsonl_path is None or not jsonl_path.exists():
                    raise FileNotFoundError(
                        f"No generation output found for {ds_name} in {out_dir}. "
                        f"Run generation phase first (mode: generate)."
                    )

                print(f"[mmbench] scoring {ds_name} from {jsonl_path}", flush=True)
                items: "list[dict[str, Any]]" = []
                with jsonl_path.open("r", encoding="utf-8") as reader:
                    for line in reader:
                        line = line.strip()
                        if not line:
                            continue
                        items.append(json.loads(line))

                # Re-hydrate missing keys from the source TSV when needed
                # (back-compat with JSONLs produced by the pre-rewrite code).
                dataset_path = dataset_paths[ds_name]
                df = pd.read_csv(dataset_path, sep="\t") if dataset_path.exists() else pd.DataFrame()
                if not df.empty:
                    df["index"] = df["index"].astype(int)
                    df_by_idx = df.set_index("index", drop=False)
                    language_default = str(DS_COLLECTIONS.get(ds_name, {}).get("language", "en"))
                    for item in items:
                        idx = int(item["index"])
                        row = df_by_idx.loc[idx] if idx in df_by_idx.index else None
                        if "options" not in item and row is not None:
                            item["options"] = _parse_options(row)
                        if "prediction" not in item:
                            item["prediction"] = item.get("response", item.get("answer", ""))
                        if "gt_answer" not in item and row is not None:
                            if "answer" in row and not pd.isna(row["answer"]):
                                item["gt_answer"] = str(row["answer"]).strip().upper()
                        if "language" not in item:
                            item["language"] = language_default
                        if "question" not in item and row is not None and "question" in row:
                            item["question"] = str(row["question"])

                # Prepare per-item `choices` (copy of options, used by can_infer).
                for item in items:
                    item["choices"] = dict(item.get("options", {}))

                # Group by g_index = index % 1e6 → circular rotation group
                groups: "dict[int, list[dict[str, Any]]]" = {}
                for item in items:
                    g = int(item["index"]) % 1_000_000
                    groups.setdefault(g, []).append(item)
                # Sort rotations inside each group by their original index so
                # rotation k=0 (smallest) comes first, matching VLMEvalKit's order.
                for g in groups:
                    groups[g].sort(key=lambda it: int(it["index"]))

                # Fast-path: resolve as many groups as possible with can_infer only.
                group_results: "dict[int, dict[str, Any]]" = {}
                pending_groups: "list[tuple[int, list[dict[str, Any]]]]" = []
                for g, rows in groups.items():
                    pre_res, _gts, _preds = _prefetch_circular_group(rows)
                    if pre_res is not None:
                        group_results[g] = pre_res
                    else:
                        pending_groups.append((g, rows))

                # Load the judge once if any groups still need LLM extraction.
                if pending_groups and llm_model_path and judge is None:
                    judge = _JudgeBundle(llm_model_path, max_new_tokens=llm_max_new_tokens)

                if pending_groups:
                    print(
                        f"[mmbench] {ds_name}: {len(group_results)} groups decided by can_infer, "
                        f"{len(pending_groups)} need LLM judging",
                        flush=True,
                    )
                    for g, rows in tqdm(pending_groups, desc=f"mmbench/{ds_name}/judge", file=sys.stdout):
                        group_results[g] = _eval_circular_group(rows, judge=judge)
                else:
                    print(
                        f"[mmbench] {ds_name}: all {len(group_results)} groups decided by can_infer",
                        flush=True,
                    )

                df_scored: pd.DataFrame
                if not df.empty:
                    df_work = df.copy()
                    df_work["g_index"] = df_work["index"].astype(int) % 1_000_000
                    df_scored = df_work[df_work["index"] == df_work["g_index"]].copy()
                    expected_g = set(int(g) for g in df_scored["g_index"].tolist())
                    missing = expected_g - set(group_results.keys())
                    if missing:
                        raise RuntimeError(
                            f"[mmbench] {ds_name}: {len(missing)} groups in TSV missing from "
                            f"generation outputs (e.g. {sorted(missing)[:5]}). "
                            f"Generation phase skipped some rows; rerun with mode: full or "
                            f"mode: generate to refill the JSONL before scoring."
                        )
                    df_scored["hit"] = df_scored["g_index"].map(lambda g: group_results[int(g)]["hit"])
                    df_scored["log"] = df_scored["g_index"].map(lambda g: group_results[int(g)]["log"])
                else:
                    # No TSV available (unlikely): fall back to a minimal frame.
                    rows = []
                    for g, rows_list in groups.items():
                        head = rows_list[0]
                        rows.append(
                            {
                                "index": head["index"],
                                "g_index": g,
                                "hit": group_results[g]["hit"],
                            }
                        )
                    df_scored = pd.DataFrame(rows)

                metrics = _report_acc(df_scored)
                overall_vals = [v for k, v in metrics.items() if k.endswith("|Overall")]
                overall_acc_pct = round(100.0 * (overall_vals[0] if overall_vals else 0.0), 2)
                question_count = int(len(df_scored))
                hit_count = int(df_scored["hit"].sum()) if "hit" in df_scored.columns else 0
                # Write the annotated items (now with extraction+hit) back to JSONL.
                # Each item gets its group's hit attached for easy inspection.
                for item in items:
                    g = int(item["index"]) % 1_000_000
                    res = group_results.get(g, {"hit": 0, "log": ""})
                    item["group_hit"] = int(res.get("hit", 0))
                with jsonl_path.open("w", encoding="utf-8") as writer:
                    for item in items:
                        writer.write(json.dumps(item) + "\n")

                score_path = jsonl_path.with_name(f"{jsonl_path.stem}_score.json")
                score_payload = {
                    "overall": {
                        "accuracy": overall_acc_pct,
                        "hit_count": hit_count,
                        "question_count": question_count,
                        "mode": "circular",
                    },
                    "metrics": {k: round(100.0 * v, 2) for k, v in metrics.items()},
                }
                score_path.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
                print(
                    f"[mmbench] {ds_name} Overall Acc = {overall_acc_pct}% "
                    f"({hit_count}/{question_count} groups)",
                    flush=True,
                )

                # Persist the flat acc CSV too for parity with VLMEvalKit.
                if not df_scored.empty:
                    csv_path = jsonl_path.with_name(f"{jsonl_path.stem}_acc.csv")
                    acc_rows = [{"metric": k, "accuracy": round(100.0 * v, 2)} for k, v in metrics.items()]
                    pd.DataFrame(acc_rows).to_csv(csv_path, index=False)
                    summary[f"{ds_name}_acc_csv"] = str(csv_path)

                # Optional xlsx with predictions filled in (requires openpyxl).
                # Wise image ships without openpyxl; if missing, silently skip.
                is_mmbench_schema = ds_name.startswith("mmbench_") or ds_name.startswith("MMBench_")
                if not df.empty and is_mmbench_schema:
                    try:
                        import openpyxl  # noqa: F401
                        xlsx_path = jsonl_path.with_suffix(".xlsx")
                        cur_df = df.copy()
                        drop_cols = [c for c in ("hint", "category", "source", "image", "comment", "l2-category") if c in cur_df.columns]
                        if drop_cols:
                            cur_df = cur_df.drop(columns=drop_cols)
                        # For circular V11 the xlsx still lists all 4 rotations; each row
                        # gets the extracted letter from the corresponding item.
                        pred_by_idx: "dict[int, str]" = {}
                        for item in items:
                            pred_by_idx[int(item["index"])] = str(item.get("extraction", item.get("prediction", "")))
                        cur_df["prediction"] = cur_df["index"].map(lambda k: pred_by_idx.get(int(k), ""))
                        cur_df.to_excel(xlsx_path, index=False, engine="openpyxl")
                        summary[f"{ds_name}_xlsx"] = str(xlsx_path)
                    except ImportError:
                        print("[mmbench] openpyxl unavailable, skipping xlsx export", flush=True)

                summary[f"{ds_name}_score_file"] = str(score_path)
                summary[f"{ds_name}_accuracy"] = overall_acc_pct
        finally:
            if judge is not None:
                judge.close()

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MMBench summary to {score_path}")

    print(f"[umm eval] completed MMBench (mode={mode}) for backbone={backbone}, outputs={out_dir}")
    return 0
