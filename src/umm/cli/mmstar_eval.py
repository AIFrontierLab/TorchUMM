from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from umm.core.config import load_config
from umm.inference import InferencePipeline


DS_COLLECTIONS = {
    "MMStar": {
        "root": "Lin-Chen/MMStar",
        "split": "val",
        "cache_dir": "data/MMStar",
        "max_new_tokens": 20,
    }
}


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


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mmstar_cfg = raw_cfg.get("mmstar", {}) if isinstance(raw_cfg.get("mmstar"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmstar_cfg, inference_cfg


def _build_prompt(question: str, answer_instruction: str) -> str:
    question = question.strip()
    if answer_instruction.lower() in question.lower():
        return question
    return f"{question}\n{answer_instruction}".strip()


def _parse_options(question: str) -> dict[str, str]:
    if "Options:" not in question:
        return {}
    option_text = question.split("Options:", 1)[1]
    matches = list(re.finditer(r"\b([A-H])\s*:\s*", option_text))
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(option_text)
        options[match.group(1).upper()] = option_text[start:end].strip().rstrip(",")
    return options


def _post_process(prediction: str, options: dict[str, str]) -> str:
    candidates = set(options) if options else set("ABCDEFGH")
    text = str(prediction or "")
    for pattern in (
        r"(?:final answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-H])\)?",
        r"^\s*\(?([A-H])\)?\s*(?:[:.)]|$)",
        r"\b([A-H])\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in candidates:
            return match.group(1).upper()
    lowered = text.lower()
    for letter, value in options.items():
        if str(value).strip().lower() in lowered:
            return letter
    return ""


def _safe_id(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"


def _write_image(sample: dict[str, Any], image_dir: Path, sample_id: str) -> str:
    image = sample.get("image")
    if isinstance(image, Image.Image):
        image_dir.mkdir(parents=True, exist_ok=True)
        out_path = image_dir / f"{_safe_id(sample_id)}.png"
        image.convert("RGB").save(out_path, format="PNG")
        return str(out_path)
    if isinstance(image, str):
        path = Path(image).expanduser()
        if path.exists():
            return str(path)
    filename = getattr(image, "filename", None)
    if filename:
        path = Path(str(filename)).expanduser()
        if path.exists():
            return str(path)
    raise ValueError(f"MMStar sample {sample_id} does not contain a usable image.")


def _load_outputs(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    outputs: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))
    return outputs


def _accuracy_summary(outputs: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    total = len(outputs)
    correct = sum(1 for item in outputs if bool(item.get("answer_ok")))
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "overall": {
            "accuracy": round(100.0 * correct / max(total, 1), 2),
            "correct": int(correct),
            "total": int(total),
        },
    }
    for key, label in (("category", "category"), ("l2_category", "l2_category")):
        grouped: dict[str, dict[str, int]] = {}
        for item in outputs:
            group_name = str(item.get(key) or "unknown")
            bucket = grouped.setdefault(group_name, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(bool(item.get("answer_ok")))
        summary[label] = {
            group: {
                "accuracy": round(100.0 * values["correct"] / max(values["total"], 1), 2),
                "correct": values["correct"],
                "total": values["total"],
            }
            for group, values in sorted(grouped.items())
        }
    return summary


def run_mmstar_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mmstar_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mmstar":
        raise ValueError(f"Expected `eval.benchmark: mmstar`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MMStar eval.")
    backbone = _normalize_backbone_name(backbone_raw)

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a dict when provided.")

    request_cfg = inference_cfg.get("request", {})
    request_params: dict[str, Any] = {}
    if isinstance(request_cfg, dict):
        params = request_cfg.get("params", {})
        if isinstance(params, dict):
            request_params = dict(params)

    datasets_value = mmstar_cfg.get("datasets", ["MMStar"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["MMStar"]
    if not datasets:
        raise ValueError("`mmstar.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mmstar_cfg.get("out_dir", f"output/mmstar/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mmstar_cfg.get("image_dir", out_dir / "images")), repo_root)
    cache_dir = _resolve_path(str(mmstar_cfg.get("cache_dir", "data/MMStar")), repo_root)
    score_output_path = mmstar_cfg.get("score_output_path")
    max_samples = int(mmstar_cfg.get("max_samples", 0) or 0)
    resume = bool(mmstar_cfg.get("resume", True))
    answer_instruction = str(
        mmstar_cfg.get(
            "answer_instruction",
            "Answer with the option's letter from the given choices directly.",
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    run_summary: dict[str, Any] = {
        "benchmark": "mmstar",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
    }

    for ds_name in datasets:
        entry = DS_COLLECTIONS.get(ds_name, {})
        dataset_root = str(mmstar_cfg.get("root", entry.get("root", "Lin-Chen/MMStar")))
        split = str(mmstar_cfg.get("split", entry.get("split", "val")))
        max_new_tokens = int(mmstar_cfg.get("max_new_tokens", entry.get("max_new_tokens", 20)))

        checkpoint_jsonl = out_dir / f"{ds_name}_checkpoint.jsonl"
        outputs = _load_outputs(checkpoint_jsonl) if resume else []
        done_ids = {str(item.get("index")) for item in outputs}
        print(f"[mmstar] loading {dataset_root} split={split} cache={cache_dir}", flush=True)
        dataset = load_dataset(dataset_root, split=split, cache_dir=str(cache_dir))
        print(f"[mmstar] {ds_name}: {len(dataset)} total, {len(done_ids)} done", flush=True)

        dataset_request_params = dict(request_params)
        if backbone == "bagel":
            dataset_request_params.setdefault("max_think_token_n", max_new_tokens)
        else:
            dataset_request_params.setdefault("max_new_tokens", max_new_tokens)

        with checkpoint_jsonl.open("a", encoding="utf-8") as writer:
            for row_idx, raw_sample in enumerate(tqdm(dataset, desc=f"mmstar/{ds_name}", file=sys.stdout)):
                sample = dict(raw_sample)
                sample_id = sample.get("index", row_idx)
                sample_id_str = str(sample_id)
                if sample_id_str in done_ids:
                    continue
                if max_samples > 0 and len(outputs) >= max_samples:
                    break

                question = str(sample["question"])
                options = _parse_options(question)
                prompt = _build_prompt(question, answer_instruction)
                image_path = _write_image(sample, image_dir=image_dir, sample_id=sample_id_str)

                payload = {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": prompt,
                    "images": [image_path],
                    "params": dataset_request_params,
                    "metadata": {"index": sample_id, "dataset": ds_name},
                }
                response = _extract_text(pipeline.run(payload))
                pred = _post_process(response, options)
                answer = str(sample.get("answer", "")).strip().upper()
                item = {
                    "index": sample_id,
                    "question": question,
                    "prediction": pred,
                    "raw_prediction": response,
                    "gt_answer": answer,
                    "answer_ok": pred == answer,
                    "category": sample.get("category"),
                    "l2_category": sample.get("l2_category"),
                    "meta_info": sample.get("meta_info"),
                }
                outputs.append(item)
                done_ids.add(sample_id_str)
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.flush()

        outputs.sort(key=lambda item: str(item.get("index", "")))
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_jsonl = out_dir / f"{ds_name}_{time_prefix}.jsonl"
        results_json = out_dir / f"{ds_name}_{time_prefix}.json"
        with results_jsonl.open("w", encoding="utf-8") as writer:
            for item in outputs:
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
        results_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

        score = _accuracy_summary(outputs, ds_name)
        score_path = out_dir / f"{ds_name}_{time_prefix}_score.json"
        score_path.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "score.json").write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")

        if checkpoint_jsonl.exists() and (max_samples <= 0 or len(outputs) >= min(max_samples, len(dataset))):
            checkpoint_jsonl.unlink()

        run_summary[f"{ds_name}_output_jsonl"] = str(results_jsonl)
        run_summary[f"{ds_name}_output_json"] = str(results_json)
        run_summary[f"{ds_name}_score_path"] = str(score_path)
        run_summary[f"{ds_name}_score"] = score
        overall = score["overall"]
        print(
            f"[mmstar] {ds_name}: accuracy={overall['accuracy']:.2f} "
            f"({overall['correct']}/{overall['total']})",
            flush=True,
        )

    if isinstance(score_output_path, str) and score_output_path:
        summary_path = _resolve_path(score_output_path, repo_root)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[umm eval] wrote MMStar summary to {summary_path}")

    print(f"[umm eval] completed MMStar for backbone={backbone}, outputs={out_dir}")
    return 0
