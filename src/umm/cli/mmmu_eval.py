from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from datasets import concatenate_datasets, load_dataset
from PIL import Image

from umm.core.config import load_config
from umm.inference import InferencePipeline
from umm.eval.internvl_chat.eval.mmmu import data_utils, eval_utils


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
    if isinstance(output, list):
        for item in output:
            text = _extract_text(item)
            if text:
                return text
    return ""


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mmmu_cfg = raw_cfg.get("mmmu", {}) if isinstance(raw_cfg.get("mmmu"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmmu_cfg, inference_cfg


def _load_mmmu_dataset(root: str, split: str, cache_dir: str | None) -> Any:
    datasets_list = []
    for subject in data_utils.CAT_SHORT2LONG.values():
        datasets_list.append(load_dataset(root, subject, split=split, cache_dir=cache_dir))
    return concatenate_datasets(datasets_list)


def _iter_images(images: Iterable[Any]) -> Iterable[Any]:
    for image in images:
        if image is None:
            continue
        yield image


def _coerce_image_paths(
    images: Iterable[Any],
    image_dir: Path,
    data_id: str,
    max_images: int,
) -> list[str]:
    paths: list[str] = []
    safe_id = data_id.replace("/", "_")
    for idx, image in enumerate(_iter_images(images)):
        if len(paths) >= max_images:
            break
        if isinstance(image, str):
            path = Path(image).expanduser()
            if path.exists():
                paths.append(str(path))
                continue
        filename = getattr(image, "filename", None)
        if filename:
            path = Path(str(filename)).expanduser()
            if path.exists():
                paths.append(str(path))
                continue
        if isinstance(image, Image.Image):
            image_dir.mkdir(parents=True, exist_ok=True)
            out_path = image_dir / f"{safe_id}_{idx}.png"
            image.save(out_path, format="PNG")
            paths.append(str(out_path))
    return paths


def _build_prompt(question: str, question_type: str, options: list[str], prompts: dict[str, str]) -> str:
    prompt_suffix = prompts.get(question_type, prompts["open"])
    if options:
        choice_list: list[str] = []
        multiple_choices = "ABCDEFGHIJKLM"
        for i, option in enumerate(options):
            choice_list.append(f"{multiple_choices[i]}. {option.strip()}")
        choice_txt = "\n".join(choice_list)
        prompt = f"{question.strip()}\n{choice_txt}\n{prompt_suffix}".strip()
    else:
        prompt = f"{question.strip()}\n{prompt_suffix}".strip()
    return prompt


def run_mmmu_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mmmu_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mmmu":
        raise ValueError(f"Expected `eval.benchmark: mmmu`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MMMU eval.")
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

    datasets_value = mmmu_cfg.get("datasets", ["MMMU_validation"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["MMMU_validation"]
    if not datasets:
        raise ValueError("`mmmu.datasets` must contain at least one dataset name.")

    prompt_cfg = {
        "multiple-choice": "Answer with the option's letter from the given choices directly.",
        "open": "Answer the question using a single word or phrase.",
    }
    user_prompts = mmmu_cfg.get("prompts")
    if isinstance(user_prompts, dict):
        prompt_cfg.update({str(k): str(v) for k, v in user_prompts.items()})

    dataset_root = str(mmmu_cfg.get("root", "MMMU/MMMU"))
    cache_dir = mmmu_cfg.get("cache_dir")
    cache_dir_path = _resolve_path(str(cache_dir), repo_root) if cache_dir else repo_root / "data" / "MMMU"

    out_dir = _resolve_path(str(mmmu_cfg.get("out_dir", f"output/mmmu/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mmmu_cfg.get("image_dir", out_dir / "images")), repo_root)
    score_output_path = mmmu_cfg.get("score_output_path")
    run_calculation = bool(mmmu_cfg.get("run_calculation", True))
    calculation_script = _resolve_path(
        str(mmmu_cfg.get("calculation_script", "src/umm/eval/internvl_chat/eval/mmmu/main_eval_only.py")),
        repo_root,
    )
    answer_path = _resolve_path(
        str(mmmu_cfg.get("answer_path", "src/umm/eval/internvl_chat/eval/mmmu/answer_dict_val.json")),
        repo_root,
    )
    max_samples = int(mmmu_cfg.get("max_samples", 0) or 0)
    max_images = int(mmmu_cfg.get("max_images", 1) or 1)
    seed = int(mmmu_cfg.get("seed", 0) or 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

    summary: dict[str, Any] = {
        "benchmark": "mmmu",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
    }

    random.seed(seed)
    for ds_name in datasets:
        split = "validation"
        if ds_name.endswith("test"):
            split = "test"
        elif ds_name.endswith("dev"):
            split = "dev"
        elif ds_name.endswith("validation"):
            split = "validation"

        dataset = _load_mmmu_dataset(
            root=dataset_root,
            split=split,
            cache_dir=str(cache_dir_path),
        )
        outputs = []
        for idx, sample in enumerate(dataset, start=1):
            data = data_utils.process_single_sample(sample)
            question = str(data["question"]).strip()
            question_type = str(data["question_type"])
            options = eval(data["options"]) if isinstance(data.get("options"), str) else data.get("options", [])
            if not isinstance(options, list):
                options = []

            prompt = _build_prompt(question, question_type, options, prompt_cfg)
            index2ans, all_choices = data_utils.get_multi_choice_info(options) if options else ({}, [])

            image_paths = _coerce_image_paths(
                data.get("image", []),
                image_dir=image_dir,
                data_id=str(data["id"]),
                max_images=max_images,
            )

            payload = {
                "backbone": backbone,
                "task": "understanding",
                "prompt": prompt,
                "images": image_paths,
                "params": request_params,
                "metadata": {"question_type": question_type, "data_id": str(data["id"])},
            }
            output = pipeline.run(payload)
            response = _extract_text(output)
            if question_type == "multiple-choice" and all_choices and index2ans:
                pred = eval_utils.parse_multi_choice_response(response, all_choices, index2ans)
            else:
                pred = response

            outputs.append(
                {
                    "question": question,
                    "answer": pred,
                    "gt_answers": data.get("answer"),
                    "data_id": data.get("id"),
                    "question_type": question_type,
                    "prompt": prompt,
                    "raw_response": response,
                }
            )

            if max_samples > 0 and idx >= max_samples:
                break

        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        output_json = out_dir / f"{ds_name}_{time_prefix}.json"
        output_jsonl = out_dir / f"{ds_name}_{time_prefix}.jsonl"

        output_dict = {item["data_id"]: item["answer"] for item in outputs}
        output_json.write_text(json.dumps(output_dict, indent=4), encoding="utf-8")
        with output_jsonl.open("w", encoding="utf-8") as writer:
            for item in outputs:
                writer.write(json.dumps(item) + "\n")

        summary[f"{ds_name}_output_path"] = str(output_json)
        summary[f"{ds_name}_output_jsonl"] = str(output_jsonl)

        if run_calculation and split == "validation":
            cmd = [
                sys.executable,
                str(calculation_script),
                "--output_path",
                str(output_json),
                "--answer_path",
                str(answer_path),
            ]
            proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
            print(proc.stdout)
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"MMMU calculation failed with return code {proc.returncode}")
            summary[f"{ds_name}_calculation_stdout"] = proc.stdout

    if isinstance(score_output_path, str) and score_output_path:
        score_path = _resolve_path(score_output_path, repo_root)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[umm eval] wrote MMMU summary to {score_path}")

    print(f"[umm eval] completed MMMU for backbone={backbone}, outputs={out_dir}")
    return 0
