from __future__ import annotations

import argparse
import base64
import json
import os
import re
import string
import subprocess
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from umm.backbones.bagel.adapter import BagelBackbone
from umm.core.config import load_config
from umm.post_training.unipath.planner.bagel_lora_runtime import install_lora_adapter_for_bagel
from umm.post_training.unipath.planner.bagel_features import build_online_planner_feature, path_prompt
from umm.post_training.unipath.planner.models import PATHS, load_planner, planner_probs, slice_bagel_features
from umm.post_training.unipath.planner.qfcp_policy import (
    QFCP_REFINED_PRAGMATIC_SAFE_POLICY,
    build_mmmu_query,
    choose_qfcp_refined_pragmatic_safe_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online UniPath planner routing evaluation for understanding datasets.")
    parser.add_argument("--config", default="", help="TorchUMM eval config. Overrides are intentionally minimal.")
    parser.add_argument("--benchmark", choices=["mmmu", "mmbench", "mme", "mathvista", "mmstar"], default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--planner-path", default="")
    parser.add_argument("--policy", choices=["qfcp_refined_pragmatic_safe", "raw_planner"], default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--bagel-root", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_path(path_str: str | Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _get_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "answer", "response", "output", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        for container_key in ("results", "understandings"):
            container = output.get(container_key)
            if isinstance(container, dict):
                text = _get_text(container)
                if text:
                    return text
            if isinstance(container, list):
                for item in container:
                    text = _get_text(item)
                    if text:
                        return text
    if isinstance(output, list):
        for item in output:
            text = _get_text(item)
            if text:
                return text
    return ""


def extract_answer_text(text: str) -> str:
    raw = str(text or "")
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]
    match = re.search(r"(?:final answer|answer)\s*[:：]\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else raw.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _default_config() -> dict[str, Any]:
    return {
        "benchmark": "mmmu",
        "policy": "qfcp_refined_pragmatic_safe",
        "query_mode": "dataset",
        "model_path": os.environ.get("UNIPATH_BAGEL_MODEL_PATH", ""),
        "adapter_dir": os.environ.get("UNIPATH_BAGEL_ADAPTER_DIR", ""),
        "planner_path": os.environ.get("UNIPATH_PLANNER_PATH", ""),
        "bagel_root": "src/umm/backbones/bagel/Bagel",
        "output_dir": "outputs/unipath_online_route",
        "cache_dir": "",
        "offload_folder": "tmp/unipath_online_route_offload",
        "max_mem_per_gpu": "80GiB",
        "max_new_tokens": 2048,
        "max_samples": 0,
        "skip_samples": 0,
        "num_shards": 1,
        "shard_index": 0,
        "progress_every": 25,
        "seed": 42,
        "dry_run": False,
    }


def _merge_resolved(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, str) and "${" in value:
            continue
        dst[key] = value


def _load_config(config_path: str, cli_args: argparse.Namespace | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    cfg = _default_config()
    raw: dict[str, Any] = {}
    if config_path:
        raw = load_config(config_path)
        route_cfg = raw.get("online_route")
        if not isinstance(route_cfg, dict):
            route_cfg = raw.get("unipath_online_route")
        if isinstance(route_cfg, dict):
            _merge_resolved(cfg, route_cfg)
        inference_cfg = raw.get("inference", {}) if isinstance(raw.get("inference"), dict) else {}
        backbone_cfg = inference_cfg.get("backbone_cfg", {}) if isinstance(inference_cfg.get("backbone_cfg"), dict) else {}
        for key in ("model_path", "bagel_root", "offload_folder", "max_mem_per_gpu", "seed", "adapter_dir"):
            if key in backbone_cfg and backbone_cfg[key] not in (None, "") and not (isinstance(backbone_cfg[key], str) and "${" in backbone_cfg[key]):
                cfg[key] = backbone_cfg[key]
        request = inference_cfg.get("request", {}) if isinstance(inference_cfg.get("request"), dict) else {}
        params = request.get("params", {}) if isinstance(request.get("params"), dict) else {}
        if "max_think_token_n" in params:
            cfg["max_new_tokens"] = params["max_think_token_n"]

    if cli_args is not None:
        for key in ("benchmark", "output_dir", "planner_path", "policy", "model_path", "adapter_dir", "bagel_root"):
            value = getattr(cli_args, key, None)
            if value not in (None, ""):
                cfg[key] = value
        if cli_args.max_samples is not None:
            cfg["max_samples"] = cli_args.max_samples
        if cli_args.dry_run:
            cfg["dry_run"] = True

    for key in ("model_path", "adapter_dir", "planner_path", "bagel_root", "output_dir", "cache_dir", "offload_folder"):
        value = str(cfg.get(key, "") or "")
        if value and "${" not in value:
            cfg[key] = str(_resolve_path(value, repo_root))
    return cfg


def keep_sample(position_zero_based: int, cfg: dict[str, Any]) -> bool:
    num_shards = int(cfg.get("num_shards", 1) or 1)
    shard_index = int(cfg.get("shard_index", 0) or 0)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return num_shards == 1 or position_zero_based % num_shards == shard_index


def _choose_margin_path(scores: dict[str, float], margin: float) -> str:
    best = max(float(v) for v in scores.values())
    candidates = [path for path in PATHS if float(scores[path]) >= best - margin]
    return min(candidates, key=lambda path: PATHS.index(path))


def _prepare_feature_for_planner(feature: torch.Tensor, payload: dict[str, Any]) -> torch.Tensor:
    feature_layout = str(payload.get("feature_layout") or "")
    input_dim = int(payload["input_dim"])
    if feature_layout in {"all_abs", "direct_plus_delta"}:
        x = feature.float()
    else:
        mode = str(payload.get("bagel_feature_mode") or "joint_text_mean_last_image")
        x = slice_bagel_features(feature.unsqueeze(0), mode).squeeze(0)
    if int(x.numel()) != input_dim:
        raise ValueError(f"Planner feature dim mismatch: got {x.numel()}, expected {input_dim}")
    return x


class OnlineRouter:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.output_dir = Path(str(cfg["output_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cases_path = self.output_dir / "cases.jsonl"
        self.policy = str(cfg.get("policy") or "qfcp_refined_pragmatic_safe")
        self.model, self.planner_payload = load_planner(cfg["planner_path"], device="cpu")
        self.margin = float(self.planner_payload.get("margin", 0.03))
        self.feature_layout = str(self.planner_payload.get("feature_layout") or "")
        self.path_counts: Counter[str] = Counter()
        self.bucket_counts: Counter[str] = Counter()
        self.selector_counts: Counter[str] = Counter()
        self.cache_dir = Path(str(cfg.get("cache_dir") or self.output_dir / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.planner_cache_path = self.cache_dir / "planner_predictions.jsonl"
        self.response_cache_path = self.cache_dir / "responses.jsonl"
        self.planner_cache = {str(row["sample_key"]): row for row in _read_jsonl(self.planner_cache_path) if "sample_key" in row}
        self.response_cache = {
            (str(row["sample_key"]), str(row["path"])): row for row in _read_jsonl(self.response_cache_path) if "sample_key" in row and "path" in row
        }

        if bool(cfg.get("dry_run", False)):
            self.backbone = None
            self.inferencer = None
            return

        self.backbone = BagelBackbone(
            model_path=str(cfg["model_path"]),
            bagel_root=str(cfg.get("bagel_root") or ""),
            max_mem_per_gpu=str(cfg.get("max_mem_per_gpu") or "80GiB"),
            offload_folder=str(cfg.get("offload_folder") or self.output_dir / "offload"),
            seed=int(cfg.get("seed", 42)),
        )
        self.backbone.load(
            {
                "understanding_cfg": {"max_think_token_n": int(cfg.get("max_new_tokens", 2048)), "do_sample": False},
            }
        )
        self.inferencer = self.backbone.inferencer
        if self.inferencer is None:
            raise RuntimeError("BAGEL inferencer is not initialized.")
        adapter_dir = str(cfg.get("adapter_dir") or "")
        if adapter_dir:
            self.inferencer.model = install_lora_adapter_for_bagel(self.inferencer.model, adapter_dir)

    def planner_scores(self, sample_key: str, query: str, image_paths: list[str]) -> dict[str, float]:
        cached = self.planner_cache.get(sample_key)
        if cached is not None:
            return {path: float(cached["planner_scores"][path]) for path in PATHS}
        if self.inferencer is None:
            raise RuntimeError("Dry run has no inferencer; cannot compute planner scores.")
        feature = build_online_planner_feature(self.inferencer, query, image_paths, feature_layout=self.feature_layout)
        x = _prepare_feature_for_planner(feature, self.planner_payload).unsqueeze(0)
        with torch.no_grad():
            probs = planner_probs(self.model, x)[0].detach().cpu()
        scores = {PATHS[idx]: float(probs[idx].item()) for idx in range(len(PATHS))}
        row = {"sample_key": sample_key, "planner_scores": scores, "feature_layout": self.feature_layout}
        self.planner_cache[sample_key] = row
        _append_jsonl(self.planner_cache_path, row)
        return scores

    def select_path(self, scores: dict[str, float], query: str) -> tuple[str, str, str]:
        default_path = _choose_margin_path(scores, self.margin)
        if self.policy == "raw_planner":
            return default_path, "raw_planner", "raw_planner"
        selected, bucket, selector = choose_qfcp_refined_pragmatic_safe_path(scores, query)
        return selected, bucket, selector

    def _understand(self, prompt: str, image_paths: list[str]) -> str:
        if self.backbone is None:
            raise RuntimeError("Dry run has no backbone; cannot run understanding.")
        result = self.backbone.understanding(
            prompt=prompt,
            images=image_paths,
            understanding_cfg={
                "think": False,
                "do_sample": False,
                "max_think_token_n": int(self.cfg.get("max_new_tokens", 2048)),
            },
        )
        return _get_text(result)

    def generate(self, sample_key: str, query: str, image_paths: list[str]) -> dict[str, Any]:
        scores = self.planner_scores(sample_key, query, image_paths)
        selected_path, bucket, selector = self.select_path(scores, query)
        cached = self.response_cache.get((sample_key, selected_path))
        if cached is not None:
            raw_response = str(cached.get("raw_response") or "")
            answer_text = str(cached.get("answer_text") or extract_answer_text(raw_response))
        else:
            prompt = path_prompt(selected_path, query)
            raw_response = self._understand(prompt, image_paths)
            answer_text = extract_answer_text(raw_response)
            cached = {
                "sample_key": sample_key,
                "path": selected_path,
                "prompt_used": prompt,
                "raw_response": raw_response,
                "answer_text": answer_text,
            }
            self.response_cache[(sample_key, selected_path)] = cached
            _append_jsonl(self.response_cache_path, cached)
        self.path_counts[selected_path] += 1
        self.bucket_counts[bucket] += 1
        self.selector_counts[selector] += 1
        return {
            "selected_path": selected_path,
            "bucket": bucket,
            "selector": selector,
            "planner_scores": scores,
            "prompt_used": cached.get("prompt_used") or path_prompt(selected_path, query),
            "raw_response": raw_response,
            "answer_text": answer_text,
        }


def _save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.convert("RGB").save(path, format="PNG")
    return str(path)


def _run_mmmu(cfg: dict[str, Any], router: OnlineRouter) -> dict[str, Any]:
    from datasets import concatenate_datasets, load_dataset

    from umm.cli.mmmu_eval import _coerce_image_paths
    from umm.eval.internvl_chat.eval.mmmu import data_utils, eval_utils

    root = str(cfg.get("mmmu_root") or cfg.get("root") or "MMMU/MMMU")
    split = str(cfg.get("mmmu_split") or cfg.get("split") or "validation")
    cache_dir = cfg.get("mmmu_cache_dir") or cfg.get("dataset_cache_dir")
    datasets_list = [load_dataset(root, subject, split=split, cache_dir=cache_dir) for subject in data_utils.CAT_SHORT2LONG.values()]
    dataset = concatenate_datasets(datasets_list)
    image_dir = Path(str(cfg.get("image_dir") or router.output_dir / "images"))
    correct = 0
    cases = 0
    started = time.time()
    max_samples = int(cfg.get("max_samples", 0) or 0)
    skip_samples = int(cfg.get("skip_samples", 0) or 0)
    max_images = int(cfg.get("mmmu_max_images", 1) or 1)
    for idx, sample in enumerate(dataset, start=1):
        if idx <= skip_samples or not keep_sample(idx - 1, cfg):
            continue
        data = data_utils.process_single_sample(sample)
        options = eval(data["options"]) if isinstance(data.get("options"), str) else data.get("options", [])
        options = options if isinstance(options, list) else []
        query = build_mmmu_query(str(data["question"]), str(data["question_type"]), options)
        image_paths = _coerce_image_paths(data.get("image", []), image_dir=image_dir, data_id=str(data["id"]), max_images=max_images)
        if not image_paths:
            continue
        generated = router.generate(f"mmmu:{data['id']}", query, image_paths[:1])
        index2ans, all_choices = data_utils.get_multi_choice_info(options) if options else ({}, [])
        if str(data["question_type"]) == "multiple-choice" and all_choices:
            pred = eval_utils.parse_multi_choice_response(generated["answer_text"], all_choices, index2ans)
        else:
            pred = generated["answer_text"]
        ok = str(pred).strip() == str(data.get("answer")).strip()
        correct += int(ok)
        cases += 1
        _append_jsonl(
            router.cases_path,
            {
                "dataset": "mmmu",
                "data_id": data["id"],
                "question": data["question"],
                "options": options,
                "gt_answer": data.get("answer"),
                "pred_answer": pred,
                "answer_ok": ok,
                "image_paths": image_paths[:1],
                **generated,
            },
        )
        if cases % max(int(cfg.get("progress_every", 25)), 1) == 0:
            print(f"[online_route:mmmu] {cases} acc={correct / max(cases, 1):.4f} paths={dict(router.path_counts)}", flush=True)
        if max_samples > 0 and cases >= max_samples:
            break
    return {"benchmark": "mmmu", "split": split, "num_samples": cases, "correct": correct, "accuracy": correct / max(cases, 1), "elapsed_seconds": time.time() - started}


def _mmbench_options(row: pd.Series) -> dict[str, str]:
    return {c: str(row[c]) for c in string.ascii_uppercase if c in row and not pd.isna(row[c])}


def _decode_mmbench_image(value: str, image_dir: Path, row_index: int) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_b64 = str(value).strip()
    if "," in image_b64 and image_b64.lower().startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    image_b64 += "=" * (-len(image_b64) % 4)
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    path = image_dir / f"{row_index}.png"
    if not path.exists():
        image.save(path, format="PNG")
    return str(path)


def _run_mmbench(cfg: dict[str, Any], router: OnlineRouter) -> dict[str, Any]:
    from umm.cli.mmbench_eval import DS_COLLECTIONS, _build_prompt, _can_infer

    dataset_name = str(cfg.get("dataset") or "mmbench_dev_20230712")
    data_path = cfg.get("data_path") or cfg.get("dataset_path")
    if not data_path:
        data_path = DS_COLLECTIONS[dataset_name]["root"]
    data_path = Path(str(data_path))
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parents[4] / data_path
    df = pd.read_csv(data_path, sep="\t")
    image_dir = Path(str(cfg.get("image_dir") or router.output_dir / "images"))
    image_map = {int(row["index"]): str(row["image"]) for _, row in df.iterrows() if "image" in row and not pd.isna(row["image"])}
    correct = 0
    total = 0
    outputs: list[dict[str, Any]] = []
    max_samples = int(cfg.get("max_samples", 0) or 0)
    started = time.time()
    for pos, (_, row) in enumerate(df.iterrows()):
        if not keep_sample(pos, cfg):
            continue
        row_index = int(row["index"])
        image_value = image_map.get(row_index, str(row.get("image", "")))
        if len(str(image_value)) <= 64:
            try:
                image_value = image_map.get(int(str(image_value)), str(image_value))
            except ValueError:
                pass
        image_path = _decode_mmbench_image(str(image_value), image_dir, row_index)
        options = _mmbench_options(row)
        hint = str(row["hint"]) if "hint" in row and not pd.isna(row["hint"]) else None
        query = _build_prompt(str(row["question"]), options, hint)
        generated = router.generate(f"mmbench:{dataset_name}:{row_index}", query, [image_path])
        pred = _can_infer(generated["answer_text"], options) or "Z"
        gt = str(row["answer"]).strip() if "answer" in row and not pd.isna(row["answer"]) else ""
        ok = bool(gt) and str(pred).strip() == gt
        if gt:
            total += 1
            correct += int(ok)
        case = {"dataset": dataset_name, "index": row_index, "question": query, "prediction": pred, "gt_answer": gt or None, "answer_ok": ok if gt else None, "image_path": image_path, **generated}
        outputs.append(case)
        _append_jsonl(router.cases_path, case)
        if len(outputs) % max(int(cfg.get("progress_every", 25)), 1) == 0:
            print(f"[online_route:mmbench] {len(outputs)} acc={correct / max(total, 1):.4f} paths={dict(router.path_counts)}", flush=True)
        if max_samples > 0 and len(outputs) >= max_samples:
            break
    out_jsonl = router.output_dir / f"{dataset_name}_predictions.jsonl"
    out_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in outputs) + ("\n" if outputs else ""), encoding="utf-8")
    return {"benchmark": "mmbench", "dataset": dataset_name, "num_samples": len(outputs), "correct": correct, "total_with_gt": total, "accuracy": correct / max(total, 1), "prediction_jsonl": str(out_jsonl), "elapsed_seconds": time.time() - started}


def _run_mme(cfg: dict[str, Any], router: OnlineRouter) -> dict[str, Any]:
    from umm.cli.mme_eval import _post_process

    repo_root = Path(__file__).resolve().parents[4]
    question_root = _resolve_path(str(cfg.get("question_root") or cfg.get("mme_root") or "src/umm/eval/internvl_chat/eval/mme/Your_Results"), repo_root)
    image_root = _resolve_path(str(cfg.get("image_root") or "data/mme/MME_Benchmark_release_version"), repo_root)
    prompt_suffix = str(cfg.get("prompt_suffix") or "Answer the question using a single word or phrase.")
    max_samples = int(cfg.get("max_samples", 0) or 0)
    samples = 0
    started = time.time()
    mme_out_dir = router.output_dir / "mme_results"
    mme_out_dir.mkdir(parents=True, exist_ok=True)
    for task_txt in sorted(question_root.glob("*.txt")):
        task_name = task_txt.stem
        with task_txt.open("r", encoding="utf-8") as fin, (mme_out_dir / task_txt.name).open("w", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin, start=1):
                row = line.strip().split("\t")
                if len(row) != 3:
                    continue
                img, question, gt = row
                image_path = image_root / task_name / img
                if not image_path.exists():
                    image_path = image_root / task_name / "images" / img
                if not image_path.exists():
                    continue
                query = f"{question} {prompt_suffix}".strip()
                key = f"mme:{task_name}:{img}:{line_idx}"
                generated = router.generate(key, query, [str(image_path)])
                response = _post_process(generated["answer_text"])
                print(img, query, gt, response, sep="\t", file=fout)
                fout.flush()
                _append_jsonl(router.cases_path, {"dataset": "mme", "task": task_name, "image": img, "question": query, "gt_answer": gt, "prediction": response, "image_path": str(image_path), **generated})
                samples += 1
                if samples % max(int(cfg.get("progress_every", 25)), 1) == 0:
                    print(f"[online_route:mme] {samples} paths={dict(router.path_counts)}", flush=True)
                if max_samples > 0 and samples >= max_samples:
                    break
        if max_samples > 0 and samples >= max_samples:
            break
    summary: dict[str, Any] = {"benchmark": "mme", "num_samples": samples, "mme_results_dir": str(mme_out_dir), "elapsed_seconds": time.time() - started}
    if bool(cfg.get("run_calculation", False)):
        script = _resolve_path(str(cfg.get("calculation_script") or "src/umm/eval/internvl_chat/eval/mme/calculation.py"), repo_root)
        proc = subprocess.run([sys.executable, str(script), "--results_dir", str(mme_out_dir)], cwd=str(repo_root), capture_output=True, text=True)
        summary.update({"calculation_stdout": proc.stdout, "calculation_stderr": proc.stderr, "calculation_returncode": proc.returncode})
        if proc.returncode != 0:
            raise RuntimeError(f"MME calculation failed: {proc.stderr}")
    return summary


def _run_mathvista(cfg: dict[str, Any], router: OnlineRouter) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_name = str(cfg.get("dataset") or "MathVista_testmini")
    root = str(cfg.get("mathvista_root") or cfg.get("root") or "AI4Math/MathVista")
    split = str(cfg.get("split") or ("testmini" if dataset_name == "MathVista_testmini" else "test"))
    cache_dir = cfg.get("dataset_cache_dir") or cfg.get("cache_dir_hf")
    data = load_dataset(root, cache_dir=cache_dir)[split]
    image_dir = Path(str(cfg.get("image_dir") or router.output_dir / "images"))
    results: dict[str, Any] = {}
    max_samples = int(cfg.get("max_samples", 0) or 0)
    started = time.time()
    for pos, item in enumerate(data):
        if not keep_sample(pos, cfg):
            continue
        pid = str(item["pid"])
        image = item.get("decoded_image")
        if not isinstance(image, Image.Image):
            continue
        image_path = _save_image(image, image_dir / f"{pid}.png")
        query = str(item["query"])
        generated = router.generate(f"mathvista:{dataset_name}:{pid}", query, [image_path])
        out = dict(item)
        out.pop("decoded_image", None)
        out.update({"response": generated["raw_response"], "extraction": extract_answer_text(generated["answer_text"]), "selected_path": generated["selected_path"], "planner_scores": generated["planner_scores"]})
        results[pid] = out
        _append_jsonl(router.cases_path, {"dataset": dataset_name, "pid": pid, "question": query, "image_path": image_path, **generated})
        if len(results) % max(int(cfg.get("progress_every", 25)), 1) == 0:
            print(f"[online_route:mathvista] {len(results)} paths={dict(router.path_counts)}", flush=True)
        if max_samples > 0 and len(results) >= max_samples:
            break
    out_json = router.output_dir / f"{dataset_name}_predictions.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary: dict[str, Any] = {"benchmark": "mathvista", "dataset": dataset_name, "split": split, "num_samples": len(results), "prediction_json": str(out_json), "elapsed_seconds": time.time() - started}
    gt_file = str(cfg.get("gt_file") or "")
    if gt_file:
        score_file = router.output_dir / f"{dataset_name}_score.json"
        repo_root = Path(__file__).resolve().parents[4]
        proc = subprocess.run(
            [
                sys.executable,
                "src/umm/eval/internvl_chat/eval/mathvista/calculate_score.py",
                "--output_file",
                out_json.name,
                "--output_dir",
                str(router.output_dir),
                "--score_file",
                score_file.name,
                "--gt_file",
                gt_file,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        summary.update({"score_file": str(score_file), "score_stdout": proc.stdout, "score_stderr": proc.stderr, "score_returncode": proc.returncode})
        if proc.returncode != 0:
            raise RuntimeError(f"MathVista scoring failed: {proc.stderr}")
    return summary


def _mmstar_query(question: str) -> str:
    suffix = "Answer with the option's letter from the given choices directly."
    return question if suffix.lower() in question.lower() else f"{question.strip()}\n{suffix}"


def _mmstar_options(question: str) -> dict[str, str]:
    if "Options:" not in question:
        return {}
    option_text = question.split("Options:", 1)[1]
    matches = list(re.finditer(r"\b([A-H])\s*:\s*", option_text))
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(option_text)
        out[match.group(1).upper()] = option_text[start:end].strip().rstrip(",")
    return out


def _extract_option(text: str, options: dict[str, str]) -> str:
    candidates = set(options) if options else set("ABCDEFGH")
    for pattern in (r"(?:final answer|answer|option|choice)\s*(?:is|:|：)?\s*\(?([A-H])\)?", r"^\s*\(?([A-H])\)?\s*(?:[:：.)]|$)", r"\b([A-H])\b"):
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match and match.group(1).upper() in candidates:
            return match.group(1).upper()
    lowered = str(text).lower()
    for letter, value in options.items():
        if str(value).strip().lower() in lowered:
            return letter
    return ""


def _run_mmstar(cfg: dict[str, Any], router: OnlineRouter) -> dict[str, Any]:
    from datasets import load_dataset

    root = str(cfg.get("mmstar_root") or cfg.get("root") or os.environ.get("UNIPATH_MMSTAR_ROOT") or "Lin-Chen/MMStar")
    split = str(cfg.get("split") or "val")
    data = load_dataset(root, split=split)
    image_dir = Path(str(cfg.get("image_dir") or router.output_dir / "images"))
    correct = 0
    total = 0
    max_samples = int(cfg.get("max_samples", 0) or 0)
    started = time.time()
    for pos, item in enumerate(data):
        if not keep_sample(pos, cfg):
            continue
        sample_index = int(item["index"])
        question = str(item["question"])
        query = _mmstar_query(question)
        image = item.get("image")
        if not isinstance(image, Image.Image):
            continue
        image_path = _save_image(image, image_dir / f"{sample_index}.png")
        generated = router.generate(f"mmstar:{split}:{sample_index}", query, [image_path])
        pred = _extract_option(generated["answer_text"], _mmstar_options(question))
        expected = str(item["answer"]).strip().upper()
        ok = pred == expected
        correct += int(ok)
        total += 1
        _append_jsonl(router.cases_path, {"dataset": "mmstar", "index": sample_index, "question": question, "prediction": pred, "gt_answer": expected, "answer_ok": ok, "image_path": image_path, **generated})
        if total % max(int(cfg.get("progress_every", 25)), 1) == 0:
            print(f"[online_route:mmstar] {total} acc={correct / max(total, 1):.4f} paths={dict(router.path_counts)}", flush=True)
        if max_samples > 0 and total >= max_samples:
            break
    return {"benchmark": "mmstar", "split": split, "num_samples": total, "correct": correct, "accuracy": correct / max(total, 1), "elapsed_seconds": time.time() - started}


RUNNERS = {
    "mmmu": _run_mmmu,
    "mmbench": _run_mmbench,
    "mme": _run_mme,
    "mathvista": _run_mathvista,
    "mmstar": _run_mmstar,
}


def run_online_route_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    benchmark = str(cfg.get("benchmark") or "").lower()
    if benchmark not in RUNNERS:
        raise ValueError(f"Unsupported online route benchmark: {benchmark}")
    router = OnlineRouter(cfg)
    if bool(cfg.get("dry_run", False)):
        summary = {
            "benchmark": benchmark,
            "dry_run": True,
            "planner_path": str(cfg.get("planner_path")),
            "model_path": str(cfg.get("model_path")),
            "adapter_dir": str(cfg.get("adapter_dir")),
            "policy": str(cfg.get("policy")),
            "output_dir": str(cfg.get("output_dir")),
            "feature_layout": str(router.planner_payload.get("feature_layout") or ""),
        }
        router.output_dir.mkdir(parents=True, exist_ok=True)
        (router.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    summary = RUNNERS[benchmark](cfg, router)
    summary.update(
        {
            "planner_path": str(cfg.get("planner_path")),
            "model_path": str(cfg.get("model_path")),
            "adapter_dir": str(cfg.get("adapter_dir")),
            "policy": router.policy,
            "qfcp_policy": QFCP_REFINED_PRAGMATIC_SAFE_POLICY if router.policy == "qfcp_refined_pragmatic_safe" else None,
            "path_counts": dict(router.path_counts),
            "bucket_counts": dict(router.bucket_counts),
            "selector_counts": dict(router.selector_counts),
            "cases_jsonl": str(router.cases_path),
            "planner_cache_jsonl": str(router.planner_cache_path),
            "response_cache_jsonl": str(router.response_cache_path),
        }
    )
    metrics_path = router.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def run_online_route_eval_command(args: Any) -> int:
    cfg = _load_config(str(args.config), None)
    run_online_route_eval(cfg)
    return 0


def main() -> int:
    args = parse_args()
    cfg = _load_config(args.config, args)
    run_online_route_eval(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
