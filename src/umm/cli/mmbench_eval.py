from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from umm.core.config import load_config
from umm.eval.distributed import (
    barrier,
    cleanup_distributed,
    cleanup_shards,
    load_shard_items,
    maybe_init_distributed,
    merge_shards,
    rank_shard_path,
    sum_across_ranks,
)
from umm.eval.runner import run_sharded_inference
from umm.inference import InferencePipeline


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


def _post_process(pred: str, option: dict[str, str]) -> str:
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    if len(pred) > 1 and pred[0] in option_candidate:
        return pred[0]
    if len(pred) > 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k
    return pred


def _load_eval_cfg(config_path: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_cfg = load_config(config_path)
    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    mmbench_cfg = raw_cfg.get("mmbench", {}) if isinstance(raw_cfg.get("mmbench"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmbench_cfg, inference_cfg


def _build_prompt(question: str, options: dict[str, str], hint: str | None, language: str) -> str:
    if hint:
        question = f"{hint}\n{question}"
    for key, item in options.items():
        question += f"\n{key}. {item}"
    if language == "cn":
        suffix = "请直接回答选项字母。"
    else:
        suffix = "Answer with the option's letter from the given choices directly."
    return f"{question}\n{suffix}".strip()


def _decode_image(image_b64: str, image_dir: Path, row_index: int) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    out_path = image_dir / f"{row_index}.png"
    image.save(out_path, format="PNG")
    return str(out_path)


def _get_dataset_paths(
    datasets: list[str],
    repo_root: Path,
    override_paths: dict[str, Any],
) -> dict[str, Path]:
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


def _options_for(row: pd.Series) -> dict[str, str]:
    options: dict[str, str] = {}
    for cand in ["A", "B", "C", "D", "E"]:
        if cand in row and not pd.isna(row[cand]):
            options[cand] = row[cand]
    return options


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
    request_params: dict[str, Any] = {}
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

    dataset_paths = _get_dataset_paths(
        datasets=datasets,
        repo_root=repo_root,
        override_paths=mmbench_cfg.get("dataset_paths", {}) if isinstance(mmbench_cfg.get("dataset_paths"), dict) else {},
    )

    dist_info = maybe_init_distributed()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

        summary: dict[str, Any] = {
            "benchmark": "mmbench",
            "backbone": backbone,
            "out_dir": str(out_dir),
            "datasets": datasets,
            "world_size": dist_info.world_size,
        }

        if dist_info.world_size > 1:
            print(
                f"[mmbench] distributed inference enabled: rank={dist_info.rank}, "
                f"local_rank={dist_info.local_rank}, world_size={dist_info.world_size}",
                flush=True,
            )

        local_total_written = 0
        for ds_name in datasets:
            dataset_path = dataset_paths[ds_name]
            if not dataset_path.exists():
                raise FileNotFoundError(f"MMBench dataset not found: {dataset_path}")

            entry = DS_COLLECTIONS.get(ds_name, {})
            language = str(entry.get("language", "en"))
            df = pd.read_csv(dataset_path, sep="\t")

            checkpoint_jsonl = out_dir / f"{ds_name}_checkpoint.jsonl"
            shard_path = rank_shard_path(checkpoint_jsonl, dist_info.rank, dist_info.world_size)

            done_indices: set[Any] = set()
            if resume:
                shard_items = load_shard_items(shard_path)
                if shard_items:
                    done_indices = {int(item["index"]) for item in shard_items}
                    print(
                        f"[mmbench] {ds_name}: rank {dist_info.rank} resume from "
                        f"checkpoint: {len(done_indices)} done",
                        flush=True,
                    )
                elif dist_info.world_size <= 1:
                    # Single-card fallback: recover from a previously-completed
                    # output JSONL. Multi-card runs rely on the rank shard only.
                    jsonl_path: Path | None = None
                    if isinstance(resume_jsonl, str) and resume_jsonl:
                        jsonl_path = _resolve_path(resume_jsonl, repo_root)
                    else:
                        candidates = sorted(out_dir.glob(f"{ds_name}_*.jsonl"))
                        candidates = [c for c in candidates if "_checkpoint" not in c.name]
                        if candidates:
                            jsonl_path = max(candidates, key=lambda p: p.stat().st_mtime)
                    if jsonl_path and jsonl_path.exists():
                        prior = load_shard_items(jsonl_path)
                        done_indices = {int(item["index"]) for item in prior}
                        print(
                            f"[mmbench] resume from {jsonl_path}: "
                            f"{len(done_indices)} done",
                            flush=True,
                        )

            assigned_total = (
                (len(df) + dist_info.world_size - 1 - dist_info.rank) // dist_info.world_size
                if dist_info.world_size > 1
                else len(df)
            )
            print(
                f"[mmbench] {ds_name}: total={len(df)}, rank={dist_info.rank}, "
                f"assigned={assigned_total}, done={len(done_indices)}, "
                f"remaining={max(0, assigned_total - len(done_indices))}",
                flush=True,
            )

            def payload_fn(pair: tuple[Any, pd.Series]) -> dict[str, Any]:
                _, row = pair
                row_index = int(row["index"])
                image_path = _decode_image(str(row["image"]), image_dir, row_index=row_index)
                options = _options_for(row)
                hint = None
                if "hint" in row and not pd.isna(row["hint"]):
                    hint = row["hint"]
                question = _build_prompt(str(row["question"]), options, hint, language=language)
                return {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": question,
                    "images": [image_path],
                    "params": request_params,
                    "metadata": {"index": row_index, "dataset": ds_name},
                }

            def record_fn(pair: tuple[Any, pd.Series], raw: Any, _idx: int) -> dict[str, Any]:
                _, row = pair
                row_index = int(row["index"])
                options = _options_for(row)
                hint = None
                if "hint" in row and not pd.isna(row["hint"]):
                    hint = row["hint"]
                question = _build_prompt(str(row["question"]), options, hint, language=language)
                response = _extract_text(raw)
                pred = _post_process(response, options)
                return {
                    "question": question,
                    "answer": pred,
                    "gt_answers": row["answer"] if "answer" in row else None,
                    "index": row_index,
                }

            n_written = run_sharded_inference(
                infer_fn=pipeline.run,
                dist_info=dist_info,
                shard_path=shard_path,
                samples=df.iterrows(),
                total=len(df),
                payload_fn=payload_fn,
                record_fn=record_fn,
                sample_id_fn=lambda pair: int(pair[1]["index"]),
                done_ids=done_indices,
                max_samples=max_samples,
                log_prefix=f"mmbench/{ds_name}/rank{dist_info.rank}",
            )
            local_total_written += n_written

            barrier(dist_info)

            time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
            results_file = f"{ds_name}_{time_prefix}.xlsx"
            output_path = out_dir / results_file
            jsonl_path_out = out_dir / f"{ds_name}_{time_prefix}.jsonl"

            if dist_info.rank == 0:
                merged_outputs = merge_shards(checkpoint_jsonl)

                cur_df = df.copy()
                if "mmbench" in ds_name:
                    cur_df = cur_df.drop(columns=["hint", "category", "source", "image", "comment", "l2-category"])
                    cur_df.insert(6, "prediction", None)
                else:
                    cur_df = cur_df.drop(columns=["category", "image"])
                    cur_df.insert(8, "prediction", None)

                for item in merged_outputs:
                    cur_df.loc[df["index"] == item["index"], "prediction"] = item["answer"]

                cur_df.to_excel(output_path, index=False, engine="openpyxl") # pip install openpyxl
                with jsonl_path_out.open("w", encoding="utf-8") as writer:
                    for item in merged_outputs:
                        writer.write(json.dumps(item) + "\n")

                cleanup_shards(checkpoint_jsonl)
                if dist_info.world_size <= 1 and checkpoint_jsonl.exists():
                    checkpoint_jsonl.unlink()

                summary[f"{ds_name}_output_path"] = str(output_path)
                summary[f"{ds_name}_output_jsonl"] = str(jsonl_path_out)

            barrier(dist_info)

        total_written_all = sum_across_ranks(local_total_written, dist_info)
        if dist_info.rank != 0:
            print(
                f"[umm eval] rank {dist_info.rank} finished MMBench shard: "
                f"samples_written={local_total_written}",
                flush=True,
            )
            return 0

        summary["samples_written"] = total_written_all
        if isinstance(score_output_path, str) and score_output_path:
            score_path = _resolve_path(score_output_path, repo_root)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[umm eval] wrote MMBench summary to {score_path}")

        print(
            f"[umm eval] completed MMBench for backbone={backbone}, outputs={out_dir}, "
            f"samples_written={total_written_all}, world_size={dist_info.world_size}"
        )
        return 0
    finally:
        cleanup_distributed(dist_info)
