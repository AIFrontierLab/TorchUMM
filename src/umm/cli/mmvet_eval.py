from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

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
    "mmvet": {
        "max_new_tokens": 1000,
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
    mmvet_cfg = raw_cfg.get("mmvet", {}) if isinstance(raw_cfg.get("mmvet"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mmvet_cfg, inference_cfg


def _normalize_output_key(question_id: Any) -> str:
    qid = str(question_id)
    return qid if qid.startswith("v1_") else f"v1_{qid}"


def run_mmvet_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mmvet_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mmvet":
        raise ValueError(f"Expected `eval.benchmark: mmvet`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MM-Vet eval.")
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

    datasets_value = mmvet_cfg.get("datasets", ["mmvet"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["mmvet"]
    if not datasets:
        raise ValueError("`mmvet.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mmvet_cfg.get("out_dir", f"output/mmvet/{backbone}")), repo_root)
    score_output_path = mmvet_cfg.get("score_output_path")
    max_samples = int(mmvet_cfg.get("max_samples", 0) or 0)

    dataset_paths = mmvet_cfg.get("dataset_paths", {})
    if not isinstance(dataset_paths, dict):
        dataset_paths = {}

    dist_info = maybe_init_distributed()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

        summary: dict[str, Any] = {
            "benchmark": "mmvet",
            "backbone": backbone,
            "out_dir": str(out_dir),
            "datasets": datasets,
            "world_size": dist_info.world_size,
        }

        if dist_info.world_size > 1:
            print(
                f"[mmvet] distributed inference enabled: rank={dist_info.rank}, "
                f"local_rank={dist_info.local_rank}, world_size={dist_info.world_size}",
                flush=True,
            )

        local_total_written = 0
        for ds_name in datasets:
            entry = DS_COLLECTIONS.get(ds_name)
            if not entry and ds_name not in dataset_paths:
                raise ValueError(f"Unknown MM-Vet dataset: {ds_name}")
            image_root_value = dataset_paths.get("image_root")
            question_value = dataset_paths.get("question")
            if not image_root_value or not question_value:
                raise ValueError(
                    "MM-Vet requires `mmvet.dataset_paths.image_root` and "
                    "`mmvet.dataset_paths.question` to be set in the YAML config."
                )
            image_root = _resolve_path(str(image_root_value), repo_root)
            question_path = _resolve_path(str(question_value), repo_root)
            if not image_root.exists():
                raise FileNotFoundError(f"MM-Vet image root not found: {image_root}")
            if not question_path.exists():
                raise FileNotFoundError(f"MM-Vet question file not found: {question_path}")

            checkpoint_jsonl = out_dir / f"{ds_name}_checkpoint.jsonl"
            shard_path = rank_shard_path(checkpoint_jsonl, dist_info.rank, dist_info.world_size)

            done_keys = {
                str(it.get("output_key", "")) for it in load_shard_items(shard_path)
            }
            if done_keys:
                print(
                    f"[mmvet] {ds_name}: rank {dist_info.rank} resuming after "
                    f"{len(done_keys)} shard items",
                    flush=True,
                )

            lines = [l.strip() for l in question_path.read_text("utf-8").splitlines() if l.strip()]
            print(
                f"[mmvet] {ds_name}: total={len(lines)}, rank={dist_info.rank}, "
                f"done={len(done_keys)}",
                flush=True,
            )

            def iter_rows() -> Iterator[dict[str, Any]]:
                for line in lines:
                    yield json.loads(line)

            def payload_fn(row: dict[str, Any]) -> dict[str, Any]:
                image_name = row["image"]
                question = row["text"]
                question_id = row["question_id"]
                image_path = image_root / image_name
                if not image_path.exists():
                    raise FileNotFoundError(f"MM-Vet image not found: {image_path}")
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as exc:
                    raise RuntimeError(f"Failed to open image {image_path}: {exc}") from exc
                return {
                    "backbone": backbone,
                    "task": "understanding",
                    "prompt": question,
                    "images": [str(image_path)],
                    "params": request_params,
                    "metadata": {"question_id": question_id, "dataset": ds_name},
                }

            def record_fn(row: dict[str, Any], raw: Any, _idx: int) -> dict[str, Any]:
                response = _extract_text(raw)
                output_key = _normalize_output_key(row["question_id"])
                return {
                    "output_key": output_key,
                    "response": response,
                }

            def sample_id_fn(row: dict[str, Any]) -> str:
                return _normalize_output_key(row["question_id"])

            n_written = run_sharded_inference(
                infer_fn=pipeline.run,
                dist_info=dist_info,
                shard_path=shard_path,
                samples=iter_rows(),
                total=len(lines),
                payload_fn=payload_fn,
                record_fn=record_fn,
                sample_id_fn=sample_id_fn,
                done_ids=done_keys,
                max_samples=max_samples,
                log_prefix=f"mmvet/{ds_name}/rank{dist_info.rank}",
            )
            local_total_written += n_written

            barrier(dist_info)

            time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
            results_file = out_dir / f"{ds_name}_{time_prefix}.json"

            if dist_info.rank == 0:
                merged = merge_shards(checkpoint_jsonl)
                outputs: dict[str, str] = {item["output_key"]: item["response"] for item in merged}
                results_file.write_text(json.dumps(outputs, indent=2), encoding="utf-8")

                cleanup_shards(checkpoint_jsonl)
                if dist_info.world_size <= 1 and checkpoint_jsonl.exists():
                    checkpoint_jsonl.unlink()

                summary[f"{ds_name}_output_path"] = str(results_file)

            barrier(dist_info)

        total_written_all = sum_across_ranks(local_total_written, dist_info)
        if dist_info.rank != 0:
            print(
                f"[umm eval] rank {dist_info.rank} finished MM-Vet shard: "
                f"samples_written={local_total_written}",
                flush=True,
            )
            return 0

        summary["samples_written"] = total_written_all
        if isinstance(score_output_path, str) and score_output_path:
            score_path = _resolve_path(score_output_path, repo_root)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[umm eval] wrote MM-Vet summary to {score_path}")

        print(
            f"[umm eval] completed MM-Vet for backbone={backbone}, outputs={out_dir}, "
            f"samples_written={total_written_all}, world_size={dist_info.world_size}"
        )
        return 0
    finally:
        cleanup_distributed(dist_info)
