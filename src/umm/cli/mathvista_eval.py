from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

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


DS_COLLECTIONS = {
    "MathVista_testmini": {
        "root": "AI4Math/MathVista",
        "split": "testmini",
        "max_new_tokens": 4096,
    },
    "MathVista_test": {
        "root": "AI4Math/MathVista",
        "split": "test",
        "max_new_tokens": 4096,
    },
}


COT_INSTRUCTION = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


# Load few-shot demo prompt from the canonical source
from umm.eval.internvl_chat.eval.mathvista.prompts.ext_ans import demo_prompt as _EXTRACT_DEMO_PROMPT


def _quick_extract(response: str, problem: dict) -> str | None:
    if not response:
        return ""
    question_type = problem.get("question_type", "")
    answer_type = problem.get("answer_type", "")
    choices = problem.get("choices") or []

    if question_type == "multi_choice" and response in choices:
        return response
    if answer_type == "integer":
        try:
            return str(int(response))
        except (ValueError, TypeError):
            pass
    if answer_type == "float":
        try:
            return str(float(response))
        except (ValueError, TypeError):
            pass

    match = re.search(r"(?:Final answer:|Answer:)\s*(.*)", response, re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
        if ans:
            return ans

    return None


def _build_extract_prompt(query: str, response: str) -> str:
    test_prompt = f"{query}\n\n{response}"
    return f"{_EXTRACT_DEMO_PROMPT.strip()}\n\n{test_prompt}\n\nExtracted answer: "


def _run_llm_extraction(
    results: dict[str, Any],
    model_path: str,
    max_new_tokens: int = 256,
    use_quick_extract: bool = True,
) -> dict[str, Any]:
    """Extract answers from model responses using a local LLM (e.g. Qwen3-32B)."""
    already_done = 0
    need_llm = []
    for pid, problem in results.items():
        if "choices" not in problem:
            continue
        if "extraction" in problem and problem["extraction"]:
            already_done += 1
            continue
        response = problem.get("response", "")
        if use_quick_extract:
            quick = _quick_extract(response, problem)
            if quick is not None:
                problem["extraction"] = quick
                continue
        need_llm.append(pid)

    if already_done > 0:
        print(
            f"[mathvista] {already_done} already extracted from previous run",
            flush=True,
        )

    print(
        f"[mathvista] {len(results) - len(need_llm) - already_done} extracted by rules, "
        f"{len(need_llm)} need LLM",
        flush=True,
    )

    if not need_llm:
        print("[mathvista] all items already extracted, skipping LLM load", flush=True)
        return results

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[mathvista] loading extraction LLM: {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    for pid in tqdm(need_llm, desc="mathvista/extract", file=sys.stdout):
        problem = results[pid]
        query = problem.get("query", "")
        response = problem.get("response", "")
        prompt = _build_extract_prompt(query, response)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        extraction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        think_match = re.search(r"</think>\s*(.*)", extraction, re.DOTALL)
        if think_match:
            extraction = think_match.group(1).strip()
        extraction = extraction.split("\n")[0].strip()
        problem["extraction"] = extraction

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[mathvista] extraction done for {len(results)} items", flush=True)
    return results


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
    mathvista_cfg = raw_cfg.get("mathvista", {}) if isinstance(raw_cfg.get("mathvista"), dict) else {}
    inference_cfg = raw_cfg.get("inference", {}) if isinstance(raw_cfg.get("inference"), dict) else {}
    if not eval_cfg and "benchmark" in raw_cfg:
        eval_cfg = {"benchmark": raw_cfg.get("benchmark")}
    return eval_cfg, mathvista_cfg, inference_cfg


def _find_latest_results(out_dir: Path, ds_name: str) -> Path | None:
    candidates = sorted(out_dir.glob(f"{ds_name}_*.json"))
    candidates = [c for c in candidates if "_checkpoint" not in c.name and "_score" not in c.name]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def run_mathvista_eval_command(args: Any) -> int:
    config_path = str(args.config)
    eval_cfg, mathvista_cfg, inference_cfg = _load_eval_cfg(config_path)
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if benchmark != "mathvista":
        raise ValueError(f"Expected `eval.benchmark: mathvista`, got: {benchmark or '<empty>'}")

    repo_root = Path(__file__).resolve().parents[3]

    backbone_raw = inference_cfg.get("backbone")
    if not isinstance(backbone_raw, str) or not backbone_raw:
        raise ValueError("`inference.backbone` is required for MathVista eval.")
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

    datasets_value = mathvista_cfg.get("datasets", ["MathVista_testmini"])
    if isinstance(datasets_value, str):
        datasets = [name.strip() for name in datasets_value.split(",") if name.strip()]
    elif isinstance(datasets_value, list):
        datasets = [str(name).strip() for name in datasets_value if str(name).strip()]
    else:
        datasets = ["MathVista_testmini"]
    if not datasets:
        raise ValueError("`mathvista.datasets` must contain at least one dataset name.")

    out_dir = _resolve_path(str(mathvista_cfg.get("out_dir", f"output/mathvista/{backbone}")), repo_root)
    image_dir = _resolve_path(str(mathvista_cfg.get("image_dir", out_dir / "images")), repo_root)
    score_output_path = mathvista_cfg.get("score_output_path")
    cache_dir = mathvista_cfg.get("cache_dir")
    max_samples = int(mathvista_cfg.get("max_samples", 0) or 0)
    use_cot = bool(mathvista_cfg.get("cot", False))
    gt_file = mathvista_cfg.get("gt_file")
    resume = bool(mathvista_cfg.get("resume", False))

    mode = str(mathvista_cfg.get("mode", "full")).strip().lower()
    if mode not in ("full", "generate", "score"):
        print(f"[mathvista] unknown mode '{mode}', defaulting to 'full'", flush=True)
        mode = "full"

    run_gen = mode in ("full", "generate")
    run_score = mode in ("full", "score")

    llm_extract_cfg = mathvista_cfg.get("llm_extract", {})
    if not isinstance(llm_extract_cfg, dict):
        llm_extract_cfg = {}
    llm_model_path = str(llm_extract_cfg.get("model_path", "")).strip()
    llm_max_new_tokens = int(llm_extract_cfg.get("max_new_tokens", 2048))

    run_extract_legacy = bool(mathvista_cfg.get("run_extract", False))
    run_calculation_legacy = bool(mathvista_cfg.get("run_calculation", False))
    openai_api_key = mathvista_cfg.get("openai_api_key")

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark": "mathvista",
        "backbone": backbone,
        "out_dir": str(out_dir),
        "datasets": datasets,
        "cot": use_cot,
        "mode": mode,
    }

    dist_info = maybe_init_distributed()
    try:
        summary["world_size"] = dist_info.world_size
        if dist_info.world_size > 1 and run_gen:
            print(
                f"[mathvista] distributed inference enabled: rank={dist_info.rank}, "
                f"local_rank={dist_info.local_rank}, world_size={dist_info.world_size}",
                flush=True,
            )

        local_total_written = 0

        # ── Phase 1: Generation ──
        if run_gen:
            from datasets import load_dataset
            from PIL import Image

            from umm.inference import InferencePipeline

            pipeline = InferencePipeline(backbone_name=backbone, backbone_cfg=backbone_cfg)

            for ds_name in datasets:
                entry = DS_COLLECTIONS.get(ds_name)
                if not entry:
                    raise ValueError(f"Unknown MathVista dataset: {ds_name}")

                dataset_root = str(mathvista_cfg.get("root", entry["root"]))
                split = str(mathvista_cfg.get("split", entry["split"]))
                dataset = load_dataset(
                    dataset_root,
                    cache_dir=str(_resolve_path(cache_dir, repo_root)) if cache_dir else None,
                )
                data = dataset[split]

                checkpoint_jsonl = out_dir / f"{ds_name}_checkpoint.jsonl"
                shard_path = rank_shard_path(checkpoint_jsonl, dist_info.rank, dist_info.world_size)

                done_pids: set[str] = set()
                results_file: Path | None = None

                if resume:
                    shard_items = load_shard_items(shard_path)
                    if shard_items:
                        done_pids = {str(it.get("pid", "")) for it in shard_items}
                        print(
                            f"[mathvista] {ds_name}: rank {dist_info.rank} resume "
                            f"from shard: {len(done_pids)} done",
                            flush=True,
                        )
                    elif dist_info.world_size <= 1:
                        # Single-card fallback: a previous completed run can be loaded
                        # whole. Multi-card runs rely on the rank shard only.
                        prior = _find_latest_results(out_dir, ds_name)
                        if prior:
                            print(
                                f"[mathvista] resume: using completed file {prior}",
                                flush=True,
                            )
                            results_file = prior

                if results_file is None:
                    print(
                        f"[mathvista] {ds_name}: total={len(data)}, rank={dist_info.rank}, "
                        f"done={len(done_pids)}",
                        flush=True,
                    )

                    def payload_fn(data_item: Any) -> dict[str, Any]:
                        pid = data_item.get("pid")
                        if pid is None:
                            raise ValueError("MathVista sample missing `pid`.")
                        image = data_item.get("decoded_image")
                        if image is None:
                            raise ValueError("MathVista sample missing `decoded_image`.")
                        if not isinstance(image, Image.Image):
                            raise ValueError("Expected `decoded_image` to be a PIL image.")
                        image_dir.mkdir(parents=True, exist_ok=True)
                        image_path = image_dir / f"{pid}.png"
                        image.save(image_path, format="PNG")
                        question = data_item.get("query")
                        if question is None:
                            raise ValueError("MathVista sample missing `query`.")
                        prompt = COT_INSTRUCTION.format(question=question) if use_cot else question
                        return {
                            "backbone": backbone,
                            "task": "understanding",
                            "prompt": prompt,
                            "images": [str(image_path)],
                            "params": request_params,
                            "metadata": {"pid": pid, "dataset": ds_name},
                        }

                    def record_fn(data_item: Any, raw: Any, _idx: int) -> dict[str, Any]:
                        response = _extract_text(raw)
                        item = dict(data_item)
                        item.pop("decoded_image", None)
                        item["response"] = response
                        # Ensure pid is JSON-friendly so resume works.
                        item["pid"] = str(item.get("pid", ""))
                        return item

                    n_written = run_sharded_inference(
                        infer_fn=pipeline.run,
                        dist_info=dist_info,
                        shard_path=shard_path,
                        samples=data,
                        total=len(data),
                        payload_fn=payload_fn,
                        record_fn=record_fn,
                        sample_id_fn=lambda d: str(d.get("pid", "")),
                        done_ids=done_pids,
                        max_samples=max_samples,
                        log_prefix=f"mathvista/{ds_name}/rank{dist_info.rank}",
                    )
                    local_total_written += n_written

                    barrier(dist_info)

                    time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
                    results_file = out_dir / f"{ds_name}_{time_prefix}.json"

                    if dist_info.rank == 0:
                        merged = merge_shards(checkpoint_jsonl)
                        results: dict[str, Any] = {item["pid"]: item for item in merged}
                        results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
                        cleanup_shards(checkpoint_jsonl)
                        if dist_info.world_size <= 1 and checkpoint_jsonl.exists():
                            checkpoint_jsonl.unlink()

                    barrier(dist_info)

                summary[f"{ds_name}_output_path"] = str(results_file)

            if mode == "generate":
                print(f"[mathvista] generation phase done, outputs={out_dir}", flush=True)

            # Free GPU memory from generation pipeline before scoring (rank 0 only
            # touches scoring; other ranks return after the barriers above).
            del pipeline
            import gc
            gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            print(
                f"[mathvista] rank {dist_info.rank}: released generation pipeline GPU memory",
                flush=True,
            )

        # Multi-card: ranks > 0 finish here; only rank 0 proceeds with scoring +
        # summary writing (scoring loads its own LLM with device_map="auto").
        local_written_all = sum_across_ranks(local_total_written, dist_info)
        if dist_info.rank != 0:
            print(
                f"[umm eval] rank {dist_info.rank} finished MathVista shard: "
                f"samples_written={local_total_written}",
                flush=True,
            )
            return 0

        summary["samples_written"] = local_written_all

        # ── Phase 2: Scoring (extract + calculate) — rank 0 only ──
        if run_score:
            for ds_name in datasets:
                results_file = None
                if f"{ds_name}_output_path" in summary:
                    results_file = Path(summary[f"{ds_name}_output_path"])
                else:
                    results_file = _find_latest_results(out_dir, ds_name)
                if results_file is None or not results_file.exists():
                    raise FileNotFoundError(
                        f"No results file found for {ds_name} in {out_dir}. "
                        f"Run generation phase first (mode: generate)."
                    )

                print(f"[mathvista] scoring {ds_name} from {results_file}", flush=True)
                results = json.loads(results_file.read_text("utf-8"))

                if llm_model_path:
                    results = _run_llm_extraction(
                        results,
                        model_path=llm_model_path,
                        max_new_tokens=llm_max_new_tokens,
                        use_quick_extract=use_cot,
                    )
                    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    print(f"[mathvista] saved extractions to {results_file}", flush=True)
                elif run_extract_legacy:
                    cmd = [
                        sys.executable,
                        "src/umm/eval/internvl_chat/eval/mathvista/extract_answer.py",
                        "--output_file",
                        results_file.name,
                        "--output_dir",
                        str(out_dir),
                    ]
                    if use_cot:
                        cmd.append("--quick_extract")
                    env = None
                    if isinstance(openai_api_key, str) and openai_api_key.strip():
                        env = dict(os.environ)
                        env["OPENAI_API_KEY"] = openai_api_key.strip()
                    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env)
                    print(proc.stdout)
                    if proc.returncode != 0:
                        if proc.stderr:
                            print(proc.stderr, file=sys.stderr)
                        raise RuntimeError(f"MathVista extract_answer failed with return code {proc.returncode}")
                    summary[f"{ds_name}_extract_stdout"] = proc.stdout

                score_file = results_file.with_name(f"{results_file.stem}_score.json")
                cmd = [
                    sys.executable,
                    "src/umm/eval/internvl_chat/eval/mathvista/calculate_score.py",
                    "--output_file",
                    results_file.name,
                    "--output_dir",
                    str(out_dir),
                    "--score_file",
                    score_file.name,
                ]
                if isinstance(gt_file, str) and gt_file.strip():
                    cmd.extend(["--gt_file", gt_file.strip()])
                proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
                print(proc.stdout)
                if proc.returncode != 0:
                    if proc.stderr:
                        print(proc.stderr, file=sys.stderr)
                    raise RuntimeError(f"MathVista calculate_score failed with return code {proc.returncode}")
                summary[f"{ds_name}_score_file"] = str(score_file)
                summary[f"{ds_name}_score_stdout"] = proc.stdout

        if isinstance(score_output_path, str) and score_output_path:
            score_path = _resolve_path(score_output_path, repo_root)
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[umm eval] wrote MathVista summary to {score_path}")

        print(
            f"[umm eval] completed MathVista (mode={mode}) for backbone={backbone}, "
            f"outputs={out_dir}, world_size={dist_info.world_size}"
        )
        return 0
    finally:
        cleanup_distributed(dist_info)
