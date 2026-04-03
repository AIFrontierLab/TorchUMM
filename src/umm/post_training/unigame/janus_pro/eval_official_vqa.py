import argparse
import json
import logging
import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from umm.post_training.unigame.janus_pro.data import VQAHFDataset
from umm.post_training.unigame.janus_pro.framework import JanusProAdvFramework

try:
    from eval.vlm.eval.vqa.textvqa_eval import TextVQAAccuracyEvaluator
except Exception as exc:
    raise ImportError(
        "Failed to import TextVQAAccuracyEvaluator from eval/vlm. "
        "Run from repo root and ensure repo root is on PYTHONPATH."
    ) from exc


LOGGER = logging.getLogger("umm.unigame.janus_pro.official_eval")
_OFFICIAL_PROMPT_SUFFIX = "Answer the question using a single word or phrase."


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _subset_hf_dataset(ds, keep: float, seed: int):
    keep = float(keep)
    if keep >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * keep))
    LOGGER.info("subset eval split: keep=%.3f size=%d -> %d", keep, n, k)
    return ds.shuffle(seed=seed).select(range(k))


def _answers_to_str_list(answers: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(answers, list):
        return out
    for item in answers:
        if isinstance(item, dict):
            v = item.get("answer")
            if v is None:
                v = item.get("text")
            if v is not None:
                s = str(v).strip()
                if s:
                    out.append(s)
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def _ensure_len10(answers: list[str]) -> list[str]:
    # TextVQAAccuracyEvaluator expects exactly 10 answers like VQAv2 annotations.
    if len(answers) == 10:
        return answers
    if len(answers) == 0:
        return [""] * 10
    if len(answers) > 10:
        return answers[:10]
    pad = [answers[-1]] * (10 - len(answers))
    return answers + pad


def _clean_pred_answer(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""

    s = s.replace("Ġ", " ").replace("▁", " ")
    s = " ".join(s.split())

    low = s.lower()
    # Remove common answer prefixes.
    for p in ("answer:", "answer is", "the answer is", "final answer:"):
        if low.startswith(p):
            s = s[len(p) :].strip(" :")
            low = s.lower()

    # Drop the echoed instruction suffix if present.
    marker = _OFFICIAL_PROMPT_SUFFIX.lower()
    pos = low.find(marker)
    if pos >= 0:
        s = s[:pos].strip(" .,:;")

    # Keep the first line/sentence for short-form VQA style outputs.
    for sep in ("\n", ". ", "! ", "? "):
        if sep in s:
            s = s.split(sep, 1)[0].strip(" .,:;")
            break

    return " ".join(s.split())


def _build_loader(ds, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    # Keep raw PIL images for evaluation. Janus image processor handles resize/
    # crop internally; forcing square resize here can hurt VQA alignment.
    dataset = VQAHFDataset(ds, ans2id={}, image_size=image_size, return_pil=True)

    def _collate(batch):
        images = [x["image"] for x in batch]
        questions = [x["question"] for x in batch]
        answers = [x.get("answers", []) for x in batch]
        return {
            "image": images,
            "question": questions,
            "answers": answers,
        }

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate,
    }
    if num_workers > 0:
        kwargs.update({
            "pin_memory": True,
            "persistent_workers": True,
        })
    return DataLoader(**kwargs)


def _load_ckpt(model: JanusProAdvFramework, ckpt_path: str) -> None:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("model", obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    LOGGER.info(
        "checkpoint loaded: %s | missing=%d unexpected=%d",
        ckpt_path,
        len(missing),
        len(unexpected),
    )


def run(args: argparse.Namespace) -> None:
    _setup_logging(args.log_level)
    torch.set_grad_enabled(False)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `datasets` (likely pyarrow/libstdc++ mismatch). "
            "Try: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` and/or "
            "`conda install -y -c conda-forge libstdcxx-ng` in the active env."
        ) from exc

    hf_kwargs = {"trust_remote_code": True}
    ds = load_dataset(args.dataset_path, split=args.split, **hf_kwargs)
    ds = _subset_hf_dataset(ds, args.subset_keep, args.seed)
    LOGGER.info("loaded dataset | path=%s split=%s size=%d", args.dataset_path, args.split, len(ds))

    loader = _build_loader(
        ds=ds,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = JanusProAdvFramework(
        model_path=args.model_path,
        num_answers=args.num_answers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        finetune_last_k=0,
        dtype=dtype,
        eps_max=args.eps_max,
        low_cpu_mem_usage=False,
        gradient_checkpointing=False,
    )
    model.eval()

    if args.ckpt:
        _load_ckpt(model, args.ckpt)

    evaluator = TextVQAAccuracyEvaluator()
    pred_list = []
    rows = []

    pbar = tqdm(total=len(loader), desc="OfficialEval[VQAv2]", dynamic_ncols=True)
    for batch in loader:
        images = batch["image"]
        questions = [f"{q} {_OFFICIAL_PROMPT_SUFFIX}".strip() for q in batch["question"]]

        preds = model.infer_answers_batch(
            images=images,
            questions=questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        for q, pred, answers in zip(batch["question"], preds, batch["answers"]):
            pred_clean = _clean_pred_answer(pred)
            gt_answers = _ensure_len10(_answers_to_str_list(answers))
            pred_list.append({
                "pred_answer": pred_clean,
                "gt_answers": gt_answers,
            })
            rows.append({
                "question": q,
                "pred_answer": pred_clean,
                "pred_answer_raw": pred,
                "gt_answers": gt_answers,
            })
        pbar.update(1)
    pbar.close()

    score = evaluator.eval_pred_list(pred_list, disable_tqdm=True)
    result = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "samples": len(pred_list),
        "vqa_score": float(score),
        "vqa_score_percent": float(score) * 100.0,
        "prompt_suffix": _OFFICIAL_PROMPT_SUFFIX,
        "ckpt": args.ckpt,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, "official_vqa_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    pred_path = os.path.join(args.out_dir, "official_vqa_preds.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("official vqa done | samples=%d vqa_score=%.6f (%.2f%%)", len(pred_list), score, score * 100.0)
    LOGGER.info("saved result: %s", result_path)
    LOGGER.info("saved preds : %s", pred_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Janus-Pro official-style VQA evaluation")
    p.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-7B")
    p.add_argument("--ckpt", type=str, default="", help="Path to UniGame training checkpoint (.pt)")
    p.add_argument("--dataset-path", type=str, default="merve/vqav2-small")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--subset-keep", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--num-answers", type=int, default=3129)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eps-max", type=float, default=0.02)

    p.add_argument("--out-dir", type=str, default="logs_januspro_official_eval")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    run(parser.parse_args())
