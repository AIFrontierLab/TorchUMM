from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from umm.post_training.unigame.janus_pro.data import (
    VQAHFDataset,
    build_sample_adapter,
    extract_answers_from_example,
    load_mmmu_split,
)
from umm.post_training.unigame.janus_pro.trainer import AdvTrainer


LOGGER = logging.getLogger("umm.unigame.janus_pro")


def _setup_logging(cfg: dict[str, Any]) -> None:
    log_file = str(cfg.get("debug_log_file", "logs_januspro/metrics/debug.log"))
    log_path = Path(log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if bool(cfg.get("debug", True)) else logging.INFO
    LOGGER.handlers.clear()
    LOGGER.setLevel(level)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    LOGGER.addHandler(sh)

    LOGGER.info("debug log file: %s", log_path)


def _expand_user_cfg_path(cfg: dict[str, Any], key: str, env_key: str) -> None:
    value = cfg.get(key)
    if not value:
        return
    expanded = str(Path(str(value)).expanduser())
    cfg[key] = expanded
    os.environ[env_key] = expanded


def _resolve_auto_download_ids(cfg: dict[str, Any]) -> None:
    auto_download = bool(cfg.get("auto_download", True))
    if not auto_download:
        return

    hf_model_id = cfg.get("hf_model_id")
    hf_dataset_id = cfg.get("hf_dataset_id")

    if hf_model_id:
        cfg["model_path"] = str(hf_model_id)
    if hf_dataset_id:
        cfg["dataset_path"] = str(hf_dataset_id)

    _expand_user_cfg_path(cfg, "hf_home", "HF_HOME")
    _expand_user_cfg_path(cfg, "transformers_cache", "TRANSFORMERS_CACHE")
    _expand_user_cfg_path(cfg, "datasets_cache", "HF_DATASETS_CACHE")

    LOGGER.info(
        "auto_download=%s model=%s dataset=%s",
        auto_download,
        cfg.get("model_path"),
        cfg.get("dataset_path"),
    )


def _init_ddp(cfg: dict[str, Any]) -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        default_backend = "nccl" if torch.cuda.is_available() else "gloo"
        backend = str(cfg.get("ddp_backend", default_backend)).strip().lower()
        timeout_s = int(cfg.get("ddp_timeout_s", 900))
        LOGGER.info(
            "DDP pre-init | backend=%s rank=%s local_rank=%s world_size=%s master=%s:%s timeout_s=%d",
            backend,
            os.environ.get("RANK"),
            os.environ.get("LOCAL_RANK"),
            os.environ.get("WORLD_SIZE"),
            os.environ.get("MASTER_ADDR", "<unset>"),
            os.environ.get("MASTER_PORT", "<unset>"),
            timeout_s,
        )
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_s),
        )
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            LOGGER.info(
                "DDP init ok | backend=%s rank=%d local_rank=%d world_size=%s cuda_device=%d",
                backend,
                rank,
                local_rank,
                os.environ.get("WORLD_SIZE"),
                torch.cuda.current_device(),
            )
        else:
            LOGGER.info(
                "DDP init ok | backend=%s rank=%d local_rank=%d world_size=%s (cpu)",
                backend,
                rank,
                local_rank,
                os.environ.get("WORLD_SIZE"),
            )
        return True
    return False


def _build_vocab_fast_hf(hf_train: Any, topk: int, sample_adapter: Any = None) -> tuple[list[str], dict[str, int]]:
    cnt = Counter()
    for ex in hf_train:
        if sample_adapter is None:
            answers = extract_answers_from_example(ex)
        else:
            answers = sample_adapter(ex).get("answers", [])
        for a in answers:
            ans = (a.get("answer") or "").strip().lower()
            if ans:
                cnt[ans] += 1
    vocab = [a for a, _ in cnt.most_common(topk)]
    ans2id = {a: i for i, a in enumerate(vocab)}
    return vocab, ans2id


def _majority_answer(answers_list: list[dict[str, Any]]) -> str:
    toks = [a.get("answer").strip().lower() for a in answers_list if a.get("answer")]
    return Counter(toks).most_common(1)[0][0] if toks else "unknown"


def _vqa_collate_generative(batch: list[dict[str, Any]]) -> dict[str, Any]:
    imgs, qs, ans_txt, answers = [], [], [], []
    for item in batch:
        imgs.append(item["image"])
        qs.append(item["question"])
        ans_list = item.get("answers") or []
        answers.append(ans_list)
        ans_txt.append(_majority_answer(ans_list))
    return {
        "image": torch.stack(imgs, 0),
        "question": qs,
        "answer_text": ans_txt,
        "answers": answers,
    }


def _make_loader(
    dataset: Any,
    batch_size: int,
    sampler: Any,
    num_workers: int,
    collate_fn: Any,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    if num_workers > 0:
        kwargs.update(pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _subset_hf_dataset(dataset: Any, keep_ratio: float, seed: int, name: str) -> Any:
    keep = float(keep_ratio)
    if keep >= 1.0:
        return dataset
    if keep <= 0.0:
        raise ValueError(f"{name} keep ratio must be > 0, got {keep}")
    subset = dataset.train_test_split(test_size=max(0.0, 1.0 - keep), seed=seed)["train"]
    LOGGER.info("subset %s for smoke eval: keep=%.3f size=%d -> %d", name, keep, len(dataset), len(subset))
    return subset


def _apply_split_keep(dataset: Any, keep_ratio: float, seed: int, split_name: str) -> Any:
    """Apply keep ratio safely for train/val/test split datasets."""
    keep = float(keep_ratio)
    if keep >= 1.0:
        return dataset
    if keep <= 0.0:
        raise ValueError(f"{split_name}_split_keep must be > 0, got {keep}")
    subset = dataset.train_test_split(test_size=(1.0 - keep), seed=seed)["train"]
    LOGGER.info("subset %s split: keep=%.3f size=%d -> %d", split_name, keep, len(dataset), len(subset))
    return subset


def _get_hf_load_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    # New datasets versions discourage trust_remote_code. Only pass it when explicitly enabled.
    kwargs: dict[str, Any] = {}
    if bool(cfg.get("trust_remote_code", False)):
        kwargs["trust_remote_code"] = True

    # Allow loading local tabular/json datasets with explicit data files.
    # Example:
    #   dataset_path: csv
    #   hf_data_files:
    #     train: /path/train.tsv
    #     validation: /path/val.tsv
    #     test: /path/test.tsv
    data_files = cfg.get("hf_data_files")
    if isinstance(data_files, dict) and data_files:
        kwargs["data_files"] = {str(k): str(v) for k, v in data_files.items()}
    elif isinstance(data_files, list) and data_files:
        kwargs["data_files"] = [str(v) for v in data_files]
    elif isinstance(data_files, str) and data_files.strip():
        kwargs["data_files"] = data_files.strip()

    data_dir = cfg.get("hf_data_dir")
    if isinstance(data_dir, str) and data_dir.strip():
        kwargs["data_dir"] = data_dir.strip()

    # Common CSV/TSV knobs.
    delimiter = cfg.get("hf_delimiter")
    if isinstance(delimiter, str) and delimiter:
        kwargs["delimiter"] = delimiter

    column_names = cfg.get("hf_column_names")
    if isinstance(column_names, list) and column_names:
        kwargs["column_names"] = [str(c) for c in column_names]

    features = cfg.get("hf_features")
    if features is not None:
        kwargs["features"] = features

    return kwargs


def _pick_available_split(requested: str, available: list[str], role: str) -> str:
    if requested in available:
        return requested

    # Prefer common canonical alternatives before falling back to first available split.
    fallbacks = {
        "train": ["validation", "test"],
        "val": ["validation", "train", "test"],
        "test": ["test", "validation", "train"],
    }
    for candidate in fallbacks.get(role, []):
        if candidate in available:
            LOGGER.warning(
                "requested %s split '%s' not found; fallback to '%s' (available=%s)",
                role,
                requested,
                candidate,
                available,
            )
            return candidate

    if available:
        LOGGER.warning(
            "requested %s split '%s' not found; fallback to first available '%s' (available=%s)",
            role,
            requested,
            available[0],
            available,
        )
        return available[0]

    raise ValueError("No available dataset splits returned by Hugging Face datasets.")


def run_janus_pro_train(cfg: dict[str, Any]) -> None:
    _setup_logging(cfg)
    _resolve_auto_download_ids(cfg)

    ddp = _init_ddp(cfg)
    LOGGER.info("DDP enabled: %s", ddp)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        LOGGER.info("CUDA available, TF32 enabled")

    dataset_name = str(cfg.get("dataset_name", "vqav2")).strip().lower()
    sample_adapter = build_sample_adapter(cfg)
    num_answers = int(cfg.get("num_answers", 3129))
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("`train.dataset_path` is required for UniGame Janus-Pro training.")
    LOGGER.info("dataset_name=%s dataset_path=%s num_answers=%d", dataset_name, dataset_path, num_answers)

    train_split = str(cfg.get("train_split", "train"))
    val_split = str(cfg.get("val_split", "validation"))
    test_split = str(cfg.get("test_split", "test"))
    train_split_keep = float(cfg.get("train_split_keep", 0.9))
    val_split_keep = float(cfg.get("val_split_keep", 0.9))
    test_split_keep = float(cfg.get("test_split_keep", 1.0))
    split_seed = int(cfg.get("split_seed", 42))

    hf_kwargs = _get_hf_load_kwargs(cfg)
    if dataset_name == "mmmu":
        available_splits = ["dev", "validation", "test"]
        LOGGER.info("MMMU dataset selected, using canonical splits: %s", available_splits)
    else:
        data_files = hf_kwargs.get("data_files")
        if isinstance(data_files, dict) and data_files:
            # For local/webdataset data_files, avoid get_dataset_split_names() which may
            # open tar members during inspection and fail before actual loading.
            available_splits = [str(k) for k in data_files.keys()]
            LOGGER.info("using splits from hf_data_files keys: %s", available_splits)
        else:
            try:
                available_splits = list(get_dataset_split_names(dataset_path, **hf_kwargs))
            except Exception as exc:
                LOGGER.warning(
                    "failed to inspect dataset splits (%s); fallback to configured splits",
                    exc,
                )
                available_splits = [train_split, val_split, test_split]
    LOGGER.info("available dataset splits: %s", available_splits)

    picked_train_split = _pick_available_split(train_split, available_splits, role="train")
    picked_val_split = _pick_available_split(val_split, available_splits, role="val")
    picked_test_split = _pick_available_split(test_split, available_splits, role="test")

    enable_test_eval = bool(cfg.get("enable_test_eval", True))
    used_splits = {picked_train_split, picked_val_split}
    if enable_test_eval:
        used_splits.add(picked_test_split)

    # If all roles map to the same source split (common on tiny benchmark datasets),
    # build disjoint train/val/test subsets from that one split.
    ds_test = None
    if len(used_splits) == 1:
        base_split = next(iter(used_splits))
        if dataset_name == "mmmu":
            base_ds = load_mmmu_split(
                root=str(dataset_path),
                split=base_split,
                cache_dir=cfg.get("dataset_cache_dir"),
            )
        else:
            base_ds = load_dataset(dataset_path, split=base_split, **hf_kwargs)

        train_w = max(0.0, train_split_keep)
        val_w = max(0.0, val_split_keep)
        test_w = max(0.0, test_split_keep) if enable_test_eval else 0.0
        total_w = train_w + val_w + test_w
        if total_w <= 0:
            raise ValueError("train/val/test weights are all zero; adjust *_split_keep values.")

        train_ratio = train_w / total_w
        rest = base_ds.train_test_split(test_size=max(0.0, 1.0 - train_ratio), seed=split_seed)
        ds_train = rest["train"]
        remainder = rest["test"]

        if enable_test_eval:
            rem_total = val_w + test_w
            if rem_total <= 0:
                raise ValueError("val/test weights are zero while test eval is enabled.")
            val_ratio_in_remainder = val_w / rem_total
            vt = remainder.train_test_split(test_size=max(0.0, 1.0 - val_ratio_in_remainder), seed=split_seed + 1)
            ds_val = vt["train"]
            ds_test = vt["test"]
        else:
            ds_val = remainder

        LOGGER.warning(
            "single source split '%s' detected; created disjoint subsets train=%d val=%d test=%s",
            base_split,
            len(ds_train),
            len(ds_val),
            ("N/A" if ds_test is None else str(len(ds_test))),
        )
    else:
        if dataset_name == "mmmu":
            ds_train = load_mmmu_split(
                root=str(dataset_path),
                split=picked_train_split,
                cache_dir=cfg.get("dataset_cache_dir"),
            )
            ds_train = _apply_split_keep(ds_train, train_split_keep, split_seed, "train")

            ds_val = load_mmmu_split(
                root=str(dataset_path),
                split=picked_val_split,
                cache_dir=cfg.get("dataset_cache_dir"),
            )
            ds_val = _apply_split_keep(ds_val, val_split_keep, split_seed, "validation")
        else:
            ds_train = load_dataset(dataset_path, split=picked_train_split, **hf_kwargs)
            ds_train = _apply_split_keep(ds_train, train_split_keep, split_seed, "train")

            ds_val = load_dataset(dataset_path, split=picked_val_split, **hf_kwargs)
            ds_val = _apply_split_keep(ds_val, val_split_keep, split_seed, "validation")

        if enable_test_eval:
            try:
                if dataset_name == "mmmu":
                    ds_test = load_mmmu_split(
                        root=str(dataset_path),
                        split=picked_test_split,
                        cache_dir=cfg.get("dataset_cache_dir"),
                    )
                    ds_test = _apply_split_keep(ds_test, test_split_keep, split_seed, "test")
                else:
                    ds_test = load_dataset(dataset_path, split=picked_test_split, **hf_kwargs)
                    ds_test = _apply_split_keep(ds_test, test_split_keep, split_seed, "test")
            except Exception as exc:
                LOGGER.warning(
                    "test split unavailable, skip test eval | requested=%s picked=%s err=%s",
                    test_split,
                    picked_test_split,
                    exc,
                )

    LOGGER.info(
        "dataset loaded train=%d val=%d test=%s",
        len(ds_train),
        len(ds_val),
        ("N/A" if ds_test is None else str(len(ds_test))),
    )

    eval_subset_keep = float(cfg.get("eval_subset_keep", 1.0))
    if eval_subset_keep < 1.0:
        ds_val = _subset_hf_dataset(ds_val, eval_subset_keep, split_seed + 11, "validation")
        if ds_test is not None:
            ds_test = _subset_hf_dataset(ds_test, eval_subset_keep, split_seed + 13, "test")

    _, ans2id = _build_vocab_fast_hf(ds_train, topk=num_answers, sample_adapter=sample_adapter)
    image_size = int(cfg.get("image_size", 384))
    train_dataset = VQAHFDataset(
        ds_train,
        ans2id,
        image_size=image_size,
        return_pil=False,
        sample_adapter=sample_adapter,
    )
    val_dataset = VQAHFDataset(
        ds_val,
        ans2id,
        image_size=image_size,
        return_pil=False,
        sample_adapter=sample_adapter,
    )
    test_dataset = (
        VQAHFDataset(
            ds_test,
            ans2id,
            image_size=image_size,
            return_pil=False,
            sample_adapter=sample_adapter,
        )
        if ds_test is not None
        else None
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp else None
    test_sampler = (
        DistributedSampler(test_dataset, shuffle=False, drop_last=False)
        if (ddp and test_dataset is not None)
        else None
    )

    num_workers = int(cfg.get("num_workers", 4))
    train_bs = int(cfg.get("batch_size", 8))
    val_bs = int(cfg.get("val_batch_size", train_bs))
    LOGGER.info(
        "dataloader batch_size=%d val_batch_size=%d workers=%d image_size=%d",
        train_bs,
        val_bs,
        num_workers,
        image_size,
    )

    train_loader = _make_loader(
        dataset=train_dataset,
        batch_size=train_bs,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=_vqa_collate_generative,
    )
    val_loader = _make_loader(
        dataset=val_dataset,
        batch_size=val_bs,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=_vqa_collate_generative,
    )
    test_loader = (
        _make_loader(
            dataset=test_dataset,
            batch_size=val_bs,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=_vqa_collate_generative,
        )
        if test_dataset is not None
        else None
    )

    trainer = AdvTrainer(cfg)

    resume_ckpt = str(cfg.get("resume_from_ckpt", "") or "").strip()
    if resume_ckpt:
        resume_load_optim = bool(cfg.get("resume_load_optim", True))
        resume_strict = bool(cfg.get("resume_strict", False))
        LOGGER.info(
            "resume requested | ckpt=%s load_optim=%s strict=%s",
            resume_ckpt,
            resume_load_optim,
            resume_strict,
        )
        trainer.load_ckpt(
            resume_ckpt,
            load_optim=resume_load_optim,
            strict=resume_strict,
        )
        if ddp:
            dist.barrier()

    # ── Checkpoint evaluation dataset (e.g. VQAv2) ──────────────────────
    ckpt_eval_loader = None
    if bool(cfg.get("ckpt_eval_enabled", False)):
        ckpt_eval_path = cfg.get("ckpt_eval_dataset")
        if ckpt_eval_path:
            ckpt_eval_split = str(cfg.get("ckpt_eval_split", "validation"))
            ckpt_eval_cache = cfg.get("ckpt_eval_cache_dir") or cfg.get("dataset_cache_dir")
            LOGGER.info("loading ckpt eval dataset: %s split=%s", ckpt_eval_path, ckpt_eval_split)
            ckpt_hf = load_dataset(ckpt_eval_path, split=ckpt_eval_split, cache_dir=ckpt_eval_cache)
            ckpt_eval_keep = float(cfg.get("ckpt_eval_subset_keep", 1.0))
            if ckpt_eval_keep < 1.0:
                ckpt_hf = _subset_hf_dataset(ckpt_hf, ckpt_eval_keep, split_seed + 99, "ckpt_eval")
            from umm.post_training.unigame.janus_pro.data import VQADefaultAdapter
            ckpt_eval_ds = VQAHFDataset(
                ckpt_hf, ans2id, image_size=image_size,
                return_pil=False, sample_adapter=VQADefaultAdapter(),
            )
            ckpt_eval_loader = _make_loader(
                dataset=ckpt_eval_ds,
                batch_size=val_bs,
                sampler=None,
                num_workers=num_workers,
                collate_fn=_vqa_collate_generative,
            )
            LOGGER.info("ckpt eval dataset ready: %d samples", len(ckpt_hf))

    do_eval_before_train = bool(cfg.get("run_eval_before_train", False))
    eval_max_batches = int(cfg.get("eval_max_batches", 0) or 0)
    if do_eval_before_train and ddp:
        # Keep all ranks in the same phase: rank0 runs pre-eval while others wait.
        dist.barrier()
    if do_eval_before_train and trainer.rank == 0:
        LOGGER.info("start pre-train validation evaluation")
        eval_val_loader_pre = val_loader
        if ddp:
            eval_val_loader_pre = _make_loader(
                dataset=val_dataset,
                batch_size=val_bs,
                sampler=None,
                num_workers=num_workers,
                collate_fn=_vqa_collate_generative,
            )
        val_metrics_pre = trainer.evaluate(
            eval_val_loader_pre,
            split_name="validation_pretrain",
            max_batches=eval_max_batches,
        )
        LOGGER.info("pre-train validation metrics: %s", val_metrics_pre)
    if do_eval_before_train and ddp:
        dist.barrier()

    LOGGER.info("trainer initialized, start train loop")
    print("Starting training", "(DDP)" if ddp else "")
    trainer.train(train_loader, val_loader, ckpt_eval_loader=ckpt_eval_loader)

    do_eval = bool(cfg.get("run_eval_after_train", True))
    if do_eval and trainer.rank == 0:
        LOGGER.info("start post-train evaluation")

        # In DDP, distributed samplers shard data. Build full loaders for rank0-only eval.
        eval_val_loader = val_loader
        eval_test_loader = test_loader
        if ddp:
            eval_val_loader = _make_loader(
                dataset=val_dataset,
                batch_size=val_bs,
                sampler=None,
                num_workers=num_workers,
                collate_fn=_vqa_collate_generative,
            )
            if test_dataset is not None:
                eval_test_loader = _make_loader(
                    dataset=test_dataset,
                    batch_size=val_bs,
                    sampler=None,
                    num_workers=num_workers,
                    collate_fn=_vqa_collate_generative,
                )

        val_metrics = trainer.evaluate(eval_val_loader, split_name="validation", max_batches=eval_max_batches)
        LOGGER.info("validation metrics: %s", val_metrics)

        if eval_test_loader is not None:
            test_metrics = trainer.evaluate(eval_test_loader, split_name="test", max_batches=eval_max_batches)
            LOGGER.info("test metrics: %s", test_metrics)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
