from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import get_dataset_split_names, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from umm.post_training.unigame.bagel.data import VQAHFDataset, build_sample_adapter, vqa_collate
from umm.post_training.unigame.bagel.trainer import AdvTrainer


LOGGER = logging.getLogger("umm.unigame.bagel")


def _setup_logging(cfg: dict[str, Any]) -> None:
	log_file = str(cfg.get("debug_log_file", "logs_bagel_unigame/metrics/debug.log"))
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


def _init_ddp() -> bool:
	if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
		backend = "nccl" if torch.cuda.is_available() else "gloo"
		dist.init_process_group(backend=backend, init_method="env://")
		if torch.cuda.is_available():
			torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
		return True
	return False


def _majority_answer(answers_list: list[dict[str, Any]]) -> str:
	toks = [a.get("answer").strip().lower() for a in answers_list if a.get("answer")]
	return Counter(toks).most_common(1)[0][0] if toks else "unknown"


def _vqa_collate_generative(batch: list[dict[str, Any]]) -> dict[str, Any]:
	out = vqa_collate(batch)
	if "answer_text" not in out:
		out["answer_text"] = [_majority_answer(x.get("answers") or []) for x in batch]
	return out


def _make_loader(dataset: Any, batch_size: int, sampler: Any, num_workers: int) -> DataLoader:
	kwargs = {
		"dataset": dataset,
		"batch_size": batch_size,
		"sampler": sampler,
		"num_workers": num_workers,
		"collate_fn": _vqa_collate_generative,
	}
	if num_workers > 0:
		kwargs.update(pin_memory=True, persistent_workers=True, prefetch_factor=2)
	return DataLoader(**kwargs)


def _apply_split_keep(dataset: Any, keep_ratio: float, seed: int, split_name: str) -> Any:
	keep = float(keep_ratio)
	if keep >= 1.0:
		return dataset
	if keep <= 0.0:
		raise ValueError(f"{split_name}_split_keep must be > 0, got {keep}")
	subset = dataset.train_test_split(test_size=(1.0 - keep), seed=seed)["train"]
	LOGGER.info("subset %s split: keep=%.3f size=%d -> %d", split_name, keep, len(dataset), len(subset))
	return subset


def _get_hf_load_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
	kwargs: dict[str, Any] = {}
	if bool(cfg.get("trust_remote_code", False)):
		kwargs["trust_remote_code"] = True

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

	fallbacks = {
		"train": ["validation", "test"],
		"val": ["validation", "test", "train"],
		"test": ["test", "validation", "train"],
	}
	for candidate in fallbacks.get(role, []):
		if candidate in available:
			LOGGER.warning("requested %s split '%s' missing, fallback to '%s'", role, requested, candidate)
			return candidate
	if not available:
		raise ValueError(f"No dataset splits available for role={role}")
	LOGGER.warning("requested %s split '%s' missing, fallback to first available '%s'", role, requested, available[0])
	return available[0]


def _run_internal_bagel_unigame(cfg: dict[str, Any]) -> None:
	_setup_logging(cfg)
	_resolve_auto_download_ids(cfg)

	ddp = _init_ddp()
	LOGGER.info("DDP enabled: %s", ddp)

	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.set_float32_matmul_precision("high")

	dataset_path = cfg.get("dataset_path")
	if not dataset_path:
		raise ValueError("`train.dataset_path` is required for UniGame Bagel training.")

	dataset_name = str(cfg.get("dataset_name", "vqav2")).strip().lower()
	sample_adapter = build_sample_adapter(cfg)
	train_split = str(cfg.get("train_split", "train"))
	val_split = str(cfg.get("val_split", "validation"))
	test_split = str(cfg.get("test_split", "test"))
	train_split_keep = float(cfg.get("train_split_keep", 0.9))
	val_split_keep = float(cfg.get("val_split_keep", 0.9))
	test_split_keep = float(cfg.get("test_split_keep", 1.0))
	split_seed = int(cfg.get("split_seed", 42))
	hf_kwargs = _get_hf_load_kwargs(cfg)

	data_files = hf_kwargs.get("data_files")
	if isinstance(data_files, dict) and data_files:
		available_splits = [str(k) for k in data_files.keys()]
		LOGGER.info("using splits from hf_data_files keys: %s", available_splits)
	else:
		try:
			available_splits = list(get_dataset_split_names(dataset_path, **hf_kwargs))
		except Exception as exc:
			LOGGER.warning("failed to inspect dataset splits (%s); fallback to configured splits", exc)
			available_splits = [train_split, val_split, test_split]
	LOGGER.info("dataset_name=%s dataset_path=%s available_splits=%s", dataset_name, dataset_path, available_splits)

	picked_train_split = _pick_available_split(train_split, available_splits, role="train")
	picked_val_split = _pick_available_split(val_split, available_splits, role="val")
	picked_test_split = _pick_available_split(test_split, available_splits, role="test")

	enable_test_eval = bool(cfg.get("enable_test_eval", True))
	used_splits = {picked_train_split, picked_val_split}
	if enable_test_eval:
		used_splits.add(picked_test_split)

	ds_test = None
	if len(used_splits) == 1:
		base_split = next(iter(used_splits))
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
		ds_train = load_dataset(dataset_path, split=picked_train_split, **hf_kwargs)
		ds_train = _apply_split_keep(ds_train, train_split_keep, split_seed, "train")

		ds_val = load_dataset(dataset_path, split=picked_val_split, **hf_kwargs)
		ds_val = _apply_split_keep(ds_val, val_split_keep, split_seed, "validation")

		if enable_test_eval:
			try:
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

	image_size = int(cfg.get("image_size", 512))
	train_dataset = VQAHFDataset(
		ds_train,
		image_size=image_size,
		return_pil=False,
		sample_adapter=sample_adapter,
		skip_overlong_samples=bool(cfg.get("skip_overlong_samples", False)),
		max_question_chars=int(cfg.get("max_question_chars", 0)),
		max_answer_chars=int(cfg.get("max_answer_chars", 0)),
		max_resample_attempts=int(cfg.get("max_resample_attempts", 32)),
	)
	val_dataset = VQAHFDataset(
		ds_val,
		image_size=image_size,
		return_pil=False,
		sample_adapter=sample_adapter,
		skip_overlong_samples=bool(cfg.get("skip_overlong_samples", False)),
		max_question_chars=int(cfg.get("max_question_chars", 0)),
		max_answer_chars=int(cfg.get("max_answer_chars", 0)),
		max_resample_attempts=int(cfg.get("max_resample_attempts", 32)),
	)
	test_dataset = (
		VQAHFDataset(
			ds_test,
			image_size=image_size,
			return_pil=False,
			sample_adapter=sample_adapter,
			skip_overlong_samples=bool(cfg.get("skip_overlong_samples", False)),
			max_question_chars=int(cfg.get("max_question_chars", 0)),
			max_answer_chars=int(cfg.get("max_answer_chars", 0)),
			max_resample_attempts=int(cfg.get("max_resample_attempts", 32)),
		)
		if ds_test is not None
		else None
	)

	train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if ddp else None
	val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if ddp else None
	test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False) if (ddp and test_dataset is not None) else None

	num_workers = int(cfg.get("num_workers", 2))
	train_bs = int(cfg.get("batch_size", 1))
	val_bs = int(cfg.get("val_batch_size", train_bs))

	train_loader = _make_loader(train_dataset, train_bs, train_sampler, num_workers)
	val_loader = _make_loader(val_dataset, val_bs, val_sampler, num_workers)
	test_loader = _make_loader(test_dataset, val_bs, test_sampler, num_workers) if test_dataset is not None else None

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
	LOGGER.info("trainer initialized, start train loop")
	trainer.train(train_loader, val_loader)

	do_eval = bool(cfg.get("run_eval_after_train", True))
	eval_max_batches = int(cfg.get("eval_max_batches", 0) or 0)
	if do_eval and trainer.rank == 0:
		eval_val_loader = val_loader
		eval_test_loader = test_loader
		if ddp:
			eval_val_loader = _make_loader(val_dataset, val_bs, None, num_workers)
			if test_dataset is not None:
				eval_test_loader = _make_loader(test_dataset, val_bs, None, num_workers)

		val_metrics = trainer.evaluate(eval_val_loader, split_name="validation", max_batches=eval_max_batches)
		LOGGER.info("validation metrics: %s", val_metrics)

		if eval_test_loader is not None:
			test_metrics = trainer.evaluate(eval_test_loader, split_name="test", max_batches=eval_max_batches)
			LOGGER.info("test metrics: %s", test_metrics)

	if ddp:
		dist.barrier()
		dist.destroy_process_group()


def _find_repo_root(start: Path) -> Path | None:
	for parent in [start, *start.parents]:
		if (parent / "pyproject.toml").exists():
			return parent
	return None


def _resolve_cwd(config_path: str | None, cfg: dict[str, Any]) -> tuple[Path, Path]:
	base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
	repo_root = _find_repo_root(base_dir) or base_dir

	bagel_root_cfg = cfg.get("bagel_root")
	if isinstance(bagel_root_cfg, str) and bagel_root_cfg.strip():
		bagel_root = Path(bagel_root_cfg).expanduser()
		if not bagel_root.is_absolute():
			bagel_root = (repo_root / bagel_root).resolve()
	else:
		bagel_root = (repo_root / "src" / "umm" / "backbones" / "bagel" / "Bagel").resolve()

	cwd_cfg = cfg.get("cwd")
	if isinstance(cwd_cfg, str) and cwd_cfg.strip():
		cwd = Path(cwd_cfg).expanduser()
		if not cwd.is_absolute():
			candidate = (base_dir / cwd).resolve()
			cwd = candidate if candidate.exists() else (repo_root / cwd).resolve()
	else:
		cwd = bagel_root

	if not bagel_root.exists():
		raise FileNotFoundError(f"Bagel root not found: {bagel_root}")
	if not cwd.exists():
		raise FileNotFoundError(f"Working directory not found: {cwd}")
	return cwd, repo_root


def _build_args(args: dict[str, Any]) -> list[str]:
	out: list[str] = []
	for key, value in args.items():
		flag = f"--{key}"
		if value is None:
			continue
		if isinstance(value, bool):
			out.extend([flag, "True" if value else "False"])
		elif isinstance(value, (list, tuple)):
			for item in value:
				out.extend([flag, str(item)])
		else:
			out.extend([flag, str(value)])
	return out


def _build_launcher_prefix(torchrun_cfg: dict[str, Any], use_torchrun: bool) -> list[str]:
	if not use_torchrun:
		return [sys.executable]

	torchrun_bin = shutil.which("torchrun")
	if torchrun_bin:
		cmd: list[str] = [torchrun_bin]
	else:
		cmd = [sys.executable, "-m", "torch.distributed.run"]

	defaults = {
		"nnodes": 1,
		"node_rank": 0,
		"nproc_per_node": 1,
	}
	for key, default in defaults.items():
		value = torchrun_cfg.get(key, default)
		cmd.append(f"--{key}={value}")

	for key in ("master_addr", "master_port", "rdzv_backend", "rdzv_endpoint"):
		value = torchrun_cfg.get(key)
		if value is not None:
			cmd.append(f"--{key}={value}")

	extra = torchrun_cfg.get("extra_args", [])
	if isinstance(extra, list):
		cmd.extend([str(x) for x in extra])
	return cmd


def _run_external_bagel_launcher(cfg: dict[str, Any], config_path: str | None) -> None:
	cwd_path, repo_root = _resolve_cwd(config_path, cfg)

	script = cfg.get("script")
	module = cfg.get("module")
	if not isinstance(script, str):
		script = ""
	if not isinstance(module, str):
		module = ""
	if script and module:
		raise ValueError("Specify only one of `script` or `module` for bagel training.")
	if not script and not module:
		script = "train/pretrain_unified_navit.py"

	torchrun_cfg = cfg.get("torchrun", {})
	if torchrun_cfg is None:
		torchrun_cfg = {}
	if not isinstance(torchrun_cfg, dict):
		raise ValueError("`torchrun` must be a dict if provided.")

	use_torchrun = bool(cfg.get("use_torchrun", True))
	cmd = _build_launcher_prefix(torchrun_cfg=torchrun_cfg, use_torchrun=use_torchrun)

	if module:
		cmd.extend(["-m", module])
	else:
		script_path = Path(script).expanduser()
		if not script_path.is_absolute():
			script_path = (cwd_path / script_path).resolve()
		if not script_path.exists():
			raise FileNotFoundError(f"Bagel training script not found: {script_path}")
		cmd.append(str(script_path))

	args_dict = cfg.get("args", {})
	if args_dict is None:
		args_dict = {}
	if not isinstance(args_dict, dict):
		raise ValueError("`args` must be a dict if provided.")
	cmd.extend(_build_args(args_dict))

	extra_args = cfg.get("extra_args", [])
	if isinstance(extra_args, list):
		cmd.extend([str(x) for x in extra_args])

	env = os.environ.copy()
	env_update = cfg.get("env", {})
	if isinstance(env_update, dict):
		for key, value in env_update.items():
			if value is not None:
				env[str(key)] = str(value)

	pythonpath_parts: list[str] = [str((repo_root / "src").resolve()), str(cwd_path)]
	existing_pythonpath = env.get("PYTHONPATH")
	if existing_pythonpath:
		pythonpath_parts.append(existing_pythonpath)
	env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

	print(f"[umm train] backbone=bagel cwd: {cwd_path}")
	print(f"[umm train] running: {shlex.join(cmd)}")
	subprocess.run(cmd, cwd=str(cwd_path), env=env, check=True)


def run_bagel_train(cfg: dict[str, Any], config_path: str | None = None) -> None:
	"""Run UniGame Bagel training.

	Default behavior is internal UniGame training implementation.
	Set `external_runner: true` or provide `script/module` to use external launcher mode.
	"""

	external_runner = bool(cfg.get("external_runner", False))
	has_external_entry = bool(cfg.get("script") or cfg.get("module"))
	if external_runner or has_external_entry:
		_run_external_bagel_launcher(cfg, config_path)
		return

	_run_internal_bagel_unigame(cfg)
