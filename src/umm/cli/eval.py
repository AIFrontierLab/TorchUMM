from __future__ import annotations

from typing import Any

from umm.core.config import load_config


def run_eval_command(args: Any) -> int:
    """Route evaluation command based on benchmark type."""
    config_path = args.config

    raw_cfg = load_config(config_path)

    eval_cfg = raw_cfg.get("eval", {}) if isinstance(raw_cfg.get("eval"), dict) else {}
    benchmark = str(eval_cfg.get("benchmark", "")).strip().lower()
    if not benchmark and isinstance(raw_cfg.get("benchmark"), str):
        benchmark = str(raw_cfg.get("benchmark", "")).strip().lower()

    # Lazy imports to avoid pulling heavy deps that may not be installed
    # in every container image (e.g. geneval image lacks `datasets`).
    if benchmark == "dpg_bench" or "dpg_bench" in raw_cfg:
        from umm.cli.dpg_bench import run_eval_command as _fn
        return _fn(args)
    if benchmark == "mme" or "mme" in raw_cfg:
        from umm.cli.mme_eval import run_mme_eval_command as _fn
        return _fn(args)
    if benchmark == "mmmu" or "mmmu" in raw_cfg:
        from umm.cli.mmmu_eval import run_mmmu_eval_command as _fn
        return _fn(args)
    if benchmark == "mmbench" or "mmbench" in raw_cfg:
        from umm.cli.mmbench_eval import run_mmbench_eval_command as _fn
        return _fn(args)
    if benchmark == "mmvet" or "mmvet" in raw_cfg:
        from umm.cli.mmvet_eval import run_mmvet_eval_command as _fn
        return _fn(args)
    if benchmark == "mathvista" or "mathvista" in raw_cfg:
        from umm.cli.mathvista_eval import run_mathvista_eval_command as _fn
        return _fn(args)
    if benchmark == "uni_mmmu" or "uni_mmmu" in raw_cfg:
        from umm.cli.uni_mmmu import run_eval_command as _fn
        return _fn(args)
    if benchmark == "wise" or "wise" in raw_cfg:
        from umm.cli.wise import run_wise_eval_command as _fn
        return _fn(args)
    if benchmark == "ueval" or "ueval" in raw_cfg:
        from umm.cli.ueval_eval import run_ueval_eval_command as _fn
        return _fn(args)
    if benchmark == "imgedit" or "imgedit" in raw_cfg:
        from umm.cli.imgedit import run_imgedit_eval_command as _fn
        return _fn(args)
    if benchmark == "gedit" or "gedit" in raw_cfg:
        from umm.cli.gedit import run_gedit_eval_command as _fn
        return _fn(args)
    if benchmark == "geneval" or "geneval" in raw_cfg:
        from umm.cli.geneval import run_eval_command as _fn
        return _fn(args)
    raise NotImplementedError(f"`umm eval` benchmark '{benchmark}' is not supported yet (config: {args.config}).")
