from __future__ import annotations

from typing import Any

from umm.core.config import load_config
from umm.post_training.recA.pipeline import run_reca_train
from umm.post_training.unicot.pipeline import run_unicot_train
from umm.post_training.sft.bagel.pipeline import run_bagel_train
from umm.post_training.IRG.pipeline import run_irg_train
from umm.post_training.unigame.pipeline import run_unigame_train


def _unwrap_train_block(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("train", "posttrain", "post_training"):
        block = config.get(key)
        if isinstance(block, dict):
            return block
    return config


def run_train_command(args: Any) -> int:
    raw_cfg = load_config(args.config)
    cfg = _unwrap_train_block(raw_cfg)

    pipeline = cfg.get("pipeline", "recA")
    if pipeline == "recA":
        run_reca_train(cfg, config_path=args.config)
        return 0
    if pipeline == "unicot":
        run_unicot_train(cfg, config_path=args.config)
        return 0
    if pipeline == "bagel":
        run_bagel_train(cfg, config_path=args.config)
        return 0
    if pipeline == "irg":
        run_irg_train(cfg, config_path=args.config)
        return 0
    if pipeline == "unigame":
        run_unigame_train(cfg, config_path=args.config)
        return 0
    raise ValueError(f"Unsupported training pipeline: {pipeline}")
