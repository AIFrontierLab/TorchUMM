"""Reuse TorchUMM benchmarks and add the UniMore-local RealUnify entry point."""

from __future__ import annotations

import os
import subprocess
import sys

from .config import PACKAGE_ROOT, TORCHUMM_ROOT, load_config


def evaluate(args):
    config = load_config(args.config)
    if config.get("eval", {}).get("benchmark") != "realunify":
        from umm.cli.eval import run_eval_command

        return run_eval_command(args)
    return run_realunify(config)


def run_realunify(config):
    real = config["realunify"]
    model = config["inference"]["backbone_cfg"]
    if real["split"] not in ("GEU_direct", "UEG_direct"):
        raise ValueError("The UniMore presets use GEU_direct or UEG_direct")
    common = [
        "--checkpoint-name", "unimore", "--data-root", str(real["data_root"]),
        "--output-root", str(real["out_dir"]), "--splits", real["split"],
        "--max-samples", str(real.get("max_samples", 0)),
    ]
    script = PACKAGE_ROOT / "scripts/eval_realunify_bagel.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(TORCHUMM_ROOT / "src"), env.get("PYTHONPATH", ""))))
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    subprocess.run([
        sys.executable, "-B", str(script), "run", *common,
        "--model-path", model["model_path"], "--bagel-root", model["bagel_root"],
        "--image-size", "512", "--base-seed", "42", "--max-mem-per-gpu", "78GiB",
        "--offload-folder", model["offload_folder"],
    ], env=env, check=True)
    subprocess.run([sys.executable, "-B", str(script), "aggregate", *common, "--require-complete"],
                   env=env, check=True)
    return 0
