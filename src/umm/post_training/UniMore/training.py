"""Launch the model-native packed BAGEL trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .config import PACKAGE_ROOT, load_config


def argument_list(values):
    result = []
    for name, value in values.items():
        if value is not None:
            result.extend([f"--{name}", str(value)])
    return result


def train(config_path: Path, *, dry_run=False):
    config = load_config(config_path)
    repo = PACKAGE_ROOT
    args = dict(config["args"])
    if config["route"] not in ("t2i", "i2t"):
        raise ValueError("route must be t2i or i2t")
    if Path(args["model_path"]).resolve() != Path(args["resume_from"]).resolve():
        raise ValueError("Each branch must start independently from the same base model")
    if bool(args["visual_gen"]) == bool(args["visual_und"]):
        raise ValueError("Train exactly one branch at a time")
    if bool(args["visual_gen"]) != (config["route"] == "t2i"):
        raise ValueError("The selected route must match the enabled training branch")
    if int(config["num_processes"]) != int(args["num_shard"]) * int(args["num_replicate"]):
        raise ValueError("num_processes must equal num_shard * num_replicate")
    output = Path(args["results_dir"]).expanduser().resolve()
    if not dry_run and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output must be empty: {output}")
    data_path = Path(config["data_config"])
    if not data_path.is_absolute():
        data_path = repo / data_path
    data = load_config(data_path)
    resolved = output / "data.yaml"
    args["dataset_config_file"] = str(resolved)
    command = [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={config['num_processes']}",
        "train/pretrain_unified_navit.py",
        *argument_list(args),
    ]
    print(shlex.join(command), flush=True)
    if dry_run:
        return command
    import yaml

    output.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(PYTHONPATH=".", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="offline")
    subprocess.run(command, cwd=repo / "third_party/bagel_training", env=env, check=True)
