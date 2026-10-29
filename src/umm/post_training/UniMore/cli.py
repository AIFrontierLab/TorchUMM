"""Train, export, merge and evaluate UniMore without modifying TorchUMM's CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m umm.post_training.UniMore", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train one independent branch")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dry-run", action="store_true")
    for name in ("infer", "eval"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "infer":
            command.add_argument("--output-json")
    export = commands.add_parser("export", help="Export one LoRA branch as BAGEL weights")
    export.add_argument("--base", required=True, type=Path)
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--route", choices=("t2i", "i2t"), required=True)
    export.add_argument("--output", required=True, type=Path)
    merge = commands.add_parser("merge", help="Apply the fixed UniMore task-vector merge")
    for name in ("base", "t2i", "i2t", "output"):
        merge.add_argument(f"--{name}", required=True, type=Path)
    data = commands.add_parser("prepare-data", help="Relocate the released training manifests")
    data.add_argument("--t2i-manifest", required=True, type=Path)
    data.add_argument("--i2t-manifest", required=True, type=Path)
    data.add_argument("--path-map", required=True, type=Path)
    data.add_argument("--output", required=True, type=Path)
    download = commands.add_parser("download", help="Download a released checkpoint")
    download.add_argument("--model", choices=("base", "unimore", "t2i", "i2t"), default="unimore")
    download.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .training import train

        train(args.config, dry_run=args.dry_run)
    elif args.command == "infer":
        from .config import load_config

        load_config(args.config)
        from umm.cli.infer import run_infer_command

        return run_infer_command(args)
    elif args.command == "eval":
        from .evaluation import evaluate

        return evaluate(args)
    elif args.command == "prepare-data":
        from .data import prepare_data

        prepare_data(args.t2i_manifest, args.i2t_manifest, args.path_map, args.output)
    else:
        from . import checkpoints

        if args.command == "export":
            checkpoints.export_branch(args.base, args.checkpoint, args.route, args.output)
        elif args.command == "merge":
            checkpoints.merge_branches(args.base, args.t2i, args.i2t, args.output)
        else:
            checkpoints.download_model(args.model, args.output)
    return 0
