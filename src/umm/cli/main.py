from __future__ import annotations

import argparse
import sys

from umm.cli.eval import run_eval_command
from umm.cli.infer import run_infer_command
# from umm.cli.train import run_train_command
# try:
#     import debugpy
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach on port 9501 ...")
#     debugpy.wait_for_client()
#     print("Debugger attached!")
# except Exception as e:
#     print(f"[debugpy] skipped: {e}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umm")
    sub = parser.add_subparsers(dest="cmd")

    infer = sub.add_parser("infer")
    infer.add_argument("--config", required=True)
    infer.add_argument("--output-json", default=None, help="Optional path to dump serializable outputs.")
    infer.set_defaults(handler=run_infer_command)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--config", required=True)
    evaluate.set_defaults(handler=run_eval_command)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    # train.set_defaults(handler=run_train_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 1
    if not hasattr(args, "handler"):
        parser.error(f"No handler configured for command: {args.cmd}")
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
