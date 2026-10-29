"""Verify the fixed merge against a released checkpoint without exporting weights."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from ..checkpoints import merge_tensor, weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "t2i", "i2t", "released"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tensor", action="append", help="Check only named tensors for a smoke test")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    import torch
    from safetensors import safe_open

    torch.set_num_threads(args.threads)
    elements = 0
    with ExitStack() as stack:
        files = {name: stack.enter_context(safe_open(str(weights(getattr(args, name))), framework="pt", device="cpu"))
                 for name in ("base", "t2i", "i2t", "released")}
        keys = set(files["base"].keys())
        if any(set(file.keys()) != keys for file in files.values()):
            raise ValueError("All four checkpoints must contain the same tensors")
        selected = sorted(set(args.tensor or keys))
        if not selected or not set(selected) <= keys:
            raise ValueError("Requested tensor names are missing")
        for key in selected:
            actual = merge_tensor(*(files[name].get_tensor(key) for name in ("base", "t2i", "i2t")))
            expected = files["released"].get_tensor(key)
            if actual.dtype != expected.dtype or not torch.equal(actual, expected):
                raise ValueError(f"Released checkpoint mismatch: {key}")
            elements += actual.numel()
    print(json.dumps({"passed": True, "scope": "smoke" if args.tensor else "full",
                      "tensors_checked": len(selected), "total_tensors": len(keys), "elements_checked": elements}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
