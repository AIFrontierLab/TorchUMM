#!/usr/bin/env python
import argparse
import os
import torch
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="HF repo ID or local folder to load config+weights")
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint dir containing model.pt")
    parser.add_argument("--output_dir", required=True, help="Where to write HF-style model")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if not os.path.isdir(args.ckpt_dir):
        raise SystemExit(f"ckpt_dir not found: {args.ckpt_dir}")
    model_path = os.path.join(args.ckpt_dir, "model.pt")
    if not os.path.isfile(model_path):
        raise SystemExit(f"model.pt not found in {args.ckpt_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # load base model (config + architecture)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    state = torch.load(model_path, map_location="cpu")
    msg = model.load_state_dict(state, strict=False)
    print(f"Loaded ckpt: {msg}")

    model.save_pretrained(args.output_dir)

    # save processor/tokenizer assets
    try:
        processor = VLChatProcessor.from_pretrained(args.base_model)
        processor.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"WARNING: failed to save processor: {e}")

    print(f"Saved HF model to: {args.output_dir}")


if __name__ == "__main__":
    main()
