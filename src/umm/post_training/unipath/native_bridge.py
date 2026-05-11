from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

BAGEL_CODE_ROOT = add_bagel_code_to_sys_path(REPO_ROOT)

from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer


DEFAULT_TRAINABLE_PREFIXES = [
    "vae2llm",
    "llm2vae",
    "latent_pos_embed",
    "time_embedder",
]


def build_full_bagel(model_path: Path, *, visual_gen: bool = True, visual_und: bool = True) -> tuple[Bagel, Qwen2Tokenizer]:
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", None)
    rope_scaling = getattr(llm_config, "rope_scaling", None)
    if rope_scaling is None:
        llm_config.rope_scaling = {"rope_type": "linear", "factor": 1.0}
    else:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type in (None, "default"):
            llm_config.rope_scaling = {
                **rope_scaling,
                "rope_type": "linear",
                "factor": rope_scaling.get("factor", 1.0),
            }
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.freeze_und = True

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    _, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        if visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, _, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    return model, tokenizer


def load_full_bagel_dispatch(
    model_path: Path,
    *,
    checkpoint_file: str = "ema.safetensors",
    max_mem_per_gpu: str = "80GiB",
    offload_folder: str = "./tmp/offload",
    dtype: torch.dtype = torch.bfloat16,
    visual_gen: bool = True,
    visual_und: bool = True,
) -> tuple[Bagel, Qwen2Tokenizer]:
    model, tokenizer = build_full_bagel(model_path, visual_gen=visual_gen, visual_und=visual_und)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("Native Bagel loading requires at least one CUDA device.")
    max_memory = {i: max_mem_per_gpu for i in range(gpu_count)}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for module_name in same_device_modules:
        device_map[module_name] = device_map.get(module_name, first_device)

    checkpoint_path = model_path / checkpoint_file
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint_path),
        device_map=device_map,
        offload_buffers=True,
        dtype=dtype,
        force_hooks=True,
        offload_folder=offload_folder,
    )
    return model.eval(), tokenizer


def freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _matches_native_moe_gen(name: str) -> bool:
    return "moe_gen" in name or name.endswith("_moe_gen.weight") or name.endswith("_moe_gen.bias")


def select_native_visual_trainables(
    model: Bagel,
    include_moe_gen: bool = True,
    extra_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    freeze_all(model)
    prefixes = list(DEFAULT_TRAINABLE_PREFIXES)
    if extra_prefixes:
        prefixes.extend(extra_prefixes)

    selected_names: list[str] = []
    selected_counter: Counter[str] = Counter()

    for name, param in model.named_parameters():
        should_train = False
        if _matches_prefix(name, prefixes):
            should_train = True
        if include_moe_gen and _matches_native_moe_gen(name):
            should_train = True
        if name.startswith("connector."):
            should_train = True
        if should_train:
            param.requires_grad = True
            selected_names.append(name)
            selected_counter[name.split(".")[0]] += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "selected_names": selected_names,
        "selected_counter": selected_counter,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the native-visual trainable subset for UniPath on full Bagel.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--extra-prefix", action="append", default=[])
    parser.add_argument("--mode", choices=["bridge_only", "bridge_plus_moe"], default="bridge_plus_moe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    model, tokenizer = build_full_bagel(model_path)
    stats = select_native_visual_trainables(
        model,
        include_moe_gen=(args.mode == "bridge_plus_moe"),
        extra_prefixes=args.extra_prefix,
    )
    print(f"[native_bridge] tokenizer_vocab={len(tokenizer)}")
    print(f"[native_bridge] mode={args.mode}")
    print(
        f"[native_bridge] trainable={stats['trainable']} total={stats['total']} "
        f"pct={100.0 * stats['trainable'] / max(stats['total'], 1):.4f}"
    )
    print(f"[native_bridge] top_level_param_breakdown={dict(stats['selected_counter'])}")
    preview = stats["selected_names"][:40]
    for name in preview:
        print(f"[native_bridge] train {name}")
    if len(stats["selected_names"]) > len(preview):
        print(f"[native_bridge] ... and {len(stats['selected_names']) - len(preview)} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
