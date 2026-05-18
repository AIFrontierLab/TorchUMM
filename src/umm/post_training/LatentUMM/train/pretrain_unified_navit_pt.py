# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import sys
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Optional, Tuple, List

from peft import LoraConfig, get_peft_model

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

# Allow running this script directly from outside repo root.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens, patchify
from data.transforms import ImageTransform
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, fsdp_with_lora_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)
from data.data_utils import prepare_attention_mask_per_sample
from safetensors.torch import load_file


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


def _parse_float_pair(value: str, default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    return default


def _extract_prompt_ids(
    packed_text_ids: torch.LongTensor,
    bos_token_id: int,
    eos_token_id: int,
    max_tokens: int,
) -> Optional[List[int]]:
    ids = packed_text_ids.tolist()
    try:
        bos_idx = ids.index(bos_token_id)
    except ValueError:
        bos_idx = 0
    try:
        eos_idx = ids.index(eos_token_id, bos_idx + 1)
    except ValueError:
        eos_idx = min(len(ids), bos_idx + 1 + max_tokens)
    prompt_ids = ids[bos_idx + 1:eos_idx]
    if max_tokens is not None and len(prompt_ids) > max_tokens:
        prompt_ids = prompt_ids[:max_tokens]
    if len(prompt_ids) == 0:
        return None
    return prompt_ids


def _mean_text_embedding(model: Bagel, prompt_ids: List[int], new_token_ids: dict) -> torch.Tensor:
    ids = torch.tensor(
        [new_token_ids["bos_token_id"]] + prompt_ids + [new_token_ids["eos_token_id"]],
        dtype=torch.long,
        device=model.language_model.model.embed_tokens.weight.device,
    )
    emb = model.language_model.model.embed_tokens(ids)
    if emb.shape[0] > 2:
        emb = emb[1:-1]
    return emb.mean(dim=0)


def _latent_mean_embedding(model: Bagel, packed_latent_tokens: torch.Tensor) -> torch.Tensor:
    latent = packed_latent_tokens.to(model.vae2llm.weight.dtype)
    latent_tokens = model.vae2llm(latent)
    return latent_tokens.mean(dim=0)


def _load_shared_latent_teacher(ckpt_path: str, device: int, logger) -> Optional[torch.nn.Module]:
    if ckpt_path is None or ckpt_path == "" or not os.path.exists(ckpt_path):
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict):
        logger.warning(f"Shared latent teacher load skipped: unsupported checkpoint format at {ckpt_path}")
        return None

    # Stage2 checkpoints store Stage1 backbone as "backbone.*".
    if any(k.startswith("backbone.") for k in state_dict.keys()):
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                stripped[key[len("backbone."):]] = value
        state_dict = stripped

    required_keys = (
        "aligner.text_proj.weight",
        "aligner.image_proj.weight",
        "aligner.fuse.4.weight",
    )
    if not all(k in state_dict for k in required_keys):
        logger.warning(
            f"Shared latent teacher load skipped: checkpoint does not contain Stage1/2 aligner keys ({ckpt_path})"
        )
        return None

    from stage1_shared_latent_modules.model import Stage1Config, Stage1SharedLatentModel

    text_dim = int(state_dict["aligner.text_proj.weight"].shape[1])
    image_dim = int(state_dict["aligner.image_proj.weight"].shape[1])
    hidden_dim = int(state_dict["aligner.text_proj.weight"].shape[0])
    latent_dim = int(state_dict["aligner.fuse.4.weight"].shape[0])
    output_dim = int(state_dict["generator.net.2.weight"].shape[0]) if "generator.net.2.weight" in state_dict else text_dim

    teacher = Stage1SharedLatentModel(
        Stage1Config(
            text_dim=text_dim,
            image_dim=image_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.1,
        )
    )
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    teacher_device = torch.device("cuda", device) if isinstance(device, int) and torch.cuda.is_available() else device
    teacher.to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    logger.info(f"Loaded shared latent teacher from {ckpt_path}")
    logger.info(f"Shared latent teacher dims: text={text_dim}, image={image_dim}, latent={latent_dim}")
    if len(missing) > 0:
        logger.info(f"Shared latent teacher missing keys: {len(missing)}")
    if len(unexpected) > 0:
        logger.info(f"Shared latent teacher unexpected keys: {len(unexpected)}")
    return teacher


def _vit_preprocess_tensor(image_tensor: torch.Tensor, vit_transform: ImageTransform) -> torch.Tensor:
    # image_tensor: [C, H, W] in [0, 1]
    image_tensor = vit_transform.resize_transform(image_tensor)
    image_tensor = vit_transform.normalize_transform(image_tensor)
    return image_tensor


def _image_understanding_embedding(
    model: Bagel,
    vit_transform: ImageTransform,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    image_tensor = _vit_preprocess_tensor(image_tensor, vit_transform)
    vit_tokens = patchify(image_tensor, model.vit_patch_size)
    vit_position_ids = model.get_flattened_position_ids(
        image_tensor.size(1),
        image_tensor.size(2),
        model.vit_patch_size,
        max_num_patches_per_side=model.vit_max_num_patch_per_side,
    )

    vit_token_seqlens = torch.tensor([vit_tokens.shape[0]], dtype=torch.int)
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
    max_seqlen = vit_token_seqlens.max().item()

    device = next(model.vit_model.parameters()).device
    dtype = next(model.vit_model.parameters()).dtype
    vit_tokens = vit_tokens.to(device=device, dtype=dtype)
    vit_position_ids = vit_position_ids.to(device=device)
    vit_token_seqlens = vit_token_seqlens.to(device=device)
    cu_seqlens = cu_seqlens.to(device=device)

    packed_vit_token_embed = model.vit_model(
        packed_pixel_values=vit_tokens,
        packed_flattened_position_ids=vit_position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    packed_vit_token_embed = model.connector(packed_vit_token_embed)
    packed_vit_token_embed = packed_vit_token_embed + model.vit_pos_embed(vit_position_ids)

    return packed_vit_token_embed.mean(dim=0)


def _latent_understanding_embedding(
    model: Bagel,
    vae_model: torch.nn.Module,
    vit_transform: ImageTransform,
    packed_latent_tokens: torch.Tensor,
    image_shape: Tuple[int, int],
) -> torch.Tensor:
    H, W = image_shape
    h, w = H // model.latent_downsample, W // model.latent_downsample
    p = model.latent_patch_size
    c = model.latent_channel
    latent = _unpatchify_latent_tokens(packed_latent_tokens, h, w, p, c).unsqueeze(0)

    vae_device = next(vae_model.parameters()).device
    vae_dtype = next(vae_model.parameters()).dtype
    latent = latent.to(device=vae_device, dtype=vae_dtype)

    image = vae_model.decode(latent)
    image = (image * 0.5 + 0.5).clamp(0, 1)[0]

    return _image_understanding_embedding(model, vit_transform, image)


def _unpatchify_latent_tokens(
    packed_latent_tokens: torch.Tensor,
    h: int,
    w: int,
    p: int,
    c: int,
) -> torch.Tensor:
    latent = packed_latent_tokens.reshape(h, w, p, p, c)
    latent = latent.permute(4, 0, 2, 1, 3).contiguous()
    latent = latent.reshape(c, h * p, w * p)
    return latent


def _build_rollout_model_from_base(base: Bagel, state_dict: Optional[dict] = None) -> Bagel:
    llm_config = deepcopy(base.config.llm_config)
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = None
    if base.config.visual_und and base.config.vit_config is not None:
        vit_config = deepcopy(base.config.vit_config)
        vit_model = SiglipVisionModel(vit_config)
    rollout = Bagel(language_model, vit_model, base.config)
    if base.config.visual_und and vit_model is not None:
        rollout.vit_model.vision_model.embeddings.convert_conv2d_to_linear(base.config.vit_config)
    if state_dict is not None:
        # Drop pos embeds if present to avoid resolution mismatches.
        state_dict.pop('latent_pos_embed.pos_embed', None)
        state_dict.pop('vit_pos_embed.pos_embed', None)
        rollout.load_state_dict(state_dict, strict=False)
    return rollout


def _load_rollout_state_dict(ckpt_dir: str) -> Optional[dict]:
    if ckpt_dir is None:
        return None
    ema_path = os.path.join(ckpt_dir, "ema.safetensors")
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(ema_path):
        return load_file(ema_path, device="cpu")
    if os.path.exists(model_path):
        return load_file(model_path, device="cpu")
    return None


def _build_text_image_inputs(
    model: torch.nn.Module,
    prompt_ids: List[int],
    packed_latent_tokens: torch.Tensor,
    image_shape: Tuple[int, int],
    new_token_ids: dict,
    device: torch.device,
) -> dict:
    module = model.module if hasattr(model, "module") else model
    H, W = image_shape
    p = module.latent_patch_size
    h = H // module.latent_downsample
    w = W // module.latent_downsample
    num_img_tokens = h * w

    # text split
    text_ids = [new_token_ids["bos_token_id"]] + prompt_ids + [new_token_ids["eos_token_id"]]
    text_len = len(text_ids)

    # sequence layout:
    # [text tokens] [<vision_start>] [image tokens] [<vision_end>]
    packed_text_ids = text_ids + [new_token_ids["start_of_image"], new_token_ids["end_of_image"]]
    packed_text_indexes = list(range(text_len)) + [text_len, text_len + num_img_tokens + 1]
    packed_vae_token_indexes = list(range(text_len + 1, text_len + 1 + num_img_tokens))

    total_len = text_len + num_img_tokens + 2
    sample_lens = [total_len]

    split_lens = [text_len, num_img_tokens + 2]
    attn_modes = ["causal", "noise"]
    nested_attention_masks = [prepare_attention_mask_per_sample(split_lens, attn_modes, device=device)]

    packed_position_ids = list(range(text_len)) + [text_len] * (num_img_tokens + 2)
    packed_position_ids = torch.tensor(packed_position_ids, dtype=torch.long, device=device)

    packed_latent_position_ids = module.get_flattened_position_ids(
        H, W, module.latent_downsample, module.max_latent_size
    ).to(device)

    # build padded_latent
    latent = _unpatchify_latent_tokens(
        packed_latent_tokens,
        h=h,
        w=w,
        p=p,
        c=module.latent_channel,
    )
    padded_latent = latent.unsqueeze(0)

    packed_timesteps = torch.randn(num_img_tokens, device=device)
    mse_loss_indexes = torch.zeros(total_len, dtype=torch.bool, device=device)
    mse_loss_indexes[torch.tensor(packed_vae_token_indexes, dtype=torch.long, device=device)] = True

    return dict(
        sequence_length=total_len,
        packed_text_ids=torch.tensor(packed_text_ids, dtype=torch.long, device=device),
        packed_text_indexes=torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
        sample_lens=sample_lens,
        nested_attention_masks=nested_attention_masks,
        packed_position_ids=packed_position_ids,
        padded_latent=padded_latent,
        patchified_vae_latent_shapes=[(h, w)],
        packed_latent_position_ids=packed_latent_position_ids,
        packed_vae_token_indexes=torch.tensor(packed_vae_token_indexes, dtype=torch.long, device=device),
        packed_timesteps=packed_timesteps,
        mse_loss_indexes=mse_loss_indexes,
    )


def _compute_diffusion_logp(
    model: torch.nn.Module,
    prompt_ids: List[int],
    packed_latent_tokens: torch.Tensor,
    image_shape: Tuple[int, int],
    new_token_ids: dict,
    device,
    rng_seed: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(device, torch.device):
        if torch.cuda.is_available():
            device = torch.device("cuda", int(device))
        else:
            device = torch.device("cpu")
    module = model.module if hasattr(model, "module") else model
    restore_visual_und = None
    if rng_seed is not None:
        if device.type == "cuda":
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(rng_seed)
                torch.cuda.manual_seed(rng_seed)
                inputs = _build_text_image_inputs(
                    model, prompt_ids, packed_latent_tokens, image_shape, new_token_ids, device
                )
        else:
            with torch.random.fork_rng():
                torch.manual_seed(rng_seed)
                inputs = _build_text_image_inputs(
                    model, prompt_ids, packed_latent_tokens, image_shape, new_token_ids, device
                )
    else:
        inputs = _build_text_image_inputs(
            model, prompt_ids, packed_latent_tokens, image_shape, new_token_ids, device
        )
    restore_lm_training = None
    try:
        if hasattr(module, "config") and getattr(module.config, "visual_und", False):
            restore_visual_und = module.config.visual_und
            module.config.visual_und = False
        if hasattr(module, "language_model"):
            restore_lm_training = module.language_model.training
            module.language_model.train()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            loss_dict = model(**inputs)
    finally:
        if restore_visual_und is not None:
            module.config.visual_und = restore_visual_und
        if restore_lm_training is not None:
            module.language_model.train(restore_lm_training)
    mse = loss_dict["mse"]
    if mse is None:
        return torch.tensor(0.0, device=device)
    return -mse.mean()


@torch.no_grad()
def _generate_latent_no_tqdm(
    model: Bagel,
    prompt_text: str,
    tokenizer: Qwen2Tokenizer,
    new_token_ids: dict,
    image_shape: Tuple[int, int],
    num_timesteps: int,
    cfg_text_scale: float,
    cfg_img_scale: float,
    cfg_interval: Tuple[float, float],
    timestep_shift: float,
    cfg_renorm_min: float,
    cfg_renorm_type: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    if not hasattr(model.language_model.model, "enable_taylorseer"):
        model.language_model.model.enable_taylorseer = False

    def _to_device(mapping):
        for k, v in mapping.items():
            if torch.is_tensor(v):
                mapping[k] = v.to(device)
        return mapping

    gen_context = {
        "kv_lens": [0],
        "ropes": [0],
        "past_key_values": NaiveCache(model.config.llm_config.num_hidden_layers),
    }
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)

    generation_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=gen_context["kv_lens"],
        curr_rope=gen_context["ropes"],
        prompts=[prompt_text],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = _to_device(generation_input)
    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
        gen_context["past_key_values"] = model.forward_cache_update_text(
            gen_context["past_key_values"], **generation_input
        )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    cfg_img_context = deepcopy(gen_context)

    generation_input = model.prepare_vae_latent(
        curr_kvlens=gen_context["kv_lens"],
        curr_rope=gen_context["ropes"],
        image_sizes=[image_shape],
        new_token_ids=new_token_ids,
    )
    generation_input = _to_device(generation_input)
    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_context["kv_lens"],
        curr_rope=cfg_text_context["ropes"],
        image_sizes=[image_shape],
    )
    generation_input_cfg_text = _to_device(generation_input_cfg_text)
    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_context["kv_lens"],
        curr_rope=cfg_img_context["ropes"],
        image_sizes=[image_shape],
    )
    generation_input_cfg_img = _to_device(generation_input_cfg_img)

    x_t = generation_input["packed_init_noises"]
    timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            v_t = model._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
                packed_vae_position_ids=generation_input["packed_vae_position_ids"],
                packed_text_ids=generation_input["packed_text_ids"],
                packed_text_indexes=generation_input["packed_text_indexes"],
                packed_position_ids=generation_input["packed_position_ids"],
                packed_indexes=generation_input["packed_indexes"],
                packed_seqlens=generation_input["packed_seqlens"],
                key_values_lens=generation_input["key_values_lens"],
                past_key_values=gen_context["past_key_values"],
                packed_key_value_indexes=generation_input["packed_key_value_indexes"],
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=generation_input_cfg_text["cfg_packed_position_ids"],
                cfg_text_packed_query_indexes=generation_input_cfg_text["cfg_packed_query_indexes"],
                cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text["cfg_packed_key_value_indexes"],
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=generation_input_cfg_img["cfg_packed_position_ids"],
                cfg_img_packed_query_indexes=generation_input_cfg_img["cfg_packed_query_indexes"],
                cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img["cfg_packed_key_value_indexes"],
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

    return x_t


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # --- RIR + Consistency DPO ---
    rir_enable: bool = field(
        default=False,
        metadata={"help": "Enable rollout + RIR scoring + consistency DPO training."}
    )
    rir_every: int = field(
        default=1,
        metadata={"help": "Run RIR/DPO update every N steps."}
    )
    rir_prompt_max_tokens: int = field(
        default=128,
        metadata={"help": "Max prompt token length used for rollout and scoring."}
    )
    rir_image_height: int = field(
        default=1024,
        metadata={"help": "Height of generated image for RIR rollout (pixels)."}
    )
    rir_image_width: int = field(
        default=1024,
        metadata={"help": "Width of generated image for RIR rollout (pixels)."}
    )
    rir_num_timesteps: int = field(
        default=50,
        metadata={"help": "Diffusion steps for RIR rollout generation."}
    )
    rir_timestep_shift: float = field(
        default=3.0,
        metadata={"help": "Timestep shift for RIR rollout generation."}
    )
    rir_cfg_text_scale: float = field(
        default=4.0,
        metadata={"help": "CFG text scale for RIR rollout generation."}
    )
    rir_cfg_img_scale: float = field(
        default=1.0,
        metadata={"help": "CFG image scale for RIR rollout generation."}
    )
    rir_cfg_interval: str = field(
        default="0.4,1.0",
        metadata={"help": "CFG interval as 'start,end' for RIR rollout generation."}
    )
    rir_cfg_renorm_min: float = field(
        default=0.0,
        metadata={"help": "CFG renorm min for RIR rollout generation."}
    )
    rir_cfg_renorm_type: str = field(
        default="global",
        metadata={"help": "CFG renorm type for RIR rollout generation."}
    )
    rir_beta: float = field(
        default=0.1,
        metadata={"help": "Beta temperature for consistency DPO."}
    )
    rir_dpo_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for consistency DPO loss."}
    )
    rir_grounding_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for latent grounding loss."}
    )
    rir_use_visual_und: bool = field(
        default=True,
        metadata={"help": "Use image-understanding loopback (VAE -> image -> ViT) for RIR scoring/grounding."}
    )
    rir_separate_backward: bool = field(
        default=True,
        metadata={"help": "Backprop RIR losses in a separate backward pass to reduce peak memory."}
    )
    rir_rollout_device: str = field(
        default="",
        metadata={"help": "Device for a separate rollout model (e.g., 'cuda:0'). Empty uses no separate model."}
    )
    rir_rollout_lazy: bool = field(
        default=True,
        metadata={"help": "Lazily create the rollout model at first use to reduce startup memory."}
    )
    rir_rollout_free_each_step: bool = field(
        default=False,
        metadata={"help": "Free the rollout model after each RIR step to reduce RAM usage (slower)."}
    )
    rir_skip_logp: bool = field(
        default=False,
        metadata={"help": "Skip DPO log-prob computation to save memory (uses grounding loss only)."}
    )
    # --- LoRA ---
    lora_rank: int = field(
        default=0,
        metadata={"help": "LoRA rank (0 disables LoRA)."}
    )
    lora_alpha: int = field(
        default=0,
        metadata={"help": "LoRA alpha scaling (ignored if lora_rank=0)."}
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "Comma-separated module names to apply LoRA to."}
    )
    shared_latent_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for text-image shared latent alignment loss."},
    )
    shared_latent_normalize: bool = field(
        default=True,
        metadata={"help": "L2-normalize text/image latent summaries before alignment."},
    )
    shared_latent_stopgrad: bool = field(
        default=True,
        metadata={"help": "Use symmetric stop-grad alignment loss for stability."},
    )
    shared_latent_prompt_max_tokens: int = field(
        default=256,
        metadata={"help": "Maximum prompt token length used for shared-latent alignment."},
    )
    shared_latent_teacher_ckpt: str = field(
        default="<path-to-stage2-rollout-model-ckpt>",
        metadata={"help": "Optional Stage1/Stage2 checkpoint to provide a frozen shared-latent teacher."},
    )
    shared_latent_teacher_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for teacher-guided latent target loss (0 disables teacher term)."},
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    use_lora = training_args.lora_rank is not None and training_args.lora_rank > 0

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = None if use_lora else deepcopy(model)
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    if use_lora:
        # Freeze base model params; only LoRA layers train.
        for param in model.parameters():
            param.requires_grad = False
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            task_type=None,
            target_modules=[
                "q_proj", "v_proj", "q_proj_moe_gen", "v_proj_moe_gen",
                "gate_proj", "up_proj", "down_proj"
            ] if training_args.lora_target_modules is None else training_args.lora_target_modules.split(","),
        )
        model = get_peft_model(model, peft_config)
        if dist.get_rank() == 0:
            model.print_trainable_parameters()
    rollout_model = None
    rollout_device = None
    if training_args.rir_enable and training_args.rir_rollout_device:
        if training_args.rir_rollout_device in ("cuda", "cuda:local", "local"):
            rollout_device = torch.device("cuda", torch.cuda.current_device())
        else:
            rollout_device = torch.device(training_args.rir_rollout_device)
    if ema_model is not None:
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    if use_lora:
        fsdp_model = fsdp_with_lora_wrapper(model, fsdp_config)
    else:
        fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    shared_latent_teacher = None
    if training_args.shared_latent_teacher_weight > 0:
        shared_latent_teacher = _load_shared_latent_teacher(
            training_args.shared_latent_teacher_ckpt,
            device,
            logger,
        )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    rir_vit_transform = None
    if training_args.rir_enable and training_args.rir_use_visual_und:
        if not training_args.visual_und:
            if dist.get_rank() == 0:
                logger.warning("RIR visual-understanding loopback requested but visual_und=False; falling back to latent scoring.")
        else:
            rir_vit_max_image_size = model_args.vit_patch_size * model_args.vit_max_num_patch_per_side
            rir_vit_min_image_size = model_args.vit_patch_size * 16
            rir_vit_transform = ImageTransform(
                rir_vit_max_image_size,
                rir_vit_min_image_size,
                model_args.vit_patch_size,
            )
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    shared_teacher_dim_warned = False
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                loss_dict = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        # Shared latent alignment (text summaries vs image latent summaries).
        shared_latent_loss = torch.tensor(0.0, device=device)
        shared_teacher_loss = torch.tensor(0.0, device=device)
        if training_args.shared_latent_weight > 0 and training_args.visual_gen:
            prompt_ids = _extract_prompt_ids(
                data["packed_text_ids"],
                new_token_ids["bos_token_id"],
                new_token_ids["eos_token_id"],
                training_args.shared_latent_prompt_max_tokens,
            )
            if prompt_ids is not None and "packed_latent_tokens" in data:
                base_model = fsdp_model.module if hasattr(fsdp_model, "module") else fsdp_model
                z_text = _mean_text_embedding(base_model, prompt_ids, new_token_ids)
                z_image = _latent_mean_embedding(base_model, data["packed_latent_tokens"])

                if training_args.shared_latent_normalize:
                    z_text = torch.nn.functional.normalize(z_text, dim=0)
                    z_image = torch.nn.functional.normalize(z_image, dim=0)

                if training_args.shared_latent_stopgrad:
                    loss_t = torch.mean((z_text - z_image.detach()) ** 2)
                    loss_i = torch.mean((z_image - z_text.detach()) ** 2)
                    shared_latent_loss = loss_t + loss_i
                else:
                    shared_latent_loss = torch.mean((z_text - z_image) ** 2)

                if shared_latent_teacher is not None and training_args.shared_latent_teacher_weight > 0:
                    teacher_text_dim = int(shared_latent_teacher.aligner.text_proj.in_features)
                    teacher_image_dim = int(shared_latent_teacher.aligner.image_proj.in_features)
                    student_dim = int(z_text.shape[0])
                    teacher_latent_dim = int(shared_latent_teacher.aligner.fuse[-1].out_features)
                    if (
                        student_dim == teacher_text_dim
                        and student_dim == teacher_image_dim
                        and student_dim == teacher_latent_dim
                    ):
                        with torch.no_grad():
                            z_teacher = shared_latent_teacher.aligner(
                                z_text.unsqueeze(0), z_image.unsqueeze(0)
                            ).squeeze(0)
                        z_student = 0.5 * (z_text + z_image)
                        shared_teacher_loss = torch.mean((z_student - z_teacher) ** 2)
                        loss = loss + training_args.shared_latent_teacher_weight * shared_teacher_loss
                    elif not shared_teacher_dim_warned:
                        logger.warning(
                            "Shared latent teacher dims do not match Bagel latent summaries; "
                            f"teacher(text={teacher_text_dim}, image={teacher_image_dim}, latent={teacher_latent_dim}), "
                            f"student={student_dim}. Teacher term will be skipped."
                        )
                        shared_teacher_dim_warned = True

                loss = loss + training_args.shared_latent_weight * shared_latent_loss

        loss_dict["shared_latent"] = shared_latent_loss.detach()
        loss_dict["shared_latent_teacher"] = shared_teacher_loss.detach()

        base_loss = loss / training_args.gradient_accumulation_steps
        base_backprop_done = False
        if training_args.rir_separate_backward:
            base_loss.backward()
            base_backprop_done = True

        # RIR + Consistency DPO
        if training_args.rir_enable and training_args.visual_gen and (curr_step % training_args.rir_every == 0):
            cfg_interval = _parse_float_pair(training_args.rir_cfg_interval, (0.4, 1.0))
            prompt_ids = _extract_prompt_ids(
                data["packed_text_ids"],
                new_token_ids["bos_token_id"],
                new_token_ids["eos_token_id"],
                training_args.rir_prompt_max_tokens,
            )
            if prompt_ids is not None:
                prompt_text = tokenizer.decode(prompt_ids)

                if rollout_model is None:
                    if rollout_device is None:
                        if dist.get_rank() == 0:
                            logger.warning("RIR rollout skipped: set --rir_rollout_device to enable separate rollout model.")
                        continue
                    if rollout_device.type == "cpu":
                        if dist.get_rank() == 0:
                            logger.warning("RIR rollout skipped: CPU rollout is not supported because flash-attn requires CUDA.")
                        continue
                    if dist.get_rank() == 0:
                        logger.info(f"Creating rollout model on {rollout_device} (lazy={training_args.rir_rollout_lazy}).")
                    try:
                        ckpt_source = resume_from or model_args.model_path
                        rollout_state = _load_rollout_state_dict(ckpt_source)
                        if rollout_state is None:
                            if dist.get_rank() == 0:
                                logger.warning(f"RIR rollout skipped: no safetensors found in {ckpt_source}.")
                            continue
                        rollout_model = _build_rollout_model_from_base(model, state_dict=rollout_state)
                        rollout_model.to(rollout_device, dtype=torch.bfloat16)
                        rollout_model.eval()
                        for param in rollout_model.parameters():
                            param.requires_grad = False
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if dist.get_rank() == 0:
                                logger.warning("RIR rollout model OOM; skipping rollout for this step.")
                            rollout_model = None
                            if rollout_device.type == "cuda":
                                torch.cuda.empty_cache()
                            continue
                        raise

                # Generate two candidate latents (separate rollout model)
                with torch.no_grad():
                    lat1 = _generate_latent_no_tqdm(
                        rollout_model,
                        prompt_text,
                        tokenizer,
                        new_token_ids,
                        (training_args.rir_image_height, training_args.rir_image_width),
                        training_args.rir_num_timesteps,
                        training_args.rir_cfg_text_scale,
                        training_args.rir_cfg_img_scale,
                        cfg_interval,
                        training_args.rir_timestep_shift,
                        training_args.rir_cfg_renorm_min,
                        training_args.rir_cfg_renorm_type,
                    )
                    lat2 = _generate_latent_no_tqdm(
                        rollout_model,
                        prompt_text,
                        tokenizer,
                        new_token_ids,
                        (training_args.rir_image_height, training_args.rir_image_width),
                        training_args.rir_num_timesteps,
                        training_args.rir_cfg_text_scale,
                        training_args.rir_cfg_img_scale,
                        cfg_interval,
                        training_args.rir_timestep_shift,
                        training_args.rir_cfg_renorm_min,
                        training_args.rir_cfg_renorm_type,
                    )

                # RIR scoring (cosine)
                prompt_emb = _mean_text_embedding(rollout_model, prompt_ids, new_token_ids)
                use_rir_understanding = (
                    training_args.rir_use_visual_und
                    and rir_vit_transform is not None
                    and rollout_model is not None
                    and rollout_model.config.visual_und
                )
                if use_rir_understanding:
                    try:
                        lat1_emb = _latent_understanding_embedding(
                            rollout_model,
                            vae_model,
                            rir_vit_transform,
                            lat1,
                            (training_args.rir_image_height, training_args.rir_image_width),
                        )
                        lat2_emb = _latent_understanding_embedding(
                            rollout_model,
                            vae_model,
                            rir_vit_transform,
                            lat2,
                            (training_args.rir_image_height, training_args.rir_image_width),
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if dist.get_rank() == 0:
                                logger.warning("RIR understanding embedding OOM; falling back to latent embeddings for this step.")
                            torch.cuda.empty_cache()
                            lat1_emb = _latent_mean_embedding(rollout_model, lat1)
                            lat2_emb = _latent_mean_embedding(rollout_model, lat2)
                        else:
                            raise
                else:
                    lat1_emb = _latent_mean_embedding(rollout_model, lat1)
                    lat2_emb = _latent_mean_embedding(rollout_model, lat2)
                score1 = torch.nn.functional.cosine_similarity(lat1_emb.float(), prompt_emb.float(), dim=0)
                score2 = torch.nn.functional.cosine_similarity(lat2_emb.float(), prompt_emb.float(), dim=0)

                if score1 >= score2:
                    lat_w, lat_l = lat1, lat2
                    score_w, score_l = score1, score2
                    lat_w_emb, lat_l_emb = lat1_emb, lat2_emb
                else:
                    lat_w, lat_l = lat2, lat1
                    score_w, score_l = score2, score1
                    lat_w_emb, lat_l_emb = lat2_emb, lat1_emb

                dpo_loss = torch.tensor(0.0, device=device)
                if not training_args.rir_skip_logp:
                    # Diffusion logp (current + reference) for DPO
                    seed_base = (curr_step + 1) * 1000003 + dist.get_rank()
                    train_device = torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
                    lat_w = lat_w.to(train_device)
                    lat_l = lat_l.to(train_device)
                    try:
                        logp_w = _compute_diffusion_logp(
                            fsdp_model,
                            prompt_ids,
                            lat_w,
                            (training_args.rir_image_height, training_args.rir_image_width),
                            new_token_ids,
                            device,
                            rng_seed=seed_base,
                        )
                        logp_l = _compute_diffusion_logp(
                            fsdp_model,
                            prompt_ids,
                            lat_l,
                            (training_args.rir_image_height, training_args.rir_image_width),
                            new_token_ids,
                            device,
                            rng_seed=seed_base + 1,
                        )
                        if ema_model is None:
                            logp_w_ref = logp_w.detach()
                            logp_l_ref = logp_l.detach()
                        else:
                            with torch.no_grad():
                                logp_w_ref = _compute_diffusion_logp(
                                    ema_model,
                                    prompt_ids,
                                    lat_w,
                                    (training_args.rir_image_height, training_args.rir_image_width),
                                    new_token_ids,
                                    device,
                                    rng_seed=seed_base,
                                )
                                logp_l_ref = _compute_diffusion_logp(
                                    ema_model,
                                    prompt_ids,
                                    lat_l,
                                    (training_args.rir_image_height, training_args.rir_image_width),
                                    new_token_ids,
                                    device,
                                    rng_seed=seed_base + 1,
                                )

                        dpo_term = training_args.rir_beta * (
                            (logp_w - logp_l) - (logp_w_ref - logp_l_ref)
                        )
                        dpo_loss = -torch.nn.functional.logsigmoid(dpo_term)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if dist.get_rank() == 0:
                                logger.warning("RIR DPO logp OOM; skipping logp for this step.")
                            torch.cuda.empty_cache()
                        else:
                            raise

                grounding_loss = torch.mean((lat_w_emb.float() - prompt_emb.float()) ** 2)

                loss_dict["rir_score_w"] = score_w.detach()
                loss_dict["rir_score_l"] = score_l.detach()
                loss_dict["rir_dpo"] = dpo_loss.detach()
                loss_dict["rir_grounding"] = grounding_loss.detach()

                rir_loss = training_args.rir_dpo_weight * dpo_loss + training_args.rir_grounding_weight * grounding_loss
                if training_args.rir_separate_backward:
                    if rir_loss.requires_grad:
                        (rir_loss / training_args.gradient_accumulation_steps).backward()
                    else:
                        if dist.get_rank() == 0:
                            logger.warning("RIR loss has no grad; skipping RIR backward (likely rir_skip_logp=True).")
                else:
                    loss = loss + rir_loss
                
                # Release rollout tensors early to reduce peak memory.
                del lat1, lat2, lat_w, lat_l
                torch.cuda.empty_cache()
                if rollout_model is not None and training_args.rir_rollout_free_each_step:
                    if dist.get_rank() == 0:
                        logger.info("Freeing rollout model after RIR step.")
                    rollout_model = None
                    if rollout_device is not None and rollout_device.type == "cuda":
                        torch.cuda.empty_cache()

        if not base_backprop_done:
            (loss / training_args.gradient_accumulation_steps).backward()

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
            optimizer.zero_grad()
        
        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Clear caches and ensure all CUDA operations complete before checkpoint
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
            except RuntimeError as e:
                logger.error(f"Error during gather_object at step {curr_step}: {e}")
                gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            # Clear CUDA cache and force garbage collection after checkpoint to free memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # comment out as an alternative to save the ema model in pt format
            # ema_state_dict = {}
            # for name, param in ema_model.named_parameters():
            #     ema_state_dict[name] = param.detach().cpu()
            
            # torch.save(
            #     ema_state_dict, 
            #     os.path.join(training_args.checkpoint_dir, f"{curr_step:07d}", "ema_standard.pt")
            # )
    
    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        # Clear caches and ensure all CUDA operations complete before final checkpoint
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
        except RuntimeError as e:
            logger.error(f"Error during final gather_object: {e}")
            gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()
        
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, 
            train_steps=curr_step, 
            model=fsdp_model, 
            ema_model=ema_model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list
        )
        # Clear CUDA cache and force garbage collection after final checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
