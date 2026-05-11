from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from umm.post_training.unipath.planner.models import PATHS
from umm.post_training.unipath.prompts import DEFAULT_LEVEL_PROMPTS


def path_prompt(level: str, query: str) -> str:
    if level == "direct":
        return query.strip()
    prompt = DEFAULT_LEVEL_PROMPTS[level].strip()
    return f"{prompt}\n\nQuery:\n{query.strip()}".strip()


def _update_text_with_output(inferencer: Any, text: str, gen_context: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    model = inferencer.model
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    generation_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        prompts=[text],
        tokenizer=inferencer.tokenizer,
        new_token_ids=inferencer.new_token_ids,
    )

    text_device = model.language_model.model.embed_tokens.weight.device
    packed_text_ids = generation_input["packed_text_ids"].to(text_device)
    packed_text_position_ids = generation_input["packed_text_position_ids"].to(text_device)
    text_token_lens = generation_input["text_token_lens"].to(text_device)
    packed_text_indexes = generation_input["packed_text_indexes"].to(text_device)
    packed_key_value_indexes = generation_input["packed_key_value_indexes"].to(text_device)
    key_values_lens = generation_input["key_values_lens"].to(text_device)
    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)

    extra_inputs = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_text_embedding,
        query_lens=text_token_lens,
        packed_query_position_ids=packed_text_position_ids,
        packed_query_indexes=packed_text_indexes,
        past_key_values=past_key_values,
        packed_key_value_indexes=packed_key_value_indexes,
        key_values_lens=key_values_lens,
        update_past_key_values=True,
        is_causal=True,
        **extra_inputs,
    )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = output.past_key_values
    return gen_context, output.packed_query_sequence


def _update_image_with_output(inferencer: Any, image: Image.Image, gen_context: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    model = inferencer.model
    past_key_values = gen_context["past_key_values"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]
    generation_input, kv_lens, ropes = model.prepare_vit_images(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        images=[image],
        transforms=inferencer.vit_transform,
        new_token_ids=inferencer.new_token_ids,
    )

    text_device = model.language_model.model.embed_tokens.weight.device
    vit_device = next(model.vit_model.parameters()).device
    vit_dtype = next(model.vit_model.parameters()).dtype
    packed_text_ids = generation_input["packed_text_ids"].to(text_device)
    packed_text_indexes = generation_input["packed_text_indexes"].to(text_device)
    packed_vit_token_indexes = generation_input["packed_vit_token_indexes"].to(text_device)
    packed_vit_position_ids = generation_input["packed_vit_position_ids"].to(vit_device)
    vit_token_seqlens = generation_input["vit_token_seqlens"].to(vit_device)
    packed_vit_tokens = generation_input["packed_vit_tokens"].to(device=vit_device, dtype=vit_dtype)
    packed_position_ids = generation_input["packed_position_ids"].to(text_device)
    packed_seqlens = generation_input["packed_seqlens"].to(text_device)
    packed_indexes = generation_input["packed_indexes"].to(text_device)
    packed_key_value_indexes = generation_input["packed_key_value_indexes"].to(text_device)
    key_values_lens = generation_input["key_values_lens"].to(text_device)

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros((int(packed_seqlens.sum().item()), model.hidden_size))
    packed_sequence[packed_text_indexes] = packed_text_embedding

    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
    max_seqlen = torch.max(vit_token_seqlens).item()
    packed_vit_token_embed = model.vit_model(
        packed_pixel_values=packed_vit_tokens,
        packed_flattened_position_ids=packed_vit_position_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    packed_vit_token_embed = model.connector(packed_vit_token_embed)
    packed_vit_token_embed = packed_vit_token_embed + model.vit_pos_embed(packed_vit_position_ids)
    if packed_vit_token_embed.dtype != packed_sequence.dtype:
        packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
    packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

    extra_inputs = {"mode": "und"} if model.use_moe else {}
    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=packed_position_ids,
        packed_query_indexes=packed_indexes,
        past_key_values=past_key_values,
        packed_key_value_indexes=packed_key_value_indexes,
        key_values_lens=key_values_lens,
        update_past_key_values=True,
        is_causal=False,
        **extra_inputs,
    )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = output.past_key_values
    return gen_context, output.packed_query_sequence


def _split_direct_feature(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if feature.ndim != 1 or feature.numel() % 3 != 0:
        raise ValueError(f"Expected 1D [text_last,text_mean,image_mean], got shape={tuple(feature.shape)}")
    hidden = feature.numel() // 3
    return feature[:hidden], feature[hidden : 2 * hidden], feature[2 * hidden :]


def _split_text_feature(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if feature.ndim != 1 or feature.numel() % 2 != 0:
        raise ValueError(f"Expected 1D [text_last,text_mean], got shape={tuple(feature.shape)}")
    hidden = feature.numel() // 2
    return feature[:hidden], feature[hidden:]


@torch.no_grad()
def encode_multimodal_feature(inferencer: Any, prompt: str, image_paths: list[str]) -> torch.Tensor:
    from data.data_utils import pil_img2rgb

    gen_context = inferencer.init_gen_context()
    image_seq: torch.Tensor | None = None
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for image_path in image_paths:
            image = Image.open(Path(image_path)).convert("RGB")
            image = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
            gen_context, image_seq = _update_image_with_output(inferencer, image, gen_context)
        gen_context, text_seq = _update_text_with_output(inferencer, prompt, gen_context)

    text_seq = text_seq.detach().float().cpu()
    text_last = text_seq[-1]
    text_mean = text_seq.mean(dim=0)
    image_mean = torch.zeros_like(text_mean) if image_seq is None else image_seq.detach().float().cpu().mean(dim=0)
    return torch.cat([text_last, text_mean, image_mean], dim=0)


@torch.no_grad()
def encode_text_feature(inferencer: Any, prompt: str) -> torch.Tensor:
    gen_context = inferencer.init_gen_context()
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        _, text_seq = _update_text_with_output(inferencer, prompt, gen_context)
    text_seq = text_seq.detach().float().cpu()
    return torch.cat([text_seq[-1], text_seq.mean(dim=0)], dim=0)


def build_online_planner_feature(
    inferencer: Any,
    query: str,
    image_paths: list[str],
    *,
    feature_layout: str,
) -> torch.Tensor:
    direct_prompt = path_prompt("direct", query)
    direct_feature = encode_multimodal_feature(inferencer, direct_prompt, image_paths)
    direct_last, direct_mean, image_mean = _split_direct_feature(direct_feature)

    if feature_layout == "all_abs":
        parts = [image_mean]
        for path_name in PATHS:
            text_last, text_mean = _split_text_feature(encode_text_feature(inferencer, path_prompt(path_name, query)))
            parts.extend([text_last, text_mean])
        return torch.cat(parts, dim=0)

    if feature_layout == "direct_plus_delta":
        parts = [image_mean, direct_last, direct_mean]
        for path_name in PATHS[1:]:
            text_last, text_mean = _split_text_feature(encode_text_feature(inferencer, path_prompt(path_name, query)))
            parts.extend([text_last - direct_last, text_mean - direct_mean])
        return torch.cat(parts, dim=0)

    return direct_feature
