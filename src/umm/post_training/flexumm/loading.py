"""Construction and checkpoint loading for the FlexUMM BAGEL runtime."""

from __future__ import annotations

import warnings
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

from .bootstrap import ensure_bagel_importable
from .config import DynamicDepthConfig
from .modeling import DynamicBagel, enable_dynamic_depth
from .overlays import apply_overlay_to_model
from .protocols import VLM_EVAL_VIT_TRANSFORM

ensure_bagel_importable()

from data.data_utils import add_special_tokens  # noqa: E402
from data.transforms import ImageTransform  # noqa: E402
from inferencer import InterleaveInferencer  # noqa: E402
from modeling.autoencoder import load_ae  # noqa: E402
from modeling.bagel import (  # noqa: E402
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402


_TORCHUMM_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "FLEXUMM_MODEL_PATH",
        str(_TORCHUMM_ROOT / "model_cache" / "bagel" / "models" / "BAGEL-7B-MoT"),
    )
)
DEFAULT_DEPTH_CONFIG = Path(__file__).resolve().parent / "configs" / "dynamic_depth.json"


@dataclass
class DynamicBagelRuntime:
    model: DynamicBagel
    inferencer: InterleaveInferencer
    vae_model: Any
    tokenizer: Any
    new_token_ids: dict[str, int]


def load_dynamic_bagel(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    checkpoint_path: str | Path | None = None,
    shared_overlay_path: str | Path | None = None,
    depth_config: DynamicDepthConfig | str | Path | None = DEFAULT_DEPTH_CONFIG,
    max_memory_per_gpu: str = "80GiB",
    offload_folder: str | Path = "./tmp/offload",
    device_id: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    vit_max_image_size: int = VLM_EVAL_VIT_TRANSFORM["max_image_size"],
    vit_min_image_size: int = VLM_EVAL_VIT_TRANSFORM["min_image_size"],
    vit_image_stride: int = VLM_EVAL_VIT_TRANSFORM["image_stride"],
    vit_max_pixels: int = VLM_EVAL_VIT_TRANSFORM["max_pixels"],
) -> DynamicBagelRuntime:
    """Load BAGEL-7B-MoT and enable the configured FlexUMM route.

    The default configuration enables the current TAFE FFN-pool path. Passing
    ``depth_config=None`` creates the original full-depth policy with routing
    disabled, retaining the compatibility API.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL-7B-MoT loading requires at least one CUDA device")

    model_path = Path(model_path).expanduser().resolve()
    checkpoint = _resolve_checkpoint(model_path, checkpoint_path)

    if depth_config is None:
        resolved_depth_config = DynamicDepthConfig()
    elif isinstance(depth_config, DynamicDepthConfig):
        resolved_depth_config = depth_config
    else:
        resolved_depth_config = DynamicDepthConfig.from_json(depth_config)
    checkpoint_keys: set[str] = set()
    checkpoint_has_residual_gates = False
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
        checkpoint_has_residual_gates = any(
            key.endswith((".gen_attn_gate", ".gen_mlp_gate"))
            for key in checkpoint_keys
        )
        if (
            resolved_depth_config.router_enabled
            and resolved_depth_config.router_candidates_understanding
        ):
            has_understanding_router = any(
                ".depth_router." in key or key.startswith("depth_router.")
                for key in checkpoint_keys
            )
        else:
            has_understanding_router = False
        if (
            resolved_depth_config.router_enabled
            and resolved_depth_config.router_candidates_generation
        ):
            has_generation_router = any(
                ".generation_depth_router." in key
                or key.startswith("generation_depth_router.")
                for key in checkpoint_keys
            )
        else:
            has_generation_router = False
    if resolved_depth_config.router_enabled:
        if (
            resolved_depth_config.router_candidates_understanding
            and not has_understanding_router
        ):
            raise ValueError(
                f"{checkpoint} has no trained FlexUMM router parameters. "
                "Use a post-training checkpoint, or select a rule-only depth config."
            )
        if (
            resolved_depth_config.router_candidates_generation
            and not has_generation_router
        ):
            raise ValueError(
                f"{checkpoint} has no trained FlexUMM generation-router "
                "parameters. Use a dual-gate checkpoint, or select a config "
                "without router_candidates_generation."
            )

    llm_config = Qwen2Config.from_json_file(model_path / "llm_config.json")
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.use_residual_gates = checkpoint_has_residual_gates

    vit_config = SiglipVisionConfig.from_json_file(model_path / "vit_config.json")
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=model_path / "ae.safetensors")
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
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
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )
        model = enable_dynamic_depth(model, resolved_depth_config)
        _materialize_missing_router_risk_head(model, checkpoint_keys, dtype)
        _materialize_missing_tafe_parameters(
            model,
            checkpoint_keys,
            resolved_depth_config,
            dtype,
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    # Match eval/vlm/utils.py and the MoE benchmark runner exactly. The
    # interactive demo uses a smaller 224-pixel minimum, but that is not the
    # reported VLM benchmark protocol.
    vit_transform = ImageTransform(
        max_image_size=int(vit_max_image_size),
        min_image_size=int(vit_min_image_size),
        image_stride=int(vit_image_stride),
        max_pixels=int(vit_max_pixels),
    )

    if device_id is not None:
        device_id = int(device_id)
        if not 0 <= device_id < torch.cuda.device_count():
            raise ValueError(
                f"device_id={device_id} is not available; "
                f"CUDA device count is {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(device_id)
        # A Unison worker is assigned one physical GPU.  The model fits on a
        # 96-GiB device, so bind the complete runtime to that worker GPU
        # instead of letting accelerate greedily place it on cuda:0.
        device_map = {"": f"cuda:{device_id}"}
    else:
        max_memory = {
            index: max_memory_per_gpu for index in range(torch.cuda.device_count())
        }
        # Keep low-memory runs in host RAM rather than Accelerate's disk
        # offload path.  Disk offload leaves custom FlexUMM controller inputs
        # on meta tensors and is also needlessly slow on a single A100.
        max_memory["cpu"] = "160GiB"
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["DynamicBagel", "Qwen2MoTDecoderLayer"],
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
    if resolved_depth_config.router_enabled:
        same_device_modules.append("language_model.model.depth_router")
        same_device_modules.append(
            "language_model.model.generation_depth_router"
        )
    if resolved_depth_config.tafe_enabled:
        same_device_modules.extend(
            [
                "language_model.model.tafe_gate_understanding",
                "language_model.model.tafe_gate_generation",
            ]
        )
    if resolved_depth_config.tafe_attention_enabled:
        same_device_modules.extend(
            [
                "language_model.model.tafe_attention_gate_understanding",
                "language_model.model.tafe_attention_gate_generation",
            ]
        )
    if resolved_depth_config.fusion_sealing_enabled:
        same_device_modules.extend(
            [
                "language_model.model.fusion_sealing_compilers",
                "language_model.model.fusion_sealing_realizers",
                "language_model.model.fusion_release_controller",
            ]
        )
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for module_name in same_device_modules:
            device_map[module_name] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for module_name in same_device_modules:
            if module_name in device_map:
                device_map[module_name] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint),
        device_map=device_map,
        offload_buffers=True,
        dtype=dtype,
        force_hooks=True,
        offload_folder=str(offload_folder),
    ).eval()
    # ``action_costs`` is a persistent checkpoint buffer for compatibility,
    # but the dynamic-depth config is the authoritative runtime policy.  A
    # checkpoint may have been trained with a different cost vector, so apply
    # the requested inference/training policy after dispatch.  This keeps
    # action-cost sweeps from silently reusing stale checkpoint costs.
    if resolved_depth_config.tafe_enabled:
        tafe_model = model.language_model.model
        for gate_name in (
            "tafe_gate_understanding",
            "tafe_gate_generation",
        ):
            gate = getattr(tafe_model, gate_name, None)
            if gate is None or not hasattr(gate, "action_costs"):
                continue
            gate.action_costs.copy_(
                torch.as_tensor(
                    resolved_depth_config.tafe_action_costs,
                    device=gate.action_costs.device,
                    dtype=gate.action_costs.dtype,
                )
            )
    # The BAGEL autoencoder is intentionally kept in FP32.  Move it to the
    # execution device here; otherwise the interleaved generation path sends
    # BF16 latents to a CPU VAE and fails at its first convolution.
    execution_device = next(
        parameter.device
        for parameter in model.parameters()
        if parameter.device.type != "meta"
    )
    vae_model = vae_model.to(execution_device).eval()
    if shared_overlay_path is not None:
        apply_overlay_to_model(model, shared_overlay_path)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return DynamicBagelRuntime(
        model=model,
        inferencer=inferencer,
        vae_model=vae_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )


def _materialize_missing_router_risk_head(
    model: torch.nn.Module,
    checkpoint_keys: set[str],
    dtype: torch.dtype,
) -> None:
    """Backfill the optional risk head for older 13-tensor router checkpoints."""

    state = model.state_dict()
    missing = sorted(
        key
        for key in state
        if ".depth_router.risk_output." in key and key not in checkpoint_keys
    )
    if not missing:
        return

    warnings.warn(
        "Checkpoint is missing depth_router.risk_output tensors; initializing "
        "the backward-compatible risk head defaults for load-time dispatch.",
        RuntimeWarning,
        stacklevel=2,
    )
    for key in missing:
        shape = tuple(state[key].shape)
        value = torch.zeros(shape, dtype=dtype)
        if key.endswith("risk_output.3.bias") and shape == (2,):
            value = torch.tensor([0.0, -2.0], dtype=dtype)
            set_module_tensor_to_device(model, key, "cpu", value=value)


def _materialize_missing_tafe_parameters(
    model: torch.nn.Module,
    checkpoint_keys: set[str],
    config: DynamicDepthConfig,
    dtype: torch.dtype,
) -> None:
    """Initialize new TAFE attention/FFN-bank tensors absent from the base model.

    The pretrained BAGEL checkpoint contains the original FFN projections but
    cannot contain the newly introduced specialized banks or TAFE controllers.
    Materializing those tensors before Accelerate dispatch keeps a base model
    usable for Stage-1 post-training.
    """

    state = model.state_dict()
    missing = sorted(
        key
        for key in state
        if key not in checkpoint_keys
        and (
            ".tafe_gate_understanding." in key
            or ".tafe_gate_generation." in key
            or ".tafe_attention_gate_understanding." in key
            or ".tafe_attention_gate_generation." in key
            or ".specialized_banks." in key
        )
    )
    if not missing:
        return

    warnings.warn(
        "Checkpoint is missing FlexUMM TAFE/FFN-pool tensors; initializing "
        "them for post-training.",
        RuntimeWarning,
        stacklevel=2,
    )
    for key in missing:
        shape = tuple(state[key].shape)
        value = torch.zeros(shape, dtype=dtype)
        if key.endswith(".down.weight"):
            value.normal_(mean=0.0, std=0.02)
        if (
            ("tafe_gate_" in key or "tafe_attention_gate_" in key)
            and key.endswith("utility_head.3.bias")
            and shape == (3,)
        ):
            value = torch.tensor(
                config.tafe_initial_action_bias,
                dtype=dtype,
            )
        elif (
            ("tafe_gate_" in key or "tafe_attention_gate_" in key)
            and key.endswith("action_costs")
            and shape == (3,)
        ):
            value = torch.tensor(config.tafe_action_costs, dtype=dtype)
        elif (
            ("tafe_gate_" in key or "tafe_attention_gate_" in key)
            and key.endswith("initial_action_bias")
            and shape == (3,)
        ):
            value = torch.tensor(
                config.tafe_initial_action_bias,
                dtype=dtype,
            )
        set_module_tensor_to_device(model, key, "cpu", value=value)


def _resolve_checkpoint(
    model_path: Path, checkpoint_path: str | Path | None
) -> Path:
    if checkpoint_path is None:
        checkpoint = model_path / "ema.safetensors"
    else:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if checkpoint.is_dir():
            ema_checkpoint = checkpoint / "ema.safetensors"
            model_checkpoint = checkpoint / "model.safetensors"
            checkpoint = (
                ema_checkpoint if ema_checkpoint.is_file() else model_checkpoint
            )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing BAGEL checkpoint: {checkpoint}")
    return checkpoint
