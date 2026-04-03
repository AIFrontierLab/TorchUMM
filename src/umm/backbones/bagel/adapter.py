from __future__ import annotations

import importlib
import json
import os
import random
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from safetensors.torch import load_file as st_load_file, save_file as st_save_file


def _img_format(path: str | Path) -> str:
    """Infer PIL save format from file extension, defaulting to PNG."""
    ext = Path(path).suffix.lower()
    return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext.lstrip("."), "PNG")


class BagelBackbone:
    name = "bagel"

    def __init__(
        self,
        model_path: Optional[str] = None,
        bagel_root: Optional[str] = None,
        max_mem_per_gpu: str = "80GiB",
        offload_folder: str = "./tmp/offload",
        seed: int = 42,
        distributed_single_gpu: bool = False,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.bagel_root = (
            Path(bagel_root).expanduser()
            if bagel_root
            else repo_root / "src" / "umm" / "backbones" / "bagel" / "Bagel"
        )
        self.model_path = (
            str(Path(model_path).expanduser())
            if model_path
            else str(repo_root / "model_cache" / "bagel" / "models" / "BAGEL-7B-MoT")
        )
        self.max_mem_per_gpu = max_mem_per_gpu
        self.offload_folder = offload_folder
        self.seed = seed
        self.distributed_single_gpu = distributed_single_gpu
        self.checkpoint_file = "ema.safetensors"
        self.lora_scaling: Optional[float] = None  # set via config for PEFT checkpoints

        self.default_generation_cfg: dict[str, Any] = {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "global",
        }
        self.default_editing_cfg: dict[str, Any] = {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 50,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "text_channel",
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_think_token_n": 1000,
            "do_sample": False,
        }
        self.inferencer: Any = None

    def load(self, cfg: dict[str, Any]) -> None:
        cfg_bagel_root = cfg.get("bagel_root")
        if cfg_bagel_root:
            self.bagel_root = Path(cfg_bagel_root).expanduser()
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(self._normalize_path(cfg_model_path))
        self.max_mem_per_gpu = cfg.get("max_mem_per_gpu", self.max_mem_per_gpu)
        self.offload_folder = cfg.get("offload_folder", self.offload_folder)
        self.seed = cfg.get("seed", self.seed)
        self.distributed_single_gpu = bool(cfg.get("distributed_single_gpu", self.distributed_single_gpu))
        self.checkpoint_file = cfg.get("checkpoint_file", self.checkpoint_file)
        lora_scaling = cfg.get("lora_scaling")
        if lora_scaling is not None:
            self.lora_scaling = float(lora_scaling)
        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        editing_cfg = cfg.get("editing_cfg")
        if isinstance(editing_cfg, dict):
            self.default_editing_cfg.update(editing_cfg)
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

        self._set_seed(self.seed)
        self.inferencer = self._build_inferencer()

    @staticmethod
    def _normalize_path(path_like: str) -> Path:
        path = Path(path_like).expanduser()
        if path.exists():
            return path

        parts = path.parts
        if len(parts) >= 6 and parts[1] == "sciclone" and parts[2] == "home" and parts[4] == "data10":
            fallback = Path("/") / "sciclone" / "data10" / parts[3] / Path(*parts[5:])
            if fallback.exists():
                return fallback
        return path

    def encode(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError("BagelBackbone.encode is not implemented yet.")

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected batch['prompt'] as a non-empty string.")
        return self.generation(prompt=prompt, generation_cfg=gen_cfg)

    def generation(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.inferencer is None:
            self.load({})

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        output_dict = self.inferencer(text=prompt, **config)
        if output_path is not None:
            output_dict["image"].save(output_path, format=_img_format(output_path))
        return output_dict

    def edit(self, batch: dict[str, Any], edit_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt")
        images = batch.get("images", [])
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected batch['prompt'] as a non-empty string.")
        if not isinstance(images, list) or not images:
            raise ValueError("Expected batch['images'] as a non-empty list.")
        return self.editing(
            prompt=prompt,
            images=images,
            output_path=batch.get("output_path"),
            editing_cfg=edit_cfg,
        )

    def editing(
        self,
        prompt: str,
        images: list[str],
        output_path: Optional[str] = None,
        editing_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.inferencer is None:
            self.load({})
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected `prompt` as a non-empty string.")
        if not images:
            raise ValueError("Expected `images` as a non-empty list.")

        if isinstance(images[0], Image.Image):
            image = images[0].convert("RGB")
        else:
            image_path = Path(images[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Editing image not found: {image_path}")
            image = Image.open(image_path).convert("RGB")

        config = dict(self.default_editing_cfg)
        if editing_cfg:
            config.update(editing_cfg)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output_dict = self.inferencer(image=image, text=prompt, **config)
        if output_path is not None:
            output_dict["image"].save(output_path, format=_img_format(output_path))
        return output_dict
    
    def understand(self, batch: dict[str, Any], understanding_cfg: dict[str, Any]) -> Any:
        return self.understanding(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            videos=batch.get("videos", []),
            understanding_cfg=understanding_cfg,
        )

    def understanding(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.inferencer is None:
            self.load({})

        image_list = images or []
        video_list = videos or []
        if video_list:
            raise NotImplementedError("BagelBackbone.understanding currently does not support videos.")

        if prompt is None and not image_list:
            raise ValueError("Understanding requires at least one input: prompt or images.")

        image_obj: Optional[Image.Image] = None
        if image_list:
            image_path = Path(image_list[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Understanding image not found: {image_path}")
            image_obj = Image.open(image_path).convert("RGB")

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        return self.inferencer(image=image_obj, text=prompt, understanding_output=True, **config)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _is_peft_checkpoint(path: str) -> bool:
        """Check if a safetensors file uses PEFT key format (base_model.model. prefix)."""
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        keys = [k for k in header if k != "__metadata__"]
        return any(k.startswith("base_model.model.") for k in keys[:5])

    @staticmethod
    def _merge_peft_checkpoint(path: str, scaling: float) -> str:
        """Load a PEFT checkpoint, merge LoRA weights in memory, save to a temp file.

        For each LoRA-adapted layer the merged weight is:
            weight = base_layer + lora_B @ lora_A * scaling

        Returns the path to a temporary safetensors file with standard key names.
        """
        print(f"[bagel] merging PEFT/LoRA checkpoint (scaling={scaling}) ...", flush=True)
        state_dict = st_load_file(path)

        prefix = "base_model.model."
        # Collect LoRA pairs keyed by their base parameter name
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}
        base_layers: dict[str, torch.Tensor] = {}
        plain: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            # Strip PEFT prefix
            short = key[len(prefix):] if key.startswith(prefix) else key
            if ".lora_A.default.weight" in short:
                param = short.replace(".lora_A.default.weight", "")
                lora_a[param] = tensor
            elif ".lora_B.default.weight" in short:
                param = short.replace(".lora_B.default.weight", "")
                lora_b[param] = tensor
            elif ".base_layer.weight" in short:
                param = short.replace(".base_layer.weight", ".weight")
                base_layers[param] = tensor
            elif ".base_layer.bias" in short:
                param = short.replace(".base_layer.bias", ".bias")
                base_layers[param] = tensor
            else:
                plain[short] = tensor

        # Merge LoRA into base weights
        merged: dict[str, torch.Tensor] = {}
        merged.update(plain)
        for param, base_w in base_layers.items():
            merged[param] = base_w
        for param in lora_a:
            weight_key = param + ".weight"
            if weight_key in merged and param in lora_b:
                a = lora_a[param].float()   # [rank, in]
                b = lora_b[param].float()   # [out, rank]
                merged[weight_key] = merged[weight_key].float() + (b @ a) * scaling

        # Cast to bfloat16 for consistency with standard checkpoints
        for k in merged:
            merged[k] = merged[k].to(torch.bfloat16)

        tmp = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        tmp.close()
        st_save_file(merged, tmp.name)
        n_lora = len(lora_a)
        print(f"[bagel] PEFT merge done: {len(merged)} params, {n_lora} LoRA pairs merged", flush=True)
        return tmp.name

    def _build_inferencer(self) -> Any:
        modules = self._load_bagel_modules()
        model_path = self.model_path

        llm_config = modules["Qwen2Config"].from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = modules["SiglipVisionConfig"].from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = modules["load_ae"](local_path=os.path.join(model_path, "ae.safetensors"))

        config = modules["BagelConfig"](
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
            language_model = modules["Qwen2ForCausalLM"](llm_config)
            vit_model = modules["SiglipVisionModel"](vit_config)
            model = modules["Bagel"](language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = modules["Qwen2Tokenizer"].from_pretrained(model_path)
        tokenizer, new_token_ids, _ = modules["add_special_tokens"](tokenizer)

        vae_transform = modules["ImageTransform"](1024, 512, 16)
        vit_transform = modules["ImageTransform"](980, 224, 14)

        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise RuntimeError("BagelBackbone requires at least one CUDA device.")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and self.distributed_single_gpu:
            local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))
            local_rank = max(0, min(local_rank, gpu_count - 1))
            max_memory = {local_rank: self.max_mem_per_gpu}
        else:
            max_memory = {i: self.max_mem_per_gpu for i in range(gpu_count)}

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

        import io, contextlib, logging
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("safetensors").setLevel(logging.ERROR)

        ckpt_path = os.path.join(model_path, self.checkpoint_file)
        tmp_ckpt: Optional[str] = None
        if self.lora_scaling is not None and self._is_peft_checkpoint(ckpt_path):
            tmp_ckpt = self._merge_peft_checkpoint(ckpt_path, self.lora_scaling)
            ckpt_path = tmp_ckpt

        print("[bagel] loading model weights...", flush=True)
        with contextlib.redirect_stderr(io.StringIO()):
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=ckpt_path,
                device_map=device_map,
                offload_buffers=True,
                dtype=torch.bfloat16,
                force_hooks=True,
                offload_folder=self.offload_folder,
            )
        print("[bagel] model weights loaded", flush=True)

        if tmp_ckpt is not None:
            os.unlink(tmp_ckpt)
        model = model.eval()

        # Move VAE decoder to GPU (without this it stays on CPU and decode_image takes ~134s instead of <1s)
        vae_model = vae_model.to("cuda").eval()
        # Wrap encode and decode to auto-cast input dtype/device to match VAE parameters
        _original_encode = vae_model.encode
        _original_decode = vae_model.decode
        _vae_dtype = next(vae_model.parameters()).dtype
        _vae_device = next(vae_model.parameters()).device
        def _encode_with_cast(x, *args, **kwargs):
            return _original_encode(x.to(device=_vae_device, dtype=_vae_dtype), *args, **kwargs)
        def _decode_with_cast(latent, *args, **kwargs):
            return _original_decode(latent.to(device=_vae_device, dtype=_vae_dtype), *args, **kwargs)
        vae_model.encode = _encode_with_cast
        vae_model.decode = _decode_with_cast

        return modules["InterleaveInferencer"](
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

    def _load_bagel_modules(self) -> dict[str, Any]:
        bagel_root_str = str(self.bagel_root.resolve())
        if bagel_root_str not in sys.path:
            sys.path.insert(0, bagel_root_str)

        data_transforms = importlib.import_module("data.transforms")
        data_utils = importlib.import_module("data.data_utils")
        modeling_bagel = importlib.import_module("modeling.bagel")
        modeling_qwen2 = importlib.import_module("modeling.qwen2")
        modeling_autoencoder = importlib.import_module("modeling.autoencoder")
        inferencer_module = importlib.import_module("inferencer")

        return {
            "ImageTransform": data_transforms.ImageTransform,
            "add_special_tokens": data_utils.add_special_tokens,
            "Bagel": modeling_bagel.Bagel,
            "BagelConfig": modeling_bagel.BagelConfig,
            "Qwen2Config": modeling_bagel.Qwen2Config,
            "Qwen2ForCausalLM": modeling_bagel.Qwen2ForCausalLM,
            "SiglipVisionConfig": modeling_bagel.SiglipVisionConfig,
            "SiglipVisionModel": modeling_bagel.SiglipVisionModel,
            "Qwen2Tokenizer": modeling_qwen2.Qwen2Tokenizer,
            "load_ae": modeling_autoencoder.load_ae,
            "InterleaveInferencer": inferencer_module.InterleaveInferencer,
        }
