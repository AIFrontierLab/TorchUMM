from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any, Optional


# Wrapper script that intercepts wandb.log to save images deterministically.
_GEN_WRAPPER = Path(__file__).resolve().parent / "_gen_wrapper.py"


class ShowOBackbone:
    name = "show_o2"

    def __init__(
        self,
        model_path: Optional[str] = None,
        show_o_root: Optional[str] = None,
        vae_path: Optional[str] = None,
        seed: int = 42,
        torch_dtype: str = "bfloat16",
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        if show_o_root:
            self.show_o_root = Path(show_o_root).expanduser()
        else:
            alt = repo_root / "src" / "umm" / "backbones" / "show_o" / "Show-o"
            default = repo_root / "model" / "Show-o"
            self.show_o_root = alt if alt.exists() else default

        self.model_path = (
            str(Path(model_path).expanduser()) if model_path else str(repo_root / "model_cache" / "show_o" / "models" / "show_o2_7B")
        )
        # VAE weight path (v2 only) — auto-resolve from model cache if not specified
        if vae_path:
            self.vae_path: Optional[str] = str(Path(vae_path).expanduser())
        else:
            default_vae = Path("/model_cache/show_o2/Wan2.1_VAE.pth")
            self.vae_path = str(default_vae) if default_vae.exists() else None
        # VQ model path (v1 only) — MagVITv2 discrete tokenizer
        self.vq_model_path: Optional[str] = None
        self.seed = seed
        self.torch_dtype = torch_dtype
        self.version = 1
        self._python_executable: Optional[str] = None

        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 2048,
            "top_k": 1,
            "temperature": 0.8,
            "do_sample": False,
            "use_clip_vit": True,
        }
        self.default_generation_cfg: dict[str, Any] = {
            "guidance_scale": 3.0,
            "generation_timesteps": 50,
            "batch_size": 1,
        }

        # v1 in-process model components (populated by _build_v1_model)
        self._v1_model: Any = None
        self._v1_vq_model: Any = None
        self._v1_vision_tower: Any = None
        self._v1_clip_processor: Any = None
        self._v1_tokenizer: Any = None
        self._v1_uni_prompting: Any = None
        self._v1_config: Any = None
        self._v1_image_transform_fn: Any = None
        self._v1_conv_templates: Any = None
        self._v1_create_attention_mask_for_mmu: Any = None
        self._v1_create_attention_mask_for_mmu_vit: Any = None

        # v2 in-process model components (populated by _build_v2_model)
        self._v2_model: Any = None
        self._v2_vae: Any = None
        self._v2_tokenizer: Any = None
        self._v2_showo_token_ids: dict[str, int] = {}
        self._v2_config: Any = None
        self._v2_num_mmu_image_tokens: int = 0
        self._v2_image_transform_fn: Any = None
        self._v2_omni_attn_mask_fn: Any = None
        # Precomputed chat template token IDs
        self._v2_sys_prompt_ids: list[int] = []
        self._v2_role_a_ids: list[int] = []
        self._v2_role_b_ids: list[int] = []

    def _get_python(self) -> str:
        return self._python_executable or sys.executable

    def load(self, cfg: dict[str, Any]) -> None:
        # minimal load: store cfg values for later subprocess call
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(Path(cfg_model_path).expanduser())
        cfg_show_o_root = cfg.get("show_o_root")
        if cfg_show_o_root:
            self.show_o_root = Path(cfg_show_o_root).expanduser()
        cfg_vae_path = cfg.get("vae_path")
        if cfg_vae_path:
            self.vae_path = str(Path(cfg_vae_path).expanduser())
        cfg_vq_path = cfg.get("vq_model_path")
        if cfg_vq_path:
            self.vq_model_path = str(Path(cfg_vq_path).expanduser())
        if isinstance(cfg.get("python_executable"), str):
            self._python_executable = str(Path(cfg["python_executable"]).expanduser())
        self.seed = int(cfg.get("seed", self.seed))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)
        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)

        # Version detection: explicit config > auto-detect from model config.json
        v = cfg.get("version")
        if v is not None:
            self.version = int(v)
        elif self._auto_detect_v2():
            self.version = 2

        # Load model components in-process for fast inference
        if self.version == 2:
            self._build_v2_model()
        else:
            self._build_v1_model()

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def _auto_detect_v2(self) -> bool:
        config_json = Path(self.model_path) / "config.json"
        if not config_json.exists():
            return False
        try:
            data = json.loads(config_json.read_text())
            return "Showo2" in data.get("_class_name", "")
        except Exception:
            return False

    def _get_cwd(self) -> Path:
        if self.version == 2:
            return self.show_o_root / "show-o2"
        return self.show_o_root

    def _is_1_5b(self) -> bool:
        """Check if loaded model is the 1.5B variant based on model_path."""
        return "1.5B" in self.model_path or "1_5B" in self.model_path

    def _get_config_name(self, gen_cfg: Optional[dict[str, Any]] = None) -> str:
        if self.version == 2:
            if self._is_1_5b():
                return "showo2_1.5b_demo_432x432.yaml"
            return "showo2_7b_demo_432x432.yaml"
        use_clip_vit = True
        if gen_cfg is not None:
            use_clip_vit = bool(gen_cfg.get("use_clip_vit", True))
        else:
            use_clip_vit = bool(self.default_understanding_cfg.get("use_clip_vit", True))
        return "showo_demo_w_clip_vit_512x512.yaml" if use_clip_vit else "showo_demo_512x512.yaml"

    # ------------------------------------------------------------------
    # v1 in-process model loading and inference
    # ------------------------------------------------------------------

    _V1_SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    _V1_SYSTEM_PROMPT_LEN = 28

    def _build_v1_model(self) -> None:
        """Load v1 model (Showo + MAGVITv2 + CLIP) in-process once."""
        import torch

        show_o_v1_root = str(self.show_o_root.resolve())
        if show_o_v1_root not in sys.path:
            sys.path.insert(0, show_o_v1_root)

        # Import v1 modules from the Show-o repo
        from models import Showo, MAGVITv2, CLIPVisionTower
        from training.prompting_utils import (
            UniversalPrompting,
            create_attention_mask_for_mmu,
            create_attention_mask_for_mmu_vit,
        )
        from training.utils import get_config, image_transform
        from transformers import AutoTokenizer, CLIPImageProcessor

        # Store functions needed at inference time
        self._v1_image_transform_fn = image_transform
        self._v1_create_attention_mask_for_mmu = create_attention_mask_for_mmu
        self._v1_create_attention_mask_for_mmu_vit = create_attention_mask_for_mmu_vit

        # Load conversation templates (avoid mutating module-level global)
        conv_mod = importlib.import_module("llava.llava.conversation")
        self._v1_conv_templates = conv_mod.conv_templates

        # Build OmegaConf config
        cfg_name = self._get_config_name()
        saved_argv = sys.argv[:]
        try:
            cfg_abs = os.path.join(show_o_v1_root, "configs", cfg_name)
            sys.argv = ["adapter", f"config={cfg_abs}"]
            if self.model_path:
                sys.argv.append(f"model.showo.pretrained_model_path={self.model_path}")
            if self.vq_model_path:
                sys.argv.append(f"model.vq_model.vq_model_name={self.vq_model_path}")
            config = get_config()
        finally:
            sys.argv = saved_argv
        self._v1_config = config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        print("[show_o v1] Loading tokenizer ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.showo.llm_model_path, padding_side="left",
        )
        self._v1_tokenizer = tokenizer

        # UniversalPrompting (adds special tokens to tokenizer)
        self._v1_uni_prompting = UniversalPrompting(
            tokenizer,
            max_text_len=config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=config.training.cond_dropout_prob,
        )

        # VQ model (MagVITv2)
        print("[show_o v1] Loading VQ model ...", flush=True)
        vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        self._v1_vq_model = vq_model

        # CLIP vision tower + processor (only when w_clip_vit)
        if config.model.showo.w_clip_vit:
            clip_name = "openai/clip-vit-large-patch14-336"
            print(f"[show_o v1] Loading CLIP vision tower ({clip_name}) ...", flush=True)
            self._v1_vision_tower = CLIPVisionTower(clip_name).to(device)
            self._v1_clip_processor = CLIPImageProcessor.from_pretrained(clip_name)

        # Showo model
        print(f"[show_o v1] Loading Showo model from {config.model.showo.pretrained_model_path} ...", flush=True)
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
        model.eval()
        self._v1_model = model
        print("[show_o v1] Model loaded in-process", flush=True)

    def _v1_infer_image(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 100,
        top_k: int = 1,
        temperature: float = 0.8,
    ) -> str:
        """Run v1 image understanding using pre-loaded model (no subprocess)."""
        import torch
        from PIL import Image

        model = self._v1_model
        vq_model = self._v1_vq_model
        tokenizer = self._v1_tokenizer
        uni_prompting = self._v1_uni_prompting
        config = self._v1_config

        device = next(model.parameters()).device
        resolution = config.dataset.params.resolution

        image_ori = Image.open(image_path).convert("RGB")
        image = self._v1_image_transform_fn(image_ori, resolution=resolution).to(device)
        image = image.unsqueeze(0)

        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        with torch.no_grad():
            if config.model.showo.w_clip_vit:
                pixel_values = self._v1_clip_processor(
                    image_ori, return_tensors="pt",
                )["pixel_values"][0]

                # Use "phi1.5" (conv_phi_v0) which has system="" — the
                # system prompt is already prepended separately via
                # input_ids_system.  Using "v1" (vicuna_v1) would
                # duplicate the system prompt in the conv output,
                # corrupting the input and causing empty responses
                # on complex prompts (MMMU, MathVista).
                conv = self._v1_conv_templates["phi1.5"].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()

                input_ids_system = uni_prompting.text_tokenizer(
                    self._V1_SYSTEM_PROMPT, return_tensors="pt", padding="longest",
                ).input_ids.to(device)

                input_ids = uni_prompting.text_tokenizer(
                    prompt_question.strip(), return_tensors="pt", padding="longest",
                ).input_ids.to(device)

                input_ids_llava = torch.cat([
                    torch.tensor([[int(uni_prompting.sptids_dict["<|mmu|>"])]]).to(device),
                    input_ids_system,
                    torch.tensor([[int(uni_prompting.sptids_dict["<|soi|>"])]]).to(device),
                    torch.tensor([[int(uni_prompting.sptids_dict["<|eoi|>"])]]).to(device),
                    input_ids,
                ], dim=1).long()

                images_embeddings = self._v1_vision_tower(pixel_values[None].to(device))
                images_embeddings = model.mm_projector(images_embeddings)

                text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

                # Splice image embeddings after [mmu + system_prompt + soi], before [eoi + question]
                part1 = text_embeddings[:, :2 + self._V1_SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + self._V1_SYSTEM_PROMPT_LEN:, :]
                input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)

                attention_mask = self._v1_create_attention_mask_for_mmu_vit(
                    input_embeddings, system_prompt_len=self._V1_SYSTEM_PROMPT_LEN,
                )

                cont_toks_list = model.mmu_generate(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask[0].unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    eot_token=tokenizer.eos_token_id,
                )
            else:
                # Non-CLIP path: use VQ image tokens directly
                input_ids = uni_prompting.text_tokenizer(
                    ["USER: \n" + question + " ASSISTANT:"],
                )["input_ids"]
                input_ids = torch.tensor(input_ids).to(device)

                input_ids = torch.cat([
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|mmu|>"]).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|soi|>"]).to(device),
                    image_tokens,
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|eoi|>"]).to(device),
                    (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict["<|sot|>"]).to(device),
                    input_ids,
                ], dim=1).long()

                attention_mask = self._v1_create_attention_mask_for_mmu(
                    input_ids.to(device),
                    eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                )

                cont_toks_list = model.mmu_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    eot_token=uni_prompting.sptids_dict["<|eot|>"],
                )

        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
        text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        return text[0] if text else ""

    # ------------------------------------------------------------------
    # v2 in-process model loading and inference
    # ------------------------------------------------------------------

    def _build_v2_model(self) -> None:
        """Load v2 model, VAE, and tokenizer in-process (once)."""
        import torch

        show_o2_root = str((self.show_o_root / "show-o2").resolve())
        cfg_name = self._get_config_name()

        # Add show-o2 to sys.path for model's internal imports
        if show_o2_root not in sys.path:
            sys.path.insert(0, show_o2_root)

        # Import show-o2 modules (models/utils won't conflict with std libs)
        showo2_models = importlib.import_module("models")
        models_misc = importlib.import_module("models.misc")
        showo2_utils = importlib.import_module("utils")

        # Import image_transform from file to avoid HuggingFace datasets conflict
        ds_utils_path = os.path.join(show_o2_root, "datasets", "utils.py")
        spec = importlib.util.spec_from_file_location("_showo2_ds_utils", ds_utils_path)
        ds_utils_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ds_utils_mod)
        self._v2_image_transform_fn = ds_utils_mod.image_transform

        # Store attention mask function
        self._v2_omni_attn_mask_fn = showo2_models.omni_attn_mask_naive

        # Build config via OmegaConf (requires sys.argv manipulation)
        saved_argv = sys.argv[:]
        try:
            cfg_abs = os.path.join(show_o2_root, "configs", cfg_name)
            sys.argv = ["adapter", f"config={cfg_abs}"]
            if self.model_path:
                sys.argv.append(f"model.showo.pretrained_model_path={self.model_path}")
            if self.vae_path:
                sys.argv.append(f"model.vae_model.pretrained_model_path={self.vae_path}")
            config = showo2_utils.get_config()
        finally:
            sys.argv = saved_argv
        self._v2_config = config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        weight_type = _dtype_map.get(self.torch_dtype, torch.bfloat16)
        print(f"[show_o2] Using weight dtype: {weight_type}", flush=True)

        # Load VAE
        print("[show_o2] Loading VAE in-process ...", flush=True)
        self._v2_vae = showo2_models.WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type,
            device=device,
        )

        # Load tokenizer
        print("[show_o2] Loading tokenizer in-process ...", flush=True)
        text_tokenizer, showo_token_ids = models_misc.get_text_tokenizer(
            config.model.showo.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=showo2_utils.path_to_llm_name[config.model.showo.llm_model_path],
        )
        config.model.showo.llm_vocab_size = len(text_tokenizer)
        self._v2_tokenizer = text_tokenizer
        self._v2_showo_token_ids = showo_token_ids

        # Load model
        print(f"[show_o2] Loading model in-process from {config.model.showo.pretrained_model_path} ...", flush=True)
        if config.model.showo.load_from_showo:
            # Suppress noisy warnings from accelerate/torch during from_pretrained:
            # - meta-parameter copy no-op warnings (accelerate resolves them internally)
            # - torch.load weights_only FutureWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
                warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
                model = showo2_models.Showo2Qwen2_5.from_pretrained(
                    config.model.showo.pretrained_model_path, use_safetensors=False
                )
            # Safety check: if any meta parameters remain, reload with assign=True
            has_meta = any(p.is_meta for p in model.parameters())
            if has_meta:
                print("[show_o2] Detected meta parameters, reloading state_dict with assign=True ...", flush=True)
                state_dict = showo2_utils.load_state_dict(config.model.showo.pretrained_model_path)
                model.load_state_dict(state_dict, assign=True)
            model = model.to(device)
        else:
            model = showo2_models.Showo2Qwen2_5(**config.model.showo).to(device)
            state_dict = showo2_utils.load_state_dict(config.model_path)
            model.load_state_dict(state_dict, assign=True)
        model.to(weight_type)
        model.eval()
        self._v2_model = model
        print("[show_o2] Model loaded in-process", flush=True)

        # Compute num_mmu_image_tokens
        if config.model.showo.add_time_embeds:
            config.dataset.preprocessing.num_t2i_image_tokens += 1
            config.dataset.preprocessing.num_mmu_image_tokens += 1
            config.dataset.preprocessing.num_video_tokens += 1

        (_, num_mmu_image_tokens, *_rest) = showo2_utils.get_hyper_params(
            config, text_tokenizer, showo_token_ids
        )
        self._v2_num_mmu_image_tokens = num_mmu_image_tokens

        # Precompute chat template token IDs
        self._v2_sys_prompt_ids = text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>",
            add_special_tokens=False,
        )["input_ids"]
        self._v2_role_a_ids = text_tokenizer(
            "\n<|im_start|>user\n", add_special_tokens=False,
        )["input_ids"]
        self._v2_role_b_ids = text_tokenizer(
            "\n<|im_start|>assistant\n", add_special_tokens=False,
        )["input_ids"]

    def _v2_infer_image(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 300,
        top_k: int = 1,
        temperature: float = 1.0,
    ) -> str:
        """Run v2 image understanding using pre-loaded model (no subprocess)."""
        import torch
        from PIL import Image

        model = self._v2_model
        vae_model = self._v2_vae
        tokenizer = self._v2_tokenizer
        token_ids = self._v2_showo_token_ids
        config = self._v2_config

        device = next(model.parameters()).device
        weight_type = next(model.parameters()).dtype

        # Process image
        image_ori = Image.open(image_path).convert("RGB")
        image = self._v2_image_transform_fn(
            image_ori, resolution=config.dataset.preprocessing.resolution
        ).to(device).unsqueeze(0)

        # VAE encode
        image_latents = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)

        # Dual-path embeddings + fusion
        image_embeds_und = model.image_embedder_und(image_latents)
        image_embeds_gen = model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
        image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
        image_embeds = model.fusion_proj(
            torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
        )

        # Build input token IDs
        question_ids = tokenizer(question, add_special_tokens=False).input_ids
        text_tokens_a = torch.tensor(
            [token_ids["bos_id"]] + self._v2_sys_prompt_ids + self._v2_role_a_ids
        ).to(device)[None, :]
        text_tokens_b = torch.tensor(
            [token_ids["boi_id"], token_ids["eoi_id"]]
            + question_ids + self._v2_role_b_ids
        ).to(device)[None, :]

        # Get text embeddings
        text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
        text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

        # Assemble input embeddings (with optional time embeddings)
        if config.model.showo.add_time_embeds:
            time_embeds = model.time_embed(
                torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype
            )
            if hasattr(model, "time_embed_proj"):
                time_embeds = model.time_embed_proj(time_embeds)
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],   # boi token
                time_embeds,
                image_embeds,
                text_embeds_b[:, 1:],   # eoi + question + role_b
            ], dim=1).to(weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 2, self._v2_num_mmu_image_tokens]
            )[None, None, :].to(device)
        else:
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],   # boi token
                image_embeds,
                text_embeds_b[:, 1:],   # eoi + question + role_b
            ], dim=1).to(weight_type)
            modality_positions = torch.tensor(
                [text_tokens_a.shape[1] + 1, self._v2_num_mmu_image_tokens]
            )[None, None, :].to(device)

        # Attention mask
        attention_mask = self._v2_omni_attn_mask_fn(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=device,
            inverted=True,
        ).to(input_embeds.dtype)

        # Generate
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        with torch.no_grad():
            output_tokens = model.mmu_generate(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                eos_token=tokenizer.eos_token_id,
            )

        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return text[0] if text else ""

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
        if videos:
            raise NotImplementedError("Videos not supported.")

        if prompt is None and not images:
            raise ValueError("Understanding requires a prompt or images.")

        image_list = images or []
        if not image_list:
            # Pure text understanding via lm_generate (no image needed)
            return self._text_only_understanding(prompt, understanding_cfg)

        image_path = Path(image_list[0]).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cfg = dict(self.default_understanding_cfg)
        if understanding_cfg:
            cfg.update(understanding_cfg)

        cfg_name = self._get_config_name(cfg)

        question = prompt if isinstance(prompt, str) else "Please describe this image in detail."
        max_new_tokens = int(cfg.get("max_new_tokens", 100))
        do_sample = bool(cfg.get("do_sample", False))
        top_k = int(cfg.get("top_k", 1))
        # Greedy decoding: force top_k=1 so multinomial always picks the argmax
        if not do_sample:
            top_k = 1

        if self.version == 2:
            response = self._v2_infer_image(
                image_path=str(image_path),
                question=question,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                temperature=float(cfg.get("temperature", 1.0)),
            )
            return {"text": response}

        response = self._v1_infer_image(
            image_path=str(image_path),
            question=question,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=float(cfg.get("temperature", 0.8)),
        )
        return {"text": response}

    def _text_only_understanding(
        self,
        prompt: Optional[str],
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Pure text understanding (single item) — delegates to batch."""
        results = self._run_text_only_batch([prompt or ""], understanding_cfg)
        return {"text": results[0].get("text", "")}

    # ------------------------------------------------------------------
    # v2 image understanding (inline runner)
    # ------------------------------------------------------------------

    def _run_v2_image_understanding(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 300,
        top_k: int = 1,
        temperature: float = 1.0,
        cfg_name: str = "showo2_7b_demo_432x432.yaml",
    ) -> dict[str, Any]:
        """Run v2 multimodal understanding via inline runner subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            params_file = tmpdir_path / "params.json"
            results_file = tmpdir_path / "results.json"
            runner_file = tmpdir_path / "runner_mmu.py"

            params_file.write_text(json.dumps({
                "image_path": image_path,
                "question": question,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "temperature": temperature,
                "seed": self.seed,
                "model_path": self.model_path,
                "vae_path": self.vae_path,
                "config_name": cfg_name,
            }))
            runner_file.write_text(self._runner_code_mmu())

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)

            proc = subprocess.run(
                [self._get_python(), str(runner_file), str(params_file), str(results_file)],
                cwd=str(self._get_cwd()),
                env=env,
            )

            result: dict[str, Any] = {"text": "", "returncode": proc.returncode}
            if results_file.exists():
                try:
                    result.update(json.loads(results_file.read_text()))
                except Exception:
                    pass

            if proc.returncode != 0 and not result.get("text"):
                print(f"[show_o2] mmu runner exited rc={proc.returncode}")

            return result

    @staticmethod
    def _runner_code_mmu() -> str:
        """Inline runner for v2 image understanding (mirrors inference_mmu.py)."""
        return textwrap.dedent('''\
            import json
            import os
            import sys
            import traceback
            sys.path.insert(0, os.getcwd())
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            def log(msg):
                print(msg, flush=True)

            params_file = sys.argv[1]
            results_file = sys.argv[2]
            with open(params_file, "r") as f:
                params = json.load(f)

            image_path = params["image_path"]
            question = params["question"]
            max_new_tokens = params.get("max_new_tokens", 300)
            top_k = params.get("top_k", 1)
            temperature = params.get("temperature", 1.0)
            seed = params.get("seed", 42)
            model_path = params.get("model_path")
            vae_path = params.get("vae_path")
            config_name = params["config_name"]

            log(f"[show_o2-mmu] image={image_path}, question={question[:80]}")

            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                from PIL import Image
                from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
                from models.misc import get_text_tokenizer
                from datasets.utils import image_transform
                from utils import get_config, get_hyper_params, path_to_llm_name, load_state_dict

                sys.argv = ["runner", f"config=configs/{config_name}"]
                if model_path:
                    sys.argv.append(f"model.showo.pretrained_model_path={model_path}")
                if vae_path:
                    sys.argv.append(f"model.vae_model.pretrained_model_path={vae_path}")
                config = get_config()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                weight_type = torch.float32
                log(f"[show_o2-mmu] device={device}")

                # Load VAE
                log("[show_o2-mmu] Loading VAE ...")
                vae_model = WanVAE(
                    vae_pth=config.model.vae_model.pretrained_model_path,
                    dtype=weight_type, device=device,
                )

                # Load tokenizer
                log("[show_o2-mmu] Loading tokenizer ...")
                text_tokenizer, showo_token_ids = get_text_tokenizer(
                    config.model.showo.llm_model_path,
                    add_showo_tokens=True,
                    return_showo_token_ids=True,
                    llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                )
                config.model.showo.llm_vocab_size = len(text_tokenizer)

                # Load model
                log(f"[show_o2-mmu] Loading model from {config.model.showo.pretrained_model_path} ...")
                if config.model.showo.load_from_showo:
                    model = Showo2Qwen2_5.from_pretrained(
                        config.model.showo.pretrained_model_path, use_safetensors=False
                    )
                    # Fix meta params from accelerate init_empty_weights
                    if any(p.is_meta for p in model.parameters()):
                        log("[show_o2-mmu] Fixing meta parameters with assign=True ...")
                        sd = load_state_dict(config.model.showo.pretrained_model_path)
                        model.load_state_dict(sd, assign=True)
                    model = model.to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict, assign=True)
                model.to(weight_type)
                model.eval()
                log("[show_o2-mmu] Model loaded")

                # Time embedding adjustment
                if config.model.showo.add_time_embeds:
                    config.dataset.preprocessing.num_t2i_image_tokens += 1
                    config.dataset.preprocessing.num_mmu_image_tokens += 1
                    config.dataset.preprocessing.num_video_tokens += 1

                (_, num_mmu_image_tokens, _, _, _, _, _, _, _,
                 _, _, _, _, _, _, _, _, _, _) = get_hyper_params(
                    config, text_tokenizer, showo_token_ids
                )

                # Prepare chat template tokens
                sys_prompt_ids = text_tokenizer(
                    "system\\nYou are a helpful assistant.<|im_end|>",
                    add_special_tokens=False,
                )["input_ids"]
                role_a = text_tokenizer(
                    "\\n<|im_start|>user\\n", add_special_tokens=False
                )["input_ids"]
                role_b = text_tokenizer(
                    "\\n<|im_start|>assistant\\n", add_special_tokens=False
                )["input_ids"]

                # Process image
                image_ori = Image.open(image_path).convert("RGB")
                image = image_transform(
                    image_ori, resolution=config.dataset.preprocessing.resolution
                ).to(device)
                image = image.unsqueeze(0)

                # Encode image through VAE
                image_latents = vae_model.sample(image.unsqueeze(2)).squeeze(2).to(weight_type)

                # Dual-path image embeddings + fusion
                image_embeds_und = model.image_embedder_und(image_latents)
                image_embeds_gen = model.image_embedder_gen(image_latents)
                image_embeds_und = image_embeds_und + model.position_embedding(model.image_position_ids)
                image_embeds_und = model.und_trans(image_embeds_und)["last_hidden_state"]
                image_embeds = model.fusion_proj(
                    torch.cat([image_embeds_und, image_embeds_gen], dim=-1)
                )

                # Build input embeddings
                question_ids = text_tokenizer(question, add_special_tokens=False).input_ids
                text_tokens_a = torch.tensor(
                    [showo_token_ids["bos_id"]] + sys_prompt_ids + role_a
                ).to(device)[None, :]
                text_tokens_b = torch.tensor(
                    [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]]
                    + question_ids + role_b
                ).to(device)[None, :]
                text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
                text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

                if config.model.showo.add_time_embeds:
                    time_embeds = model.time_embed(
                        torch.Tensor([[1.0]]).to(device), text_embeds_a.dtype
                    )
                    if hasattr(model, "time_embed_proj"):
                        time_embeds = model.time_embed_proj(time_embeds)
                    input_embeds = torch.cat([
                        text_embeds_a,
                        text_embeds_b[:, :1],
                        time_embeds,
                        image_embeds,
                        text_embeds_b[:, 1:],
                    ], dim=1).to(weight_type)
                    modality_positions = torch.tensor(
                        [text_tokens_a.shape[1] + 2, num_mmu_image_tokens]
                    )[None, None, :].to(device)
                else:
                    input_embeds = torch.cat([
                        text_embeds_a,
                        text_embeds_b[:, :1],
                        image_embeds,
                        text_embeds_b[:, 1:],
                    ], dim=1).to(weight_type)
                    modality_positions = torch.tensor(
                        [text_tokens_a.shape[1] + 1, num_mmu_image_tokens]
                    )[None, None, :].to(device)

                # Attention mask
                attention_mask = omni_attn_mask_naive(
                    B=input_embeds.size(0),
                    LEN=input_embeds.size(1),
                    modalities=modality_positions,
                    device=device, inverted=True,
                ).to(input_embeds.dtype)

                # Generate
                log("[show_o2-mmu] Generating ...")
                with torch.no_grad():
                    output_tokens = model.mmu_generate(
                        input_embeds=input_embeds,
                        attention_mask=attention_mask,
                        top_k=top_k,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        eos_token=text_tokenizer.eos_token_id,
                    )

                output_tokens = torch.stack(output_tokens).squeeze()[None]
                text = text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
                response = text[0] if text else ""
                log(f"[show_o2-mmu] Response: {response[:200]}")

                with open(results_file, "w") as f:
                    json.dump({"text": response}, f, ensure_ascii=False)

            except Exception as e:
                log(f"[show_o2-mmu] FATAL ERROR: {e}")
                traceback.print_exc()
                sys.exit(1)
            ''')

    # ------------------------------------------------------------------
    # Batch methods (single subprocess for all items)
    # ------------------------------------------------------------------

    def understand_batch(
        self,
        items: list[dict[str, Any]],
        understanding_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Batch text-only understanding — one subprocess, one model load."""
        questions = [item.get("prompt", "") or "" for item in items]
        return self._run_text_only_batch(questions, understanding_cfg)

    def _run_text_only_batch(
        self,
        questions: list[str],
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Run text-only understanding for all questions in a single subprocess."""
        if self.version != 2:
            raise NotImplementedError("Text-only understanding requires Show-o2 (version 2).")

        cfg = dict(self.default_understanding_cfg)
        if understanding_cfg:
            cfg.update(understanding_cfg)

        max_new_tokens = int(cfg.get("max_new_tokens", 512))
        cfg_name = self._get_config_name(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            params_file = tmpdir_path / "params.json"
            results_file = tmpdir_path / "results.json"
            runner_file = tmpdir_path / "runner_text_batch.py"

            params_file.write_text(json.dumps({
                "questions": questions,
                "max_new_tokens": max_new_tokens,
                "seed": self.seed,
                "model_path": self.model_path,
                "vae_path": self.vae_path,
                "config_name": cfg_name,
            }))
            runner_file.write_text(self._runner_code_text_batch())

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)

            proc = subprocess.run(
                [self._get_python(), str(runner_file), str(params_file), str(results_file)],
                cwd=str(self._get_cwd()),
                env=env,
            )

            # Read whatever results were saved (supports partial results from interrupted runs)
            partial: list[dict[str, Any]] = []
            if results_file.exists():
                try:
                    partial = json.loads(results_file.read_text())
                except Exception:
                    pass

            if proc.returncode != 0:
                print(f"[show_o2] text batch exited rc={proc.returncode}, got {len(partial)}/{len(questions)} results")

            # Pad with empty results if incomplete
            while len(partial) < len(questions):
                partial.append({"text": ""})
            return partial

    @staticmethod
    def _runner_code_text_batch() -> str:
        """Embedded runner: load model once, process all questions."""
        return textwrap.dedent('''\
            import json
            import os
            import sys
            import traceback
            sys.path.insert(0, os.getcwd())
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            def log(msg):
                print(msg, flush=True)

            params_file = sys.argv[1]
            results_file = sys.argv[2]
            with open(params_file, "r") as f:
                params = json.load(f)

            questions = params["questions"]
            max_new_tokens = params.get("max_new_tokens", 512)
            seed = params.get("seed", 42)
            model_path = params.get("model_path")
            vae_path = params.get("vae_path")
            config_name = params["config_name"]

            log(f"[show_o2] Starting text batch runner: {len(questions)} questions")
            log(f"[show_o2] config={config_name}, model_path={model_path}")

            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                from models import Showo2Qwen2_5
                from models.misc import get_text_tokenizer
                from utils import get_config, path_to_llm_name, load_state_dict

                sys.argv = ["runner", f"config=configs/{config_name}"]
                if model_path:
                    sys.argv.append(f"model.showo.pretrained_model_path={model_path}")
                if vae_path:
                    sys.argv.append(f"model.vae_model.pretrained_model_path={vae_path}")
                config = get_config()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                log(f"[show_o2] device={device}, cuda_available={torch.cuda.is_available()}")
                weight_type = torch.bfloat16

                log("[show_o2] Loading tokenizer ...")
                text_tokenizer, showo_token_ids = get_text_tokenizer(
                    config.model.showo.llm_model_path,
                    add_showo_tokens=True,
                    return_showo_token_ids=True,
                    llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                )
                config.model.showo.llm_vocab_size = len(text_tokenizer)

                log(f"[show_o2] Loading model from {config.model.showo.pretrained_model_path} ...")
                if config.model.showo.load_from_showo:
                    model = Showo2Qwen2_5.from_pretrained(
                        config.model.showo.pretrained_model_path, use_safetensors=False
                    )
                    # Fix meta params from accelerate init_empty_weights
                    if any(p.is_meta for p in model.parameters()):
                        log("[show_o2] Fixing meta parameters with assign=True ...")
                        sd = load_state_dict(config.model.showo.pretrained_model_path)
                        model.load_state_dict(sd, assign=True)
                    model = model.to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict, assign=True)
                model.to(weight_type)
                model.eval()
                log("[show_o2] Model loaded successfully")

                boi_id = showo_token_ids["boi_id"]

                # Precompute chat template tokens (shared across all questions)
                sys_prompt_ids = text_tokenizer(
                    "system\\nYou are a helpful assistant.<|im_end|>",
                    add_special_tokens=False,
                )["input_ids"]
                role_a = text_tokenizer(
                    "\\n<|im_start|>user\\n", add_special_tokens=False
                )["input_ids"]
                role_b = text_tokenizer(
                    "\\n<|im_start|>assistant\\n", add_special_tokens=False
                )["input_ids"]

                results = []
                for i, question in enumerate(questions):
                    question_ids = text_tokenizer(question, add_special_tokens=False)["input_ids"]
                    input_ids = (
                        [showo_token_ids["bos_id"]]
                        + sys_prompt_ids + role_a + question_ids + role_b
                    )
                    log(f"[show_o2] [{i+1}/{len(questions)}] {question[:80]} ...")
                    try:
                        with torch.no_grad():
                            response = model.lm_generate(
                                input_ids=input_ids,
                                tokenizer=text_tokenizer,
                                max_new_tokens=max_new_tokens,
                                boi_token=boi_id,
                                device=device,
                            )
                        results.append({"text": response})
                        log(f"[show_o2]   -> {len(response)} chars")
                    except Exception as e:
                        log(f"[show_o2]   Error: {e}")
                        results.append({"text": ""})

                    # Save incrementally so partial results survive interruption
                    with open(results_file, "w") as f:
                        json.dump(results, f, ensure_ascii=False)

                log(f"[show_o2] Batch text understanding done: {len(results)} items")

            except Exception as e:
                log(f"[show_o2] FATAL ERROR: {e}")
                traceback.print_exc()
                sys.exit(1)
            ''')

    @staticmethod
    def _runner_code_unified_batch() -> str:
        """Embedded runner: load model ONCE, do text understanding + image generation."""
        return textwrap.dedent('''\
            import json
            import os
            import sys
            import traceback
            import warnings
            import numpy as np
            from pathlib import Path
            # Suppress meta-parameter copy warnings from CLIP vision model loading
            # (vision model is unused in text-understanding + t2i pipeline)
            warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
            sys.path.insert(0, os.getcwd())
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            def log(msg):
                print(msg, flush=True)

            params_file = sys.argv[1]
            with open(params_file, "r") as f:
                params = json.load(f)

            questions = params["questions"]
            item_ids = params["item_ids"]
            max_new_tokens_und = params.get("max_new_tokens_und", 100)
            max_new_tokens_gen = params.get("max_new_tokens_gen", 512)
            seed = params.get("seed", 42)
            model_path = params.get("model_path")
            vae_path = params.get("vae_path")
            config_name = params["config_name"]
            guidance_scale_override = params.get("guidance_scale")
            num_inference_steps = params.get("num_inference_steps", 50)
            gen_batch_size = params.get("batch_size", 1)
            output_dir = params["output_dir"]
            text_results_file = params["text_results_file"]
            gen_results_file = params["gen_results_file"]

            os.makedirs(output_dir, exist_ok=True)

            log(f"[show_o2-unified] {len(questions)} items, config={config_name}")

            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                from PIL import Image
                from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
                from models.misc import get_text_tokenizer, prepare_gen_input
                from utils import get_config, get_hyper_params, denorm, path_to_llm_name, load_state_dict
                from transport import Sampler, create_transport
                if torch.cuda.is_available():
                    from torch.nn.attention.flex_attention import flex_attention
                    flex_attention = torch.compile(flex_attention)

                # ---- Load config ----
                sys.argv = ["runner", f"config=configs/{config_name}"]
                if model_path:
                    sys.argv.append(f"model.showo.pretrained_model_path={model_path}")
                if vae_path:
                    sys.argv.append(f"model.vae_model.pretrained_model_path={vae_path}")
                config = get_config()

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                weight_type = torch.bfloat16
                log(f"[show_o2-unified] device={device}")

                # ---- Load tokenizer ----
                log("[show_o2-unified] Loading tokenizer ...")
                text_tokenizer, showo_token_ids = get_text_tokenizer(
                    config.model.showo.llm_model_path,
                    add_showo_tokens=True,
                    return_showo_token_ids=True,
                    llm_name=path_to_llm_name[config.model.showo.llm_model_path],
                )
                config.model.showo.llm_vocab_size = len(text_tokenizer)

                # ---- Load model (ONCE) ----
                log(f"[show_o2-unified] Loading model from {config.model.showo.pretrained_model_path} ...")
                if config.model.showo.load_from_showo:
                    model = Showo2Qwen2_5.from_pretrained(
                        config.model.showo.pretrained_model_path, use_safetensors=False
                    )
                    # Fix meta params from accelerate init_empty_weights
                    if any(p.is_meta for p in model.parameters()):
                        log("[show_o2-unified] Fixing meta parameters with assign=True ...")
                        sd = load_state_dict(config.model.showo.pretrained_model_path)
                        model.load_state_dict(sd, assign=True)
                    model = model.to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict, assign=True)
                model.to(weight_type)
                model.eval()
                log("[show_o2-unified] Model loaded successfully")

                # ---- Load VAE ----
                log("[show_o2-unified] Loading VAE ...")
                vae_model = WanVAE(
                    vae_pth=config.model.vae_model.pretrained_model_path,
                    dtype=weight_type, device=device,
                )
                log("[show_o2-unified] VAE loaded")

                boi_id = showo_token_ids["boi_id"]

                # ==================================================================
                # Phase 1: Text Understanding
                # ==================================================================
                log("[show_o2-unified] === Phase 1: Text Understanding ===")

                sys_prompt_ids = text_tokenizer(
                    "system\\nYou are a helpful assistant.<|im_end|>",
                    add_special_tokens=False,
                )["input_ids"]
                role_a = text_tokenizer(
                    "\\n<|im_start|>user\\n", add_special_tokens=False
                )["input_ids"]
                role_b = text_tokenizer(
                    "\\n<|im_start|>assistant\\n", add_special_tokens=False
                )["input_ids"]

                text_results = []
                for i, question in enumerate(questions):
                    question_ids = text_tokenizer(question, add_special_tokens=False)["input_ids"]
                    input_ids = (
                        [showo_token_ids["bos_id"]]
                        + sys_prompt_ids + role_a + question_ids + role_b
                    )
                    log(f"[show_o2-unified] text [{i+1}/{len(questions)}] {question[:80]} ...")
                    try:
                        with torch.no_grad():
                            response = model.lm_generate(
                                input_ids=input_ids,
                                tokenizer=text_tokenizer,
                                max_new_tokens=max_new_tokens_und,
                                boi_token=boi_id,
                                device=device,
                            )
                        text_results.append({"text": response})
                        log(f"[show_o2-unified]   -> {len(response)} chars")
                    except Exception as e:
                        log(f"[show_o2-unified]   Error: {e}")
                        text_results.append({"text": ""})
                    # Incremental save
                    with open(text_results_file, "w") as f:
                        json.dump(text_results, f, ensure_ascii=False)

                n_text_ok = sum(1 for r in text_results if r.get("text"))
                log(f"[show_o2-unified] Phase 1 done: {n_text_ok}/{len(questions)} have text")

                torch.cuda.empty_cache()

                # ==================================================================
                # Phase 2: Image Generation
                # ==================================================================
                log("[show_o2-unified] === Phase 2: Image Generation ===")

                # Adjust for time embeddings
                if config.model.showo.add_time_embeds:
                    config.dataset.preprocessing.num_t2i_image_tokens += 1
                    config.dataset.preprocessing.num_mmu_image_tokens += 1
                    config.dataset.preprocessing.num_video_tokens += 1

                (num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens,
                 max_seq_len, max_text_len, image_latent_dim, patch_size,
                 latent_width, latent_height, pad_id, bos_id, eos_id,
                 boi_id_hp, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id,
                 guidance_scale_cfg) = get_hyper_params(config, text_tokenizer, showo_token_ids)

                guidance_scale = guidance_scale_override if guidance_scale_override is not None else guidance_scale_cfg
                config.transport.num_inference_steps = num_inference_steps

                transport = create_transport(
                    path_type=config.transport.path_type,
                    prediction=config.transport.prediction,
                    loss_weight=config.transport.loss_weight,
                    train_eps=config.transport.train_eps,
                    sample_eps=config.transport.sample_eps,
                    snr_type=config.transport.snr_type,
                    do_shift=config.transport.do_shift,
                    seq_len=num_t2i_image_tokens,
                )
                sampler = Sampler(transport)

                # Build gen prompts from text answers
                gen_prompts = []
                for i, question in enumerate(questions):
                    text_answer = text_results[i].get("text", "") if i < len(text_results) else ""
                    if text_answer:
                        gen_prompts.append(
                            f"{question}\\n\\n"
                            f"Based on the following description, generate an image:\\n"
                            f"{text_answer}"
                        )
                    else:
                        gen_prompts.append(question)

                gen_results = []
                for step in range(0, len(gen_prompts), gen_batch_size):
                    batch_prompts = gen_prompts[step:step + gen_batch_size]
                    batch_ids = item_ids[step:step + gen_batch_size]
                    log(f"[show_o2-unified] gen [{step+1}-{step+len(batch_prompts)}/{len(gen_prompts)}]")

                    try:
                        (batch_text_tokens, batch_text_tokens_null,
                         batch_mod_pos, batch_mod_pos_null) = prepare_gen_input(
                                batch_prompts, text_tokenizer, num_t2i_image_tokens,
                                bos_id, eos_id, boi_id_hp, eoi_id, pad_id, img_pad_id,
                                max_text_len, device,
                            )

                        z = torch.randn(
                            len(batch_prompts), image_latent_dim,
                            latent_height * patch_size, latent_width * patch_size,
                        ).to(weight_type).to(device)

                        if guidance_scale > 0:
                            z = torch.cat([z, z], dim=0)
                            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
                            modality_positions = torch.cat([batch_mod_pos, batch_mod_pos_null], dim=0)
                        else:
                            text_tokens = batch_text_tokens
                            modality_positions = batch_mod_pos

                        block_mask = omni_attn_mask_naive(
                            text_tokens.size(0), max_seq_len, modality_positions, device,
                        ).to(weight_type)

                        model_kwargs = dict(
                            text_tokens=text_tokens,
                            attention_mask=block_mask,
                            modality_positions=modality_positions,
                            output_hidden_states=True,
                            max_seq_len=max_seq_len,
                            guidance_scale=guidance_scale,
                        )

                        sample_fn = sampler.sample_ode(
                            sampling_method=config.transport.sampling_method,
                            num_steps=config.transport.num_inference_steps,
                            atol=config.transport.atol,
                            rtol=config.transport.rtol,
                            reverse=config.transport.reverse,
                            time_shifting_factor=config.transport.time_shifting_factor,
                        )

                        with torch.no_grad():
                            samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]

                        samples = torch.chunk(samples, 2)[0]
                        samples = samples.unsqueeze(2)
                        images = vae_model.batch_decode(samples)
                        images = images.squeeze(2)
                        images = denorm(images)

                        for j, img_arr in enumerate(images):
                            iid = batch_ids[j]
                            pil_img = Image.fromarray(img_arr)
                            out_path = os.path.join(output_dir, f"{iid}.png")
                            pil_img.save(out_path, format="PNG")
                            gen_results.append({"ok": True})
                            log(f"[show_o2-unified]   saved {out_path}")

                    except Exception as e:
                        log(f"[show_o2-unified]   gen batch error: {e}")
                        traceback.print_exc()
                        for _ in batch_ids:
                            gen_results.append({"ok": False})

                    # Incremental save
                    with open(gen_results_file, "w") as f:
                        json.dump(gen_results, f, ensure_ascii=False)

                log(f"[show_o2-unified] Phase 2 done: {sum(1 for r in gen_results if r.get('ok'))}/{len(gen_prompts)} images")
                log("[show_o2-unified] All done.")

            except Exception as e:
                log(f"[show_o2-unified] FATAL ERROR: {e}")
                traceback.print_exc()
                sys.exit(1)
            ''')

    def run_unified_batch(
        self,
        items: list[dict[str, Any]],
        images_dir: Path,
        understanding_params: dict[str, Any],
        gen_params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run text understanding + image generation in a single subprocess (one model load)."""
        if self.version != 2:
            raise NotImplementedError("Unified batch requires Show-o2 (version 2).")

        und_cfg = dict(self.default_understanding_cfg)
        cfg_name = self._get_config_name(und_cfg)
        max_new_tokens_und = int(und_cfg.get("max_new_tokens", 100))

        # Merge gen defaults with runtime overrides
        gen_cfg = dict(self.default_generation_cfg)
        if gen_params:
            gen_cfg.update(gen_params)
        guidance_scale = gen_cfg.get("guidance_scale")
        num_steps = gen_cfg.get("num_inference_steps") or gen_cfg.get("generation_timesteps")
        gen_batch_size = int(gen_cfg.get("batch_size", 1))

        questions = [item["prompt_text"] for item in items]
        item_ids = [item["item_id"] for item in items]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            params_file = tmpdir_path / "params.json"
            text_results_file = tmpdir_path / "text_results.json"
            gen_results_file = tmpdir_path / "gen_results.json"
            runner_file = tmpdir_path / "runner_unified.py"

            params_file.write_text(json.dumps({
                "questions": questions,
                "item_ids": item_ids,
                "max_new_tokens_und": max_new_tokens_und,
                "seed": self.seed,
                "model_path": self.model_path,
                "vae_path": self.vae_path,
                "config_name": cfg_name,
                "guidance_scale": guidance_scale,
                "num_inference_steps": int(num_steps) if num_steps is not None else 50,
                "batch_size": gen_batch_size,
                "output_dir": str(images_dir),
                "text_results_file": str(text_results_file),
                "gen_results_file": str(gen_results_file),
            }))
            runner_file.write_text(self._runner_code_unified_batch())

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)

            proc = subprocess.run(
                [self._get_python(), str(runner_file), str(params_file)],
                cwd=str(self._get_cwd()),
                env=env,
            )

            # Read text results
            text_results: list[dict[str, Any]] = []
            if text_results_file.exists():
                try:
                    text_results = json.loads(text_results_file.read_text())
                except Exception:
                    pass

            # Read gen results
            gen_results: list[dict[str, Any]] = []
            if gen_results_file.exists():
                try:
                    gen_results = json.loads(gen_results_file.read_text())
                except Exception:
                    pass

            if proc.returncode != 0:
                print(f"[show_o2] unified batch exited rc={proc.returncode}, "
                      f"text={len(text_results)}/{len(questions)}, "
                      f"gen={len(gen_results)}/{len(questions)}")

            # Pad with empty results if incomplete
            while len(text_results) < len(questions):
                text_results.append({"text": ""})
            while len(gen_results) < len(questions):
                gen_results.append({"ok": False})

            return text_results, gen_results

    def generate_batch(
        self,
        prompt_items: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Batch image generation — one subprocess, one model load.

        Uses inference_t2i.py which already loops over all prompts internally.
        The _gen_wrapper.py intercepts wandb.log and saves images as 000000.png,
        000001.png, ... in order, so we map them back by index.
        """
        if not prompt_items:
            return []

        prompts = [item["prompt"] for item in prompt_items]
        output_paths = [item.get("output_path", "") for item in prompt_items]

        cfg = dict(self.default_generation_cfg)
        if gen_cfg:
            cfg.update(gen_cfg)
        cfg_name = self._get_config_name(cfg)
        batch_size = int(cfg.get("batch_size", 1))
        guidance_scale = cfg.get("guidance_scale")
        model_path = cfg.get("model_path") or self.model_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            prompts_file = tmpdir_path / "validation_prompts.txt"
            prompts_file.write_text("\n".join(prompts))

            gen_img_dir = tmpdir_path / "generated_images"
            gen_img_dir.mkdir()

            if self.version == 2:
                cmd = [
                    self._get_python(),
                    str(_GEN_WRAPPER),
                    "inference_t2i.py",
                    f"config=configs/{cfg_name}",
                    f"dataset.params.validation_prompts_file={str(prompts_file)}",
                    f"validation_prompts_file={str(prompts_file)}",
                    f"batch_size={batch_size}",
                    f"model.showo.pretrained_model_path={model_path}",
                ]
                if self.vae_path:
                    cmd.append(f"model.vae_model.pretrained_model_path={self.vae_path}")
                if guidance_scale is not None:
                    cmd.append(f"guidance_scale={guidance_scale}")
                num_steps = cfg.get("num_inference_steps") or cfg.get("generation_timesteps")
                if num_steps is not None:
                    cmd.append(f"num_inference_steps={num_steps}")
            else:
                cmd = [
                    self._get_python(),
                    str(_GEN_WRAPPER),
                    "inference_t2i.py",
                    f"config=configs/{cfg_name}",
                    "mode=t2i",
                    f"validation_prompts_file={str(prompts_file)}",
                    f"batch_size={batch_size}",
                    f"model.showo.pretrained_model_path={model_path}",
                ]
                if self.vq_model_path:
                    cmd.append(f"model.vq_model.vq_model_name={self.vq_model_path}")
                if guidance_scale is not None:
                    cmd.append(f"guidance_scale={guidance_scale}")
                generation_timesteps = cfg.get("generation_timesteps")
                if generation_timesteps is not None:
                    cmd.append(f"generation_timesteps={generation_timesteps}")

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)
            env["UMM_OUTPUT_DIR"] = str(gen_img_dir)

            proc = subprocess.run(
                cmd,
                cwd=str(self._get_cwd()),
                env=env,
            )

            # Images saved as 000000.png, 000001.png, ... by _gen_wrapper
            found_images = sorted(gen_img_dir.glob("*.png"))

            if not found_images:
                print(f"[show_o2] batch gen failed rc={proc.returncode}")
                return [{"ok": False}] * len(prompt_items)

            # Map generated images to individual output_paths by index
            results: list[dict[str, Any]] = []
            for i, out_path in enumerate(output_paths):
                if i < len(found_images) and out_path:
                    dest = Path(out_path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(found_images[i], dest)
                    results.append({"ok": dest.is_file()})
                elif i < len(found_images):
                    results.append({"ok": True})
                else:
                    results.append({"ok": False})

            return results

    # ------------------------------------------------------------------
    # Single-item generation (kept for non-batch callers)
    # ------------------------------------------------------------------
    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        # Subprocess wrapper for inference_t2i.py
        prompt = batch.get("prompt") or gen_cfg.get("prompt")
        if prompt is None:
            raise ValueError("Generation requires a prompt in batch or gen_cfg.")

        # prompts can be a single string or list of strings
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]

        cfg = dict(self.default_generation_cfg)
        if gen_cfg:
            cfg.update(gen_cfg)
        cfg_name = self._get_config_name(cfg)

        batch_size = int(cfg.get("batch_size", 1))
        guidance_scale = cfg.get("guidance_scale")
        model_path = cfg.get("model_path") or self.model_path

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            prompts_file = tmpdir_path / "validation_prompts.txt"
            prompts_file.write_text("\n".join([str(p) for p in prompts]))

            # Dedicated subdir for generated images — filled by _gen_wrapper.py
            gen_img_dir = tmpdir_path / "generated_images"
            gen_img_dir.mkdir()

            if self.version == 2:
                cmd = [
                    self._get_python(),
                    str(_GEN_WRAPPER),
                    "inference_t2i.py",
                    f"config=configs/{cfg_name}",
                    f"dataset.params.validation_prompts_file={str(prompts_file)}",
                    f"validation_prompts_file={str(prompts_file)}",
                    f"batch_size={batch_size}",
                    f"model.showo.pretrained_model_path={model_path}",
                ]
                if self.vae_path:
                    cmd.append(f"model.vae_model.pretrained_model_path={self.vae_path}")
                if guidance_scale is not None:
                    cmd.append(f"guidance_scale={guidance_scale}")
                num_steps = cfg.get("num_inference_steps") or cfg.get("generation_timesteps")
                if num_steps is not None:
                    cmd.append(f"num_inference_steps={num_steps}")
            else:
                cmd = [
                    self._get_python(),
                    str(_GEN_WRAPPER),
                    "inference_t2i.py",
                    f"config=configs/{cfg_name}",
                    "mode=t2i",
                    f"validation_prompts_file={str(prompts_file)}",
                    f"batch_size={batch_size}",
                ]
                if guidance_scale is not None:
                    cmd.append(f"guidance_scale={guidance_scale}")
                generation_timesteps = cfg.get("generation_timesteps")
                if generation_timesteps is not None:
                    cmd.append(f"generation_timesteps={generation_timesteps}")
                cmd.append(f"model.showo.pretrained_model_path={model_path}")
                if cfg.get("vq_model_name"):
                    cmd.append(f"model.vq_model.vq_model_name={cfg.get('vq_model_name')}")

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)
            env["UMM_OUTPUT_DIR"] = str(gen_img_dir)

            proc = subprocess.run(
                cmd,
                cwd=str(self._get_cwd()),
                capture_output=True,
                text=True,
                env=env,
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # Read images directly from the output_dir we specified
            found_images = sorted(gen_img_dir.glob("*.png"))

            if not found_images:
                # Diagnostic: always print stderr tail when no images captured
                stderr_tail = stderr[-1200:] if stderr else "(empty)"
                print(f"[show_o generate] no images captured. rc={proc.returncode}")
                print(f"[show_o generate] stderr tail:\n{stderr_tail}")
                if stdout:
                    stdout_tail = stdout[-600:]
                    print(f"[show_o generate] stdout tail:\n{stdout_tail}")
                return {
                    "images": [],
                    "saved_paths": [],
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": proc.returncode,
                }

            # Determine output location
            output_path = batch.get("output_path")
            output_dir_from_cfg = cfg.get("output_dir")
            img_exts = {".png", ".jpg", ".jpeg", ".webp"}
            saved_paths: list[str] = []

            if output_path and Path(output_path).suffix.lower() in img_exts:
                # Caller wants a specific file — save the first generated image there
                dest = Path(output_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found_images[0], dest)
                saved_paths.append(str(dest))
            else:
                out_dir = Path(
                    output_dir_from_cfg
                    or output_path
                    or (Path(__file__).resolve().parents[4] / "output" / "show_o")
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(found_images):
                    dest = out_dir / f"showo_generated_{i:03d}{src.suffix}"
                    try:
                        shutil.copy2(src, dest)
                        saved_paths.append(str(dest))
                    except Exception:
                        saved_paths.append(str(src))

            return {
                "images": [str(p) for p in found_images],
                "saved_paths": saved_paths,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
            }

    def edit(self, batch: dict[str, Any], edit_cfg: dict[str, Any]) -> Any:
        raise NotImplementedError("Editing not supported by subprocess ShowO adapter.")
