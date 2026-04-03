from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


class JanusFlowBackbone:
    name = "janus_flow"

    def __init__(
        self,
        model_path: Optional[str] = None,
        janus_root: Optional[str] = None,
        vae_path: Optional[str] = None,
        seed: int = 42,
        torch_dtype: str = "bfloat16",
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.janus_root = (
            Path(janus_root).expanduser()
            if janus_root
            else repo_root / "src" / "umm" / "backbones" / "janus_flow" / "Janus"
        )
        self.model_path = (
            str(Path(model_path).expanduser())
            if model_path
            else str(repo_root / "model_cache" / "janus_flow" / "JanusFlow-1.3B")
        )
        self.vae_path = (
            str(Path(vae_path).expanduser())
            if vae_path
            else str(repo_root / "model_cache" / "janus_flow" / "sdxl-vae")
        )
        self.seed = seed
        self.torch_dtype = torch_dtype

        self.default_generation_cfg: dict[str, Any] = {
            "cfg_weight": 2.0,
            "num_inference_steps": 30,
            "parallel_size": 5,
            "img_size": 384,
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        self.vl_chat_processor: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.vae: Any = None

    def load(self, cfg: dict[str, Any]) -> None:
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(Path(cfg_model_path).expanduser())
        cfg_janus_root = cfg.get("janus_root")
        if cfg_janus_root:
            self.janus_root = Path(cfg_janus_root).expanduser()
        cfg_vae_path = cfg.get("vae_path")
        if cfg_vae_path:
            self.vae_path = str(Path(cfg_vae_path).expanduser())
        self.seed = int(cfg.get("seed", self.seed))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

        self._set_seed(self.seed)
        self.vl_chat_processor, self.tokenizer, self.model, self.vae = (
            self._build_model()
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected batch['prompt'] as a non-empty string.")
        return self.generation(
            prompt=prompt, output_path=batch.get("output_path"), generation_cfg=gen_cfg
        )

    def generation(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.model is None or self.vl_chat_processor is None:
            self.load({})
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected `prompt` as a non-empty string.")

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        full_prompt = self._build_generation_prompt(prompt)
        images = self._generate_images(full_prompt, config)
        saved_paths = self._save_images(images, output_path)
        return {
            "image": images[0] if images else None,
            "images": images,
            "saved_paths": saved_paths,
        }

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------

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
        if self.model is None or self.vl_chat_processor is None:
            self.load({})

        image_paths = images or []
        video_paths = videos or []
        if video_paths:
            raise NotImplementedError(
                "JanusFlowBackbone.understanding currently does not support videos."
            )

        if not prompt and not image_paths:
            raise ValueError("Understanding requires at least one input: prompt or images.")

        pil_images: list[Image.Image] = []
        if image_paths:
            image_path = Path(image_paths[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Understanding image not found: {image_path}")
            pil_images.append(Image.open(image_path).convert("RGB"))

        user_prompt = prompt if isinstance(prompt, str) and prompt else "Describe the image."
        content = f"<image_placeholder>\n{user_prompt}" if pil_images else user_prompt
        conversation = [
            {
                "role": "User",
                "content": content,
                "images": image_paths[:1] if image_paths else [],
            },
            {"role": "Assistant", "content": ""},
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self.model.device, dtype=torch.bfloat16)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            do_sample=bool(config.get("do_sample", False)),
            use_cache=bool(config.get("use_cache", True)),
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return {"text": answer, "sft_format": prepare_inputs["sft_format"][0]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _build_model(self) -> tuple[Any, Any, Any, Any]:
        if not self.janus_root.exists():
            raise FileNotFoundError(f"Janus root not found: {self.janus_root}")

        janus_root_str = str(self.janus_root.resolve())
        if janus_root_str not in sys.path:
            sys.path.insert(0, janus_root_str)

        if not torch.cuda.is_available():
            raise RuntimeError("JanusFlowBackbone requires at least one CUDA device.")

        # Import JanusFlow-specific model classes
        janusflow_models = importlib.import_module("janus.janusflow.models")
        MultiModalityCausalLM = janusflow_models.MultiModalityCausalLM
        VLChatProcessor = janusflow_models.VLChatProcessor

        model_path = str(Path(self.model_path).expanduser())
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        # Force CPU as default device during model loading to prevent
        # transformers 5.x from using meta tensors internally, which
        # breaks torch.linspace(...).item() in siglip_vit.py.
        default_device_before = None
        if hasattr(torch, "set_default_device"):
            if hasattr(torch, "get_default_device"):
                default_device_before = torch.get_default_device()
            torch.set_default_device("cpu")

        try:
            model = MultiModalityCausalLM.from_pretrained(model_path)
        finally:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device(default_device_before)

        model = model.to(torch.bfloat16).cuda().eval()

        # Load SDXL VAE for decoding latents to images.
        # IMPORTANT: VAE must use bfloat16 — fp16 produces garbage output.
        diffusers_models = importlib.import_module("diffusers.models")
        vae = diffusers_models.AutoencoderKL.from_pretrained(self.vae_path)
        vae = vae.to(torch.bfloat16).cuda().eval()

        return vl_chat_processor, tokenizer, model, vae

    def _build_generation_prompt(self, prompt: str) -> str:
        conversation = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    def _generate_images(self, prompt: str, cfg: dict[str, Any]) -> list[Image.Image]:
        parallel_size = int(cfg.get("parallel_size", 5))
        cfg_weight = float(cfg.get("cfg_weight", 2.0))
        num_inference_steps = int(cfg.get("num_inference_steps", 30))
        img_size = int(cfg.get("img_size", 384))

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()

        # Duplicate tokens for CFG: first half conditioned, second half unconditional
        batch_size = parallel_size * 2
        tokens = torch.stack([input_ids] * batch_size).cuda()
        tokens[parallel_size:, 1:] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        # Remove last <bog> token — will be replaced with t_emb in ODE loop
        inputs_embeds = inputs_embeds[:, :-1, :]

        # Initialize random latent
        z = torch.randn(
            (parallel_size, 4, 48, 48), dtype=torch.bfloat16, device="cuda"
        )

        dt = 1.0 / num_inference_steps
        dt = torch.zeros_like(z) + dt

        # Build attention mask: text tokens + t_emb (1) + z_emb (576) = +577
        seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones(
            (batch_size, seq_len + 577), dtype=torch.int, device="cuda"
        )
        # Mask out text tokens for unconditional branch (keep BOS at position 0)
        attention_mask[parallel_size:, 1:seq_len] = 0

        with torch.inference_mode():
            for step in range(num_inference_steps):
                # Duplicate z for CFG
                z_input = torch.cat([z, z], dim=0)
                t = step / num_inference_steps * 1000.0
                t = torch.tensor([t] * batch_size, dtype=torch.bfloat16, device="cuda")

                # Encode z through vision generation encoder
                z_enc = self.model.vision_gen_enc_model(z_input, t)
                z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]

                # Reshape: (B, C, H, W) -> (B, H*W, C) and project to LLM dim
                z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
                z_emb = self.model.vision_gen_enc_aligner(z_emb)

                # Concatenate: [text_emb, t_emb, z_emb]
                llm_emb = torch.cat(
                    [inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1
                )

                # Full forward pass each step (no KV-cache)
                outputs = self.model.language_model.model(
                    inputs_embeds=llm_emb,
                    use_cache=False,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state

                # Extract last 576 tokens and decode back to velocity
                hidden_states = self.model.vision_gen_dec_aligner(
                    self.model.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :])
                )
                hidden_states = hidden_states.reshape(
                    batch_size, 24, 24, 768
                ).permute(0, 3, 1, 2)

                v = self.model.vision_gen_dec_model(hidden_states, hs, t_emb)

                # Classifier-free guidance
                v_cond, v_uncond = torch.chunk(v, 2)
                v = cfg_weight * v_cond - (cfg_weight - 1.0) * v_uncond

                # Euler ODE step
                z = z + dt * v

            # Decode latent with SDXL VAE
            decoded_image = self.vae.decode(
                z / self.vae.config.scaling_factor
            ).sample

        images_np = decoded_image.float().clip(-1.0, 1.0)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        images_np = ((images_np + 1) / 2.0 * 255).astype(np.uint8)

        return [
            Image.fromarray(images_np[i]).resize((img_size, img_size), Image.LANCZOS)
            for i in range(parallel_size)
        ]

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")

    def _save_images(self, images: list[Image.Image], output_path: Optional[str]) -> list[str]:
        if not output_path:
            return []

        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix if out_path.suffix else ".png"
        stem = out_path.stem if out_path.stem else "output"

        saved_paths: list[str] = []
        if len(images) == 1:
            target = out_path if out_path.suffix else out_path.with_suffix(suffix)
            images[0].save(target, format=self._fmt(target))
            saved_paths.append(str(target))
            return saved_paths

        for idx, image in enumerate(images):
            target = out_path.with_name(f"{stem}_{idx}{suffix}")
            image.save(target, format=self._fmt(target))
            saved_paths.append(str(target))
        return saved_paths

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        lowered = dtype_name.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
