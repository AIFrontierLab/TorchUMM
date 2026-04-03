from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
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
            "max_new_tokens": 100,
            "top_k": 1,
            "temperature": 0.8,
            "use_clip_vit": True,
        }

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

        # Version detection: explicit config > auto-detect from model config.json
        v = cfg.get("version")
        if v is not None:
            self.version = int(v)
        elif self._auto_detect_v2():
            self.version = 2

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
        # Run the original Show-o inference script as a subprocess to avoid
        # importing heavy model code here. The Show-o repo was copied under
        # src/umm/backbones/show_o/Show-o and will be invoked with CLI args.

        if videos:
            raise NotImplementedError("Videos not supported by subprocess adapter.")

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

        if self.version == 2:
            return self._run_v2_image_understanding(
                image_path=str(image_path),
                question=question,
                max_new_tokens=max_new_tokens,
                top_k=int(cfg.get("top_k", 1)),
                temperature=float(cfg.get("temperature", 1.0)),
                cfg_name=cfg_name,
            )

        # v1: subprocess to inference_mmu.py (stdout parsing works for v1)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            shutil.copy(image_path, tmpdir_path / image_path.name)

            cmd = [
                self._get_python(),
                "inference_mmu.py",
                f"config=configs/{cfg_name}",
                f"max_new_tokens={max_new_tokens}",
                f"mmu_image_root={str(tmpdir_path)}",
                f"question={question}",
                f"model.showo.pretrained_model_path={self.model_path}",
            ]
            if self.vq_model_path:
                cmd.append(f"model.vq_model.vq_model_name={self.vq_model_path}")

            env = dict(os.environ)
            env.setdefault("WANDB_MODE", "offline")
            env["WANDB_DIR"] = str(tmpdir_path)
            proc = subprocess.run(
                cmd,
                cwd=str(self._get_cwd()),
                capture_output=True,
                text=True,
                env=env,
            )

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            extracted = ""
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    content = line[1:-1].strip()
                    if content.startswith('"') or content.startswith("'"):
                        extracted = content.strip().strip('"').strip("'")
                    else:
                        extracted = content
                    break

            return {"text": extracted, "stdout": stdout, "stderr": stderr, "returncode": proc.returncode}

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
            model_path = params.get("model_path")
            vae_path = params.get("vae_path")
            config_name = params["config_name"]

            log(f"[show_o2-mmu] image={image_path}, question={question[:80]}")

            try:
                import torch
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
                    ).to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict)
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
            model_path = params.get("model_path")
            vae_path = params.get("vae_path")
            config_name = params["config_name"]

            log(f"[show_o2] Starting text batch runner: {len(questions)} questions")
            log(f"[show_o2] config={config_name}, model_path={model_path}")

            try:
                import torch
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
                    ).to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict)
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
                    ).to(device)
                else:
                    model = Showo2Qwen2_5(**config.model.showo).to(device)
                    state_dict = load_state_dict(config.model_path)
                    model.load_state_dict(state_dict)
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

        # Use backbone's understanding_cfg for Phase 1 (NOT request_params which has gen values)
        und_cfg = dict(self.default_understanding_cfg)
        cfg_name = self._get_config_name(und_cfg)
        max_new_tokens_und = int(und_cfg.get("max_new_tokens", 100))

        # Gen params from request_params for Phase 2
        guidance_scale = gen_params.get("guidance_scale")
        num_steps = gen_params.get("num_inference_steps") or gen_params.get("generation_timesteps")
        gen_batch_size = int(gen_params.get("batch_size", 1))

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

        cfg_name = self._get_config_name(gen_cfg)
        batch_size = int(gen_cfg.get("batch_size", 1))
        guidance_scale = gen_cfg.get("guidance_scale")
        model_path = gen_cfg.get("model_path") or self.model_path

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
                num_steps = gen_cfg.get("num_inference_steps") or gen_cfg.get("generation_timesteps")
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

        cfg_name = self._get_config_name(gen_cfg)

        batch_size = int(gen_cfg.get("batch_size", 1))
        guidance_scale = gen_cfg.get("guidance_scale")
        model_path = gen_cfg.get("model_path") or self.model_path

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
                num_steps = gen_cfg.get("num_inference_steps") or gen_cfg.get("generation_timesteps")
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
                generation_timesteps = gen_cfg.get("generation_timesteps")
                if generation_timesteps is not None:
                    cmd.append(f"generation_timesteps={generation_timesteps}")
                cmd.append(f"model.showo.pretrained_model_path={model_path}")
                if gen_cfg.get("vq_model_name"):
                    cmd.append(f"model.vq_model.vq_model_name={gen_cfg.get('vq_model_name')}")

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
            output_dir_from_cfg = gen_cfg.get("output_dir") if gen_cfg else None
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
