from __future__ import annotations

import importlib
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from PIL import Image


def _timeout_handler(signum, frame):
    raise TimeoutError("Image generation timed out")


# Default sampling params matching Emu3.5 configs/config.py
_DEFAULT_SAMPLING_PARAMS = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=5120,
    guidance_scale=1.0,
    use_differential_sampling=True,
    do_sample=True,
    num_beams=1,
)

_SPECIAL_TOKENS = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)


def _build_unc_and_template(task: str, with_image: bool):
    """Build unconditional prompt and template for Emu3.5."""
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {question} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


class Emu3dot5Backbone:
    name = "emu3_5"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vq_path: Optional[str] = None,
        emu3_5_root: Optional[str] = None,
        use_vllm: bool = True,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.7,
        vq_device: str = "cuda:0",
        classifier_free_guidance: float = 5.0,
        max_new_tokens: int = 5120,
        image_area: int = 1048576,
        seed: int = 6666,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "Emu3.5"
        self.emu3_5_root = Path(emu3_5_root) if emu3_5_root else (
            packaged if packaged.exists() else repo_root / "model" / "Emu3.5"
        )
        self.model_path = model_path or str(repo_root / "model_cache" / "emu3_5" / "Emu3.5-Image")
        self.vq_path = vq_path or str(repo_root / "model_cache" / "emu3_5" / "Emu3.5-VisionTokenizer")
        self.tokenizer_path = str(self.emu3_5_root / "src" / "tokenizer_emu3_ibq")
        self.use_vllm = use_vllm
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vq_device = vq_device
        self.classifier_free_guidance = classifier_free_guidance
        self.max_new_tokens = max_new_tokens
        self.image_area = image_area
        self.seed = seed

        # Lazily populated by load()
        self.model: Any = None
        self.tokenizer: Any = None
        self.vq_model: Any = None
        self._backend: str = ""  # "vllm" or "transformers"

    def load(self, cfg: dict[str, Any]) -> None:
        # Update config from dict
        for key in (
            "model_path", "vq_path", "emu3_5_root", "use_vllm",
            "tensor_parallel_size", "gpu_memory_utilization", "vq_device",
            "classifier_free_guidance", "max_new_tokens", "image_area", "seed",
        ):
            if key in cfg and cfg[key] is not None:
                val = cfg[key]
                if key == "emu3_5_root":
                    self.emu3_5_root = Path(val)
                    self.tokenizer_path = str(self.emu3_5_root / "src" / "tokenizer_emu3_ibq")
                else:
                    setattr(self, key, val)

        # Add Emu3.5 repo to sys.path so its modules are importable
        emu_root_str = str(self.emu3_5_root.resolve())
        if emu_root_str not in sys.path:
            sys.path.insert(0, emu_root_str)

        # Build tokenizer (shared by both backends)
        self.tokenizer = self._build_tokenizer()

        # Build VQ vision tokenizer (does NOT depend on modeling_emu3.py)
        vt_mod = importlib.import_module("src.vision_tokenizer")
        self.vq_model = vt_mod.build_vision_tokenizer("ibq", self.vq_path, device=self.vq_device)

        if self.use_vllm:
            try:
                self.model = self._build_vllm_model()
                self._backend = "vllm"
                print("[emu3.5] Model loaded with vLLM backend.", flush=True)
            except Exception as e:
                print(f"[emu3.5] vLLM loading failed: {e}.", flush=True)
                raise

        # Build special token IDs
        self.special_token_ids = {}
        for k, v in _SPECIAL_TOKENS.items():
            self.special_token_ids[k] = self.tokenizer.encode(v)[0]

    def _build_tokenizer(self):
        """Build and configure the Emu3.5 text tokenizer."""
        import os.path as osp
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            special_tokens_file=osp.join(self.tokenizer_path, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
        tokenizer.bos_token = "<|extra_203|>"
        tokenizer.eos_token = "<|extra_204|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eol_token = "<|extra_200|>"
        tokenizer.eof_token = "<|extra_201|>"
        tokenizer.tms_token = "<|extra_202|>"
        tokenizer.img_token = "<|image token|>"
        tokenizer.boi_token = "<|image start|>"
        tokenizer.eoi_token = "<|image end|>"
        tokenizer.bss_token = "<|extra_100|>"
        tokenizer.ess_token = "<|extra_101|>"
        tokenizer.bog_token = "<|extra_60|>"
        tokenizer.eog_token = "<|extra_61|>"
        tokenizer.boc_token = "<|extra_50|>"
        tokenizer.eoc_token = "<|extra_51|>"
        return tokenizer

    def _build_vllm_model(self):
        """Build the vLLM LLM engine.

        Requires BAAI's vLLM patches to be applied at image build time
        (see modal/images.py).  The patches register a native Emu3.5
        architecture in vLLM with optimized attention kernels.
        """
        from vllm import LLM

        print(f"[emu3.5] Loading model with vLLM from {self.model_path} ...", flush=True)

        # Build resolution token map
        resolution_map = {}
        for digit_str in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*"]:
            resolution_map[self.tokenizer.encode(digit_str)[0]] = digit_str

        model = LLM(
            self.model_path,
            tokenizer=self.tokenizer_path,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_log_stats=False,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            max_num_batched_tokens=26000,
            max_num_seqs=2,
            seed=self.seed,
            generation_config="vllm",
            scheduler_cls="vllm.v1.core.sched.batch_scheduler.Scheduler",
            compilation_config={
                "full_cuda_graph": True,
                "backend": "cudagraph",
                "cudagraph_capture_sizes": [1, 2],
            },
            additional_config={
                "boi_token_id": self.tokenizer.encode("<|image start|>")[0],
                "soi_token_id": self.tokenizer.encode("<|image token|>")[0],
                "eol_token_id": self.tokenizer.encode("<|extra_200|>")[0],
                "eoi_token_id": self.tokenizer.encode("<|image end|>")[0],
                "resolution_map": resolution_map,
            },
        )
        model.set_tokenizer(self.tokenizer)
        return model

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load({})

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        self._ensure_loaded()

        prompt = batch.get("prompt") or gen_cfg.get("prompt")
        if prompt is None:
            raise ValueError("Generation requires a prompt.")
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        output_path = batch.get("output_path")

        saved_paths = []
        with torch.inference_mode():
            for i, p in enumerate(prompts):
                img = self._generate_one(p, gen_cfg)
                if img is not None and output_path and i == 0:
                    dst = Path(output_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(dst), format=self._fmt(dst))
                    saved_paths.append(str(dst))

        return {
            "images": saved_paths,
            "saved_paths": saved_paths,
            "output_path": output_path or "",
        }

    def generate_batch(
        self,
        prompt_items: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not prompt_items:
            return []
        self._ensure_loaded()

        timeout_sec = int(gen_cfg.get("timeout_per_image", 900))

        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for i, item in enumerate(prompt_items):
                prompt = item["prompt"]
                output_path = item.get("output_path", "")
                print(f"[emu3.5] [{i + 1}/{len(prompt_items)}] {prompt[:80]} ...", flush=True)

                try:
                    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_sec)
                    try:
                        img = self._generate_one(prompt, gen_cfg)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev_handler)
                except TimeoutError:
                    print(f"[emu3.5]   Timeout after {timeout_sec}s, skipping", flush=True)
                    results.append({"images": [], "ok": False})
                    continue
                except Exception as e:
                    print(f"[emu3.5]   Error: {e}", flush=True)
                    results.append({"images": [], "ok": False})
                    continue

                ok = False
                if img is not None and output_path:
                    dst = Path(output_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(dst), format=self._fmt(dst))
                    ok = dst.is_file()
                    print(f"[emu3.5]   -> saved to {dst}", flush=True)

                results.append({"images": [output_path] if ok else [], "output_path": output_path, "ok": ok})

        return results

    def _generate_one(self, prompt: str, gen_cfg: dict[str, Any]) -> Optional[Image.Image]:
        """Run a single T2I generation and return the PIL Image (or None)."""
        cfg_scale = float(gen_cfg.get("classifier_free_guidance", self.classifier_free_guidance))
        max_tokens = int(gen_cfg.get("max_new_tokens", self.max_new_tokens))
        image_area = int(gen_cfg.get("image_area", self.image_area))

        # Build prompt from template
        unc_prompt, template = _build_unc_and_template("t2i", False)
        full_prompt = template.format(question=prompt)

        # Tokenize
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        bos_id = self.special_token_ids["BOS"]
        if input_ids[0, 0] != bos_id:
            bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
            input_ids = torch.cat([bos, input_ids], dim=1)

        unconditional_ids = self.tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False)

        # Build a config namespace for the generation functions
        sampling_params = dict(_DEFAULT_SAMPLING_PARAMS)
        sampling_params["max_new_tokens"] = max_tokens
        # Override sampling params from gen_cfg if provided
        for key in ("image_top_k", "text_top_k", "text_top_p", "image_top_p"):
            if key in gen_cfg:
                sampling_params[key] = gen_cfg[key]

        cfg_ns = SimpleNamespace(
            classifier_free_guidance=cfg_scale,
            sampling_params=sampling_params,
            special_token_ids=self.special_token_ids,
            special_tokens=_SPECIAL_TOKENS,
            unconditional_type="no_text",
            image_area=image_area,
            task_type="t2i",
            streaming=False,
        )

        if self._backend == "vllm":
            return self._generate_one_vllm(cfg_ns, input_ids, unconditional_ids)
        else:
            return self._generate_one_transformers(cfg_ns, input_ids, unconditional_ids)

    def _generate_one_vllm(self, cfg_ns, input_ids, unconditional_ids) -> Optional[Image.Image]:
        """Generate one image using the vLLM backend."""
        vllm_gen = importlib.import_module("src.utils.vllm_generation_utils")
        gen_utils = importlib.import_module("src.utils.generation_utils")

        for result_tokens in vllm_gen.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
            result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
            mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
            for kind, payload in mm_out:
                if kind == "image" and isinstance(payload, Image.Image):
                    return payload
        return None

    def _generate_one_transformers(self, cfg_ns, input_ids, unconditional_ids) -> Optional[Image.Image]:
        """Generate one image using the Transformers backend."""
        gen_utils = importlib.import_module("src.utils.generation_utils")

        for result_tokens in gen_utils.generate(cfg_ns, self.model, self.tokenizer, input_ids, unconditional_ids):
            result = self.tokenizer.decode(result_tokens, skip_special_tokens=False)
            mm_out = gen_utils.multimodal_decode(result, self.tokenizer, self.vq_model)
            for kind, payload in mm_out:
                if kind == "image" and isinstance(payload, Image.Image):
                    return payload
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")
