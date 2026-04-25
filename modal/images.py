"""
Modal Image definitions for umm_codebase.

Each model repo gets its own Image because they require different:
  - Python versions (3.10 / 3.11)
  - PyTorch versions (1.12 / 2.1 / 2.2 / 2.3 / 2.5 / 2.6)
  - flash-attn versions (2.5.7 / 2.5.8 / 2.6.2)
  - Other deps

Images are lazy-loaded: only the requested model's Image is created,
so importing this module does NOT trigger builds for all models.

Usage:
    image = get_image("bagel")
"""

import modal

# =====================================================================
# Common system packages shared by most ML images
# =====================================================================
_SYS_PACKAGES = [
    "git", "libsm6", "libxext6", "libgl1",
    "ffmpeg", "build-essential", "ninja-build", "wget", "curl",
    "clang",
]


def _bagel_image():
    """BAGEL — Python 3.10, torch 2.5.1+cu124, flash-attn 2.5.8"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.49.0", "accelerate>=0.34.0",
            "safetensors==0.4.5", "sentencepiece==0.1.99",
            "scipy==1.10.1", "numpy==1.24.4",
            "einops==0.8.1", "decord==0.6.0",
            "opencv-python==4.7.0.72",
            "pyarrow", "PyYAML==6.0.2",
            "Requests==2.32.3", "Pillow",
            "huggingface_hub==0.29.1",
            "datasets",
            "matplotlib==3.7.0", "xlsxwriter",
            "wandb", "gradio", "bitsandbytes",
            "triton", "setuptools", "wheel", "ninja",
            "openpyxl",
        ])
        .pip_install("flash-attn==2.5.8", extra_options="--no-build-isolation", gpu="A100")
        .add_local_python_source("config", "volumes", "images")
    )


def _blip3o_image():
    """BLIP3o — Python 3.11, torch 2.3.0 (CUDA 12.1), flash-attn 2.6.2"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0",
            "transformers==4.51.3", "accelerate==0.28.0",
            "datasets==2.16.1", "deepspeed==0.14.4",
            "diffusers==0.34.0", "gradio==5.34.0",
            "safetensors", "sentencepiece",
            "scipy", "numpy", "Pillow",
            "opencv-python", "einops",
            "easydict", "tabulate",
            "tqdm", "pyarrow", "PyYAML",
            "wandb", "huggingface_hub",
            "setuptools", "wheel",
        ])
        .pip_install("flash-attn==2.6.2", extra_options="--no-build-isolation", gpu="A100")
        .add_local_python_source("config", "volumes", "images")
    )


def _emu3_image():
    """EMU3 — Python 3.10, torch 2.2.1, flash-attn 2.5.7"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "numpy<2",
            "torch==2.2.1", "torchvision",
            "transformers==4.44.0",
            "tiktoken==0.6.0",
            "Pillow", "gradio==4.44.0",
            "datasets",
            "huggingface_hub", "accelerate",
            "setuptools", "wheel", "ninja",
            "openpyxl",
        ])
        .run_commands("pip install --upgrade setuptools wheel")
        .pip_install("flash-attn==2.5.7", extra_options="--no-build-isolation", gpu="A100")
        .add_local_python_source("config", "volumes", "images")
    )


def _geneval_image():
    """GenEval — Python 3.10, torch 2.0.1+cu118, mmdet 2.x (H100-compatible)"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .env({"CXX": "g++", "CC": "gcc"})
        .pip_install(
            ["torch==2.0.1+cu118", "torchvision==0.15.2+cu118"],
            extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
        )
        .pip_install([
            "transformers==4.36.1",
            "diffusers==0.24.0",
            "opencv-python==4.6.0.66",
            "numpy==1.23.1", "Pillow==9.4.0",
            "scipy", "scikit-image",
            "open-clip-torch==2.20.0",
            "sentence-transformers==2.2.2",
            "clip-benchmark==1.4.0",
            "timm", "einops",
            "huggingface_hub",
            "pandas",
            "accelerate",
            "setuptools<70", "wheel",
        ])
        # mmdet 2.x uses mmcv-full; build from source to match torch ABI
        .pip_install(["mmcv-full==1.7.2"], gpu="A100")
        .pip_install(["mmdet==2.28.2"])
        .run_commands(
            # mmdet pip install omits configs/; evaluate_images.py expects them
            # at <site-packages>/configs/ relative to mmdet package.
            "apt-get update && apt-get install -y git",
            "git clone --depth 1 --branch v2.28.2 --filter=blob:none --sparse"
            " https://github.com/open-mmlab/mmdetection.git /tmp/mmdet_repo &&"
            " cd /tmp/mmdet_repo && git sparse-checkout set configs",
            "DEST=$(python -c \"import os, mmdet; print(os.path.normpath(os.path.join(os.path.dirname(mmdet.__file__), '..', 'configs')))\") &&"
            " cp -r /tmp/mmdet_repo/configs \"$DEST\" &&"
            " rm -rf /tmp/mmdet_repo",
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _janus_pro_image():
    """Janus Pro — Python 3.10, torch 2.5.1+cu124, transformers 4.45, flash-attn 2.7.4"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.45.0", "accelerate",
            "timm>=0.9.16", "sentencepiece",
            "attrdict", "einops",
            "datasets", "numpy<2", "Pillow",
            "huggingface_hub",
            "setuptools", "wheel", "ninja",
            "openpyxl",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _janus_flow_image():
    """JanusFlow — Python 3.10, torch 2.5.1+cu124, transformers 4.45, diffusers"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.45.0", "accelerate",
            "diffusers==0.30.3",
            "timm>=0.9.16", "sentencepiece",
            "attrdict", "einops",
            "datasets", "numpy<2", "Pillow",
            "huggingface_hub",
            "setuptools", "wheel",
            "openpyxl",
        ])
        .add_local_python_source("config", "volumes", "images")
    )


def _deepgen_image():
    """DeepGen — Python 3.10, torch 2.8.0, diffusers 0.35.2, flash-attn (source build)"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.8.0", "torchvision==0.23.0",
            "transformers==4.56.1", "accelerate",
            "diffusers==0.35.2", "datasets",
            "einops", "peft",
            "qwen-vl-utils", "sentencepiece",
            "opencv-python-headless", "scipy",
            "Pillow", "huggingface_hub",
            "setuptools", "wheel", "ninja",
        ])
        .pip_install("flash-attn", extra_options="--no-build-isolation", gpu="A100")
        .add_local_python_source("config", "volumes", "images")
    )


def _mmada_image():
    """MMaDA — Python 3.10, torch 2.5.1+cu124, transformers 4.46.0, flash-attn 2.7.4"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.46.0", "accelerate",
            "diffusers==0.32.2", "deepspeed",
            "datasets", "omegaconf",
            "jaxtyping", "typeguard",
            "pyarrow", "pandas",
            "lightning", "wandb",
            "scipy", "numpy", "Pillow",
            "huggingface_hub", "sentencepiece",
            "einops", "torchmetrics==1.6.2",
            "setuptools", "wheel", "ninja",
            "openpyxl",
            "modelscope", "addict",
        ])
        .run_commands(
            "pip install git+https://github.com/openai/CLIP.git"
        )
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _omnigen2_image():
    """OmniGen2 — Python 3.11, torch 2.6.0, flash-attn 2.7.4"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.6.0", "torchvision==0.21.0",
            "transformers==4.51.3", "accelerate",
            "diffusers", "datasets",
            "opencv-python-headless", "scipy",
            "timm", "einops",
            "omegaconf", "python-dotenv",
            "matplotlib", "tqdm",
            "wandb", "Pillow", "huggingface_hub",
            "ninja", "wheel",
            "bitsandbytes>=0.43.0",
            "openpyxl",
            "scikit-learn",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _ovis_u1_image():
    """Ovis-U1 — Python 3.10, torch 2.5.1+cu124, transformers 4.51.3, flash-attn 2.7.4"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.51.3", "accelerate",
            "datasets", "numpy<2", "Pillow",
            "huggingface_hub", "sentencepiece",
            "diffusers", "einops", "scipy",
            "setuptools", "wheel", "ninja",
            "openpyxl", "scikit-learn",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _show_o_image():
    """Show-o (v1) — Python 3.10, torch 2.2.1+cu121, xformers 0.0.25"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.2.1", "torchvision==0.17.1",
            "transformers==4.41.1", "accelerate==0.21.0",
            "diffusers==0.30.1", "deepspeed==0.14.2",
            "datasets==2.20.0", "opencv-python",
            "einops==0.6.0", "timm==1.0.3",
            "open-clip-torch==2.24.0", "kornia==0.7.2",
            "omegaconf==2.3.0",
            "lightning==2.2.3", "pytorch-lightning==2.2.3",
            "wandb", "scipy", "safetensors",
            "sentencepiece==0.2.0", "ftfy",
            "decord==0.6.0", "numpy==1.24.4",
            "pyarrow>=11.0.0", "fire==0.6.0",
            "triton==2.2.0", "xformers==0.0.25",
            "Pillow", "huggingface_hub",
            "setuptools", "wheel", "ninja",
            "openpyxl",
            "jaxtyping==0.2.28", "typeguard==2.13.3",
            "scikit-learn",
        ])
        .add_local_python_source("config", "volumes", "images")
    )


def _show_o2_image():
    """Show-o2 — Python 3.10, torch 2.5.1+cu124, flash-attn"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.47.0", "accelerate==0.23.0",
            "diffusers==0.31.0", "deepspeed==0.15.3",
            "datasets==2.20.0", "opencv-python",
            "einops==0.8.0", "timm==1.0.12",
            "open-clip-torch==2.24.0", "kornia==0.7.2",
            "omegaconf", "torchdiffeq",
            "lightning==2.4.0",
            "wandb", "scipy", "safetensors",
            "sentencepiece==0.2.0", "ftfy",
            "decord==0.6.0", "numpy==1.24.4",
            "pyarrow>=11.0.0", "fire==0.6.0",
            "triton",
            "Pillow", "huggingface_hub==0.24.0",
            "setuptools", "wheel", "ninja",
            "openpyxl",
            "scikit-learn",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _tokenflow_image():
    """TokenFlow — Python 3.10, torch 2.1.2"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.1.2", "torchvision==0.16.2",
            "transformers==4.43.4", "accelerate==0.31.0",
            "datasets==2.20.0", "diffusers",
            "sentencepiece==0.1.99",
            "einops==0.6.1", "timm==1.0.9",
            "omegaconf", "ftfy",
            "pytorch_lightning==2.2.4", "lightning",
            "tensorboardX",
            "deepspeed==0.12.6", "peft", "bitsandbytes",
            "gradio==4.16.0",
            "opencv-python", "numpy", "scipy",
            "scikit-learn==1.2.2", "scikit-image",
            "Pillow==10.4.0",
            "huggingface_hub==0.24.5",
            "wandb", "ninja",
        ])
        .add_local_python_source("config", "volumes", "images")
    )


def _ueval_image():
    """UEval — lightweight, just needs google-genai + pillow"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .pip_install([
            "google-genai", "Pillow", "datasets", "huggingface_hub",
        ])
        .add_local_python_source("config", "volumes", "images")
    )


def _uni_mmmu_image():
    """Uni-MMMU — Python 3.10, torch 2.5.1, heavy dependency tree"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES + ["libcairo2-dev", "libffi-dev"])
        .pip_install([
            "torch==2.5.1", "torchvision",
            "transformers==4.51.3", "accelerate==0.34.2",
            "diffusers==0.32.2",
            "open-clip-torch==3.1.0",
            "opencv-python==4.8.1.78",
            "bitsandbytes>=0.43.0",
            "peft==0.17.1", "timm==1.0.19",
            "einops==0.8.1", "datasets",
            "liger-kernel", "dreamsim", "lpips",
            "qwen-vl-utils", "google-genai",
            "sentencepiece", "moviepy",
            "lion-pytorch", "torchmetrics",
            "Pillow", "scipy", "huggingface_hub",
            "cairosvg", "openpyxl",
        ])
        .add_local_python_source("config", "volumes", "images")
    )


def _wise_image():
    """WISE — Python 3.10, torch 2.5.1+cu124, Qwen2.5-VL for VLM scoring"""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .env({"CXX": "g++", "CC": "gcc"})
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers==4.51.3", "accelerate",
            "qwen-vl-utils", "sentencepiece",
            "openai", "Pillow", "datasets", "huggingface_hub",
            "python-Levenshtein", "autoawq", "diffusers", "modelscope[framework]", "addict",
            "setuptools", "wheel", "ninja",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _unified_bench_image():
    """Unified-Bench scoring image — light, just needs transformers + Long-CLIP.

    Used for the scoring phase of Unified-Bench (CLIP / DINOv2 / DINOv3 /
    LongCLIP similarity). No flash-attn needed because we only run vision
    encoders. DINOv3 support requires transformers >= 4.50.
    """
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.5.1", "torchvision==0.20.1",
            "transformers>=4.50.0",
            "safetensors", "Pillow",
            "numpy", "tqdm", "PyYAML",
            "huggingface_hub",
            "ftfy", "regex",
            "setuptools", "wheel",
        ])
        # Long-CLIP model code (weights downloaded separately to the model cache).
        # The beichenzbc/Long-CLIP repo is not a pip package, so we clone it
        # to /opt/longclip; the scorer adds /opt to sys.path and imports
        # `from longclip.model import longclip`.
        .pip_install(["open_clip_torch"])  # LongCLIP depends on open_clip internals
        .run_commands(
            "git clone --depth 1 https://github.com/beichenzbc/Long-CLIP.git /opt/longclip",
            "touch /opt/longclip/__init__.py /opt/longclip/model/__init__.py",
        )
        .add_local_python_source("config", "volumes", "images")
    )


def _emu3_5_image():
    """EMU3.5 — Python 3.12, torch 2.8.0, vLLM 0.11.0, flash-attn 2.8.3

    Pin transformers to 4.56.1 (vLLM 0.11.0 requires 4.x).
    Apply BAAI's vLLM patches for native Emu3.5 support (custom attention
    kernels, scheduler, logits processor for image-token generation).
    """
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
        )
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(_SYS_PACKAGES)
        .pip_install([
            "torch==2.8.0", "torchvision==0.23.0", "torchaudio",
            "vllm==0.11.0",
            "transformers==4.56.1",
            "accelerate>=0.20.0",
            "protobuf>=3.20.0", "tiktoken>=0.12.0",
            "imageio==2.37.0", "imageio-ffmpeg==0.6.0",
            "omegaconf==2.3.0",
            "numpy", "Pillow", "tqdm",
            "datasets", "huggingface_hub",
            "setuptools", "wheel", "ninja",
            "openpyxl",
        ])
        .run_commands(
            "wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
            "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
            " -P /tmp"
            " && pip install '/tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'"
            " && rm /tmp/flash_attn-*.whl"
        )
        # Apply BAAI's vLLM patches for native Emu3.5 architecture support
        .add_local_dir("model/Emu3.5/third_party/vllm", "/tmp/emu3_5_patches", copy=True)
        .add_local_file("model/Emu3.5/src/patch/apply.py", "/tmp/emu3_5_apply.py", copy=True)
        .run_commands(
            "python /tmp/emu3_5_apply.py --patch-dir /tmp/emu3_5_patches"
        )
        .add_local_python_source("config", "volumes", "images")
    )


# =====================================================================
# Lookup table: model name -> factory function
# Call get_image(name) to create the Image object on demand.
# =====================================================================
_IMAGE_FACTORIES = {
    "bagel":     _bagel_image,
    "blip3o":    _blip3o_image,
    "deepgen":   _deepgen_image,
    "emu3":      _emu3_image,
    "emu3_5":    _emu3_5_image,
    "geneval":   _geneval_image,
    "janus_pro": _janus_pro_image,
    "janus_flow": _janus_flow_image,
    "mmada":     _mmada_image,
    "omnigen2":  _omnigen2_image,
    "ovis_u1":   _ovis_u1_image,
    "show_o":    _show_o_image,
    "show_o2":   _show_o2_image,
    "tokenflow": _tokenflow_image,
    "ueval":     _ueval_image,
    "uni_mmmu":  _uni_mmmu_image,
    "unified_bench": _unified_bench_image,
    "wise":      _wise_image,
}


def get_image(model_name: str) -> modal.Image:
    """Create and return the Image for the given model. Only this image is built."""
    if model_name not in _IMAGE_FACTORIES:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(_IMAGE_FACTORIES)}"
        )
    return _IMAGE_FACTORIES[model_name]()
