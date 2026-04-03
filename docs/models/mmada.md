# MMaDA

Masked Diffusion Adaptation (MMaDA) is an 8B multimodal foundation model from Gen-Verse that unifies text generation, image generation, and image understanding through a masked diffusion framework.

- **Original repository:** <https://github.com/Gen-Verse/MMaDA>
- **Paper:** [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)
- **Backbone key:** `mmada`
- **Capabilities:** Understanding, Generation (NO Editing)

## Architecture

Unlike autoregressive models, MMaDA uses discrete masked diffusion for all modalities --- both text and image tokens are generated via iterative demasking. Key components:

- **LLM base:** LLaDA (8B), a masked diffusion language model built on Llama3 architecture
- **Visual tokenizer:** MagVITv2 (discrete, codebook size 8192, produces 1024 tokens per 512x512 image)
- **Image generation:** MaskGIT-style parallel decoding with classifier-free guidance
- **Image understanding:** VQ-encoded image tokens concatenated with text, demasked to produce text response
- **Max resolution:** 512x512

## Model Variants

| Variant | HuggingFace ID | Description |
|---------|----------------|-------------|
| MMaDA-8B-Base | `Gen-Verse/MMaDA-8B-Base` | Base model |
| MMaDA-8B-MixCoT | `Gen-Verse/MMaDA-8B-MixCoT` | Chain-of-thought reasoning variant |

Both variants share the same architecture and loading procedure. Switch between them via `model_path` in config.

## Dependencies

The model environment is managed via the `mmada` image defined in `modal/images.py` (Python 3.10, PyTorch 2.5.1, CUDA 12.4, transformers 4.46.0). For local setup, install the dependencies listed in `model/MMaDA/requirements.txt`.

### Flash Attention (Recommended)

MMaDA benefits from Flash Attention for faster inference. The Modal image includes flash-attn 2.7.4 (pre-compiled wheel). For local installation:

```bash
# Check your environment first
python -c "import torch; print(torch.__version__)"  # e.g., 2.5.1
nvcc -V                                               # e.g., CUDA 12.x

# Download matching wheel from https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Inference

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="mmada", backbone_cfg={
    "model_path": "/path/to/MMaDA-8B-Base",
    "mmada_root": "/path/to/model/MMaDA",
    "vq_model_path": "/path/to/magvitv2",
    "seed": 42,
})

# Text-to-image generation
result = pipeline.run(InferenceRequest(
    backbone="mmada", task="generation",
    prompt="A cat sitting on a rainbow",
    params={"guidance_scale": 1.5, "timesteps": 12},
))

# Image understanding
result = pipeline.run(InferenceRequest(
    backbone="mmada", task="understanding",
    prompt="Describe this image in detail.",
    images=["path/to/image.jpg"],
    params={"max_new_tokens": 512},
))
```

### Modal (Cloud)

```bash
# Download model weights
modal run modal/download.py --model mmada

# Run generation evaluation
modal run modal/run.py --model mmada --eval-config modal_geneval_mmada

# Run understanding evaluation
modal run modal/run.py --model mmada --eval-config modal_ueval_mmada
```

## Supported Benchmarks

### Generation

| Benchmark | Config |
|-----------|--------|
| GenEval   | `configs/eval/geneval/modal_geneval_mmada.yaml` |
| WISE      | `configs/eval/wise/modal_wise_mmada.yaml` |
| DPG Bench | `configs/eval/dpg_bench/modal_dpg_bench_mmada.yaml` |

### Understanding

| Benchmark | Config |
|-----------|--------|
| UEval     | `configs/eval/ueval/modal_ueval_mmada.yaml` |
| MMBench   | `configs/eval/mmbench/modal_mmbench_mmada.yaml` |
| MME       | `configs/eval/mme/modal_mme_mmada.yaml` |
| MMMU      | `configs/eval/mmmu/modal_mmmu_mmada.yaml` |
| MM-Vet    | `configs/eval/mmvet/modal_mmvet_mmada.yaml` |
| MathVista | `configs/eval/mathvista/modal_mathvista_mmada.yaml` |

## Configuration Reference

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance_scale` | 1.5 | Classifier-free guidance scale (0 = no guidance) |
| `temperature` | 1.0 | Sampling temperature for Gumbel noise |
| `timesteps` | 12 | Number of MaskGIT decoding steps |
| `num_vq_tokens` | 1024 | Number of VQ tokens per image |
| `codebook_size` | 8192 | VQ codebook size |
| `mask_schedule` | cosine | Mask schedule (cosine, linear, sigmoid) |

### Understanding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `steps` | 256 | Demasking steps (typically max_new_tokens / 2) |
| `block_length` | 128 | Block length for semi-autoregressive generation |
| `temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `remasking` | low_confidence | Remasking strategy (low_confidence, random) |
