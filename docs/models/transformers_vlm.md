# Transformers VLMs

TorchUMM includes ten image-to-text VLM integrations through a shared lazy
Transformers adapter. Install the optional dependencies before using them:

```bash
pip install -e '.[vlm]'
```

Weights are downloaded only when the selected backbone receives its first
understanding request. Set `local_files_only: true` to require a local cache,
or set `device_map: auto` when loading a model across available devices.

## Supported models

| Backbone | Hugging Face checkpoint | Notes |
|---|---|---|
| `qwen2_5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | General image understanding |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Requires `trust_remote_code` |
| `llava_onevision_7b` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | Multi-image capable |
| `idefics3_8b` | `HuggingFaceM4/Idefics3-8B-Llama3` | Image-text conversation |
| `smolvlm_2b` | `HuggingFaceTB/SmolVLM-Instruct` | Compact Apache-2.0 model |
| `phi3_5_vision` | `microsoft/Phi-3.5-vision-instruct` | Requires `trust_remote_code` |
| `gemma3_4b` | `google/gemma-3-4b-it` | Requires accepting Gemma terms on Hugging Face |
| `molmo_7b` | `allenai/Molmo-7B-D-0924` | Requires `trust_remote_code` |
| `minicpm_v_2_6` | `openbmb/MiniCPM-V-2_6` | Requires `trust_remote_code` |
| `paligemma2_3b` | `google/paligemma2-3b-mix-224` | Uses PaliGemma task prompts; requires Google terms |

All ten currently expose image understanding through the unified `understanding`
interface. Video inputs are intentionally rejected by this adapter even for
checkpoints whose upstream model cards describe video support.

## Example

```yaml
inference:
  backbone: qwen2_5_vl_7b
  backbone_cfg:
    model_path: Qwen/Qwen2.5-VL-7B-Instruct
    device: cuda:0
    torch_dtype: bfloat16
  request:
    task: understanding
    prompt: "What is happening in this image?"
    images:
      - assets/torchumm_frame.png
    params:
      max_new_tokens: 256
```

Run it with:

```bash
umm infer --config configs/inference/qwen2_5_vl_7b_understanding.yaml
```

## Model-specific notes

### Qwen2.5-VL

Use `qwen2_5_vl_7b`. The default chat format is handled by the Transformers
image-to-text pipeline.

### InternVL3

Use `internvl3_8b`. Its upstream implementation uses custom Transformers code,
so `trust_remote_code` is enabled by default.

### LLaVA-OneVision

Use `llava_onevision_7b`. The adapter accepts one or more image paths in a
single request.

### Idefics3

Use `idefics3_8b` for image-text conversations and multi-image prompts.

### SmolVLM

Use `smolvlm_2b` for a smaller local inference footprint.

### Phi-3.5 Vision

Use `phi3_5_vision`. The model loads its custom Transformers implementation with
`trust_remote_code` enabled by default.

### Gemma 3

Use `gemma3_4b`. Hugging Face access requires accepting Google's Gemma license.

### Molmo

Use `molmo_7b`. Its custom model code is enabled with `trust_remote_code`.

### MiniCPM-V

Use `minicpm_v_2_6`. Its custom model code is enabled with `trust_remote_code`.

### PaliGemma 2

Use `paligemma2_3b`. The adapter automatically prefixes a plain question with
`answer en`; explicit PaliGemma task prefixes such as `ocr`, `describe en`, or
`answer en ...` are passed through unchanged. Hugging Face access requires
accepting Google's terms.
