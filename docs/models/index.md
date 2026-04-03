# Models Overview

TorchUMM integrates thirteen multimodal models through a unified backbone adapter interface. Each model is wrapped as a `BackboneAdapter` that exposes a common API for generation, understanding, and (where supported) editing.

---

## Capability Matrix

| Model | Parameters | Understand | Generate | Edit | Guide |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [Bagel](https://github.com/jpthu17/Bagel) | 7B | Yes | Yes | Yes | [bagel.md](bagel.md) |
| [DeepGen](https://github.com/deepgenteam/DeepGen) | 5B | No | Yes | Yes | [deepgen.md](deepgen.md) |
| [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) | 7B | Yes | Yes | Yes | [omnigen2.md](omnigen2.md) |
| [Emu3](https://github.com/baaivision/Emu3) | 8B | Yes | Yes | No | [emu3.md](emu3.md) |
| [Emu3.5](https://github.com/baaivision/Emu3.5) | 34B | Yes | Yes | Yes | [emu3_5.md](emu3_5.md) |
| [MMaDA](https://github.com/Gen-Verse/MMaDA) | 8B | Yes | Yes | No | [mmada.md](mmada.md) |
| [Janus](https://github.com/deepseek-ai/Janus) | 1.3B | Yes | Yes | No | [janus.md](janus.md) |
| [Janus-Pro](https://github.com/deepseek-ai/Janus) | 1B, 7B | Yes | Yes | No | [janus_pro.md](janus_pro.md) |
| [JanusFlow](https://github.com/deepseek-ai/Janus) | 1.3B | Yes | Yes | No | [janus_flow.md](janus_flow.md) |
| [Show-o](https://github.com/showlab/Show-o) | 1.3B | Yes | Yes | No | [show_o.md](show_o.md) |
| [Show-o2](https://github.com/showlab/Show-o) | 1.5B, 7B | Yes | Yes | No | [show_o2.md](show_o2.md) |
| [BLIP3-o](https://github.com/salesforce/BLIP3o) | 4B | No | Yes | No | [blip3o.md](blip3o.md) |
| [TokenFlow](https://github.com/ByteFlow-AI/TokenFlow) | 7B | No | Yes | No | [tokenflow.md](tokenflow.md) |

---

## Model Summaries

### Bagel

Mixture-of-Transformer (MoT) model supporting all three capabilities --- understanding, generation, and editing. Uses a shared Qwen2-based language backbone with a SigLIP vision encoder and a diffusion head for image generation. Backbone key: `bagel`. See [Bagel guide](bagel.md).

### DeepGen

A 5B multimodal model (3B VLM + 2B DiT) supporting text-to-image generation and image editing via a diffusers-compatible pipeline. Does not support understanding — the internal VLM is used only for semantic guidance during generation, not as a standalone VQA interface. Backbone key: `deepgen`. See [DeepGen guide](deepgen.md).

### OmniGen2

A unified multimodal model that handles image understanding, text-to-image generation, and image editing within a single architecture. Backbone key: `omnigen2`. See [OmniGen2 guide](omnigen2.md).

### Emu3

Predict-next-token multimodal model with separate generation and understanding model checkpoints. Backbone key: `emu3`. See [Emu3 guide](emu3.md).

### Emu3.5

Next-generation native multimodal model from BAAI with unified world modeling. Features 4-5x faster inference via vLLM, discrete diffusion adaptation (DiDA), and RL post-training. Supports T2I, X2I, and interleaved generation. Note: required a minor patch to the model repo's `modeling_emu3.py` (commenting out dead FX tracing code incompatible with transformers >= 4.55). Backbone key: `emu3_5`. See [Emu3.5 guide](emu3_5.md).

### MMaDA

Masked Diffusion Adaptation (MMaDA) is an 8B multimodal model from Gen-Verse that unifies text generation, image generation, and image understanding through a masked diffusion framework. Unlike autoregressive models, MMaDA uses discrete masked diffusion for all modalities --- both text and image tokens are generated via iterative demasking. Uses MagVITv2 as the visual tokenizer with a codebook size of 8192. Available in Base and MixCoT variants. Backbone key: `mmada`. See [MMaDA guide](mmada.md).

### Janus

Original decoupled visual encoding model from DeepSeek (1.3B), with separate vision encoders for understanding and generation. Uses VQ autoregressive token prediction for image generation. Backbone key: `janus_pro`. See [Janus guide](janus.md).

### Janus-Pro

Scaled-up version of Janus (7B) with improved training and stronger multimodal capabilities. Shares the same architecture as Janus but with significantly better performance. Backbone key: `janus_pro`. See [Janus-Pro guide](janus_pro.md).

### JanusFlow

Rectified flow variant of the Janus architecture from DeepSeek. Uses continuous ODE-based generation with an external SDXL VAE for image decoding, instead of the autoregressive VQ token approach used by Janus/Janus-Pro. Backbone key: `janus_flow`. See [JanusFlow guide](janus_flow.md).

### Show-o

Original unified transformer from Show Lab combining autoregressive text modeling with discrete diffusion for image generation. Uses Phi-1.5 as LLM base and MagVITv2 as visual tokenizer. Backbone key: `show_o`. See [Show-o guide](show_o.md).

### Show-o2

Next-generation unified model from Show Lab, replacing discrete diffusion with flow matching and upgrading to Qwen2.5 LLM with Wan2.1 3D Causal VAE. Backbone key: `show_o2`. See [Show-o2 guide](show_o2.md).

### BLIP3-o

Generation-focused model from Salesforce built on the BLIP3 architecture. Generation only --- no understanding support. Backbone key: `blip3o`. See [BLIP3-o guide](blip3o.md).

### TokenFlow

Generation-focused model from ByteFlow AI using a token-flow framework for text-to-image synthesis. Backbone key: `tokenflow`. See [TokenFlow guide](tokenflow.md).

---

## Adding a New Model

The backbone adapter system is designed to be extended. Any new multimodal model can be integrated by implementing a `BackboneAdapter` subclass and registering it — no changes to the inference pipeline or CLI are needed.

See the [Extending guide](../extending.md) for a step-by-step walkthrough.
