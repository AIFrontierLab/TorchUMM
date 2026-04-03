---
hide:
  - navigation
  - toc
---
# TorchUMM

<p align="center">
  <img src="assets/logo.png" width="300" alt="TorchUMM">
  <br>
  <span style="font-size: 14pt; color: gray;"><strong>Unified Multimodal Model Toolkit</strong></span>
</p>

**A unified toolkit for multimodal model inference, evaluation, and post-training.**

---

TorchUMM provides a single, config-driven interface for running, evaluating, and fine-tuning state-of-the-art multimodal models. It is designed to make fair, reproducible comparisons across diverse multimodal architectures straightforward --- whether you are benchmarking generation quality, measuring visual understanding, or experimenting with post-training methods.

<div class="grid cards" markdown>

- :material-puzzle: **Pluggable Architecture** --- 13 multimodal model adapters with unified interface
- :material-chart-bar: **10+ Benchmarks** --- Generation, understanding, and editing evaluation
- :material-tune: **Post-Training** --- SFT, IRG, recA, UniCot (LoRA-based)
- :material-cloud: **Cloud-Native** --- Scale to cloud GPUs via Modal
- :material-file-cog: **Config-Driven** --- YAML configs, no code changes needed

</div>

---

## Supported Models

| Model                                               | Parameters | Understand | Generate | Edit |           Docs           |
| :-------------------------------------------------- | :--------: | :--------: | :------: | :--: | :-----------------------: |
| [Bagel](https://github.com/jpthu17/Bagel)              |    7B     |    Yes    |   Yes   | Yes |   [guide](models/bagel.md)   |
| [DeepGen](https://github.com/deepgenteam/DeepGen)      |    5B     |     No    |   Yes   | Yes | [guide](models/deepgen.md)  |
| [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) |    7B     |    Yes    |   Yes   | Yes | [guide](models/omnigen2.md) |
| [Emu3](https://github.com/baaivision/Emu3)             |    8B     |    Yes    |   Yes   |  No  |   [guide](models/emu3.md)   |
| [Emu3.5](https://github.com/baaivision/Emu3.5)         |    34B    |    Yes    |   Yes   |  Yes  |  [guide](models/emu3_5.md)  |
| [MMaDA](https://github.com/Gen-Verse/MMaDA)            |    8B     |    Yes    |   Yes   |  No  |  [guide](models/mmada.md)    |
| [Janus](https://github.com/deepseek-ai/Janus)          |   1.3B    |    Yes    |   Yes   |  No  |  [guide](models/janus.md)    |
| [Janus-Pro](https://github.com/deepseek-ai/Janus)      |  1B, 7B   |    Yes    |   Yes   |  No  | [guide](models/janus_pro.md) |
| [JanusFlow](https://github.com/deepseek-ai/Janus)      |   1.3B    |    Yes    |   Yes   |  No  | [guide](models/janus_flow.md) |
| [Show-o](https://github.com/showlab/Show-o)            |   1.3B    |    Yes    |   Yes   |  No  |  [guide](models/show_o.md)   |
| [Show-o2](https://github.com/showlab/Show-o)           | 1.5B, 7B  |    Yes    |   Yes   |  No  |  [guide](models/show_o2.md)  |
| [BLIP3-o](https://github.com/salesforce/BLIP3o)        |    4B     |     No     |   Yes   |  No  |  [guide](models/blip3o.md)  |
| [TokenFlow](https://github.com/ByteFlow-AI/TokenFlow)  |    7B     |     No     |   Yes   |  No  | [guide](models/tokenflow.md) |

---

## Quick Start

**Install**

```bash
git clone --recursive https://github.com/ApiaoSamaa/umm_codebase.git
cd umm_codebase
pip install -e .
```

**CLI Inference**

```bash
PYTHONPATH=src python -m umm.cli.main infer \
    --config configs/inference/modal_bagel_generation.yaml
```

**Python API**

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(
    backbone_name="bagel",
    backbone_cfg={"model_path": "/path/to/BAGEL-7B-MoT"},
)
result = pipeline.run(InferenceRequest(
    backbone="bagel", task="generation",
    prompt="A cat sitting on a rainbow",
))
```

---

[Getting Started](getting-started.md){ .md-button .md-button--primary } [Models](models/index.md){ .md-button } [Evaluation Results](evaluation/results.md){ .md-button }
