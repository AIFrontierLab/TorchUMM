<p align="center">
  <img src="assets/logo.png" width="400" alt="TorchUMM Logo">
</p>

<h3 align="center">TorchUMM: Unified Multimodal Model Toolkit</h3>

<p align="center">
  A unified framework for unified multimodal model inference, evaluation, and post-training.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>

</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.10784">📄 Paper</a> |
  <a href="https://huggingface.co/aifronter/post_train">🤗 Post-Training Weights</a>
</p>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Supported Models](#supported-models)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [Understanding Benchmarks Data](#understanding-benchmarks-data)
  - [Generation Benchmarks Data](#generation-benchmarks-data)
  - [Other Benchmarks Data](#other-benchmarks-data)
- [Usage](#usage)
  - [Local Execution (CLI)](#local-execution-cli)
  - [AMD HPC Execution](#amd-hpc-execution)
  - [Python API](#python-api)
- [Evaluation Results](#evaluation-results)
  - [Generation Benchmarks](#generation-benchmarks)
  - [Understanding Benchmarks](#understanding-benchmarks)
  - [Editing Benchmarks](#editing-benchmarks)
    - [GEdit-Bench](#gedit-bench)
    - [ImgEdit-Bench](#imgedit-bench)
  - [Uni-MMMU Benchmark](#uni-mmmu-benchmark)
  - [Post-Training Models](#post-training-models)
  - [Detailed Sub-scores](#detailed-sub-scores)
  - [Reproducing Results](#reproducing-results)
- [Extending TorchUMM](#extending-torchumm)
  - [Adding a New Model](#adding-a-new-model)
  - [Adding a New Benchmark](#adding-a-new-benchmark)
  - [Adding a New Post-Training Method](#adding-a-new-post-training-method)
- [Post-Training Methods](#post-training-methods)
- [Disclaimers](#disclaimers)
- [Citation](#citation)

---

## Introduction

**TorchUMM** is a unified toolkit for running, evaluating, and fine-tuning state-of-the-art multimodal models under a single interface. It is designed to make fair, reproducible comparisons across diverse multimodal architectures easy.

**Key features:**

- **Pluggable backbone architecture** — 14 multimodal model adapters with a unified inference interface
- **Comprehensive evaluation** — 10+ benchmarks covering generation, understanding, and editing
- **Post-training support** — SFT, IRG, recA, UniCot, Unigame
- **Cloud-native** — seamless scaling to cloud GPUs via [Modal](https://modal.com) ([details](modal/README.md))
- **Config-driven** — all behavior controlled through YAML configs; no code changes needed to switch models or benchmarks

---

## Supported Models

| Model                                               | Parameters | Understand | Generate | Edit |              Docs              |
| :-------------------------------------------------- | :--------: | :--------: | :------: | :--: | :----------------------------: |
| [Bagel](https://github.com/jpthu17/Bagel)              |    7B     |     ✅     |    ✅    |  ✅  |   [guide](docs/models/bagel.md)   |
| [DeepGen](https://github.com/deepgenteam/DeepGen)      |    5B     |     ❌     |    ✅    |  ✅  | [guide](docs/models/deepgen.md)  |
| [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) |    7B     |     ✅     |    ✅    |  ✅  | [guide](docs/models/omnigen2.md) |
| [Emu3](https://github.com/baaivision/Emu3)             |    8B     |     ✅     |    ✅    |  ❌  |   [guide](docs/models/emu3.md)   |
| [Emu3.5](https://github.com/baaivision/Emu3.5)         |    34B    |     ✅     |    ✅    |  ✅  |  [guide](docs/models/emu3_5.md)  |
| [MMaDA](https://github.com/Gen-Verse/MMaDA)                    |    8B     |     ✅     |    ✅    |  ❌  |  [guide](docs/models/mmada.md)    |
| [Janus](https://github.com/deepseek-ai/Janus)              |   1.3B    |     ✅     |    ✅    |  ❌  |  [guide](docs/models/janus.md)     |
| [Janus-Pro](https://github.com/deepseek-ai/Janus)          |  1B, 7B   |     ✅     |    ✅    |  ❌  | [guide](docs/models/janus_pro.md)  |
| [JanusFlow](https://github.com/deepseek-ai/Janus)          |   1.3B    |     ✅     |    ✅    |  ❌  | [guide](docs/models/janus_flow.md) |
| [Show-o](https://github.com/showlab/Show-o)             |   1.3B    |     ✅     |    ✅    |  ❌  |  [guide](docs/models/show_o.md)   |
| [Show-o2](https://github.com/showlab/Show-o)           | 1.5B, 7B  |     ✅     |    ✅    |  ❌  |  [guide](docs/models/show_o2.md)  |
| [BLIP3-o](https://github.com/salesforce/BLIP3o)        |    4B     |     ❌     |    ✅    |  ❌  |  [guide](docs/models/blip3o.md)  |
| [TokenFlow](https://github.com/ByteFlow-AI/TokenFlow)  |          |     ❌     |    ✅    |  ❌  | [guide](docs/models/tokenflow.md) |
| [Ovis-U1](https://github.com/AIDC-AI/Ovis-U1)         |    3B     |     ✅     |    ✅    |  ✅  | [guide](docs/models/ovis_u1.md)  |

> See each model's [guide](docs/models/) for detailed usage instructions, configuration examples, and supported benchmarks.
>
> **Emu3.5 note:** Emu3.5 is the only model in TorchUMM that uses **native vLLM integration** via BAAI's official patches (20 patches applied at image build time). Unlike other models that use the standard `TransformersForCausalLM` wrapper, Emu3.5 runs on vLLM's optimized attention kernels with a custom batch scheduler for classifier-free guidance, achieving ~74 tokens/s on 2×A100-80GB. See the [Emu3.5 guide](docs/models/emu3_5.md) for details.
>
> **Flash Attention note:** Most models require or benefit from [Flash Attention](https://github.com/Dao-AILab/flash-attention). **Do not** `pip install flash-attn` from source (extremely slow, error-prone). Instead, download a pre-compiled wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) matching your Python/CUDA/PyTorch/ABI. All Modal images already include the correct wheel. See each model's guide for the exact wheel command:
>
> | Model | flash-attn | Status | Guide |
> |---|---|---|---|
> | Bagel | 2.5.8 | Required | [guide](docs/models/bagel.md#flash-attention-required) |
> | BLIP3-o | 2.6.2 | Required | [guide](docs/models/blip3o.md#flash-attention-required) |
> | Emu3 | 2.5.7 | Required | [guide](docs/models/emu3.md#flash-attention-required) |
> | Emu3.5 | 2.8.3 | Required | [guide](docs/models/emu3_5.md#flash-attention-required) |
> | Janus-Pro | 2.7.4 | Required | [guide](docs/models/janus_pro.md#flash-attention-required) |
> | MMaDA | 2.7.4 | Recommended | [guide](docs/models/mmada.md#flash-attention-recommended) |
> | Show-o2 | 2.7.4 | Required | [guide](docs/models/show_o2.md#flash-attention-required) |
> | OmniGen2 | 2.7.4 | Recommended | [guide](docs/models/omnigen2.md#flash-attention-recommended) |
> | DeepGen | latest | Recommended | [guide](docs/models/deepgen.md#flash-attention-recommended) |

---

## Repository Structure
<p align="center">
  <img src="assets/torchumm_frame.png" width="800" alt="TorchUMM Framework">
</p>

```
umm_codebase/
├── src/umm/                    # Core framework
│   ├── backbones/              # Model adapters (Bagel, BLIP3-o, DeepGen, Emu3, Emu3.5, Janus, Janus-Pro, JanusFlow, MMaDA, OmniGen2, Show-o, Show-o2, TokenFlow)
│   ├── cli/                    # CLI entry points (infer, eval, train)
│   ├── core/                   # Config, registry, interfaces
│   ├── data/                   # Datasets, collators, transforms
│   ├── evaluation/             # Evaluation runners and metrics
│   ├── inference/              # Inference pipeline (batching, generation)
│   ├── models/                 # Model builders, heads, processors
│   ├── post_training/          # Post-training methods (SFT, IRG, recA, UniCot)
│   └── serving/                # Serving APIs
│
├── model/                      # External model repos & evaluation toolkits (submodules)
│   ├── Bagel/, BLIP3o/, deepgen/, Emu3/, Emu3.5/, MMaDA/, OmniGen2/, Show-o/, TokenFlow/
│   └── UEval/, Uni-MMMU/, WISE/, geneval/, Step1X-Edit/
│
├── configs/                    # YAML configurations
│   ├── inference/              # Per-model inference configs
│   ├── eval/                   # Benchmark evaluation configs (modal_*, amd_*, and local)
│   └── posttrain/              # Post-training configs
│
├── modal/                      # Modal cloud infrastructure (see modal/README.md)
├── docs/                       # Per-model usage documentation
├── eval/                       # Evaluation runner scripts
├── scripts/                    # Utility scripts
└── output/                     # Evaluation results
```

---

## Installation

```bash
# Clone the repository
git clone --recursive https://github.com/AIFrontierLab/TorchUMM.git
cd TorchUMM

# Install the package
pip install -e .

# Install model-specific dependencies (example: Bagel)
pip install -r model/Bagel/requirements.txt
```

> **Note:** Each backbone model has its own dependencies and may require different Python/PyTorch versions. Install only the requirements for the model(s) you plan to use. For cloud execution via [Modal](https://modal.com), each model runs in an isolated container image with the correct environment — see [modal/README.md](modal/README.md) for details.

---

### Understanding Benchmarks Data

Understanding benchmarks data is prepared following the [InternVL evaluation data preparation](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) guide. All data is stored under `data/` at the repository root. Below is a quick-start summary — see [eval/vlm/README.md](eval/vlm/README.md) for full details.

**MME**

```bash
mkdir -p data/mme
cd data/mme
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip
cd -
```

**MMBench**

```bash
mkdir -p data/mmbench
cd data/mmbench
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv
cd -
```

**MM-Vet**

```bash
mkdir -p data/mm-vet
cd data/mm-vet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
cd -
```

**MathVista**

```bash
mkdir -p data/MathVista
cd data/MathVista
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
cd -
```

**MMMU** — auto-downloaded from HuggingFace (`MMMU/MMMU`) at evaluation time, cached in `data/MMMU/`. No manual download needed.

### Generation Benchmarks Data

These benchmarks include their data in the repository:

- **DPG Bench**: Prompts in `eval/generation/dpg_bench/prompts/` (100 prompt files)
- **GenEval**: Metadata and prompts in `model/geneval/`
- **WISE**: Benchmark data in `model/WISE/`

### Other Benchmarks Data

- **UEval**: Auto-downloaded from HuggingFace (`primerL/UEval-all`) at evaluation time. For Modal, run `modal run modal/download.py --dataset ueval`.
- **Uni-MMMU**: Requires dataset, scoring models (Qwen2.5-VL-72B-Instruct + Qwen3-32B), and DreamSim (auto-downloaded). For Modal: `modal run modal/download.py --dataset uni_mmmu` and `modal run modal/download.py --model evaluator`. See [eval/generation/uni_mmmu/README.md](eval/generation/uni_mmmu/README.md) for full setup.
- **GEdit-Bench**: Auto-downloaded from HuggingFace (`stepfun-ai/GEdit-Bench`) at evaluation time. For Modal, run `modal run modal/download.py --dataset gedit`. Scoring uses Qwen2.5-VL-72B-Instruct (same as WISE).

---

## Usage

### Local Execution (CLI)

**Inference**

```bash
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/modal_bagel_generation.yaml
```

**Evaluation**

```bash
# DPG Bench on Bagel
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/dpg_bench/dpg_bench_bagel.yaml

# GenEval on Bagel (full pipeline: generation + scoring)
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_bagel.yaml

# UEval on Bagel
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/ueval/ueval_bagel.yaml

# MME on Bagel
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_bagel.yaml
```

**Post-Training**

```bash
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/bagel_sft.yaml
```

> For cloud GPU execution via [Modal](https://modal.com), see [modal/README.md](modal/README.md).

### AMD HPC Execution

For AMD ROCm clusters, use `amd_` prefixed configs which contain AMD HPC absolute paths:

```bash
# Using local_run.sh (recommended)
bash scripts/amd_migration/local_run.sh bagel --eval-config amd_ueval_bagel

# Or directly with CLI
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/ueval/amd_ueval_bagel.yaml
```

Config naming convention:
- `modal_*.yaml` — Modal cloud (container mount paths like `/model_cache/...`)
- `amd_*.yaml` — AMD HPC (absolute paths like `/work1/jwang/yinyil/model_cache/...`)
- `*.yaml` (no prefix) — Legacy local configs (may have outdated paths)

> To regenerate AMD configs after modifying modal configs: `python scripts/generate_amd_configs.py`

**Upload Outputs to HuggingFace**

Evaluation outputs live on Modal's `umm-outputs` Volume. To upload them to HuggingFace (directly from Modal, no local download):

```bash
# Upload everything (resumable — re-run if interrupted)
modal run modal/upload_outputs.py --repo-id wenwenw945/umm_outputs

# Upload a specific subdirectory only
modal run modal/upload_outputs.py --repo-id wenwenw945/umm_outputs --subdir geneval

# Force overwrite: clear remote first, then upload
modal run modal/upload_outputs.py --clear --repo-id wenwenw945/umm_outputs
modal run modal/upload_outputs.py --repo-id wenwenw945/umm_outputs

# Dry run — list what would be uploaded
modal run modal/upload_outputs.py --repo-id wenwenw945/umm_outputs --dry-run
```

> Requires a `huggingface-secret` Modal secret with your `HF_TOKEN`.

### Python API

You can also use TorchUMM programmatically:

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

# Initialize the pipeline with a backbone model
pipeline = InferencePipeline(
    backbone_name="bagel",
    backbone_cfg={
        "model_path": "/path/to/BAGEL-7B-MoT",
        "max_mem_per_gpu": "80GiB",
        "seed": 42,
    },
)

# Text-to-image generation
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="generation",
    prompt="A cat sitting on a rainbow",
    params={"num_timesteps": 50},
))

# Image understanding
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="understanding",
    prompt="Describe this image in detail.",
    images=["path/to/image.jpg"],
    params={"max_think_token_n": 500, "do_sample": False},
))

# Image editing
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="editing",
    prompt="Make the sky purple",
    images=["path/to/image.jpg"],
    params={"num_timesteps": 25},
))

# Batch inference
results = pipeline.run_many(
    [request1, request2, request3],
    batch_size=2,
)
```

The `InferenceRequest` dataclass accepts:

| Field           | Type          | Description                                             |
| :-------------- | :------------ | :------------------------------------------------------ |
| `backbone`    | `str`       | Backbone model name (must match pipeline)               |
| `task`        | `str`       | `"generation"`, `"understanding"`, or `"editing"` |
| `prompt`      | `str`       | Text prompt                                             |
| `images`      | `list[str]` | Input image paths (for understanding/editing)           |
| `videos`      | `list[str]` | Input video paths                                       |
| `params`      | `dict`      | Task-specific parameters                                |
| `output_path` | `str`       | Path to save output                                     |

---

## Evaluation Results

> All results below are **independently reproduced** using TorchUMM. See [Disclaimers](#disclaimers).

### Generation Benchmarks

| Model     | DPG Bench | GenEval |  WISE  |
| :-------- | :-------: | :-----: | :----: |
| Bagel(14B)     |   84.11   |  78.81  | 0.3989 |
| DeepGen(5B)   |   87.44   |  86.59  | 0.5470 |
| Janus-Pro(7B) |   83.73   |  78.92  | 0.3811 |
| Janus(1.3B) |     73.526    |  40.04  | 0.2222 |
| Janus-Flow(1.3B)|   72.03    |  49.99  | 0.2964 |
| Show-o2(7B)   |   82.81   |  59.87  | 0.3595 |
| Show-o2(1.5B)   |   82.78   |  55.49  | 0.3349 |
| Show-o(1.3B)   |   78.74   |  65.06  | 0.3037 |
| Emu3(8B)      |   80.31   |    45.76    | 0.3373 |
| Emu3.5(34B)   |     72.51    |   81.83   | 0.6331 |
| OmniGen2(7B)  |   84.51   |  78.53  | 0.4029 |
| BLIP3-o(3B)   |   61.47   |  81.36  | 0.4138 |
| TokenFlow |   71.29   |  52.21  | 0.3056 |
| MMaDA     |    64.55    |   46.12   | 0.6560 |

> **DeepGen evaluation parameters** follow the [official DeepGen repository](https://github.com/deepgenteam/DeepGen) (`EVAL.md`): all benchmarks use 512×512 resolution, 50 inference steps, guidance scale 4.0 (7.5 for DPG-Bench), seed 42.
>
> **WISE evaluator note:** All WISE scores in this table are evaluated using **Qwen2.5-VL-72B-Instruct** as the VLM judge, rather than GPT-4o used in the [original WISE benchmark](https://github.com/PKU-YuanGroup/WISE) and most published papers. This leads to systematically lower absolute scores compared to paper-reported numbers (e.g., DeepGen paper reports 0.72 with GPT-4o vs. our 0.5470 with Qwen2.5-VL-72B). The gap is primarily due to: **(1)** different scoring VLMs have different evaluation biases — Qwen2.5-VL-72B tends to score more strictly than GPT-4o, especially on the Consistency dimension (weight 0.7 in WiScore); **(2)** we use the diffusers-format pipeline rather than DeepGen's native pipeline, which may introduce minor generation quality differences. Since all models are evaluated with the same evaluator, **relative rankings remain valid for fair comparison**.

### Understanding Benchmarks

| Model            | MME (Perception) | MME (Cognition) | MMMU  | MMBench | MM-Vet | MathVista |
| :--------------- | :--------------: | :-------------: | :---: | :-----: | :----: | :-------: |
| Bagel (14B)      |   **1691.5**     |   **695.4**     | **0.519** | **0.843** | **65.9** | **71.6** |
| Janus-Pro (7B)   |      1547.9      |      293.2      | 0.407 |  0.699  |  33.7  |   42.8   |
| JanusFlow (1.3B) |      1305.6      |      251.1      | 0.290 | 0.6486  |  31.8  |   34.8   |
| Janus (1.3B)     |      1221.4      |      264.3      | 0.273 | 0.4691  |  27.0  |   26.6   |
| Show-o2 (7B)     |      1619.8      |      387.5      | 0.479 |  0.430  |  47.1  |   51.5   |
| Show-o2 (1.5B)   |      1413.3      |      291.8      | 0.368 | 0.6813  |  46.1  |   37.9   |
| Show-o (1.3B)    |      1188.5      |      244.6      | 0.261 |  0.469  |  23.3  |   29.0   |
| Emu3 (8B)        |      1176.0      |      213.2      | 0.314 |    —    |  30.0  |   44.9   |
| Emu3.5 (34B)     |       781.1      |      324.6      | 0.292 |  0.183  |  28.0  |   41.7   |
| OmniGen2 (7B)    |      1584.4      |      614.6      | 0.460 |  0.782  |  62.7  |   38.9   |
| MMaDA (8B)       |       939.0      |      241.4      | 0.289 |  0.330  |  11.4  |   24.9   |

> **MathVista evaluator note:** All MathVista scores use **Qwen3-32B** for answer extraction from model responses, with rule-based normalization for scoring. Answer extraction is performed locally (no OpenAI API required). † OmniGen2 and Show-o produce empty responses on MathVista benchmark.
>
> **UEval notes:** Emu3 uses separate models for understanding and generation, making it incompatible with UEval's unified evaluation protocol.
>
> **Emu3.5 MMBench note ‡:** Emu3.5's MMBench score (18.3%) is far below its naive accuracy (43.7%) due to **severe option position bias** under MMBench's CircularEval protocol. CircularEval shuffles option order across variants and requires the model to answer correctly on *all* variants — Emu3.5 picks the same letter regardless of content 23.5% of the time (vs. Emu3's 7.1%), indicating it selects by position rather than understanding. This is an inherent limitation of the unified model architecture, not a code bug.
>
> **Emu3.5 MME note:** Emu3.5 uses `temperature=1.0` sampling for understanding, making scores hardware-dependent.

### Editing Benchmarks

#### GEdit-Bench

| Model    | EN SC | EN PQ | EN O  | CN SC | CN PQ | CN O  |
| :------- | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepGen  |  7.44 | **7.54** |  7.33 |  7.41 | **7.59** |  7.36 |
| Bagel    |  6.68 |  7.04 |  6.35 |  6.83 |  7.06 |  6.52 |
| OmniGen2 |  6.49 |  7.18 |  6.27 |  6.25 |  7.18 |  6.03 |
| Emu3.5   | **7.64** |  7.48 | **7.56** | **7.62** |  7.50 | **7.56** |

> "Intersection" = samples where both EN and CN instructions exist for the same source image.

#### ImgEdit-Bench

#### ImgEdit-Bench (Overall)

| Model    | ST   | MT   | UGE  |
| :------- | :--: | :--: | :--: |
| DeepGen  | 4.07 | 4.37 | 4.81 |
| Bagel    | 3.71 | 4.45 | 4.18 |
| OmniGen2 | 3.88 | 3.27 | 4.06 |
| Emu3.5   | **4.24** | **4.89** | **4.88** |

> ImgEdit-Bench evaluates image editing across three suites: Singleturn (9 edit types, 736 samples), UGE (unguided editing, 50 samples), and Multiturn (multi-round editing, 88 samples). All scores use Qwen2.5-VL-72B-Instruct as evaluator (scale 1–5).

### Uni-MMMU Benchmark

| Model | Jig. I | Jig. T | Maze I | Maze T | Slid. I | Slid. T | Geo I | Geo T | Sci. R | Sci. T | Sci. I | Code T | Code S | Code P |
| :---- | :----: | :----: | :----: | :----: | :-----: | :-----: | :---: | :---: | :----: | :----: | :----: | :----: | :----: | :----: |
| Bagel | 0.660 | 0.553 | 0.004 | 0.101 | 0.000 | 0.050 | 0.050 | 0.143 | 0.592 | 0.522 | 0.185 | 0.115 | 0.375 | 0.275 |
| Janus-Pro | — | — | — | — | — | — | — | — | 29.3 | 25.5 | 0.0 | 1.5 | 3.7 | 3.4 |

> **Note:** DeepGen, BLIP3-o, and TokenFlow are excluded from Uni-MMMU as they do not support image understanding. Janus-Pro cannot perform editing tasks.

### Post-Training Benchmarks

#### Generation

| Model                  | DPG  | GenEval | WISE  | UEval |
| :--------------------- | :--: | :-----: | :---: | :---: |
| Bagel (base)           | 84.11 | 78.81 | 0.399 | 30.9 |
| Bagel + RecA           | **85.20** | 83.05 | **0.423** | 31.0 |
| Bagel + UniCot         | 83.52 | 77.91 | 0.404 | 31.8 |
| Bagel + SFT            | 83.02 | 78.03 | 0.227 | **31.4** |
| Bagel + IRG            | 81.82 | 72.06 | 0.384 | 9.1 |
| Bagel + UniGame        | 65.77 | **85.80** | 0.403 | 31.0 |
| Janus-Pro + UniGame    | 83.92 | 78.65 | 0.373 | 20.65 |
| Janus-Pro + SFT        | 83.93 | 77.61 | 0.370 | 20.61 |
| OmniGen2 + SFT         | 84.78 | 77.84 | 0.405 | 25.91 |
| BLIP3-o + SFT          | 61.01 | 78.41 | 0.399 | — |
| TokenFlow + SFT        | 22.16 | 51.96 | 0.328 | — |
| Show-o2 (7B) + SFT     | 80.58 | 52.13 | 0.322 | 25.7 |

#### Understanding

| Model               | MME (P) | MME (C) | MMMU  | MMBench | MM-Vet | MathVista |
| :------------------ | :-----: | :-----: | :---: | :-----: | :----: | :-------: |
| Bagel (base)        | 1691.5 | 695.4 | 0.519 | 0.843 | 65.9 | 71.6 |
| Bagel + RecA        | 1689.1 | 695.4 | 0.523 | 0.842 | **66.1** | 51.6 |
| Bagel + UniCot      | 1690.7 | 678.2 | **0.531** | **0.845** | 64.5 | 73.0 |
| Bagel + SFT         | 1680.7 | 678.9 | 0.526 | 0.820 | 61.2 | 73.1 |
| Bagel + IRG         | 1647.5 | 650.4 | 0.480 | 0.778 | 40.7 | 68.0 |
| Bagel + UniGame     | **1692.1** | 695.4 | 0.524 | 0.843 | 60.7 | 72.2 |
| Janus-Pro + UniGame | 1554.0 | 288.9 | 0.409 | 0.698 | 32.4 | 43.9 |
| Janus-Pro + SFT     | 1549.9 | 292.9 | 0.400 | 0.700 | 33.0 | 35.4 |
| OmniGen2 + SFT      | 1573.6 | 610.0 | 0.469 | 0.782 | 62.2 | 63.5 |

#### Editing

| Model               | GEdit-EN (I/F) | GEdit-CN (I/F) | ImgEdit (S) | ImgEdit (M) | ImgEdit (U) |
| :------------------ | :------------: | :------------: | :---------: | :---------: | :---------: |
| Bagel (base)        | 6.38 / 6.35 | 6.68 / 6.52 | 3.71 | 4.45 | 4.18 |
| Bagel + RecA        | 6.89 / 6.80 | 6.87 / 6.75 | 3.89 | 4.28 | 4.15 |
| Bagel + UniCot      | **7.04 / 6.92** | 6.90 / 6.81 | 3.77 | 4.22 | 4.34 |
| Bagel + SFT         | 6.62 / 6.49 | 6.71 / 6.54 | 3.73 | **4.48** | 4.12 |
| Bagel + IRG         | 6.52 / 6.44 | 6.51 / 6.41 | 3.79 | 3.89 | **4.54** |
| Bagel + UniGame     | 6.48 / 6.48 | 6.55 / 6.38 | 3.72 | 4.46 | 4.31 |
| OmniGen2 + SFT      | 6.37 / 6.31 | 6.14 / 6.06 | 3.88 | 3.26 | 4.06 |


### Reproducing Results

Benchmarks with two-stage evaluation (GenEval, WISE, UEval, Uni-MMMU) provide separate `_generate` and `_score` configs. You can also use the base config (mode: `full`) to run both stages in one command.

**GenEval on Bagel**

```bash
# Step 1: Generate images
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_bagel_generate.yaml

# Step 2: Score generated images
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_bagel_score.yaml
```

**WISE on Bagel**

```bash
# Step 1: Generate images
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/wise/wise_bagel_generate.yaml

# Step 2: Score with Qwen models
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/wise/wise_bagel_score.yaml
```

**UEval on Bagel**

```bash
# Step 1: Generate text + image answers
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/ueval/ueval_bagel_generate.yaml

# Step 2: Score with Qwen models
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/ueval/ueval_bagel_score.yaml
```

**Single-stage benchmarks** (DPG Bench, MME, MMMU, MMBench, MM-Vet) run generation and scoring in one step:

```bash
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_bagel.yaml
```

**MathVista** is a two-stage benchmark: generation runs in the model environment, and scoring (Qwen3-32B answer extraction) runs in the `wise` environment which has `transformers>=4.51`:

```bash
# Step 1: Generate (model env)
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mathvista/mathvista_bagel.yaml
# Step 2: Score (wise env — requires transformers>=4.51 for Qwen3)
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mathvista/mathvista_bagel_score.yaml
```

---

## Extending TorchUMM

TorchUMM is designed for extensibility. Below are guides for adding new models, benchmarks, and post-training methods.

### Adding a New Model

1. **Implement the backbone adapter.** Create a new directory `src/umm/backbones/<model_name>/` with an adapter class. Your adapter must implement:

   - `load(cfg: dict)` — load model weights and initialize
   - `generation(batch, params)` — text-to-image generation
   - `understanding(batch, params)` — image understanding / VQA
   - `editing(batch, params)` — image editing (optional)

   Reference implementation: [`src/umm/backbones/bagel/adapter.py`](src/umm/backbones/bagel/adapter.py)

   **Adapter design guidelines:**

   - **Do not catch pipeline exceptions in `editing()`.** The evaluation pipeline (`generate_image_from_context`) relies on exceptions to fall back from editing to text-to-image generation. If your adapter catches and wraps errors into a return dict, the fallback is silently skipped. Only the final `generation()` method should catch exceptions.
   - **Share model components across pipelines.** If your model uses separate pipeline objects for different tasks (e.g., one for generation and one for understanding), construct them from shared component references to avoid duplicating large model weights in GPU memory.
   - **Use a task-appropriate system prompt for understanding.** If your model's default prompt biases toward image generation (common for unified models), override it with a text-focused prompt when handling understanding tasks. See the [OmniGen2 adapter](src/umm/backbones/omnigen2/adapter.py) for an example.

2. **Register the backbone.** Add a lazy-loading entry in [`src/umm/inference/pipeline.py`](src/umm/inference/pipeline.py) → `register_builtin_backbones()`:

   ```python
   if "my_model" not in registry.list_registered("backbone"):
       from umm.backbones.my_model import MyModelBackbone
       registry.register("backbone", "my_model", MyModelBackbone)
   ```
3. **Create inference configs.** Add YAML files in `configs/inference/`:

   ```yaml
   inference:
     backbone: my_model
     backbone_cfg:
       model_path: /path/to/weights
       seed: 42
     request:
       task: generation
       prompt: "A test prompt"
   ```
4. **Create evaluation configs.** Add per-benchmark configs in `configs/eval/<benchmark>/my_model.yaml`.
5. **(Optional) Add Modal support.** Define a container image in `modal/images.py` and add the repo directory mapping in `modal/run.py`. See [modal/README.md](modal/README.md#adding-a-new-model).
6. **Write documentation.** Create `docs/models/my_model.md` with usage instructions, supported benchmarks, and config examples.

### Adding a New Benchmark

1. **Create evaluation scripts.** Add a new directory under `eval/` (e.g., `eval/generation/my_benchmark/`) with the evaluation logic.
2. **Create per-model configs.** Add YAML configs in `configs/eval/my_benchmark/`:

   ```yaml
   eval:
     benchmark: my_benchmark

   inference:
     backbone: bagel
     backbone_cfg: { ... }

   my_benchmark:
     data_root: /path/to/data
     out_dir: output/my_benchmark/bagel
   ```
3. **Register in the eval router.** Add a routing entry in [`src/umm/cli/eval.py`](src/umm/cli/eval.py):

   ```python
   if benchmark == "my_benchmark" or "my_benchmark" in raw_cfg:
       from umm.cli.my_benchmark import run_eval_command as _fn
       return _fn(args)
   ```
4. **Write a data preparation README.** Create `eval/<category>/my_benchmark/README.md` with download and setup instructions.

   Reference: [`eval/generation/geneval/`](eval/generation/geneval/)

### Adding a New Post-Training Method

1. **Implement training logic.** Create `src/umm/post_training/<method>/` with your training pipeline.
2. **Create a config.** Add `configs/posttrain/<method>.yaml`:

   ```yaml
   train:
     pipeline: bagel
     cwd: src/umm/post_training/<method>/
     entrypoint: torchrun
     script: train.py
     args:
       learning_rate: 1e-5
   ```
3. **Run training:**

   ```bash
   PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/<method>.yaml
   ```

   Reference: [`src/umm/post_training/sft/`](src/umm/post_training/sft/)

---

## Post-Training Methods

TorchUMM supports multiple post-training strategies (currently targeting Bagel):

| Method           | Description                                | Config                                                      |
| :--------------- | :----------------------------------------- | :---------------------------------------------------------- |
| **SFT**    | Supervised fine-tuning                     | `configs/posttrain/bagel_sft.yaml`                        |
| **IRG**    | Interleaved Reasoning Generation (2-stage) | `configs/posttrain/irg_stage1.yaml` / `irg_stage2.yaml` |
| **recA**   | Reconstruction Alignment                   | `configs/posttrain/recA.yaml`                             |
| **UniCot** | Unified Chain-of-Thought training (LoRA)   | `configs/posttrain/unicot.yaml`                           |
| **UniGame** | Self-adversarial consistency training   | `configs/posttrain/unigame.yaml`                           |
```bash
# Example: SFT on Bagel (local)
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/bagel_sft.yaml
```

> For cloud-based post-training, see [modal/README.md](modal/README.md).

---

## Disclaimers

> **Important:** Please read before using or citing evaluation results.

1. **Unofficial results.** All evaluation results in this repository are **independently reproduced** by the TorchUMM team. They do **NOT** represent official results from the original model authors. Differences from published numbers may arise due to variations in inference settings, hardware, random seeds, or evaluation protocols.
2. **Active development.** TorchUMM is under active development. We are continuously adding support for new models, benchmarks, and post-training methods. Some results may be updated as we refine our evaluation pipelines.
3. **Contributions welcome.** We welcome bug reports, corrections, and contributions from the community. If you find discrepancies in our results or want to add support for a new model/benchmark, please open an issue or pull request.
4. **Community usage.** You are welcome to use TorchUMM for your own research and evaluation. If you do, we appreciate a citation (see [Citation](#citation)).

---

## Citation

If you find TorchUMM useful in your research, please consider citing:


```bibtex
@misc{luo2026torchummunifiedmultimodalmodel,
      title={TorchUMM: A Unified Multimodal Model Codebase for Evaluation, Analysis, and Post-training}, 
      author={Yinyi Luo and Wenwen Wang and Hayes Bai and Hongyu Zhu and Hao Chen and Pan He and Marios Savvides and Sharon Li and Jindong Wang},
      year={2026},
      eprint={2604.10784},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.10784}, 
}
```

