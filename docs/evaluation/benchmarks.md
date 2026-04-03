# Supported Benchmarks

TorchUMM supports 10+ benchmarks spanning image generation, visual understanding, and image editing.

---

## Benchmark Reference

| Benchmark | Evaluates | Required Capabilities | Data Source | Data Prep |
| :--- | :--- | :--- | :--- | :--- |
| [DPG Bench](https://github.com/TencentQQGYLab/ELLA) | Text-to-image detail preservation | Generation | Included in repo | [Details](../data-preparation.md) |
| [GenEval](https://github.com/djghosh13/geneval) | Compositional text-to-image generation | Generation | Included in repo | [Details](../data-preparation.md) |
| [WISE](https://github.com/PKU-YuanGroup/WISE) | World knowledge in image generation | Generation | Included in repo | [Details](../data-preparation.md) |
| [UEval](https://github.com/mdl-ueval/UEval) | Unified understanding + generation | Understanding + Generation | [HuggingFace](https://huggingface.co/datasets/primerL/UEval-all) | [Details](../data-preparation.md) |
| [Uni-MMMU](https://github.com/AI-in-Edtech/Uni-MMMU) | Multimodal understanding, generation, and editing | Understand + Generate + Edit | [HuggingFace](https://huggingface.co/datasets) | [Details](../data-preparation.md) |
| [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) | Multimodal perception and cognition | Understanding | [HuggingFace](https://huggingface.co/OpenGVLab/InternVL) | [Details](../data-preparation.md) |
| [MMMU](https://mmmu-benchmark.github.io/) | Massive multimodal understanding | Understanding | HuggingFace (auto) | [Details](../data-preparation.md) |
| [MMBench](https://opencompass.org.cn/leaderboard-multimodal) | VLM systematic evaluation | Understanding | [OpenMMLab](https://download.openmmlab.com/mmclassification/datasets/mmbench/) | [Details](../data-preparation.md) |
| [MM-Vet](https://github.com/yuweihao/MM-Vet) | Integrated vision-language capabilities | Understanding | [GitHub](https://github.com/yuweihao/MM-Vet) | [Details](../data-preparation.md) |
| [MathVista](https://mathvista.github.io/) | Mathematical reasoning with visuals | Understanding | [HuggingFace](https://huggingface.co/datasets/AI4Math/MathVista) | [Details](../data-preparation.md) |
| [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit) | Image editing quality (VIEScore) | Editing | [HuggingFace](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) | [Details](../data-preparation.md) |

---

## Evaluation Types

### Single-Stage Benchmarks

These benchmarks run generation and scoring in a single command:

- **DPG Bench** --- generates images and computes detail-preservation scores
- **MME** --- runs perception and cognition evaluation
- **MMMU** --- runs multimodal understanding evaluation
- **MMBench** --- runs systematic VLM evaluation
- **MM-Vet** --- runs integrated vision-language evaluation
- **MathVista** --- runs mathematical reasoning evaluation

```bash
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/dpg_bench/dpg_bench_bagel.yaml
```

### Two-Stage Benchmarks

These benchmarks separate generation from scoring, which allows using different models (or environments) for each stage:

- **GenEval** --- generate images, then score with an object detector
- **WISE** --- generate images, then score with Qwen VL models
- **UEval** --- generate text + image answers, then score with Qwen models
- **Uni-MMMU** --- generate outputs, then score
- **GEdit-Bench** --- edit images, then score with VIEScore (Qwen VL)

```bash
# Step 1: Generate
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/geneval/geneval_bagel_generate.yaml

# Step 2: Score
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/geneval/geneval_bagel_score.yaml
```

See [Reproducing Results](reproducing.md) for full two-stage examples.
