# Emu3.5

Next-generation native multimodal model from BAAI with unified world modeling and 4-5x faster inference via vLLM.

- **Original repository:** <https://github.com/baaivision/Emu3.5>
- **Backbone key:** `emu3_5`
- **Capabilities:** Understanding, Generation (NO Editing)

## Dependencies

The model environment is managed via the `emu3_5` image defined in `modal/images.py`:
- Python 3.12, PyTorch 2.8.0, vLLM 0.11.0, flash-attn 2.8.3, transformers 4.56.1
- Default GPU: A100-80GB:2 (tensor_parallel_size=2)

For local setup, install dependencies from `model/Emu3.5/requirements/vllm.txt`, then apply BAAI's vLLM patches:
```bash
cd model/Emu3.5
python src/patch/apply.py --patch-dir third_party/vllm
```

### Flash Attention (required)

Emu3.5 requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.8.3, used by vLLM). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Architecture Note

Emu3.5 uses three model variants:

- **Emu3.5** — general multimodal model (T2I, X2I, interleaved generation, understanding)
- **Emu3.5-Image** — optimized for T2I/X2I (recommended for generation benchmarks, cfg=5.0)
- **Emu3.5-VisionTokenizer** — IBQ-based vision tokenizer (shared across variants)

Model weights paths: `emu3_5/Emu3.5`, `emu3_5/Emu3.5-Image`, `emu3_5/Emu3.5-VisionTokenizer`.

Key features:
- **Native vLLM backend** with optimized attention kernels (PagedAttention, FlashAttention, CUDA graphs)
- **Custom cond/uncond batch scheduler** for classifier-free guidance (4-5x speedup)
- **Classifier-free guidance** with differential sampling
- **Protobuf output format** for multimodal sequences

## Adapter Architecture — How Emu3.5 Differs from Other Models

Emu3.5's adapter is **architecturally different** from all other models in TorchUMM:

| Aspect | Other Models | Emu3.5 |
|--------|-------------|--------|
| vLLM integration | `TransformersForCausalLM` wrapper (PyTorch forward pass) | Native `Emu3_5ForCausalLM` via BAAI patches (vLLM optimized kernels) |
| Architecture registration | auto_map in config.json or transformers built-in | 20 vLLM source patches applied at Docker image build time |
| Attention implementation | Transformers' eager/sdpa/flash_attention_2 | vLLM's `Attention` with `QKVParallelLinear`, `RowParallelLinear` |
| Batch scheduling | vLLM default scheduler | Custom `batch_scheduler.Scheduler` for CFG cond/uncond batching |
| Logits processing | Standard sampling | Custom logits processor for image token generation |

### Why Patches Instead of `TransformersForCausalLM`?

The HuggingFace checkpoint declares `model_type: "Emu3"`, which collides with transformers' built-in Emu3 (a different multimodal architecture). The `TransformersForCausalLM` fallback would:
1. Fail to load due to the model_type collision (built-in Emu3 expects `text_config`/`vision_config`)
2. Even if loaded, use PyTorch native forward pass with no vLLM kernel optimizations

BAAI's patches solve both problems by registering a dedicated `Emu3_5Config` and `Emu3_5ForCausalLM` directly in vLLM's model registry, bypassing transformers' auto-discovery entirely.

### Patch Details

The 20 patch files in `model/Emu3.5/third_party/vllm/` modify vLLM 0.11.0 to add:

- **Model architecture** (`model_executor/models/emu3_5.py`) — native vLLM implementation with `QKVParallelLinear`, `MergedColumnParallelLinear`, `RMSNorm`, `Attention`
- **Config registration** (`transformers_utils/config.py`, `configs/emu3_5.py`) — maps `model_type="Emu3"` to `Emu3_5Config`
- **Model registry** (`model_executor/models/registry.py`) — registers `Emu3_5ForCausalLM`
- **Batch scheduling** (`v1/core/sched/`) — custom scheduler for concurrent conditional/unconditional generation
- **Logits processing** (`v1/sample/logits_processor/`) — image token sampling logic
- **Engine/worker** (`v1/engine/`, `v1/worker/`) — adapted for Emu3.5's generation pattern

These patches are applied during Docker image build in `modal/images.py` via `model/Emu3.5/src/patch/apply.py`.

### Model Repo Modification (exception to `model/` no-edit rule)

The repo's `modeling_emu3.py` imports `is_torch_fx_available` from transformers, which was removed in transformers >= 4.55 (required by vLLM 0.11.0). Since this is dead code during inference, we commented out:
- **Line 55**: `from transformers.utils.import_utils import is_torch_fx_available`
- **Lines 64-70**: The `if is_torch_fx_available(): ...` block

This is the **only** modification to the `model/` directory, with zero impact on model behavior.

## Inference

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="emu3_5", backbone_cfg={
    "model_path": "/model_cache/emu3_5/Emu3.5-Image",
    "vq_path": "/model_cache/emu3_5/Emu3.5-VisionTokenizer",
    "emu3_5_root": "/workspace/model/Emu3.5",
    "use_vllm": True,
    "tensor_parallel_size": 2,
    "classifier_free_guidance": 5.0,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="emu3_5", task="generation",
    prompt="A cat sitting on a rainbow",
    params={"max_new_tokens": 5120},
))
```

## Supported Benchmarks

| Benchmark | Config (Modal) |
|-----------|----------------|
| GenEval   | `configs/eval/geneval/modal_geneval_emu3_5.yaml` |
| GenEval (score) | `configs/eval/geneval/modal_geneval_emu3_5_score.yaml` |
| WISE      | `configs/eval/wise/modal_wise_emu3_5.yaml` |
| WISE (score) | `configs/eval/wise/modal_wise_emu3_5_score.yaml` |
| DPG Bench | `configs/eval/dpg_bench/modal_dpg_bench_emu3_5.yaml` |
| Uni-MMMU (generate) | `configs/eval/uni_mmmu/uni_mmmu_emu3_5_generate.yaml` |
| Uni-MMMU (score) | `configs/eval/uni_mmmu/uni_mmmu_emu3_5_score.yaml` |

```bash
# Example: run GenEval on Modal
modal run modal/run.py --model emu3_5 --eval-config geneval/modal_geneval_emu3_5 --gpu A100-80GB:2

# Example: run WISE on Modal
modal run modal/run.py --model emu3_5 --eval-config modal_wise_emu3_5 --gpu A100-80GB:2
```

## Performance

On 2×A100-80GB:
- **Generation throughput:** ~74 tokens/s
- **Time per image:** ~110 seconds (5120 tokens × 2 for CFG)
- **Model loading:** ~42 seconds
- **First-run compilation:** ~250 seconds (cached on subsequent runs)

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vllm` | `true` | Use vLLM backend (required; native patches must be applied) |
| `tensor_parallel_size` | `2` | Number of GPUs for vLLM tensor parallelism |
| `gpu_memory_utilization` | `0.7` | vLLM GPU memory fraction |
| `classifier_free_guidance` | `5.0` | CFG scale (5.0 for Emu3.5-Image T2I, 2.0 for Emu3.5 base) |
| `max_new_tokens` | `5120` | Maximum tokens for T2I generation |
| `image_area` | `1048576` | Target image area in pixels (1024x1024) |
