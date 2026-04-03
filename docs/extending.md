# Extending TorchUMM

TorchUMM is designed for extensibility. This page provides step-by-step guides for adding new models, benchmarks, and post-training methods.

---

## Adding a New Model

### Step 1: Implement the Backbone Adapter

Create a new directory `src/umm/backbones/<model_name>/` with an adapter class. Your adapter must implement four methods:

```python
# src/umm/backbones/my_model/adapter.py

class MyModelBackbone:
    """Backbone adapter for MyModel."""

    def load(self, cfg: dict):
        """Load model weights and initialize.

        Args:
            cfg: Configuration dict with model_path, device settings, etc.
        """
        self.model = load_my_model(cfg["model_path"])

    def generation(self, batch, params):
        """Text-to-image generation.

        Args:
            batch: List of InferenceRequest objects.
            params: Generation parameters (steps, guidance scale, etc.).
        Returns:
            List of generated images.
        """
        ...

    def understanding(self, batch, params):
        """Image understanding / VQA.

        Args:
            batch: List of InferenceRequest objects with images.
            params: Understanding parameters (max_tokens, etc.).
        Returns:
            List of text responses.
        """
        ...

    def editing(self, batch, params):
        """Image editing (optional).

        Args:
            batch: List of InferenceRequest objects with images and edit prompts.
            params: Editing parameters.
        Returns:
            List of edited images.
        """
        raise NotImplementedError("MyModel does not support editing")
```

Reference implementation: `src/umm/backbones/bagel/adapter.py`

### Step 2: Register the Backbone

Add a lazy-loading entry in `src/umm/inference/pipeline.py` inside `register_builtin_backbones()`:

```python
if "my_model" not in registry.list_registered("backbone"):
    from umm.backbones.my_model import MyModelBackbone
    registry.register("backbone", "my_model", MyModelBackbone)
```

### Step 3: Create Inference Configs

Add YAML files in `configs/inference/`:

```yaml
# configs/inference/my_model_generation.yaml
inference:
  backbone: my_model
  backbone_cfg:
    model_path: /path/to/weights
    seed: 42
  request:
    task: generation
    prompt: "A test prompt"
```

### Step 4: Create Evaluation Configs

Add per-benchmark configs in `configs/eval/<benchmark>/`:

```yaml
# configs/eval/dpg_bench/dpg_bench_my_model.yaml
eval:
  benchmark: dpg_bench

inference:
  backbone: my_model
  backbone_cfg:
    model_path: /path/to/weights

dpg_bench:
  out_dir: output/dpg_bench/my_model
```

### Step 5: Add Modal Support (Optional)

1. Define a container image in `modal/images.py` specifying Python version, PyTorch version, and dependencies.
2. Add the repo directory mapping in `modal/run.py`.

### Step 6: Write Documentation

Create `docs/models/my_model.md` with usage instructions, supported benchmarks, and config examples. Follow the format of existing model pages (e.g., `docs/models/bagel.md`).

---

## Adding a New Benchmark

### Step 1: Create Evaluation Scripts

Add a new directory under `eval/` with the evaluation logic:

```
eval/generation/my_benchmark/
    __init__.py
    evaluate.py
    README.md
```

### Step 2: Create Per-Model Configs

Add YAML configs in `configs/eval/my_benchmark/`:

```yaml
# configs/eval/my_benchmark/my_benchmark_bagel.yaml
eval:
  benchmark: my_benchmark

inference:
  backbone: bagel
  backbone_cfg:
    model_path: /path/to/BAGEL-7B-MoT

my_benchmark:
  data_root: /path/to/data
  out_dir: output/my_benchmark/bagel
```

### Step 3: Register in the Eval Router

Add a routing entry in `src/umm/cli/eval.py`:

```python
if benchmark == "my_benchmark" or "my_benchmark" in raw_cfg:
    from umm.cli.my_benchmark import run_eval_command as _fn
    return _fn(args)
```

### Step 4: Write Data Preparation Docs

Create `eval/<category>/my_benchmark/README.md` with download and setup instructions.

Reference: `eval/generation/geneval/`

---

## Adding a New Post-Training Method

### Step 1: Implement Training Logic

Create a new directory with your training pipeline:

```
src/umm/post_training/my_method/
    __init__.py
    train.py
```

### Step 2: Create a Config

Add a config in `configs/posttrain/`:

```yaml
# configs/posttrain/my_method.yaml
train:
  pipeline: bagel
  cwd: src/umm/post_training/my_method/
  entrypoint: torchrun
  script: train.py
  args:
    learning_rate: 1e-5
    num_epochs: 3
    batch_size: 4
```

### Step 3: Run Training

```bash
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/my_method.yaml
```

Reference: `src/umm/post_training/sft/`
