"""
Unified runner for all model repos on Modal.

Each repo runs inside its own Image (environment) while sharing the same
model-cache and dataset-cache Volumes. This means:
  - Environments are permanently cached (Image layers)
  - Model weights are downloaded once (Volume)
  - Datasets are downloaded once (Volume)
  - No re-downloading or re-installing on subsequent runs

Usage:
    # Inference
    modal run modal/run.py --model bagel --script inferencer.py

    # Training (post-training via umm CLI)
    modal run modal/run.py --model bagel --config bagel_sft

    # Evaluation (eval via umm CLI)
    modal run modal/run.py --model bagel --eval-config modal_ueval_bagel

    # Custom command inside any model's environment
    modal run modal/run.py --model emu3 --cmd "python image_generation.py --prompt 'a cat'"

    # Run with specific GPU (default: A100-80GB)
    modal run modal/run.py --model blip3o --script inference.py --gpu A100
    modal run modal/run.py --model omnigen2 --gpu H100

    # Multi-GPU: append :N to GPU name
    modal run modal/run.py --model wise --eval-config modal_wise_bagel_score --gpu A100-80GB:2
"""

from __future__ import annotations

import sys

import modal

from config import (
    CHECKPOINT_PATH,
    DATASET_CACHE_PATH,
    MODEL_CACHE_PATH,
    OUTPUT_APR4_PATH,
    OUTPUT_PATH,
    POST_TRAIN_MODEL_CACHE_PATH,
    WORKSPACE_PATH,
)
from volumes import codebase_volume, checkpoint_volume, dataset_volume, model_volume, output_apr4_volume, output_volume, post_train_model_volume

# ---------------------------------------------------------------------------
# Parse --model from CLI args BEFORE registering any @app.function.
# This way only the requested model's image is imported and built.
# ---------------------------------------------------------------------------

_VALID_MODELS = [
    "bagel", "blip3o", "deepgen", "emu3", "emu3_5", "geneval", "janus_flow", "janus_pro",
    "mmada", "omnigen2", "ovis_u1",
    "show_o", "show_o2", "tokenflow", "ueval", "uni_mmmu", "unified_bench", "wise",
]

_IMAGE_NAMES = {
    "bagel":     "bagel_image",
    "blip3o":    "blip3o_image",
    "deepgen":   "deepgen_image",
    "emu3":      "emu3_image",
    "emu3_5":    "emu3_5_image",
    "geneval":   "geneval_image",
    "janus_flow": "janus_flow_image",
    "janus_pro": "janus_pro_image",
    "mmada":     "mmada_image",
    "omnigen2":  "omnigen2_image",
    "ovis_u1":   "ovis_u1_image",
    "show_o":    "show_o_image",
    "show_o2":   "show_o2_image",
    "tokenflow": "tokenflow_image",
    "ueval":     "ueval_image",
    "uni_mmmu":  "uni_mmmu_image",
    "unified_bench": "unified_bench_image",
    "wise":      "wise_image",
}

# Repo directory names inside the workspace
_REPO_DIRS = {
    "bagel":     "Bagel",
    "blip3o":    "BLIP3o",
    "deepgen":   "deepgen",
    "emu3":      "Emu3",
    "emu3_5":    "Emu3.5",
    "geneval":   "geneval",
    "janus_flow": "Janus",
    "janus_pro": "Janus",
    "mmada":     "MMaDA",
    "omnigen2":  "OmniGen2",
    "ovis_u1":   ".",
    "show_o":    "Show-o",
    "show_o2":   "Show-o",
    "tokenflow": "TokenFlow",
    "ueval":     "UEval",
    "uni_mmmu":  "Uni-MMMU",
    "unified_bench": "UAE",
    "wise":      "WISE",
}


def _parse_cli_arg(name: str, default: str) -> str:
    """Extract --name value from sys.argv before Modal processes args."""
    # Check both hyphen and underscore forms (Modal may normalize either way)
    variants = {f"--{name}", f"--{name.replace('-', '_')}", f"--{name.replace('_', '-')}"}
    for i, arg in enumerate(sys.argv):
        if arg in variants and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_requested_model = _parse_cli_arg("model", "bagel")
_requested_gpu = _parse_cli_arg("gpu", "A100-80GB")
if _requested_model not in _IMAGE_NAMES:
    raise ValueError(
        f"Unknown model '{_requested_model}'. Available: {', '.join(_VALID_MODELS)}"
    )

# Only create the one image we need (lazy — no other images are built)
from images import get_image
_selected_image = get_image(_requested_model)

# ---------------------------------------------------------------------------
# Detect score_model and score_gpu from eval config (for two-phase eval).
# If the YAML has `wise.score_model` (or similar), we register a second
# Modal function with that model's image for the scoring phase.
# score_gpu allows the scoring phase to use a different GPU spec.
# ---------------------------------------------------------------------------

def _detect_score_config() -> tuple:
    """Read eval config locally and return (score_model, score_gpu) if specified."""
    import os
    eval_config_name = _parse_cli_arg("eval-config", "")
    if not eval_config_name:
        return "", ""
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "eval")
    config_file = os.path.join(config_dir, f"{eval_config_name}.yaml")
    if not os.path.exists(config_file):
        return "", ""
    try:
        import yaml
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
        # Check benchmark-specific sections for score_model
        for section in ("wise", "geneval", "ueval", "uni_mmmu", "gedit", "imgedit", "mathvista", "mmbench", "unified_bench"):
            sec = data.get(section, {})
            if isinstance(sec, dict) and sec.get("score_model"):
                sm = str(sec["score_model"])
                sg = str(sec.get("score_gpu", "")).strip()
                if sm in _IMAGE_NAMES and sm != _requested_model:
                    print(f"[modal/run] detected score_model='{sm}' from eval config")
                    if sg:
                        print(f"[modal/run] detected score_gpu='{sg}' from eval config")
                    return sm, sg
    except Exception as exc:
        print(f"[modal/run] WARNING: score config detection failed: {exc}")
    return "", ""


_requested_score_model, _requested_score_gpu_raw = _detect_score_config()
_score_image = get_image(_requested_score_model) if _requested_score_model else None
_score_gpu = _requested_score_gpu_raw if _requested_score_gpu_raw else _requested_gpu


app = modal.App("umm-run")


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

def _run_impl(
    model_name: str,
    script: str = "",
    cmd: str = "",
    config: str = "",
    eval_config: str = "",
    mode_override: str = "",
) -> None:
    import os
    import subprocess
    import sys

    model_dir = f"{WORKSPACE_PATH}/model"
    cwd = f"{model_dir}/{_REPO_DIRS.get(model_name, model_name)}"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MODEL_CACHE"] = MODEL_CACHE_PATH
    env["DATASET_CACHE"] = DATASET_CACHE_PATH
    env["OUTPUT_DIR"] = OUTPUT_PATH

    if eval_config:
        # Evaluation mode: use umm CLI with an eval config
        config_path = f"{WORKSPACE_PATH}/configs/eval/{eval_config}.yaml"
        if not os.path.exists(config_path):
            avail = [f for f in os.listdir(f"{WORKSPACE_PATH}/configs/eval/") if f.endswith(".yaml")]
            raise FileNotFoundError(f"Eval config not found: {eval_config}. Available: {avail}")

        # If mode_override is set, patch the config to override the mode
        if mode_override:
            import tempfile
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            for section in ("wise", "geneval", "ueval", "uni_mmmu", "gedit", "imgedit", "mathvista", "mmbench", "unified_bench"):
                if section in cfg and isinstance(cfg[section], dict):
                    cfg[section]["mode"] = mode_override
                    break
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, prefix="modal_mode_"
            ) as tf:
                yaml.dump(cfg, tf)
                config_path = tf.name
            print(f"[modal/run] mode override: {mode_override}")

        env["PYTHONPATH"] = f"{WORKSPACE_PATH}/src"
        run_cmd = [sys.executable, "-m", "umm.cli.main", "eval", "--config", config_path]
        print(f"[modal/run] eval config={eval_config}", flush=True)
        subprocess.run(run_cmd, cwd=WORKSPACE_PATH, env=env, check=True)

    elif config:
        # Post-training mode: use umm CLI with a patched config
        import tempfile
        import yaml

        config_path = f"{WORKSPACE_PATH}/configs/posttrain/{config}.yaml"
        if not os.path.exists(config_path):
            avail = os.listdir(f"{WORKSPACE_PATH}/configs/posttrain/")
            raise FileNotFoundError(f"Config not found: {config}. Available: {avail}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        train_args = cfg.get("train", {}).get("args", {})
        train_args["results_dir"] = f"{CHECKPOINT_PATH}/{config}/results"
        if "checkpoint_dir" in train_args:
            train_args["checkpoint_dir"] = f"{CHECKPOINT_PATH}/{config}/checkpoints"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix=f"modal_{config}_"
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = tmp.name

        env["PYTHONPATH"] = f"{WORKSPACE_PATH}/src"
        run_cmd = [sys.executable, "-m", "umm.cli.main", "train", "--config", tmp_path]
        print(f"[modal/run] training config={config}")
        subprocess.run(run_cmd, cwd=WORKSPACE_PATH, env=env, check=True)

    elif script:
        # Run a specific script inside the model's repo dir
        run_cmd = [sys.executable, script]
        print(f"[modal/run] {model_name}: python {script}")
        subprocess.run(run_cmd, cwd=cwd, env=env, check=True)

    elif cmd:
        # Run an arbitrary shell command
        print(f"[modal/run] {model_name}: {cmd}")
        subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=True)

    else:
        # Interactive: just print the environment info
        print(f"[modal/run] {model_name} environment ready")
        print(f"  cwd:         {cwd}")
        print(f"  model_cache: {MODEL_CACHE_PATH}")
        print(f"  datasets:    {DATASET_CACHE_PATH}")
        print(f"  Python:      {sys.version}")
        subprocess.run(["pip", "list", "--format=columns"], cwd=cwd, env=env)


# ---------------------------------------------------------------------------
# Shared volumes & secrets (reused by both gen and score functions)
# ---------------------------------------------------------------------------

_VOLUMES = {
    WORKSPACE_PATH: codebase_volume,
    MODEL_CACHE_PATH: model_volume,
    POST_TRAIN_MODEL_CACHE_PATH: post_train_model_volume,
    DATASET_CACHE_PATH: dataset_volume,
    CHECKPOINT_PATH: checkpoint_volume,
    OUTPUT_PATH: output_volume,
    OUTPUT_APR4_PATH: output_apr4_volume,
}
_SECRETS = [
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("wandb-secret"),
]

# ---------------------------------------------------------------------------
# Register the main model function
# ---------------------------------------------------------------------------

@app.function(
    name=f"run_{_requested_model}",
    image=_selected_image,
    volumes=_VOLUMES,
    gpu=_requested_gpu,
    timeout=86400,
    secrets=_SECRETS,
)
def run_model(
    script: str = "",
    cmd: str = "",
    config: str = "",
    eval_config: str = "",
    mode_override: str = "",
    model_name: str = "",
) -> None:
    # model_name is passed explicitly from the local entrypoint to avoid
    # relying on sys.argv inside the container (which lacks CLI args).
    effective_model = model_name or _requested_model
    _run_impl(effective_model, script, cmd, config, eval_config, mode_override)


# ---------------------------------------------------------------------------
# (Optional) Register a second function for scoring with a different image.
# Only created when the eval config specifies score_model != model.
# ---------------------------------------------------------------------------

_run_score_model = None  # sentinel; overwritten below if needed


def _run_score_model_fn(
    eval_config: str = "",
    model_name: str = "",
) -> None:
    """Score phase — always defined so Modal can find it by attribute name."""
    effective_model = model_name or _requested_score_model
    _run_impl(effective_model, eval_config=eval_config, mode_override="score")


if _score_image is not None:
    _run_score_model_fn = app.function(
        name=f"score_{_requested_score_model}",
        image=_score_image,
        volumes=_VOLUMES,
        gpu=_score_gpu,
        timeout=86400,
        secrets=_SECRETS,
    )(_run_score_model_fn)

    _run_score_model = _run_score_model_fn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = "bagel",
    script: str = "",
    cmd: str = "",
    config: str = "",
    eval_config: str = "",
    gpu: str = "A100-80GB",
) -> None:
    """
    Run a command inside a model-specific Modal environment.

    Examples:
        modal run modal/run.py --model bagel --script inferencer.py
        modal run modal/run.py --model bagel --config bagel_sft
        modal run modal/run.py --model bagel --eval-config modal_ueval_bagel
        modal run modal/run.py --model emu3 --cmd "python image_generation.py"
        modal run modal/run.py --model bagel  # prints env info

        # Multi-GPU (e.g. 2x A100-80GB for Qwen-72B scoring)
        modal run modal/run.py --model wise --eval-config modal_wise_bagel_score --gpu A100-80GB:2
    """
    if _run_score_model is not None and eval_config:
        # Two-phase eval: generation in model's image, scoring in score_model's image
        print(f"[modal/run] two-phase eval: generate with '{_requested_model}', score with '{_requested_score_model}'")
        run_model.remote(eval_config=eval_config, mode_override="generate", model_name=_requested_model)
        print(f"[modal/run] generation phase done, starting scoring phase in '{_requested_score_model}' image ...")
        _run_score_model.remote(eval_config=eval_config, model_name=_requested_score_model)
        print(f"[modal/run] scoring phase done")
    else:
        run_model.remote(script=script, cmd=cmd, config=config, eval_config=eval_config, model_name=_requested_model)
