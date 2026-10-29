import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from umm.post_training.UniMore import evaluation, training
from umm.post_training.UniMore.checkpoints import merge_tensor
from umm.post_training.UniMore.config import PACKAGE_ROOT, TORCHUMM_ROOT, load_config


@pytest.fixture
def configured(monkeypatch, tmp_path):
    for name in ("UNIMORE_WORK", "BAGEL_MODEL_PATH", "UNIMORE_MODEL", "UNIMORE_DATA_ROOT",
                 "UNIMORE_BENCHMARKS", "UNIMORE_SCORERS", "UNIMORE_VISUAL_JUDGE", "UNIMORE_TEXT_JUDGE"):
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.delenv("UNIMORE_ROOT", raising=False)
    monkeypatch.delenv("TORCHUMM_ROOT", raising=False)


def test_import_and_help_do_not_load_models():
    env = dict(os.environ, PYTHONPATH=str(TORCHUMM_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-B", "-c",
                    "import sys; import umm.post_training.UniMore; assert 'torch' not in sys.modules"],
                   env=env, check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-B", "-m", "umm.post_training.UniMore", "--help"],
                            env=env, check=True, capture_output=True, text=True)
    assert "prepare-data" in result.stdout and "merge" in result.stdout


def test_missing_environment_fails_early(monkeypatch, tmp_path):
    monkeypatch.delenv("UNIMORE_MISSING_TEST_PATH", raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"path": "${UNIMORE_MISSING_TEST_PATH}/data"}))
    with pytest.raises(ValueError, match="UNIMORE_MISSING_TEST_PATH"):
        load_config(path)


def test_config_script_paths_exist(configured):
    for path in (PACKAGE_ROOT / "configs").rglob("*.yaml"):
        config = load_config(path)
        model = config.get("inference", {}).get("backbone_cfg", {})
        if "bagel_root" in model:
            assert Path(model["bagel_root"]) == PACKAGE_ROOT / "third_party/bagel"
            assert (Path(model["bagel_root"]) / "inferencer.py").is_file()
        for block in config.values():
            if isinstance(block, dict):
                for key in ("calculation_script", "answer_path", "geneval_root", "benchmark_data"):
                    if key in block:
                        assert Path(block[key]).exists(), (path, key, block[key])


def test_only_the_final_recipe_is_bundled():
    paths = {str(p.relative_to(PACKAGE_ROOT / "configs")) for p in (PACKAGE_ROOT / "configs").rglob("*.yaml")}
    assert paths == {"train/t2i.yaml", "train/i2t.yaml", "data/t2i.yaml", "data/i2t.yaml", "inference.yaml"}


@pytest.mark.parametrize("route,processes", [("t2i", 4), ("i2t", 2)])
def test_training_launch_and_dry_run(configured, monkeypatch, route, processes):
    runner = Mock()
    monkeypatch.setattr(training.subprocess, "run", runner)
    config_path = PACKAGE_ROOT / f"configs/train/{route}.yaml"
    config = load_config(config_path)
    output = Path(config["args"]["results_dir"])
    command = training.train(config_path, dry_run=True)
    assert not output.exists()
    runner.assert_not_called()
    assert f"--nproc_per_node={processes}" in command
    training.train(config_path)
    assert runner.call_args.kwargs["cwd"] == PACKAGE_ROOT / "third_party/bagel_training"
    assert runner.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert runner.call_args.kwargs["check"] is True
    data = yaml.safe_load((output / "data.yaml").read_text())
    assert data["route_manifest"]["num_used_data"] == ([2460] if route == "t2i" else [4000])
    with pytest.raises(FileExistsError):
        training.train(config_path)


@pytest.mark.parametrize("mutation", ["route", "shards", "resume"])
def test_invalid_training_layout_is_rejected(configured, tmp_path, mutation):
    config = load_config(PACKAGE_ROOT / "configs/train/t2i.yaml")
    if mutation == "route":
        config["route"] = "i2t"
    elif mutation == "shards":
        config["num_processes"] = 1
    else:
        config["args"]["resume_from"] = str(tmp_path / "merged")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        training.train(path, dry_run=True)


@pytest.mark.parametrize("benchmark", ["mmmu", "mmstar", "mmbench", "mathvista", "mme", "mmvet",
                                      "wise", "geneval", "dpg_bench", "ueval", "unified_bench", "imgedit"])
def test_existing_evaluators_are_delegated(tmp_path, monkeypatch, benchmark):
    from umm.cli import eval as umm_eval
    runner = Mock(return_value=0)
    monkeypatch.setattr(umm_eval, "run_eval_command", runner)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"eval": {"benchmark": benchmark}}))
    args = SimpleNamespace(config=str(path))
    assert evaluation.evaluate(args) == 0
    runner.assert_called_once_with(args)


def test_realunify_is_local_and_uses_fixed_protocol(configured, monkeypatch, tmp_path):
    runner = Mock()
    monkeypatch.setattr(evaluation.subprocess, "run", runner)
    config = load_config(PACKAGE_ROOT / "configs/inference.yaml")
    config["eval"] = {"benchmark": "realunify"}
    config["realunify"] = {"data_root": str(tmp_path / "data"), "out_dir": str(tmp_path / "results"), "split": "UEG_direct"}
    path = tmp_path / "realunify.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(config=str(path))
    assert evaluation.evaluate(args) == 0
    calls = runner.call_args_list
    assert len(calls) == 2
    command = calls[0].args[0]
    assert command[2] == str(PACKAGE_ROOT / "scripts/eval_realunify_bagel.py")
    assert command[command.index("--image-size") + 1] == "512"
    assert command[command.index("--base-seed") + 1] == "42"
    assert "--require-complete" in calls[1].args[0]
    assert calls[0].kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(TORCHUMM_ROOT / "src")


def test_bf16_intermediate_rounding_is_preserved():
    base = torch.linspace(-.01, .01, 127).to(torch.bfloat16)
    t2i = (base.float() + .00019).to(torch.bfloat16)
    i2t = (base.float() - .00023).to(torch.bfloat16)
    actual = merge_tensor(base, t2i, i2t)
    assert torch.equal(actual, base + ((t2i - base) + .02 * (i2t - base)))
    fp32 = (base.float() + (t2i.float() - base.float()) + .02 * (i2t.float() - base.float())).bfloat16()
    assert not torch.equal(actual, fp32)
    with pytest.raises(ValueError, match="BF16"):
        merge_tensor(base.float(), t2i.float(), i2t.float())


def test_merge_rejects_shape_dtype_and_nonfinite_inputs():
    base = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape"):
        merge_tensor(base, base[:1], base)
    with pytest.raises(ValueError, match="dtype"):
        merge_tensor(base, base.float(), base)
    with pytest.raises(ValueError, match="Non-finite"):
        merge_tensor(base, base * float("nan"), base)
