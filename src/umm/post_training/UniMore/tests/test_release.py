import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from umm.post_training.UniMore.checkpoints import export_branch, merge_branches, merge_tensor
from umm.post_training.UniMore.cli import build_parser
from umm.post_training.UniMore.config import load_config
from umm.post_training.UniMore.data import remap_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    return json.loads((ROOT / "tests/fixtures/numerical_reference.json").read_text())


def signature(node):
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                item.body
                and isinstance(item.body[0], ast.Expr)
                and isinstance(item.body[0].value, ast.Constant)
                and isinstance(item.body[0].value.value, str)
            ):
                item.body = item.body[1:]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def test_cli_contains_only_method_commands():
    parser = build_parser()
    assert parser.parse_args(["train", "--config", "config.yaml"]).command == "train"
    with pytest.raises(SystemExit):
        parser.parse_args(["search"])


def test_reference_negatives(reference):
    from data.route_manifest_dataset import RouteManifestIterableDataset

    dataset = object.__new__(RouteManifestIterableDataset)
    dataset.i2t_negative_seed = 20260702
    dataset.i2t_num_negatives = 3
    dataset.use_pseudo_grounding = False
    dataset.global_i2t_negative_sources = ["detail_drop", "wrong_detail", "random_caption"]
    dataset.global_i2t_caption_pool = [row["text"] for row in reference["records"]]
    for case in reference["negative_sampling"]:
        row = reference["records"][case["record"]]
        assert (
            dataset._negative_global_i2t_answers(row, row["text"], case["group"]) == case["answers"]
        )


def test_reference_packed_losses_and_gradients(reference):
    from train.grouped_i2t_loss import reduce_ranking_violations

    source = ast.parse(
        (ROOT / "third_party/bagel_training/train/pretrain_unified_navit.py").read_text()
    )
    node = next(
        item
        for item in ast.walk(source)
        if isinstance(item, ast.FunctionDef) and item.name == "compute_grouped_i2t_loss"
    )

    class Distributed:
        class ReduceOp:
            SUM = "sum"

        def __init__(self, replicas):
            self.replicas = replicas

        def get_world_size(self):
            return self.replicas

        def all_reduce(self, tensor, op):
            tensor.mul_(self.replicas)

    for case in reference["packed_losses"]:
        namespace = {
            "torch": torch,
            "device": "cpu",
            "dist": Distributed(case["replicas"]),
            "training_args": SimpleNamespace(ranking_margin=0.2),
            "reduce_ranking_violations": reduce_ranking_violations,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<loss>", "exec"), namespace)
        values = torch.tensor(case["token_ce"], dtype=torch.float32, requires_grad=True)
        loss = namespace["compute_grouped_i2t_loss"](
            values,
            {"sample_lens": case["lengths"], "ce_loss_indexes": torch.arange(values.numel())},
            case["metadata"],
        )
        loss.backward()
        assert loss.item() == case["loss"]
        assert values.grad.tolist() == case["gradient"]


def test_model_and_packing_math_unchanged(reference):
    for path, expected in reference["model_sources"].items():
        assert signature(ast.parse((ROOT / path).read_text())) == expected, path
    source = ast.parse((ROOT / "third_party/bagel_training/data/dataset_base.py").read_text())
    packed = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "PackedDataset"
    )
    for name, expected in reference["packing_functions"].items():
        function = next(
            node for node in packed.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert signature(function) == expected, name


def test_merge_matches_released_weights(reference):
    for point in reference["released_weights"]:
        b, t, i, expected = point["values"]
        tensors = [torch.tensor(v, dtype=torch.bfloat16) for v in (b, t, i)]
        assert merge_tensor(*tensors).item() == expected


def test_export_and_merge_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    base = tmp_path / "base"
    base.mkdir()
    state = {
        "language_model.layer.weight": torch.ones(3, 4, dtype=torch.bfloat16),
        "connector.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "vae2llm.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    save_file(state, str(base / "ema.safetensors"))
    a = torch.arange(64, dtype=torch.float32).reshape(16, 4) / 1024
    b = torch.arange(48, dtype=torch.float32).reshape(3, 16) / 512
    trained = {
        "language_model.layer.lora_A.default.weight": a,
        "language_model.layer.lora_B.default.weight": b,
        "connector.weight": torch.full((2, 2), 1.125),
        "vae2llm.weight": torch.full((2, 2), 1.5),
    }
    checkpoint = tmp_path / "trained.safetensors"
    save_file(trained, str(checkpoint))
    for route in ("t2i", "i2t"):
        export_branch(base, checkpoint, route, tmp_path / route)
    t = load_file(str(tmp_path / "t2i/ema.safetensors"))
    i = load_file(str(tmp_path / "i2t/ema.safetensors"))
    expected_lora = (
        state["language_model.layer.weight"]
        .float()
        .addmm(b, a, beta=1.0, alpha=2.0)
        .to(torch.bfloat16)
    )
    assert torch.equal(t["language_model.layer.weight"], expected_lora)
    assert torch.equal(t["connector.weight"], state["connector.weight"])
    assert torch.equal(i["vae2llm.weight"], state["vae2llm.weight"])
    assert torch.equal(t["vae2llm.weight"], trained["vae2llm.weight"].bfloat16())
    assert torch.equal(i["connector.weight"], trained["connector.weight"].bfloat16())
    merge_branches(base, tmp_path / "t2i", tmp_path / "i2t", tmp_path / "merged")
    merged = load_file(str(tmp_path / "merged/ema.safetensors"))
    assert set(merged) == set(state)
    for name in state:
        assert torch.equal(merged[name], merge_tensor(state[name], t[name], i[name]))
    with pytest.raises(FileExistsError):
        merge_branches(base, tmp_path / "t2i", tmp_path / "i2t", tmp_path / "merged")


def test_prepared_metadata_contains_only_record_counts(tmp_path):
    from umm.post_training.UniMore.data import prepare_data

    image = tmp_path / "image.jpg"
    image.touch()
    mapping = tmp_path / "paths.json"
    mapping.write_text("{}")
    for route, count in (("t2i", 2460), ("i2t", 4000)):
        records = [
            {
                "sample_id": f"{route}-{index}",
                "image_ref": {"type": "local_file", "path": str(image)},
                "local_phrase": "red cup",
                "local_crop_ref": {"type": "local_file", "path": str(image)},
            }
            for index in range(count)
        ]
        (tmp_path / f"{route}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    output = tmp_path / "prepared"
    prepare_data(tmp_path / "t2i.jsonl", tmp_path / "i2t.jsonl", mapping, output)
    assert json.loads((output / "manifest.json").read_text()) == {
        "t2i": {"records": 2460},
        "i2t": {"records": 4000},
    }


def test_relocation_does_not_change_caption_or_tar_member():
    ref = {
        "type": "image_crop",
        "bbox": [1, 2, 10, 20],
        "image_ref": {
            "type": "webdataset_tar_member",
            "shard_path": "/old/data.tar",
            "member": "photo.jpg",
        },
    }
    mapped = remap_reference(ref, {"/old": "/new"})
    assert ref["image_ref"]["shard_path"] == "/old/data.tar"
    assert mapped["image_ref"]["member"] == "photo.jpg"
    assert mapped["image_ref"]["shard_path"] == "/new/data.tar"
    assert mapped["bbox"] == ref["bbox"]


def test_all_configs_resolve_without_cluster_paths(monkeypatch, tmp_path):
    names = [
        "UNIMORE_WORK",
        "BAGEL_MODEL_PATH",
        "UNIMORE_MODEL",
        "UNIMORE_DATA_ROOT",
        "UNIMORE_BENCHMARKS",
        "UNIMORE_SCORERS",
        "UNIMORE_VISUAL_JUDGE",
        "UNIMORE_TEXT_JUDGE",
    ]
    for name in names:
        monkeypatch.setenv(name, str(tmp_path / name))
    for path in (ROOT / "configs").rglob("*.yaml"):
        text = json.dumps(load_config(path))
        assert "${" not in text, path
        raw = path.read_text()
        assert "/scratch/" not in raw and "/sciclone/" not in raw, path
    t2i = load_config(ROOT / "configs/train/t2i.yaml")
    i2t = load_config(ROOT / "configs/train/i2t.yaml")
    assert (t2i["num_processes"], t2i["args"]["total_steps"], t2i["args"]["lr"]) == (4, 1000, 5e-5)
    assert (i2t["num_processes"], i2t["args"]["total_steps"], i2t["args"]["lr"]) == (2, 600, 1.8e-5)
    for value in (t2i, i2t):
        assert value["args"]["global_seed"] == 4396
        assert value["args"]["data_seed"] == 42
        assert value["args"]["model_path"] == value["args"]["resume_from"]
