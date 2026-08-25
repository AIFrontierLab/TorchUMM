from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from extensions.text2image import (
    DiffusersTextToImageBackbone,
    TextToImageModelSpec,
    factories_from_specs,
    register,
)
from umm.core import registry


DUMMY_SPEC = TextToImageModelSpec(
    name="dummy_t2i",
    model_id="local/dummy",
    pipeline_class="DiffusionPipeline",
    default_generation_cfg={"num_inference_steps": 10, "guidance_scale": 4.0},
)


class FakeTextToImagePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (16, 16), color="navy")])


def test_adapter_runs_unified_generation_interface(tmp_path: Path) -> None:
    fake = FakeTextToImagePipeline()
    backbone = DiffusersTextToImageBackbone(DUMMY_SPEC, pipeline_factory=lambda *_: fake)
    backbone.load({"local_files_only": True})
    output_path = tmp_path / "dummy.png"

    result = backbone.generation(
        prompt="a blue square",
        output_path=str(output_path),
        generation_cfg={"num_inference_steps": 1, "height": 64, "width": 64},
    )

    assert result["model"] == "dummy_t2i"
    assert result["image_paths"] == [str(output_path)]
    assert output_path.exists()
    assert fake.calls[0]["prompt"] == "a blue square"
    assert fake.calls[0]["num_inference_steps"] == 1
    assert "generator" in fake.calls[0]


def test_load_is_lazy() -> None:
    backbone = DiffusersTextToImageBackbone(DUMMY_SPEC)
    backbone.load({"model_path": "local/model", "local_files_only": True})
    assert backbone.pipeline is None
    assert backbone.model_path == "local/model"


def test_specs_create_lazy_zero_argument_factories() -> None:
    factory = factories_from_specs({DUMMY_SPEC.name: DUMMY_SPEC})[DUMMY_SPEC.name]
    backbone = factory()
    assert isinstance(backbone, DiffusersTextToImageBackbone)
    assert backbone.pipeline is None


def test_extension_registration_is_explicit_and_lazy() -> None:
    register()
    registered = set(registry.list_registered("backbone"))
    assert {"flux1_schnell", "hidream_i1", "sana_1_5", "sd35_medium"} <= registered
    assert registry.get("backbone", "flux1_schnell")().pipeline is None


def test_owned_ephemeral_cache_is_removed() -> None:
    backbone = DiffusersTextToImageBackbone(DUMMY_SPEC)
    backbone.load({"ephemeral_cache": True})
    cache_path = Path(backbone._ensure_cache_dir() or "")
    (cache_path / "checkpoint.bin").write_bytes(b"smoke")

    backbone.close()

    assert not cache_path.exists()


def test_empty_prompt_fails_before_loading() -> None:
    backbone = DiffusersTextToImageBackbone(DUMMY_SPEC)
    with pytest.raises(ValueError, match="non-empty prompt"):
        backbone.generation(" ", None, {})
    assert backbone.pipeline is None
