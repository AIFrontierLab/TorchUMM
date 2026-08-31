from __future__ import annotations

from umm.backbones.transformers_vlm import (
    TransformersVLMBackbone,
    VLMModelSpec,
    factories_from_specs,
)
from umm.backbones.transformers_vlm.registration import discover_backbone_factories


DUMMY_SPEC = VLMModelSpec(
    name="dummy_vlm",
    model_id="local/dummy",
    default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
)


class FakeVLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [
            {
                "generated_text": [
                    {"role": "user", "content": "ignored prompt"},
                    {"role": "assistant", "content": "A test response."},
                ]
            }
        ]


def test_installed_vlm_specs_are_discoverable() -> None:
    assert set(discover_backbone_factories()) == {"qwen2_5_vl_7b", "internvl3_8b"}


def test_specs_create_lazy_zero_argument_factories() -> None:
    factory = factories_from_specs({DUMMY_SPEC.name: DUMMY_SPEC})[DUMMY_SPEC.name]
    backbone = factory()
    assert isinstance(backbone, TransformersVLMBackbone)
    assert backbone.pipeline is None


def test_vlm_adapter_is_lazy_and_uses_unified_understanding_interface() -> None:
    fake = FakeVLM()
    backbone = TransformersVLMBackbone(DUMMY_SPEC, pipeline_factory=lambda *_: fake)
    backbone.load({"device": "cpu", "torch_dtype": "float32"})

    assert backbone.pipeline is None
    result = backbone.understanding(
        prompt="What is in this image?",
        images=["assets/example.png"],
        videos=[],
        understanding_cfg={"max_new_tokens": 32},
    )

    assert result["text"] == "A test response."
    assert result["model"] == "dummy_vlm"
    assert fake.calls[0]["text"][0]["content"] == [
        {"type": "image", "url": "assets/example.png"},
        {"type": "text", "text": "What is in this image?"},
    ]
    assert fake.calls[0]["max_new_tokens"] == 32


def test_video_inputs_are_explicitly_rejected() -> None:
    backbone = TransformersVLMBackbone(DUMMY_SPEC, pipeline_factory=lambda *_: FakeVLM())
    try:
        backbone.understanding("Describe it", [], ["clip.mp4"], {})
    except NotImplementedError as exc:
        assert "video inputs are not enabled" in str(exc)
    else:
        raise AssertionError("video inputs should be rejected")
