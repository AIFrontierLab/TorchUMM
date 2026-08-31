from __future__ import annotations

from types import SimpleNamespace

from umm.backbones.transformers_vlm import TransformersVLMBackbone
from umm.backbones.transformers_vlm.integrations.standard import SPECS
from umm.backbones.transformers_vlm.registration import discover_backbone_factories


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


def test_ten_vlm_specs_are_discoverable() -> None:
    expected = {
        "qwen2_5_vl_7b",
        "internvl3_8b",
        "llava_onevision_7b",
        "idefics3_8b",
        "smolvlm_2b",
        "phi3_5_vision",
        "gemma3_4b",
        "molmo_7b",
        "minicpm_v_2_6",
        "paligemma2_3b",
    }
    assert set(SPECS) == expected
    assert set(discover_backbone_factories()) == expected


def test_vlm_adapter_is_lazy_and_uses_unified_understanding_interface() -> None:
    fake = FakeVLM()
    backbone = TransformersVLMBackbone(
        SPECS["qwen2_5_vl_7b"],
        pipeline_factory=lambda *_: fake,
    )
    backbone.load({"device": "cpu", "torch_dtype": "float32"})

    assert backbone.pipeline is None
    result = backbone.understanding(
        prompt="What is in this image?",
        images=["assets/example.png"],
        videos=[],
        understanding_cfg={"max_new_tokens": 32},
    )

    assert result["text"] == "A test response."
    assert result["model"] == "qwen2_5_vl_7b"
    assert fake.calls[0]["text"][0]["content"] == [
        {"type": "image", "url": "assets/example.png"},
        {"type": "text", "text": "What is in this image?"},
    ]
    assert fake.calls[0]["max_new_tokens"] == 32


def test_paligemma_prompt_prefix_can_be_overridden() -> None:
    backbone = TransformersVLMBackbone(SPECS["paligemma2_3b"], pipeline_factory=lambda *_: SimpleNamespace())

    assert backbone._format_prompt("What color is the car?") == "answer en What color is the car?"
    assert backbone._format_prompt("ocr") == "ocr"
    assert backbone._format_prompt("describe en") == "describe en"


def test_video_inputs_are_explicitly_rejected() -> None:
    backbone = TransformersVLMBackbone(SPECS["smolvlm_2b"], pipeline_factory=lambda *_: FakeVLM())
    try:
        backbone.understanding("Describe it", [], ["clip.mp4"], {})
    except NotImplementedError as exc:
        assert "video inputs are not enabled" in str(exc)
    else:
        raise AssertionError("video inputs should be rejected")
