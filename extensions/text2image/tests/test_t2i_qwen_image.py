from extensions.text2image.integrations.qwen_image import BACKBONES


def test_qwen_image_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"qwen_image"}
    backbone = BACKBONES["qwen_image"]()
    assert backbone.name == "qwen_image"
    assert backbone.pipeline is None
