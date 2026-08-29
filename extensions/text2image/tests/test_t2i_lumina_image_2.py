from extensions.text2image.integrations.lumina_image_2 import BACKBONES


def test_lumina_image_2_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"lumina_image_2"}
    backbone = BACKBONES["lumina_image_2"]()
    assert backbone.name == "lumina_image_2"
    assert backbone.pipeline is None
