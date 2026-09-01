from extensions.text2image.integrations.cogview4 import BACKBONES


def test_cogview4_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"cogview4"}
    backbone = BACKBONES["cogview4"]()
    assert backbone.name == "cogview4"
    assert backbone.pipeline is None
