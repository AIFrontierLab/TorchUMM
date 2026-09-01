from extensions.text2image.integrations.kolors import BACKBONES


def test_kolors_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"kolors"}
    backbone = BACKBONES["kolors"]()
    assert backbone.name == "kolors"
    assert backbone.pipeline is None
