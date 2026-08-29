from extensions.text2image.integrations.auraflow import BACKBONES


def test_auraflow_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"auraflow"}
    backbone = BACKBONES["auraflow"]()
    assert backbone.name == "auraflow"
    assert backbone.pipeline is None
