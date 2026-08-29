from extensions.text2image.integrations.hidream_i1 import BACKBONES


def test_hidream_i1_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"hidream_i1"}
    backbone = BACKBONES["hidream_i1"]()
    assert backbone.name == "hidream_i1"
    assert backbone.pipeline is None
