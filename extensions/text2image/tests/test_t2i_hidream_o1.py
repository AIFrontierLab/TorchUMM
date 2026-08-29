from extensions.text2image.integrations.hidream_o1 import BACKBONES


def test_hidream_o1_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"hidream_o1"}
    backbone = BACKBONES["hidream_o1"]()
    assert backbone.name == "hidream_o1"
    assert backbone.pipeline is None
