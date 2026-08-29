from extensions.text2image.integrations.sana_1_5 import BACKBONES


def test_sana_1_5_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"sana_1_5"}
    backbone = BACKBONES["sana_1_5"]()
    assert backbone.name == "sana_1_5"
    assert backbone.pipeline is None
