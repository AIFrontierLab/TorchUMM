from extensions.text2image.integrations.sd35_medium import BACKBONES


def test_sd35_medium_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"sd35_medium"}
    backbone = BACKBONES["sd35_medium"]()
    assert backbone.name == "sd35_medium"
    assert backbone.pipeline is None
