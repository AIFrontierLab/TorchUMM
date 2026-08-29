from extensions.text2image.integrations.flux1_schnell import BACKBONES


def test_flux1_schnell_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"flux1_schnell"}
    backbone = BACKBONES["flux1_schnell"]()
    assert backbone.name == "flux1_schnell"
    assert backbone.pipeline is None
