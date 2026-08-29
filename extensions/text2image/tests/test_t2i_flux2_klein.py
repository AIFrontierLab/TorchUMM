from extensions.text2image.integrations.flux2_klein import BACKBONES


def test_flux2_klein_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"flux2_klein"}
    backbone = BACKBONES["flux2_klein"]()
    assert backbone.name == "flux2_klein"
    assert backbone.pipeline is None
