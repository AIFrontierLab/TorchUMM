from extensions.text2image.integrations.pixart_sigma import BACKBONES


def test_pixart_sigma_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"pixart_sigma"}
    backbone = BACKBONES["pixart_sigma"]()
    assert backbone.name == "pixart_sigma"
    assert backbone.pipeline is None
