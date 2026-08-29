from extensions.text2image.integrations.z_image import BACKBONES


def test_z_image_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"z_image"}
    backbone = BACKBONES["z_image"]()
    assert backbone.name == "z_image"
    assert backbone.pipeline is None
