from extensions.text2image.integrations.group_07_hunyuan_pixart import BACKBONES


def test_group_07_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"hunyuan_image_2_1", "pixart_sigma"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
