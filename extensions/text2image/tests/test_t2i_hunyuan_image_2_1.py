from extensions.text2image.integrations.hunyuan_image_2_1 import BACKBONES


def test_hunyuan_image_2_1_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"hunyuan_image_2_1"}
    backbone = BACKBONES["hunyuan_image_2_1"]()
    assert backbone.name == "hunyuan_image_2_1"
    assert backbone.pipeline is None
