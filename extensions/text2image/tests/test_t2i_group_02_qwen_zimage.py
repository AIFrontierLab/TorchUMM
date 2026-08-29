from extensions.text2image.integrations.group_02_qwen_zimage import BACKBONES


def test_group_02_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"z_image", "qwen_image"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
