from extensions.text2image.integrations.group_04_sana_lumina import BACKBONES


def test_group_04_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"sana_1_5", "lumina_image_2"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
