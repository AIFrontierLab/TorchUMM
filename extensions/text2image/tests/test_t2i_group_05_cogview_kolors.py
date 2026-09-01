from extensions.text2image.integrations.group_05_cogview_kolors import BACKBONES


def test_group_05_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"cogview4", "kolors"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
