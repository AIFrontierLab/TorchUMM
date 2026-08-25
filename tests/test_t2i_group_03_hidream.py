from umm.backbones.diffusers_t2i.integrations.group_03_hidream import BACKBONES


def test_group_03_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"hidream_o1", "hidream_i1"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
