from umm.backbones.diffusers_t2i.integrations.group_01_flux import BACKBONES


def test_group_01_backbones_construct_without_loading_weights() -> None:
    assert set(BACKBONES) == {"flux2_klein", "flux1_schnell"}
    for name, factory in BACKBONES.items():
        backbone = factory()
        assert backbone.name == name
        assert backbone.pipeline is None
