from extensions.vlm.integrations.internvl3_8b import BACKBONES


def test_internvl3_8b_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"internvl3_8b"}
    backbone = BACKBONES["internvl3_8b"]()
    assert backbone.name == "internvl3_8b"
    assert backbone.pipeline is None
