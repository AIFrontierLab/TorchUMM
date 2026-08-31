from umm.backbones.transformers_vlm.integrations.qwen2_5_vl_7b import BACKBONES


def test_qwen2_5_vl_7b_constructs_without_loading_weights() -> None:
    assert set(BACKBONES) == {"qwen2_5_vl_7b"}
    backbone = BACKBONES["qwen2_5_vl_7b"]()
    assert backbone.name == "qwen2_5_vl_7b"
    assert backbone.pipeline is None
