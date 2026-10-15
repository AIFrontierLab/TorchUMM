"""FlexUMM routing utilities for the BAGEL shared multimodal backbone.

The package keeps model-heavy imports lazy. This allows the checkpoint-free
TAFE/FFN-pool components and their tests to run on CPU-only environments,
while preserving the historical top-level imports for full BAGEL inference.
"""

from .config import DepthPlan, DynamicDepthConfig, DynamicDepthController
from .tafe import (
    AttentionResidualSubset,
    FFNResidualSubset,
    FFNSubsetPool,
    RoutedAttention,
    RoutedFFN,
    TAFEAction,
    TAFEExecutionState,
    TAFEGate,
    TAFEOutput,
    current_tafe_state,
    tafe_execution,
)

_LAZY_LOADING = {
    "DEFAULT_DEPTH_CONFIG",
    "DEFAULT_MODEL_PATH",
    "DynamicBagelRuntime",
    "load_dynamic_bagel",
}
_LAZY_MODELING = {
    "DynamicBagel",
    "DynamicQwen2Model",
    "enable_dynamic_depth",
    "set_flexumm_training_stage",
}
_LAZY_OVERLAYS = {"apply_overlay_to_model", "apply_overlay_to_state_dict"}
_LAZY_FUSION = {
    "FusionCompiler",
    "FusionRealizationAdapter",
    "FusionReleaseController",
}
_LAZY_ROUTER = {"CandidateDepthRouter"}

__all__ = [
    "DEFAULT_DEPTH_CONFIG",
    "DEFAULT_MODEL_PATH",
    "CandidateDepthRouter",
    "AttentionResidualSubset",
    "DepthPlan",
    "DynamicBagel",
    "DynamicBagelRuntime",
    "DynamicDepthConfig",
    "DynamicDepthController",
    "DynamicQwen2Model",
    "FFNResidualSubset",
    "FFNSubsetPool",
    "FusionCompiler",
    "FusionRealizationAdapter",
    "FusionReleaseController",
    "RoutedFFN",
    "RoutedAttention",
    "TAFEAction",
    "TAFEExecutionState",
    "TAFEGate",
    "TAFEOutput",
    "apply_overlay_to_model",
    "apply_overlay_to_state_dict",
    "current_tafe_state",
    "enable_dynamic_depth",
    "load_dynamic_bagel",
    "set_flexumm_training_stage",
    "tafe_execution",
]


def __getattr__(name: str):
    if name in _LAZY_LOADING:
        from . import loading

        return getattr(loading, name)
    if name in _LAZY_MODELING:
        from . import modeling

        return getattr(modeling, name)
    if name in _LAZY_OVERLAYS:
        from . import overlays

        return getattr(overlays, name)
    if name in _LAZY_FUSION:
        from . import fusion

        return getattr(fusion, name)
    if name in _LAZY_ROUTER:
        from . import router

        return getattr(router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
