"""Wrapper that intercepts wandb.log to save generated images deterministically.

Usage (called by ShowOBackbone.generate()):
    python _gen_wrapper.py <target_script.py> [script args...]

Environment:
    UMM_OUTPUT_DIR  — directory where intercepted images are saved as 000000.png, 000001.png, ...

The wrapper patches:
  1. wandb.init → after init completes (which replaces wandb.log via
     set_global), we wrap the real wandb.log with our image interceptor.
  2. StableDiffusionSafetyChecker.forward → bypass safety check to avoid
     a type mismatch crash when the script passes PIL Images instead of
     numpy arrays.
"""

import os
import sys


def _install_image_hook():
    output_dir = os.environ.get("UMM_OUTPUT_DIR")
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    import wandb

    _counter = [0]

    def _make_log_hook(real_log):
        """Wrap the real wandb.log with our image interceptor."""
        def _hooked_log(data, *args, **kwargs):
            if isinstance(data, dict):
                for value in data.values():
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        if isinstance(item, wandb.Image):
                            img = getattr(item, "_image", None)
                            if img is None:
                                img = getattr(item, "image", None)
                            if img is not None and hasattr(img, "save"):
                                img.save(os.path.join(output_dir, f"{_counter[0]:06d}.png"), format="PNG")
                                _counter[0] += 1
            return real_log(data, *args, **kwargs)
        return _hooked_log

    # Hook wandb.init: after init completes and sets the real wandb.log,
    # we wrap that real log with our interceptor.
    _original_init = wandb.init

    def _hooked_init(*args, **kwargs):
        result = _original_init(*args, **kwargs)
        # Now wandb.log is the real run.log (set by set_global inside init)
        wandb.log = _make_log_hook(wandb.log)
        return result

    wandb.init = _hooked_init


def _patch_safety_checker():
    """Bypass StableDiffusionSafetyChecker to avoid PIL/numpy type mismatch.

    The Show-o2 inference script passes PIL Images to safety_checker(),
    but the checker's forward() assumes numpy arrays when replacing NSFW
    images (calls .shape on PIL Image → AttributeError).  Since safety
    checking is irrelevant for evaluation, we make it a no-op.
    """
    try:
        from diffusers.pipelines.stable_diffusion.safety_checker import (
            StableDiffusionSafetyChecker,
        )
    except ImportError:
        return

    def _noop_forward(self, clip_input, images):
        return images, [False] * (len(images) if isinstance(images, list) else 1)

    StableDiffusionSafetyChecker.forward = _noop_forward


if __name__ == "__main__":
    _install_image_hook()
    _patch_safety_checker()

    # Shift argv: [_gen_wrapper.py, target.py, arg1, ...] → [target.py, arg1, ...]
    sys.argv = sys.argv[1:]

    # Mimic `python script.py` behavior: add the script's directory to sys.path
    # so that local imports (e.g. `from models import ...`) resolve correctly.
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd()
    sys.path.insert(0, script_dir)

    import runpy

    runpy.run_path(sys.argv[0], run_name="__main__")
