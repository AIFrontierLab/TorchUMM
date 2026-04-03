from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


CODEBASE_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = CODEBASE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from umm.backbones.bagel import BagelBackbone


def generation(
    prompt: str,
    output_path: Optional[str] = None,
    generation_cfg: Optional[dict[str, Any]] = None,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    model = BagelBackbone(model_path=model_path, bagel_root=str(Path(__file__).resolve().parent))
    model.load({})
    return model.generation(
        prompt=prompt,
        output_path=output_path,
        generation_cfg=generation_cfg,
    )


if __name__ == "__main__":
    prompt = (
        "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress "
        "made of delicate fabrics in soft, mystical colors like emerald green and silver. "
        "She has pointed ears, a gentle, enchanting expression, and her outfit is adorned "
        "with sparkling jewels and intricate patterns. The background is a magical forest "
        "with glowing plants, mystical creatures, and a serene atmosphere."
    )
    generation(prompt=prompt, output_path="fairy_cosplayer.png")
