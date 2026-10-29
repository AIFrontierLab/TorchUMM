import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/bagel_training"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
