import os
import sys


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stage1_shared_latent_modules.train import main


if __name__ == "__main__":
    main()
