"""ME-TST entry point for the shared, height-gate-free persistence experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boostingvrme"))
from unified_persistence import main


if __name__ == "__main__":
    main(default_backbone="metst")
