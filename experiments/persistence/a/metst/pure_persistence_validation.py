"""ME-TST entry point for pure persistence and matched evidence validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boostingvrme"))
from pure_persistence_validation import main


if __name__ == "__main__":
    main(default_backbone="metst")
