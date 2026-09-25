"""ME-TST entry point for the shared, matched EquiScale validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boostingvrme"))
from equiscale_fair_validation import main


if __name__ == "__main__":
    main(default_backbone="metst")
