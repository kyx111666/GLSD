"""Control: original physical scales, original structure grid, 0.01 thresholds."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if not (HERE / 'run_structure.py').exists():
    sys.path.insert(0, str(HERE.parent / 'structure_search'))
import run_structure as base

base.TAUS = tuple(i / 100 for i in range(1, 100))
base.PROTOCOL = 'reference_residual_original_structure_tau001_v1'

if __name__ == '__main__':
    base.main()
