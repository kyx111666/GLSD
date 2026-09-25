"""Fine-threshold companion to run_expanded.py.

Same six-scale physical set and six local radii; only the common threshold
grid changes from .05 to .01. All methods receive the same refinement.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_expanded as expanded

expanded.base.TAUS = tuple(i / 100 for i in range(1, 100))
expanded.base.PROTOCOL = 'reference_residual_expanded_scales_a16_r6_tau001_v1'

if __name__ == '__main__':
    expanded.base.main()
