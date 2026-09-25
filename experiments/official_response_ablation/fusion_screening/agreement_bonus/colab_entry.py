"""Reuse the successfully used Colab kernel runtime and official pandas adapter."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, 'append'):
    def dataframe_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        elif isinstance(other, pd.Series):
            other = other.to_frame().T
        return pd.concat([self, other], ignore_index=ignore_index,
                         verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = dataframe_append

sys.path.insert(0, '/content/ME-TST')
sys.path.insert(0, str(Path(__file__).resolve().parent))
print('Runtime:', sys.executable, sys.version.split()[0], 'numpy', np.__version__, 'pandas', pd.__version__, flush=True)
from run_agreement_bonus import main

if __name__ == '__main__':
    main()
