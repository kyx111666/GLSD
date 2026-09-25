"""Subprocess entry: compatibility shim used by the existing Colab notebook."""
import pandas as pd

if not hasattr(pd.DataFrame, "append"):
    def dataframe_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        return pd.concat([self, other], ignore_index=ignore_index,
                         verify_integrity=verify_integrity, sort=sort)
    pd.DataFrame.append = dataframe_append

from run_one_to_one_full_tuning import main

if __name__ == "__main__":
    main()
