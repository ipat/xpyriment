from tokenize import String
import pandas as pd
from typing import List
from scipy.stats import chi2_contingency
import numpy as np

test = {"a": "b", "c": "d", "ab": "s", "age": 28}


def detect_srm(
    df: pd.DataFrame,
    variant_col: str = "variant",
    result_col: str = "result",
    value_cols: List[str] = [],
    detect_cols: List[str] = [],
    critical_value: float = 0.05,
):

    if detect_cols == []:
        detect_cols = df.columns.tolist().remove(value_cols)

    print(detect_cols)

    # for each col calculate SRM statistics
    for each_feature in detect_cols:

        this_df = df.pivot_table(
            index=each_feature,
            columns=variant_col,
            aggfunc="count",
            values=result_col,
        )

        stat, p, dof, expected_arr = chi2_contingency(this_df)

        if p <= critical_value:
            # TODO: Implement 2-side binomial test
            pass

        # the calculation approach is referred to Twitter engineering page:
        # https://blog.twitter.com/engineering/en_us/a/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests

        pass

    pass
