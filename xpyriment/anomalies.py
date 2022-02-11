from tokenize import String
import pandas as pd
from typing import List
from scipy.stats import chi2_contingency
import numpy as np

test = {"a": "b", "c": "d", "ab": "s", "age": 28}


# According to https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/
# they use 0.0005 as the critical value to detect if SRM exists. Therefore, we set the default critical_value to resemble to the study.
def detect_srm(
    df: pd.DataFrame,
    variant_col: str = "variant",
    result_col: str = "result",
    value_cols: List[str] = [],
    detect_cols: List[str] = [],
    critical_value: float = 0.0005,
):

    # the calculation approach is referred to Twitter engineering page:
    # https://blog.twitter.com/engineering/en_us/a/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests

    if detect_cols == []:
        detect_cols = df.columns.tolist().remove(value_cols)

    results_df = pd.DataFrame(
        columns=["detecting_col", "chi2_stat", "p_value", "is_srm"]
    )

    # for each col calculate SRM statistics
    for each_feature in detect_cols:

        this_df = df.pivot_table(
            index=each_feature,
            columns=variant_col,
            aggfunc="count",
            values=result_col,
        )

        chi2_stat, p, dof, expected_arr = chi2_contingency(this_df)

        if p <= critical_value:
            found_srm = True
        else:
            found_srm = False

        results_df = results_df.append(
            {
                "detecting_col": each_feature,
                "chi2_stat": chi2_stat,
                "p_value": p,
                "is_srm": found_srm,
            },
            ignore_index=True,
        )

    return results_df
