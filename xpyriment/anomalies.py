from tokenize import String
import pandas as pd
from typing import List

test = {"a": "b", "c": "d", "ab": "s", "age": 28}


def detect_srm(
    df: pd.DataFrame,
    value_cols: List[str] = [],
    detect_cols: List[str] = [],
):

    if detect_cols == []:
        detect_cols = df.columns.tolist().remove(value_cols)

    # for each col calculate SRM statistics
    for each_col in detect_cols:

        # the calculation approach is referred to Twitter engineering page:
        # https://blog.twitter.com/engineering/en_us/a/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests

        pass

    pass
