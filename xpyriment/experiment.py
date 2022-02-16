import pandas as pd
import numpy as np
from typing import List
import re
import math
from scipy.stats import norm


def cal_funnel_conversion(
    df: pd.DataFrame,
    group_cols: List[str] = ["variant"],
    funnel_cols: List[str] = ["step_1", "step_2", "step_3"],
):

    for this_col in funnel_cols:
        df[this_col] = df[this_col].astype(float)

    funnel_conv = df.groupby(group_cols).agg({x: "sum" for x in funnel_cols})

    # check if all funnel cols and group cols are in df
    assert set(funnel_cols) <= set(df.columns.tolist())
    assert set(group_cols) <= set(df.columns.tolist())

    funnel_conv = funnel_conv[funnel_cols]

    for prev_step, next_step in zip(funnel_cols[:-1], funnel_cols[1:]):

        step_name = prev_step + "->" + next_step

        funnel_conv[step_name] = 1.0 * funnel_conv[next_step] / funnel_conv[prev_step]

    funnel_conv = funnel_conv.reset_index()

    return funnel_conv


def cal_p_value(
    df: pd.DataFrame,
    var_col: str = "max_var",
    index_cols: List[str] = [],
    step_cols: List[str] = ["step_1->step_2", "step_2->step_3"],
    member_cols: List[str] = [],
    critical_value: int = 0.05,
):

    result_df = pd.DataFrame()

    if len(member_cols) == 0:
        for this_step in step_cols:
            member_cols.append(
                [x for x in df.columns if (this_step.split("->")[0]) in x][0]
            )

    if len(index_cols) == 0:
        df["dummy"] = 1
        index_cols = ["dummy"]

    for i, this_step in enumerate(step_cols):
        member_col = member_cols[i]

        this_result = df.pivot_table(
            columns=var_col, index=index_cols, values=[this_step, member_col]
        )

        if this_result.columns.get_level_values(0).tolist()[0] == this_step:
            this_result.columns = ["conv_A", "conv_B", "member_A", "member_B"]
        else:
            this_result.columns = ["member_A", "member_B", "conv_A", "conv_B"]

        this_result["conv_diff"] = this_result["conv_B"] - this_result["conv_A"]
        this_result["conv_diff_pect"] = (
            this_result["conv_B"] / this_result["conv_A"] - 1
        )
        this_result["pooled_conv"] = (
            this_result["conv_B"] * this_result["member_B"]
            + this_result["conv_A"] * this_result["member_A"]
        ) / (this_result["member_B"] + this_result["member_A"])

        this_result["z_score"] = this_result.apply(
            lambda x: (x.conv_diff)
            / math.sqrt(
                x.pooled_conv * (1 - x.pooled_conv) * (1 / x.member_B + 1 / x.member_A)
            ),
            axis=1,
        )
        this_result["p_value"] = this_result.z_score.apply(lambda x: 1 - norm.cdf(x))
        this_result["is_significant"] = this_result.p_value.apply(
            lambda x: True
            if ((x <= critical_value) or (x >= (1 - critical_value)))
            else False
        )
        this_result["step"] = this_step

        result_df = result_df.append(this_result.reset_index(), ignore_index=True)

    if "dummy" in result_df.columns.tolist():
        result_df = result_df.drop(columns=["dummy"])

    return result_df
