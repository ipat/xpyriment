from copy import deepcopy
from unittest import result
import pandas as pd
import numpy as np
from itertools import cycle
from typing import List


def add_skew(
    weight_arr: List[float],
    skew_degree: float = 0.0,
):
    new_weight_arr = weight_arr + np.random.random(len(weight_arr)) * skew_degree
    new_weight_arr = new_weight_arr / new_weight_arr.sum()

    return new_weight_arr


def generate_dummy_funnel_arr(
    n_rows: int = 100,
    n_cols: int = 1,
    p: List[int] = [0.6, 0.4],
    outcomes: List[int] = [0, 1],
):
    results = np.hstack(
        (np.ones((n_rows, 1)), np.random.choice(outcomes, (n_rows, n_cols), p=p))
    )

    for x, y in zip(range(0, n_cols), range(1, n_cols + 1)):
        results[:, y] = results[:, x] * results[:, y]

    return results[:, 1:].T.tolist()


def generate_sample_df(
    df_size: int = 2000,
    seed: int = 1234,
    skew_degree: float = 0.0,
    funnel_cols: List[str] = ["result"],
) -> pd.DataFrame:
    """Gernerate sample dataframe for the testing

    Args:
        df_size (int, optional): Number of sample df rows. Defaults to 2000.
        seed (int, optional): Random seed. Defaults to 1234.
        skew_degree (float, optional): Set if want to have skew the allocation within the features. Defaults to 0.
        funnel_cols (List(str), optional): Define the results from the sample in case want multiple mock up columns tha represent funnel. Defaults is ['result']

    Returns:
        pd.DataFrame: Generated dataframe
    """

    # template of user data from https://www.briandunning.com/sample-data/
    np.random.seed(seed)
    df_size_variant = int(df_size / 2)

    template = pd.read_csv("data/us-500.csv")

    state_list = template["state"].tolist()
    state_p = np.ones(len(state_list)) / len(state_list)
    first_name_list = template["first_name"].tolist()
    last_name_list = template["last_name"].tolist()

    purchase_freq_list = ["low", "medium", "high"]
    purchase_freq_p = [0.85, 0.1, 0.05]

    is_in_another_exp_list = [0, 1]
    is_in_another_exp_p = [0.8, 0.2]

    result_list = [0, 1]
    result_p = [0.6, 0.4]

    data = pd.DataFrame(
        np.vstack(
            (
                np.array(
                    [
                        np.random.choice(
                            first_name_list,
                            df_size_variant,
                        ),
                        np.random.choice(
                            last_name_list,
                            df_size_variant,
                        ),
                        np.random.choice(
                            ["A"],
                            df_size_variant,
                        ),
                        np.random.choice(
                            state_list,
                            df_size_variant,
                            p=state_p,
                        ),
                        np.random.choice(
                            purchase_freq_list,
                            df_size_variant,
                            p=purchase_freq_p,
                        ),
                        np.random.choice(
                            is_in_another_exp_list,
                            df_size_variant,
                            p=is_in_another_exp_p,
                        ),
                    ]
                    + generate_dummy_funnel_arr(
                        df_size_variant, len(funnel_cols), result_p, result_list
                    )
                ).T,
                np.array(
                    [
                        np.random.choice(
                            first_name_list,
                            df_size_variant,
                        ),
                        np.random.choice(
                            last_name_list,
                            df_size_variant,
                        ),
                        np.random.choice(
                            ["B"],
                            df_size_variant,
                        ),
                        np.random.choice(
                            state_list,
                            df_size_variant,
                            p=add_skew(state_p, skew_degree),
                        ),
                        np.random.choice(
                            purchase_freq_list,
                            df_size_variant,
                            p=add_skew(purchase_freq_p, skew_degree),
                        ),
                        np.random.choice(
                            is_in_another_exp_list,
                            df_size_variant,
                            p=add_skew(is_in_another_exp_p, skew_degree),
                        ),
                    ]
                    + generate_dummy_funnel_arr(
                        df_size_variant, len(funnel_cols), result_p, result_list
                    )
                ).T,
            )
        ),
        columns=[
            "first_name",
            "last_name",
            "variant",
            "state",
            "purchase_freq",
            "is_in_another_exp",
        ]
        + funnel_cols,
    )
    

    return data.sample(frac=1)
