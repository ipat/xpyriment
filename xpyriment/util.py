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


def generate_sample_df(
    df_size: int = 2000,
    seed: int = 1234,
    skew_degree: float = 0.0,
) -> pd.DataFrame:
    """Gernerate sample dataframe for the testing

    Args:
        df_size (int, optional): Number of sample df rows. Defaults to 2000.
        seed (int, optional): Random seed. Defaults to 1234.
        skew_degree (float, optional): Set if want to have skew the allocation within the features. Defaults to 0.

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
    result_p = [0.9, 0.1]

    data = pd.DataFrame(
        np.vstack(
            (
                np.array(
                    [
                        np.random.choice(first_name_list, df_size_variant),
                        np.random.choice(last_name_list, df_size_variant),
                        np.random.choice(["A"], df_size_variant),
                        np.random.choice(state_list, df_size_variant, p=state_p),
                        np.random.choice(
                            purchase_freq_list, df_size_variant, p=purchase_freq_p
                        ),
                        np.random.choice(
                            is_in_another_exp_list,
                            df_size_variant,
                            p=is_in_another_exp_p,
                        ),
                        np.random.choice(result_list, df_size_variant, p=result_p),
                    ]
                ).T,
                np.array(
                    [
                        np.random.choice(first_name_list, df_size_variant),
                        np.random.choice(last_name_list, df_size_variant),
                        np.random.choice(["B"], df_size_variant),
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
                        np.random.choice(result_list, df_size_variant, p=result_p),
                    ]
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
            "result",
        ],
    )

    return data.sample(frac=1)
