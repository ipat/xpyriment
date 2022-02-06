import pandas as pd
import numpy as np
from itertools import cycle


def generate_sample_df(
    df_size: int = 2000,
    seed: int = 1234,
    variant_skew: float = 0.0,
) -> pd.DataFrame:
    """Gernerate sample dataframe for the testing

    Args:
        df_size (int, optional): Number of sample df rows. Defaults to 2000.
        seed (int, optional): Random seed. Defaults to 1234.
        variant_skew (float, optional): Set if want to have skew allocation between variant Defaults to 0.

    Returns:
        pd.DataFrame: Generated dataframe
    """

    # template of user data from https://www.briandunning.com/sample-data/
    np.random.seed(seed)

    template = pd.read_csv("data/us-500.csv")

    state_list = template["state"].tolist()
    first_name_list = template["first_name"].tolist()
    last_name_list = template["last_name"].tolist()

    purchase_freq_list = ["low", "medium", "high"]
    purchase_freq_p = [0.85, 0.1, 0.05]

    is_in_another_exp_list = [0, 1]
    is_in_another_exp_p = [0.8, 0.2]

    result_list = [0, 1]
    result_p = [0.9, 0.1]

    variant_list = ["A", "B"]
    variant_p = [0.5 + variant_skew, 0.5 - variant_skew]

    data = pd.DataFrame(
        np.array(
            [
                np.random.choice(first_name_list, df_size),
                np.random.choice(last_name_list, df_size),
                np.random.choice(variant_list, df_size),
                np.random.choice(state_list, df_size),
                np.random.choice(purchase_freq_list, df_size, p=purchase_freq_p),
                np.random.choice(
                    is_in_another_exp_list, df_size, p=is_in_another_exp_p
                ),
                np.random.choice(result_list, df_size, p=result_p),
            ]
        ).T,
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

    return data
