from numpy import size
from xpyriment import __version__
from xpyriment.anomalies import detect_srm
from xpyriment.util import generate_sample_df
from xpyriment.experiment import cal_funnel_conversion, cal_p_value
import pandas as pd
import numpy as np
import pytest


def test_version():
    assert __version__ == "0.1.0"


def test_create_sample_df():
    pass

    # test default size
    test_df = generate_sample_df()
    assert test_df.shape == (2000, 7)

    # test different size
    test_df = generate_sample_df(df_size=100, funnel_cols=["step_1", "step_2"])
    assert test_df.shape == (100, 8)
    assert test_df.columns.tolist() == [
        "first_name",
        "last_name",
        "variant",
        "state",
        "purchase_freq",
        "is_in_another_exp",
        "step_1",
        "step_2",
    ]

    # test variant_skew that we allocate 1% more to A
    test_df = generate_sample_df(df_size=10000, skew_degree=1e-1)
    num_A = test_df.loc[test_df.variant == "A"].count()
    assert pytest.approx(num_A, 10) == int(10000 * 0.6)


def test_calculate_srm():

    # TODO: Complete SRM detection testing
    test_df = generate_sample_df(seed=101)
    detect_srm(test_df, detect_cols=["state", "purchase_freq", "is_in_another_exp"])

    pass


def test_funnel_conversion_cal():

    test_df = generate_sample_df(seed=101, funnel_cols=["step_1", "step_2", "step_3"])
    test_df.to_csv("data/sample_data_seed_101.csv", index=False)

    cal_conv = cal_funnel_conversion(
        test_df, group_cols=["variant"], funnel_cols=["step_1", "step_2", "step_3"]
    ).sort_values(["variant"])

    np.testing.assert_array_almost_equal(
        cal_conv["step_1->step_2"].to_numpy(), np.array([135.0 / 375.0, 162.0 / 394.0])
    )
    np.testing.assert_array_almost_equal(
        cal_conv["step_2->step_3"].to_numpy(), np.array([57.0 / 135.0, 53.0 / 162.0])
    )


def test_p_value_cal():
    test_df = pd.read_csv("data/sample_data_seed_101.csv")

    cal_conv = cal_funnel_conversion(
        test_df, group_cols=["variant"], funnel_cols=["step_1", "step_2", "step_3"]
    ).sort_values(["variant"])

    p_val_df = cal_p_value(
        cal_conv, var_col="variant", step_cols=["step_1->step_2", "step_2->step_3"]
    )

    np.testing.assert_array_almost_equal(
        p_val_df.sort_values(["step"])["p_value"].tolist(), [0.072599, 0.954413]
    )
