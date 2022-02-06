from numpy import size
from xpyriment import __version__
from xpyriment.anomalies import detect_srm
from xpyriment.util import generate_sample_df
import pandas as pd
import pytest


def test_version():
    assert __version__ == "0.1.0"


def test_create_sample_df():
    pass

    # test default size
    test_df = generate_sample_df()
    assert test_df.shape == (2000, 7)

    # test different size
    test_df = generate_sample_df(df_size=100)
    assert test_df.shape == (100, 7)
    assert test_df.columns.tolist() == [
        "first_name",
        "last_name",
        "variant",
        "state",
        "purchase_freq",
        "is_in_another_exp",
        "result",
    ]

    # test variant_skew that we allocate 1% more to A
    test_df = generate_sample_df(df_size=10000, variant_skew=1e-2)
    num_A = test_df.loc[test_df.variant == "A"].count()
    assert pytest.approx(num_A, 100) == int(10000 * 0.51)


def test_calculate_srm():
    pass
