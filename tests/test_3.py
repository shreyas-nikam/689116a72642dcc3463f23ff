import pytest
from definition_7cf2e47b4d9c4e8a8201221bdb4b7ed3 import deterministic_pv
import pandas as pd
import numpy as np

@pytest.mark.parametrize("cashflows, discount_rate, expected", [
    (pd.Series([100, 100, 100]), 0.05, 272.324798),
    (pd.Series([500, 400, 300]), 0.1, 994.74513),
    (pd.Series([1000]), 0.0, 1000.0),
    (pd.Series([]), 0.05, 0.0),
    (pd.Series([-100, -50, 0, 50, 100]), 0.02, -4.0199)
])
def test_deterministic_pv(cashflows, discount_rate, expected):
    pv = deterministic_pv(cashflows, discount_rate)
    assert np.isclose(pv, expected, rtol=1e-5)
