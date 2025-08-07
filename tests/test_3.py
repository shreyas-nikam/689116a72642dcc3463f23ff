import pytest
from definition_9c8a1e18558e46e0a308112936f5bd1d import deterministic_pv
import pandas as pd
import numpy as np

def create_series(data):
    return pd.Series(data)

@pytest.mark.parametrize("cashflows, expected", [
    (create_series([100, 100, 100]), 300.0),
    (create_series([100, -50, 25]), 75.0),
    (create_series([0, 0, 0]), 0.0),
    (create_series([1000]), 1000.0),
    (create_series([-100, -100, -100]), -300.0)
])
def test_deterministic_pv(cashflows, expected):
    assert np.isclose(deterministic_pv(cashflows), expected)
