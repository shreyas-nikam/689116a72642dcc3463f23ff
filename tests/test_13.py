import pytest
from definition_16811c44de624f7f813d4ad1a0a88d0b import adjust_discount_rate_for_credit_risk

@pytest.mark.parametrize("base_rate, rating_before, rating_after, expected", [
    (0.05, 5, 5, 0.05),  # No change in rating
    (0.05, 5, 6, 0.05),  # Rating improves (no adjustment implemented)
    (0.05, 5, 4, 0.05),  # Rating declines (no adjustment implemented)
    (0.10, 10, 5, 0.10), # Large rating change
    (0.0, 5, 5, 0.0),    # Zero base rate
])
def test_adjust_discount_rate_for_credit_risk(base_rate, rating_before, rating_after, expected):
    assert adjust_discount_rate_for_credit_risk(base_rate, rating_before, rating_after) == expected
