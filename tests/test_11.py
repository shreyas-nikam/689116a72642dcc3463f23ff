import pytest
from definition_c4e0d347ccf14eff819aebefdc78168b import _calculate_credit_spread_adjustment

@pytest.mark.parametrize("rating_worsened, base_spread_bp, expected", [
    # Test case 1: Standard scenario - rating worsened, positive base spread
    (True, 100, 0.01),
    # Test case 2: Rating not worsened - spread should be zero
    (False, 250, 0.0),
    # Test case 3: Edge case - rating worsened, but base spread is zero
    (True, 0, 0.0),
    # Test case 4: Edge case - rating worsened, large base spread
    (True, 5000, 0.5),
    # Test case 5: Edge case - incorrect type for base_spread_bp (should raise TypeError)
    (True, "200", TypeError),
])
def test_calculate_credit_spread_adjustment(rating_worsened, base_spread_bp, expected):
    try:
        result = _calculate_credit_spread_adjustment(rating_worsened, base_spread_bp)
        # Ensure the result is a float
        assert isinstance(result, float)
        # Use pytest.approx for float comparisons to account for potential precision issues
        assert result == pytest.approx(expected)
    except Exception as e:
        # If an exception is expected, check its type
        assert isinstance(e, expected)