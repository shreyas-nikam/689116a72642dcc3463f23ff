import pytest
import math
from definition_0d070d2e202f40f0bec06f92ce970991 import calculate_npv

@pytest.mark.parametrize("cashflows, discount_rate, expected", [
    # Test Case 1: Standard positive cash flows and positive discount rate
    # NPV = 100/(1.1)^1 + 100/(1.1)^2 + 100/(1.1)^3 = 90.90909 + 82.64463 + 75.13148 = 248.6852
    ([100, 100, 100], 0.10, pytest.approx(248.68519884, rel=1e-9)),
    
    # Test Case 2: Empty cash flow list - NPV should be 0
    ([], 0.05, 0),
    
    # Test Case 3: Zero discount rate - NPV should be sum of cash flows
    # NPV = 50/(1.0)^1 + 75/(1.0)^2 + 25/(1.0)^3 = 50 + 75 + 25 = 150
    ([50, 75, 25], 0.0, 150),
    
    # Test Case 4: Cash flows include negative values
    # NPV = -50/(1.05)^1 + 100/(1.05)^2 + 100/(1.05)^3 = -47.61905 + 90.70293 + 86.38376 = 129.46764
    ([-50, 100, 100], 0.05, pytest.approx(129.46763895, rel=1e-9)),
    
    # Test Case 5: Discount rate of -1.0 should lead to ZeroDivisionError (1+r = 0)
    ([100], -1.0, ZeroDivisionError),
])
def test_calculate_npv(cashflows, discount_rate, expected):
    try:
        result = calculate_npv(cashflows, discount_rate)
        assert result == expected
    except Exception as e:
        assert isinstance(e, expected)

