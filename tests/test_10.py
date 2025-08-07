import pytest
from definition_f3fe7f396da44415948be8b788c04367 import calculate_loan_payment

@pytest.mark.parametrize("principal, rate, term_months, expected", [
    (100000, 0.05, 360, 536.82),  # Typical mortgage calculation
    (5000, 0.10, 12, 439.56),  # Short-term loan
    (1000, 0.00, 60, 16.67),  # Zero-interest loan
    (100000, 0.05, 0, ZeroDivisionError),  # Zero term
    (100000, -0.05, 360, ValueError), # Negative rate
])
def test_calculate_loan_payment(principal, rate, term_months, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            calculate_loan_payment(principal, rate, term_months)
    else:
        assert round(calculate_loan_payment(principal, rate, term_months), 2) == expected
