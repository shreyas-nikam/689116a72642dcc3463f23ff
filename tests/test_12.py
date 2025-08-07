import pytest
from definition_8276418d095c4122a841ba7de4153ace import calculate_loan_payment

@pytest.mark.parametrize("principal, rate, term_months, expected", [
    (100000, 0.05, 360, 536.82),  # Typical loan
    (0, 0.05, 360, 0), # Zero principal
    (100000, 0, 360, 277.78), # Zero interest rate
    (100000, 0.05, 0, float('inf')), # Zero term
    (100000, 0.05, 360.5, TypeError) #Non-integer term
])
def test_calculate_loan_payment(principal, rate, term_months, expected):
    try:
      if expected == float('inf'):
        with pytest.raises(ZeroDivisionError):
            calculate_loan_payment(principal, rate, term_months)
      else:
        assert round(calculate_loan_payment(principal, rate, term_months), 2) == expected
    except Exception as e:
        assert isinstance(e, expected)
