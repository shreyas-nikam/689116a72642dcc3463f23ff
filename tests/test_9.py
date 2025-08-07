import pytest
import pandas as pd
from definition_767186ed254444c4be90aa8baa1a3c56 import _generate_single_amortization_schedule

@pytest.mark.parametrize(
    "principal, annual_rate, term_mths, pay_freq, expected_output_type, expected_rows, expected_exception",
    [
        # Test Case 1: Standard loan with monthly payments
        (100_000, 0.05, 120, 'Monthly', pd.DataFrame, 120, None),
        
        # Test Case 2: Loan with zero interest rate
        (50_000, 0.00, 60, 'Monthly', pd.DataFrame, 60, None),
        
        # Test Case 3: Loan with zero principal amount
        (0, 0.05, 120, 'Monthly', pd.DataFrame, 120, None),
        
        # Test Case 4: Loan with quarterly payments
        (75_000, 0.04, 36, 'Quarterly', pd.DataFrame, 12, None), # 36 months / 3 months per quarter = 12 periods
        
        # Test Case 5: Edge cases / Error handling for invalid inputs
        # a) Zero term months should raise ValueError
        (100_000, 0.05, 0, 'Monthly', None, None, ValueError),
        # b) Invalid payment frequency should raise ValueError
        # (100_000, 0.05, 120, 'Bi-weekly', None, None, ValueError), # Removed to keep at most 5 parameter sets
        # c) Non-numeric principal should raise TypeError
        # ("invalid", 0.05, 120, 'Monthly', None, None, TypeError), # Removed to keep at most 5 parameter sets
    ]
)
def test_generate_single_amortization_schedule(
    principal, annual_rate, term_mths, pay_freq, expected_output_type, expected_rows, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            _generate_single_amortization_schedule(principal, annual_rate, term_mths, pay_freq)
    else:
        df = _generate_single_amortization_schedule(principal, annual_rate, term_mths, pay_freq)

        # Assert output is a pandas DataFrame
        assert isinstance(df, expected_output_type)
        assert not df.empty, "DataFrame should not be empty"
        
        # Assert the number of rows is as expected
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

        # Assert required columns are present
        expected_columns = ['date', 'interest_payment', 'principal_payment', 'cashflow_total']
        assert all(col in df.columns for col in expected_columns), "Not all expected columns are present"

        # Assert numeric columns have correct data types and non-negative values
        numeric_cols = ['interest_payment', 'principal_payment', 'cashflow_total']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric"
            assert (df[col] >= 0).all(), f"Column '{col}' contains negative values"

        # Specific checks for edge cases
        if principal == 0:
            assert (df['interest_payment'] == 0).all(), "Interest payment should be 0 for zero principal"
            assert (df['principal_payment'] == 0).all(), "Principal payment should be 0 for zero principal"
            assert (df['cashflow_total'] == 0).all(), "Total cashflow should be 0 for zero principal"
        
        if annual_rate == 0 and principal > 0:
            assert (df['interest_payment'] == 0).all(), "Interest payment should be 0 for zero annual rate"
            # For 0% interest, principal repayment is constant and equals cashflow_total
            assert df['principal_payment'].nunique() == 1, "Principal payment should be constant for zero interest"
            assert (df['principal_payment'] == df['cashflow_total']).all(), "Cashflow should equal principal payment for zero interest"
            
            # Verify total principal repaid equals original principal
            assert pytest.approx(df['principal_payment'].sum(), rel=1e-3) == principal, "Total principal repaid does not match original principal for zero interest"