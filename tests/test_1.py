import pytest
import pandas as pd
import numpy as np
from definition_3a4d423330184a81954a6637a5978c0c import expand_cashflows

@pytest.fixture
def base_df_loans():
    """
    Provides a DataFrame with various loan scenarios:
    1. Restructured loan (monthly, with principal haircut).
    2. Non-restructured loan (monthly, no haircut, new terms are effectively original).
    3. Restructured loan (quarterly, with principal haircut).
    """
    data = {
        'loan_id': [1, 2, 3],
        'orig_principal': [100000.0, 50000.0, 75000.0],
        'orig_rate': [0.05, 0.04, 0.06],  # Annual rate
        'orig_term_mths': [60, 36, 48],
        'pay_freq': ['Monthly', 'Monthly', 'Quarterly'],
        'restructure_date': [pd.Timestamp('2023-01-01'), pd.NaT, pd.Timestamp('2024-03-01')],
        'new_rate': [0.06, np.nan, 0.05],
        'new_term_mths': [72, np.nan, 36],
        'principal_haircut_pct': [0.10, 0.0, 0.05],
        'rating_before': [3, 2, 3],
        'rating_after': [4, 2, 3],
    }
    df = pd.DataFrame(data)
    # Ensure numeric types, especially after NaNs might convert to object type
    df['new_rate'] = pd.to_numeric(df['new_rate'])
    df['new_term_mths'] = pd.to_numeric(df['new_term_mths'])
    df['principal_haircut_pct'] = pd.to_numeric(df['principal_haircut_pct'])
    return df

def get_expected_num_payments(term_mths, pay_freq):
    """Helper to calculate expected number of payments based on frequency."""
    if pay_freq == 'Monthly':
        return term_mths
    elif pay_freq == 'Quarterly':
        return term_mths // 3
    # Add other frequencies if specified in function's internal logic, otherwise default/error
    raise ValueError(f"Unsupported payment frequency: {pay_freq}")


def test_expand_cashflows_restructured_monthly_loan(base_df_loans):
    """
    Test case 1: Verify correct amortization for a monthly, restructured loan with principal haircut.
    Checks output DataFrame structure, length, and cashflow sum property.
    """
    df_loans_subset = base_df_loans.iloc[[0]].copy() # Loan 1: Monthly, restructured, haircut
    cf_orig, cf_new = expand_cashflows(df_loans_subset)

    assert isinstance(cf_orig, pd.DataFrame)
    assert isinstance(cf_new, pd.DataFrame)
    
    expected_cols = ['loan_id', 'date', 'interest', 'principal', 'cashflow']
    assert all(col in cf_orig.columns for col in expected_cols)
    assert all(col in cf_new.columns for col in expected_cols)

    loan_data = df_loans_subset.iloc[0]
    assert len(cf_orig) == get_expected_num_payments(loan_data['orig_term_mths'], loan_data['pay_freq'])
    assert len(cf_new) == get_expected_num_payments(loan_data['new_term_mths'], loan_data['pay_freq'])

    # Check that cashflow column is sum of interest and principal
    pd.testing.assert_series_equal(cf_orig['cashflow'], cf_orig['interest'] + cf_orig['principal'], check_dtype=False, check_names=False)
    pd.testing.assert_series_equal(cf_new['cashflow'], cf_new['interest'] + cf_new['principal'], check_dtype=False, check_names=False)

    # Ensure no negative cash flow components (interest, principal, total cashflow)
    assert (cf_orig[['interest', 'principal', 'cashflow']] >= -1e-6).all().all() # Allow tiny negative due to float precision
    assert (cf_new[['interest', 'principal', 'cashflow']] >= -1e-6).all().all()
    
    # Assert that restructured cashflows are different from original given changes in terms/haircut
    assert not cf_orig.equals(cf_new)


def test_expand_cashflows_no_restructuring(base_df_loans):
    """
    Test case 2: Verify that for a non-restructured loan, cf_orig and cf_new are identical.
    This implies the function correctly handles cases where new terms are effectively same as original.
    """
    df_loans_subset = base_df_loans.iloc[[1]].copy() # Loan 2: Monthly, no restructuring (principal_haircut_pct = 0, new terms NaN)
    cf_orig, cf_new = expand_cashflows(df_loans_subset)

    # For non-restructured loans, cf_new should be identical to cf_orig (within float tolerance)
    pd.testing.assert_frame_equal(cf_orig, cf_new, check_dtype=True, check_exact=False, rtol=1e-5)

    loan_data = df_loans_subset.iloc[0]
    assert len(cf_orig) == get_expected_num_payments(loan_data['orig_term_mths'], loan_data['pay_freq'])
    assert len(cf_new) == get_expected_num_payments(loan_data['orig_term_mths'], loan_data['pay_freq'])


def test_expand_cashflows_restructured_quarterly_loan(base_df_loans):
    """
    Test case 3: Verify correct amortization for a quarterly, restructured loan with principal haircut.
    Checks output DataFrame structure, length (based on quarterly frequency), and cashflow sum property.
    """
    df_loans_subset = base_df_loans.iloc[[2]].copy() # Loan 3: Quarterly, restructured, haircut
    cf_orig, cf_new = expand_cashflows(df_loans_subset)

    assert isinstance(cf_orig, pd.DataFrame)
    assert isinstance(cf_new, pd.DataFrame)

    expected_cols = ['loan_id', 'date', 'interest', 'principal', 'cashflow']
    assert all(col in cf_orig.columns for col in expected_cols)
    assert all(col in cf_new.columns for col in expected_cols)

    loan_data = df_loans_subset.iloc[0]
    assert len(cf_orig) == get_expected_num_payments(loan_data['orig_term_mths'], loan_data['pay_freq'])
    assert len(cf_new) == get_expected_num_payments(loan_data['new_term_mths'], loan_data['pay_freq'])

    pd.testing.assert_series_equal(cf_orig['cashflow'], cf_orig['interest'] + cf_orig['principal'], check_dtype=False, check_names=False)
    pd.testing.assert_series_equal(cf_new['cashflow'], cf_new['interest'] + cf_new['principal'], check_dtype=False, check_names=False)

    assert (cf_orig[['interest', 'principal', 'cashflow']] >= -1e-6).all().all()
    assert (cf_new[['interest', 'principal', 'cashflow']] >= -1e-6).all().all()
    
    assert not cf_orig.equals(cf_new)


@pytest.mark.parametrize("invalid_input", [
    None,
    "not a dataframe",
    123,
    pd.Series([1, 2, 3])
])
def test_expand_cashflows_invalid_input_type(invalid_input):
    """
    Test case 4: Ensure the function raises a TypeError for non-DataFrame inputs.
    """
    with pytest.raises(TypeError):
        expand_cashflows(invalid_input)


@pytest.mark.parametrize("missing_column", [
    'orig_principal', 'orig_rate', 'orig_term_mths', 'pay_freq',
    'restructure_date', 'new_rate', 'new_term_mths', 'principal_haircut_pct'
])
def test_expand_cashflows_missing_required_columns(base_df_loans, missing_column):
    """
    Test case 5: Verify that the function raises an appropriate error when essential columns are missing.
    Expected errors could be KeyError (direct column access) or ValueError/AttributeError (during validation/processing).
    """
    df_loans_missing_col = base_df_loans.drop(columns=[missing_column]).copy()
    with pytest.raises((KeyError, AttributeError, ValueError)):
        expand_cashflows(df_loans_missing_col)