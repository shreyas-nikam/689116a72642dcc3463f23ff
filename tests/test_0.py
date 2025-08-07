import pytest
import pandas as pd
import numpy as np

# Placeholder for your module import
from definition_47512f0e44a346b0ab2c30f9ba3f47f2 import load_raw

def test_load_raw_returns_dataframe_and_correct_size():
    """
    Test that load_raw returns a pandas DataFrame and has approximately 10 rows.
    """
    df = load_raw()
    assert isinstance(df, pd.DataFrame), "load_raw should return a pandas DataFrame."
    assert 8 <= len(df) <= 12, "DataFrame should have approximately 10 rows (e.g., between 8 and 12)."

def test_load_raw_contains_all_required_columns():
    """
    Test that the DataFrame contains all specified input columns as per the specification.
    """
    df = load_raw()
    expected_columns = [
        'loan_id', 'orig_principal', 'orig_rate', 'orig_term_mths',
        'pay_freq', 'restructure_date', 'new_rate', 'new_term_mths',
        'principal_haircut_pct', 'rating_before', 'rating_after'
    ]
    assert all(col in df.columns for col in expected_columns), \
        f"Missing required columns. Expected: {expected_columns}, Got: {df.columns.tolist()}"

def test_load_raw_has_both_restructured_and_non_restructured_loans():
    """
    Test that the synthetic dataset includes a mix of restructured and non-restructured loans,
    as specified (3-5 restructured, others non-restructured).
    """
    df = load_raw()
    # A loan is considered restructured if 'restructure_date' is not null/NaT
    restructured_loans_count = df['restructure_date'].notna().sum()
    non_restructured_loans_count = df['restructure_date'].isna().sum()

    assert restructured_loans_count >= 3, "Expected at least 3 restructured loans."
    assert non_restructured_loans_count >= 3, "Expected at least 3 non-restructured loans."
    assert (restructured_loans_count + non_restructured_loans_count) == len(df), "Loan counts do not sum to total."

def test_load_raw_non_restructured_loan_properties_edge_case():
    """
    Test that for non-restructured loans, specific fields correctly indicate 'no change'
    (e.g., None/NaN or original values). This is a crucial edge case.
    """
    df = load_raw()
    # Filter for non-restructured loans (restructure_date is NaN or NaT)
    non_restructured_df = df[df['restructure_date'].isna()]

    assert not non_restructured_df.empty, "There should be non-restructured loans to test this edge case."

    for index, row in non_restructured_df.iterrows():
        # restructure_date must be NaN/NaT
        assert pd.isna(row['restructure_date']), \
            f"restructure_date should be NaN for non-restructured loan {row['loan_id']}"
        
        # new_rate should equal orig_rate (or both NaN/None)
        if pd.isna(row['orig_rate']):
            assert pd.isna(row['new_rate']), \
                f"new_rate should be NaN if orig_rate is NaN for non-restructured loan {row['loan_id']}"
        else:
            assert np.isclose(row['new_rate'], row['orig_rate'], equal_nan=True), \
                f"new_rate ({row['new_rate']}) should equal orig_rate ({row['orig_rate']}) for non-restructured loan {row['loan_id']}"

        # new_term_mths should equal orig_term_mths (or both NaN/None)
        if pd.isna(row['orig_term_mths']):
            assert pd.isna(row['new_term_mths']), \
                f"new_term_mths should be NaN if orig_term_mths is NaN for non-restructured loan {row['loan_id']}"
        else:
            assert row['new_term_mths'] == row['orig_term_mths'], \
                f"new_term_mths ({row['new_term_mths']}) should equal orig_term_mths ({row['orig_term_mths']}) for non-restructured loan {row['loan_id']}"
        
        # principal_haircut_pct should be 0 or NaN/None
        assert np.isclose(row['principal_haircut_pct'], 0.0, equal_nan=True) or pd.isna(row['principal_haircut_pct']), \
            f"principal_haircut_pct ({row['principal_haircut_pct']}) should be 0 or NaN for non-restructured loan {row['loan_id']}"

def test_load_raw_loan_id_uniqueness_and_basic_value_validity():
    """
    Test that loan_id are unique and key numeric/categorical columns have realistic types and values.
    """
    df = load_raw()
    assert df['loan_id'].is_unique, "loan_id column should contain unique identifiers."
    
    # Check data types and positivity for key numeric columns
    numeric_cols_to_check = ['orig_principal', 'orig_rate', 'orig_term_mths']
    for col in numeric_cols_to_check:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' should be numeric."
        assert (df[col].dropna() >= 0).all(), f"Column '{col}' should contain non-negative values."

    # Check 'pay_freq' type and non-null
    assert pd.api.types.is_string_dtype(df['pay_freq']) or pd.api.types.is_object_dtype(df['pay_freq']), \
        "Column 'pay_freq' should be string/object type."
    assert not df['pay_freq'].isnull().any(), "Column 'pay_freq' should not contain null values."

    # Check 'rating_before' and 'rating_after' types and range (1-5 scale)
    rating_cols = ['rating_before', 'rating_after']
    for col in rating_cols:
        assert pd.api.types.is_integer_dtype(df[col]), f"Column '{col}' should be integer type."
        assert df[col].between(1, 5, inclusive='both').all(), \
            f"Column '{col}' values ({df[col].min()}-{df[col].max()}) should be between 1 and 5."