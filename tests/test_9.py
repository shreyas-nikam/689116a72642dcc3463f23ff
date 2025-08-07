import pytest
import pandas as pd
from definition_2376707ea15e4008980770f8b525a4c0 import generate_synthetic_loan_data

def test_generate_synthetic_loan_data_positive_num_loans():
    """Test that the function returns a Pandas DataFrame when num_loans is positive."""
    num_loans = 5
    result = generate_synthetic_loan_data(num_loans)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == num_loans

def test_generate_synthetic_loan_data_zero_num_loans():
    """Test that the function returns an empty Pandas DataFrame when num_loans is zero."""
    num_loans = 0
    result = generate_synthetic_loan_data(num_loans)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_generate_synthetic_loan_data_column_presence():
    """Test that the function returns a Pandas DataFrame with the correct columns."""
    num_loans = 1
    result = generate_synthetic_loan_data(num_loans)
    expected_columns = ['loan_id', 'orig_principal', 'orig_rate', 'orig_term_mths', 'pay_freq', 'restructure_date', 'new_rate', 'new_term_mths', 'principal_haircut_pct', 'rating_before', 'rating_after']
    for col in expected_columns:
        assert col in result.columns

def test_generate_synthetic_loan_data_data_types():
    """Test that the data types of the columns in the Pandas DataFrame are correct."""
    num_loans = 1
    result = generate_synthetic_loan_data(num_loans)
    assert result['orig_principal'].dtype == 'float64'
    assert result['orig_rate'].dtype == 'float64'
    assert result['orig_term_mths'].dtype == 'int64'

def test_generate_synthetic_loan_data_edge_case_large_num_loans():
    """Test that the function handles a large number of loans without errors."""
    num_loans = 1000
    result = generate_synthetic_loan_data(num_loans)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == num_loans
