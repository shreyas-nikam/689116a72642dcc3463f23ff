import pytest
import pandas as pd
from definition_a4b93386e65a4ae6ae703242a9f1421b import generate_synthetic_data

def test_generate_synthetic_data_positive_num_loans():
    num_loans = 5
    df = generate_synthetic_data(num_loans)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == num_loans
    assert 'loan_id' in df.columns
    assert 'orig_principal' in df.columns
    assert 'orig_rate' in df.columns

def test_generate_synthetic_data_zero_num_loans():
    num_loans = 0
    df = generate_synthetic_data(num_loans)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0

def test_generate_synthetic_data_columns_present():
    num_loans = 1
    df = generate_synthetic_data(num_loans)
    expected_columns = ['loan_id', 'orig_principal', 'orig_rate', 'orig_term_mths', 'pay_freq', 'restructure_date', 'new_rate', 'new_term_mths', 'principal_haircut_pct', 'rating_before', 'rating_after']
    for col in expected_columns:
        assert col in df.columns

def test_generate_synthetic_data_data_types():
    num_loans = 1
    df = generate_synthetic_data(num_loans)
    assert df['orig_principal'].dtype == 'float64'
    assert df['orig_rate'].dtype == 'float64'
    assert df['orig_term_mths'].dtype == 'int64'

def test_generate_synthetic_data_negative_num_loans():
    with pytest.raises(ValueError):
        generate_synthetic_data(-1)
