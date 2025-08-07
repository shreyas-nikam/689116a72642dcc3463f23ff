import pytest
import pandas as pd
from definition_27a3c0e6963f40b9932386e7362651e8 import calc_discount_rate

@pytest.fixture
def sample_loan_data():
    data = {
        'loan_id': [1, 2, 3],
        'orig_rate': [0.05, 0.06, 0.07],
        'rating_before': [6, 7, 8],
        'rating_after': [7, 8, 9]
    }
    return pd.DataFrame(data)

def test_calc_discount_rate_no_change(sample_loan_data):
    # Test case where the ratings don't change, discount rate is same as original rate
    sample_loan_data['rating_after'] = sample_loan_data['rating_before']
    discount_rates = calc_discount_rate(sample_loan_data)
    assert (discount_rates == sample_loan_data['orig_rate']).all()

def test_calc_discount_rate_with_change(sample_loan_data):
    # Test case where the ratings change, resulting in different discount rates
    # Assuming that a ratings upgrade decreases the discount rate by a constant value
    discount_rates = calc_discount_rate(sample_loan_data)
    expected_rates = pd.Series([0.05, 0.06, 0.07], dtype='float64')
    assert (discount_rates.fillna(0) == expected_rates.fillna(0)).all()

def test_calc_discount_rate_empty_dataframe():
    # Test case with an empty DataFrame
    empty_df = pd.DataFrame()
    discount_rates = calc_discount_rate(empty_df)
    assert isinstance(discount_rates, pd.Series)
    assert discount_rates.empty

def test_calc_discount_rate_missing_columns():
    # Test case where the DataFrame is missing required columns.
    data = {'loan_id': [1, 2, 3]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
        calc_discount_rate(df)

def test_calc_discount_rate_nan_values(sample_loan_data):
    # Test case where there are NaN values in the rating columns
    sample_loan_data['rating_before'][0] = float('nan')
    sample_loan_data['rating_after'][1] = float('nan')
    discount_rates = calc_discount_rate(sample_loan_data)
    assert discount_rates.isnull().any()
