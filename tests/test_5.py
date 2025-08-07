import pytest
import pandas as pd
from definition_4ccfe977efdc40d7a636573fb35df8c1 import calculate_npv

@pytest.fixture
def sample_loan_data():
    data = {
        'loan_id': [1, 2, 3],
        'orig_principal': [100000, 200000, 300000],
        'orig_rate': [0.05, 0.06, 0.07],
        'orig_term_mths': [36, 48, 60]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_discount_rates(sample_loan_data):
    return pd.Series([0.05, 0.06, 0.07], index=sample_loan_data.index)

def test_calculate_npv_basic(sample_loan_data, sample_discount_rates):
    result = calculate_npv(sample_loan_data, sample_discount_rates)
    assert isinstance(result, pd.DataFrame)
    assert 'loan_id' in result.columns
    assert 'NPV_orig' in result.columns
    assert 'NPV_new' in result.columns
    assert 'Delta_NPV' in result.columns

def test_calculate_npv_empty_loan_data():
    empty_data = pd.DataFrame()
    discount_rates = pd.Series([])
    result = calculate_npv(empty_data, discount_rates)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_calculate_npv_mismatched_index(sample_loan_data):
    discount_rates = pd.Series([0.05, 0.06], index=[0, 1])  # Different index length
    with pytest.raises(ValueError):
        calculate_npv(sample_loan_data, discount_rates)

def test_calculate_npv_zero_principal(sample_loan_data, sample_discount_rates):
    sample_loan_data.loc[0, 'orig_principal'] = 0
    result = calculate_npv(sample_loan_data, sample_discount_rates)
    assert result['NPV_orig'][0] == 0.0  # NPV should be zero for zero principal

def test_calculate_npv_negative_interest_rate(sample_loan_data, sample_discount_rates):
    sample_loan_data.loc[0, 'orig_rate'] = -0.05
    with pytest.raises(ValueError): # Expecting a ValueError for negative interest rates
        calculate_npv(sample_loan_data, sample_discount_rates)
