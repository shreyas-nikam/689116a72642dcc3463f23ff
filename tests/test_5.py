import pytest
import pandas as pd
from definition_57e4b0f62a7a4aa68954ac2d379d5f9c import calculate_npv

@pytest.fixture
def sample_loan_data():
    data = {
        'loan_id': [1, 2, 3],
        'orig_principal': [100000, 200000, 300000],
        'orig_rate': [0.05, 0.06, 0.07],
        'orig_term_mths': [36, 48, 60]
    }
    return pd.DataFrame(data)

def test_calculate_npv_empty_data(sample_loan_data):
    discount_rates = pd.Series([0.05, 0.06, 0.07], index=sample_loan_data['loan_id'])
    sample_loan_data = sample_loan_data.iloc[0:0]
    result = calculate_npv(sample_loan_data, discount_rates)
    assert result.empty

def test_calculate_npv_basic(sample_loan_data):
    discount_rates = pd.Series([0.05, 0.06, 0.07], index=sample_loan_data['loan_id'])
    result = calculate_npv(sample_loan_data, discount_rates)
    assert 'loan_id' in result.columns
    assert 'NPV_orig' in result.columns
    assert 'NPV_new' in result.columns
    assert 'Delta_NPV' in result.columns
    assert len(result) == len(sample_loan_data)

def test_calculate_npv_with_zero_discount_rate(sample_loan_data):
    discount_rates = pd.Series([0, 0, 0], index=sample_loan_data['loan_id'])
    result = calculate_npv(sample_loan_data, discount_rates)
    assert 'loan_id' in result.columns
    assert 'NPV_orig' in result.columns
    assert 'NPV_new' in result.columns
    assert 'Delta_NPV' in result.columns
    assert len(result) == len(sample_loan_data)

def test_calculate_npv_different_index(sample_loan_data):
    discount_rates = pd.Series([0.05, 0.06, 0.07], index=[4,5,6])
    with pytest.raises(KeyError):
        calculate_npv(sample_loan_data, discount_rates)

def test_calculate_npv_incorrect_discount_rates_type(sample_loan_data):
    with pytest.raises(TypeError):
        calculate_npv(sample_loan_data, [0.05, 0.06, 0.07])
