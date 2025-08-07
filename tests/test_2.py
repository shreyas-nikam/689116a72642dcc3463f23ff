import pytest
import pandas as pd
from definition_07ac11cdcf19440d93957e44c03a1c06 import calc_discount_rate

@pytest.fixture
def sample_loan_data():
    data = {
        'loan_id': [1, 2, 3, 4, 5],
        'orig_principal': [100000, 200000, 150000, 300000, 250000],
        'orig_rate': [0.05, 0.06, 0.04, 0.07, 0.055],
        'orig_term_mths': [36, 60, 48, 72, 42],
        'pay_freq': ['monthly', 'monthly', 'monthly', 'monthly', 'monthly'],
        'restructure_date': ['2024-01-01', '2024-01-01', None, '2024-01-01', None],
        'new_rate': [0.04, 0.05, None, 0.06, None],
        'new_term_mths': [48, 72, None, 84, None],
        'principal_haircut_pct': [0.0, 0.05, None, 0.10, None],
        'rating_before': ['A', 'B', 'C', 'A', 'B'],
        'rating_after': ['A', 'B', 'C', 'B', 'B']
    }
    return pd.DataFrame(data)


def test_calc_discount_rate_no_restructure(sample_loan_data):
    loan_data = sample_loan_data.copy()
    loan_data['rating_before'] = 'A'
    loan_data['rating_after'] = 'A'
    discount_rates = calc_discount_rate(loan_data)
    assert isinstance(discount_rates, pd.Series)
    assert len(discount_rates) == len(loan_data)
    assert all(discount_rates.isna() == False)


def test_calc_discount_rate_with_restructure(sample_loan_data):
    discount_rates = calc_discount_rate(sample_loan_data)
    assert isinstance(discount_rates, pd.Series)
    assert len(discount_rates) == len(sample_loan_data)
    assert all(discount_rates.isna() == False)



def test_calc_discount_rate_empty_dataframe():
    loan_data = pd.DataFrame()
    discount_rates = calc_discount_rate(loan_data)
    assert isinstance(discount_rates, pd.Series)
    assert len(discount_rates) == 0


def test_calc_discount_rate_null_values(sample_loan_data):
    loan_data = sample_loan_data.copy()
    loan_data.loc[0, 'orig_rate'] = None
    with pytest.raises(TypeError):
         calc_discount_rate(loan_data)

def test_calc_discount_rate_invalid_rating(sample_loan_data):

    loan_data = sample_loan_data.copy()
    loan_data['rating_before'] = [1,2,3,4,5]
    loan_data['rating_after'] = [1,2,3,4,5]

    with pytest.raises(TypeError):
        calc_discount_rate(loan_data)
