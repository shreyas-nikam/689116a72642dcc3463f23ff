import pytest
import pandas as pd
from definition_577c709badc64ad49f24aec1985bb57d import expand_cashflows

@pytest.fixture
def sample_loan_data():
    data = {'loan_id': [1],
            'orig_principal': [100000],
            'orig_rate': [0.05],
            'orig_term_mths': [36],
            'pay_freq': ['monthly'],
            'restructure_date': ['2024-01-01'],
            'new_rate': [0.04],
            'new_term_mths': [48],
            'principal_haircut_pct': [0.0],
            'rating_before': [700],
            'rating_after': [720]}
    return pd.DataFrame(data)

def test_expand_cashflows_empty_dataframe():
    df = pd.DataFrame()
    cf_orig, cf_new = expand_cashflows(df)
    assert cf_orig.empty
    assert cf_new.empty

def test_expand_cashflows_single_loan(sample_loan_data):
    cf_orig, cf_new = expand_cashflows(sample_loan_data)
    assert isinstance(cf_orig, pd.DataFrame)
    assert isinstance(cf_new, pd.DataFrame)
    assert not cf_orig.empty
    assert not cf_new.empty
    assert 'date' in cf_orig.columns
    assert 'interest' in cf_orig.columns
    assert 'principal' in cf_orig.columns
    assert 'cashflow' in cf_orig.columns
    assert 'date' in cf_new.columns
    assert 'interest' in cf_new.columns
    assert 'principal' in cf_new.columns
    assert 'cashflow' in cf_new.columns
    assert len(cf_orig) == 36
    assert len(cf_new) == 48


def test_expand_cashflows_no_restructure_date(sample_loan_data):
    sample_loan_data['restructure_date'] = None
    cf_orig, cf_new = expand_cashflows(sample_loan_data)
    assert isinstance(cf_orig, pd.DataFrame)
    assert isinstance(cf_new, pd.DataFrame)
    assert not cf_orig.empty
    assert cf_new.empty  # New cashflow should be empty when restructure date is not populated
    assert 'date' in cf_orig.columns
    assert 'interest' in cf_orig.columns
    assert 'principal' in cf_orig.columns
    assert 'cashflow' in cf_orig.columns

def test_expand_cashflows_negative_interest_rate(sample_loan_data):
    sample_loan_data['orig_rate'] = -0.05
    with pytest.raises(ValueError):
        expand_cashflows(sample_loan_data)

def test_expand_cashflows_zero_principal(sample_loan_data):
    sample_loan_data['orig_principal'] = 0
    cf_orig, cf_new = expand_cashflows(sample_loan_data)
    assert isinstance(cf_orig, pd.DataFrame)
    assert isinstance(cf_new, pd.DataFrame)
    assert not cf_orig.empty
    assert not cf_new.empty
    assert cf_orig['principal'].sum() == 0
    assert cf_new['principal'].sum() == 0
