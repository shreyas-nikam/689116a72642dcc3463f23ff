import pytest
import pandas as pd
from definition_9d88f5f9115e4b1085c4ec6ec35bae59 import tidy_merge

@pytest.fixture
def sample_dataframes():
    cf_orig = pd.DataFrame({
        'loan_id': [1, 1, 1],
        'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'cashflow': [100, 100, 100]
    })
    cf_new = pd.DataFrame({
        'loan_id': [1, 1],
        'date': ['2024-01-01', '2024-02-01'],
        'cashflow': [150, 150]
    })
    return cf_orig, cf_new

def test_tidy_merge_basic(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    result = tidy_merge(cf_orig, cf_new)
    assert isinstance(result, pd.DataFrame)

def test_tidy_merge_column_names(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    result = tidy_merge(cf_orig, cf_new)
    expected_columns = ['loan_id', 'date', 'cashflow', 'type']
    assert all(col in result.columns for col in expected_columns)

def test_tidy_merge_data_types(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    result = tidy_merge(cf_orig, cf_new)
    assert result['loan_id'].dtype == 'int64'
    assert result['date'].dtype == 'object'  # Expecting string due to mixed input
    assert result['cashflow'].dtype == 'int64'
    assert result['type'].dtype == 'object'

def test_tidy_merge_empty_dataframe():
    cf_orig = pd.DataFrame({'loan_id': [], 'date': [], 'cashflow': []})
    cf_new = pd.DataFrame({'loan_id': [], 'date': [], 'cashflow': []})

    result = tidy_merge(cf_orig, cf_new)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_tidy_merge_different_loan_ids(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    cf_new['loan_id'] = 2
    with pytest.raises(ValueError):
        tidy_merge(cf_orig, cf_new)
