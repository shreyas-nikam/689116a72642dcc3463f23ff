import pytest
import pandas as pd
from definition_d23e7b1e15574c7e959260caca3a27a0 import tidy_merge

@pytest.fixture
def sample_dataframes():
    # Create sample dataframes for testing
    cf_orig = pd.DataFrame({
        'loan_id': [1, 1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'cashflow': [100, 100, 100],
        'type': ['original'] * 3
    })
    cf_new = pd.DataFrame({
        'loan_id': [1, 1],
        'date': pd.to_datetime(['2024-04-01', '2024-05-01']),
        'cashflow': [150, 150],
        'type': ['restructured'] * 2
    })
    return cf_orig, cf_new

def test_tidy_merge_basic(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    merged_df = tidy_merge(cf_orig, cf_new)
    assert isinstance(merged_df, pd.DataFrame)
    assert len(merged_df) == 5  # Total rows from both dataframes
    assert 'loan_id' in merged_df.columns
    assert 'date' in merged_df.columns
    assert 'cashflow' in merged_df.columns
    assert 'type' in merged_df.columns

def test_tidy_merge_empty_cf_orig(sample_dataframes):
    _, cf_new = sample_dataframes
    cf_orig_empty = pd.DataFrame(columns=cf_new.columns)
    merged_df = tidy_merge(cf_orig_empty, cf_new)
    assert len(merged_df) == len(cf_new)
    assert merged_df.equals(cf_new)

def test_tidy_merge_empty_cf_new(sample_dataframes):
    cf_orig, _ = sample_dataframes
    cf_new_empty = pd.DataFrame(columns=cf_orig.columns)
    merged_df = tidy_merge(cf_orig, cf_new_empty)
    assert len(merged_df) == len(cf_orig)
    assert merged_df.equals(cf_orig)

def test_tidy_merge_no_shared_columns(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    cf_orig = cf_orig.rename(columns={'cashflow': 'orig_cashflow'})
    cf_new = cf_new.rename(columns={'cashflow': 'new_cashflow'})

    with pytest.raises(KeyError): # Expect a KeyError if the columns are not the same, since pd.concat will fail
        tidy_merge(cf_orig, cf_new)

def test_tidy_merge_different_loan_ids(sample_dataframes):
    cf_orig, cf_new = sample_dataframes
    cf_new['loan_id'] = 2
    merged_df = tidy_merge(cf_orig, cf_new)
    assert len(merged_df) == 5
    assert len(merged_df['loan_id'].unique()) == 2
