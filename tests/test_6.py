import pytest
import pandas as pd
from definition_1436a12fe8a540bf84f3dbb794bdd053 import assess_materiality

@pytest.fixture
def sample_npv_results():
    data = {
        'loan_id': [1, 2, 3, 4, 5],
        'NPV_orig': [100000, 200000, 30000, 400000, 500000],
        'NPV_new': [120000, 150000, 20000, 450000, 480000],
        'Delta_NPV': [20000, -50000, -10000, 50000, -20000]
    }
    return pd.DataFrame(data)

def test_assess_materiality_positive(sample_npv_results):
    result = assess_materiality(sample_npv_results)
    assert 'material' in result.columns
    assert result['material'].dtype == bool
    assert result['material'][0] == False
    assert result['material'][1] == True
    assert result['material'][3] == False

def test_assess_materiality_negative(sample_npv_results):
    result = assess_materiality(sample_npv_results)
    assert 'material' in result.columns
    assert result['material'].dtype == bool
    assert result['material'][2] == False
    assert result['material'][4] == False

def test_assess_materiality_empty_dataframe():
    empty_df = pd.DataFrame({'loan_id': [], 'NPV_orig': [], 'NPV_new': [], 'Delta_NPV': []})
    result = assess_materiality(empty_df)
    assert 'material' in result.columns
    assert len(result) == 0

def test_assess_materiality_zero_delta_npv(sample_npv_results):
    sample_npv_results['Delta_NPV'][0] = 0
    result = assess_materiality(sample_npv_results)
    assert result['material'][0] == False

def test_assess_materiality_large_delta_npv(sample_npv_results):
    sample_npv_results['Delta_NPV'][0] = 1000000
    result = assess_materiality(sample_npv_results)
    assert result['material'][0] == True
