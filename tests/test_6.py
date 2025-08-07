import pytest
import pandas as pd
from definition_9f7bd0ddae374310a2ef1740b083026f import assess_materiality

@pytest.fixture
def sample_npv_results():
    data = {'loan_id': [1, 2, 3, 4, 5],
            'NPV_orig': [100000, 50000, 75000, 120000, 60000],
            'NPV_new': [80000, 60000, 25000, 150000, 5000],
            'Delta_NPV': [-20000, 10000, -50000, 30000, -55000]}
    return pd.DataFrame(data)

def test_assess_materiality_below_threshold(sample_npv_results):
    threshold = 30000
    result = assess_materiality(sample_npv_results.copy(), threshold)
    expected = [False, False, True, True, True]
    assert (result['material'] == expected).all()

def test_assess_materiality_all_false(sample_npv_results):
    threshold = 100000
    result = assess_materiality(sample_npv_results.copy(), threshold)
    expected = [False, False, False, False, False]
    assert (result['material'] == expected).all()

def test_assess_materiality_empty_dataframe():
    df = pd.DataFrame()
    threshold = 50000
    result = assess_materiality(df.copy(), threshold)
    assert 'material' in result.columns
    assert len(result) == 0

def test_assess_materiality_zero_threshold(sample_npv_results):
    threshold = 0
    result = assess_materiality(sample_npv_results.copy(), threshold)
    expected = [True, True, True, True, True]
    assert (result['material'] == expected).all()

def test_assess_materiality_negative_threshold(sample_npv_results):
    threshold = -50000  #Negative threshold should still consider absolute values
    result = assess_materiality(sample_npv_results.copy(), threshold)
    expected = [False, False, True, False, True]
    assert (result['material'] == expected).all()
