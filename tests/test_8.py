import pytest
import pandas as pd
import matplotlib.pyplot as plt
from definition_ee6f4b8824e64cd9aa3f27bb33b8aff9 import plot_delta_npv_waterfall

@pytest.fixture
def sample_npv_results():
    data = {'loan_id': [1, 2, 3, 4, 5],
            'NPV_orig': [100000, 50000, 75000, 120000, 60000],
            'NPV_new': [90000, 60000, 80000, 110000, 70000],
            'Delta_NPV': [-10000, 10000, 5000, -10000, 10000]}
    return pd.DataFrame(data)


def test_plot_delta_npv_waterfall_valid_input(sample_npv_results, monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        plot_delta_npv_waterfall(sample_npv_results)
    except Exception as e:
        pytest.fail(f"plot_delta_npv_waterfall raised an exception: {e}")

def test_plot_delta_npv_waterfall_empty_dataframe():
    df = pd.DataFrame({'loan_id': [], 'NPV_orig': [], 'NPV_new': [], 'Delta_NPV': []})
    try:
        plot_delta_npv_waterfall(df)
    except Exception as e:
        pytest.fail(f"plot_delta_npv_waterfall raised an exception: {e}")


def test_plot_delta_npv_waterfall_missing_column(monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    data = {'loan_id': [1, 2, 3], 'NPV_orig': [100000, 50000, 75000], 'NPV_new': [90000, 60000, 80000]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
         plot_delta_npv_waterfall(df)

def test_plot_delta_npv_waterfall_non_numeric_data(monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    data = {'loan_id': [1, 2, 3], 'NPV_orig': ['a', 'b', 'c'], 'NPV_new': [90000, 60000, 80000], 'Delta_NPV': [1,2,3]}
    df = pd.DataFrame(data)
    with pytest.raises(TypeError):
        plot_delta_npv_waterfall(df)

def test_plot_delta_npv_waterfall_inf_values(sample_npv_results, monkeypatch):
    # Mock plt.show() to prevent the plot from displaying during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    sample_npv_results.loc[0, 'Delta_NPV'] = float('inf')
    try:
        plot_delta_npv_waterfall(sample_npv_results)
    except Exception as e:
        pytest.fail(f"plot_delta_npv_waterfall raised an exception with inf values: {e}")
