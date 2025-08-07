import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from definition_b166d527209c469fa71f79c3e6cda3e4 import plot_delta_npv_waterfall

@pytest.fixture
def sample_npv_results():
    data = {'loan_id': [1, 2, 3],
            'NPV_orig': [100000, 50000, 25000],
            'NPV_new': [90000, 60000, 30000],
            'Delta_NPV': [-10000, 10000, 5000]}
    return pd.DataFrame(data)

def test_plot_delta_npv_waterfall_valid_data(sample_npv_results):
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_delta_npv_waterfall(sample_npv_results)
        mock_show.assert_called_once()

def test_plot_delta_npv_waterfall_empty_dataframe():
    empty_df = pd.DataFrame()
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_delta_npv_waterfall(empty_df)
        mock_show.assert_called_once()

def test_plot_delta_npv_waterfall_zero_delta_npv(sample_npv_results):
    sample_npv_results['Delta_NPV'] = 0
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_delta_npv_waterfall(sample_npv_results)
        mock_show.assert_called_once()

def test_plot_delta_npv_waterfall_large_delta_npv(sample_npv_results):
    sample_npv_results['Delta_NPV'] = sample_npv_results['Delta_NPV'] * 1000
    with patch("matplotlib.pyplot.show") as mock_show:
        plot_delta_npv_waterfall(sample_npv_results)
        mock_show.assert_called_once()

def test_plot_delta_npv_waterfall_nan_values():
    data = {'loan_id': [1, 2, 3],
            'NPV_orig': [100000, 50000, float('nan')],
            'NPV_new': [90000, 60000, 30000],
            'Delta_NPV': [-10000, 10000, float('nan')]}
    nan_df = pd.DataFrame(data)

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_delta_npv_waterfall(nan_df)
        mock_show.assert_called_once()
