import pytest
import pandas as pd
import matplotlib.pyplot as plt
from definition_27f938d5b0cd4098bf9f94e114b00814 import plot_cashflow_timeline

@pytest.fixture
def sample_cashflow_data():
    data = {'loan_id': [1, 1, 1, 1, 2, 2, 2],
            'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-01-01', '2024-02-01', '2024-03-01']),
            'cashflow': [100, 100, 100, 100, 50, 50, 50],
            'type': ['original', 'original', 'restructured', 'restructured', 'original', 'restructured', 'restructured']}
    return pd.DataFrame(data)


def test_plot_cashflow_timeline_valid_data(sample_cashflow_data, monkeypatch):
    # Mock plt.show() to avoid displaying the plot during testing
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        plot_cashflow_timeline(sample_cashflow_data)
    except Exception as e:
        pytest.fail(f"plot_cashflow_timeline raised an exception: {e}")


def test_plot_cashflow_timeline_empty_dataframe():
    empty_df = pd.DataFrame()
    try:
        plot_cashflow_timeline(empty_df)
    except Exception as e:
        assert "ValueError" in str(type(e)), "Expected a ValueError for empty DataFrame."

def test_plot_cashflow_timeline_missing_columns():
    data = {'date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'cashflow': [100, 100]}
    df = pd.DataFrame(data)
    with pytest.raises(KeyError):
        plot_cashflow_timeline(df)

def test_plot_cashflow_timeline_non_datetime_date(sample_cashflow_data):
    # Modify the 'date' column to be strings
    sample_cashflow_data['date'] = sample_cashflow_data['date'].astype(str)

    with pytest.raises(TypeError):  # Expect TypeError as date must be datetime
        plot_cashflow_timeline(sample_cashflow_data)

def test_plot_cashflow_timeline_non_numeric_cashflow(sample_cashflow_data):
    # Modify cashflow to be strings
    sample_cashflow_data['cashflow'] = ['a','b','c','d','e','f','g']
    with pytest.raises(TypeError): #Expect TypeError since cashflow must be numeric.
        plot_cashflow_timeline(sample_cashflow_data)
