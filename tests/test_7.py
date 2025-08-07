import pytest
import pandas as pd
import matplotlib.pyplot as plt

# DO NOT REPLACE or REMOVE THE BLOCK BELOW
from definition_2263cdea51eb49d8b9b8ed8c6cfc940c import plot_cashflow_timeline
# DO NOT REPLACE or REMOVE THE BLOCK ABOVE

@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Clears matplotlib figures after each test to prevent side effects."""
    yield
    plt.close('all')

def test_plot_cashflow_timeline_valid_data():
    """Test with a valid DataFrame containing typical cash flow data."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
        'cashflow_orig': [100, 120, 110],
        'cashflow_new': [80, 110, 130]
    }
    df_long = pd.DataFrame(data)
    
    # Expected: No exception raised, a plot should be generated.
    plot_cashflow_timeline(df_long)
    
    # Assert that a figure and axes were created
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) > 0

def test_plot_cashflow_timeline_empty_dataframe():
    """Test with an empty DataFrame but with correct columns and types."""
    df_long = pd.DataFrame(columns=['date', 'cashflow_orig', 'cashflow_new'])
    # Ensure 'date' column is of datetime type, even if empty, for consistency
    df_long['date'] = df_long['date'].astype('datetime64[ns]')
    
    # Expected: No exception, an empty plot should be generated.
    plot_cashflow_timeline(df_long)
    
    # Assert that a figure and axes were created, even for an empty plot
    fig = plt.gcf()
    assert fig is not None
    assert len(fig.axes) > 0

def test_plot_cashflow_timeline_missing_orig_column():
    """Test with a DataFrame missing the 'cashflow_orig' column."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
        'cashflow_new': [80, 120]
    }
    df_long = pd.DataFrame(data)
    
    # Expected: KeyError because 'cashflow_orig' is a required column for plotting.
    with pytest.raises(KeyError, match="cashflow_orig"):
        plot_cashflow_timeline(df_long)

def test_plot_cashflow_timeline_missing_new_column():
    """Test with a DataFrame missing the 'cashflow_new' column."""
    data = {
        'date': pd.to_datetime(['2023-01-01', '2023-02-01']),
        'cashflow_orig': [100, 100]
    }
    df_long = pd.DataFrame(data)
    
    # Expected: KeyError because 'cashflow_new' is a required column for plotting.
    with pytest.raises(KeyError, match="cashflow_new"):
        plot_cashflow_timeline(df_long)

def test_plot_cashflow_timeline_invalid_input_type():
    """Test with an input that is not a pandas DataFrame (e.g., string, None)."""
    
    # Expected: AttributeError as DataFrame methods (like column access) would be called.
    with pytest.raises(AttributeError):
        plot_cashflow_timeline("this is not a dataframe")

    with pytest.raises(AttributeError):
        plot_cashflow_timeline(None)