import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

# This block should remain as is. DO NOT REPLACE or REMOVE.
from definition_8b69db58514446798a5e48bb4f5e7b95 import plot_delta_npv_waterfall


def test_plot_delta_npv_waterfall_valid_data():
    """
    Test case 1: Valid DataFrame with mixed positive and negative Delta_NPV values.
    Ensures the function executes without error and calls matplotlib's show function.
    """
    results_df = pd.DataFrame({
        'loan_id': ['L001', 'L002', 'L003', 'L004'],
        'NPV_orig': [100000, 50000, 75000, 120000],
        'NPV_new': [110000, 45000, 80000, 100000],
        'Delta_NPV': [10000, -5000, 5000, -20000]
    })
    with patch('matplotlib.pyplot.show') as mock_show:
        plot_delta_npv_waterfall(results_df)
        mock_show.assert_called_once()


def test_plot_delta_npv_waterfall_empty_dataframe():
    """
    Test case 2: Empty DataFrame.
    Ensures the function handles an empty input DataFrame gracefully without crashing,
    and still attempts to generate a plot (which would be empty).
    """
    results_df = pd.DataFrame(columns=['loan_id', 'NPV_orig', 'NPV_new', 'Delta_NPV'])
    with patch('matplotlib.pyplot.show') as mock_show:
        plot_delta_npv_waterfall(results_df)
        mock_show.assert_called_once()


@pytest.mark.parametrize("missing_col", ["NPV_orig", "NPV_new", "Delta_NPV"])
def test_plot_delta_npv_waterfall_missing_required_column(missing_col):
    """
    Test case 3: DataFrame with a missing required column (NPV_orig, NPV_new, or Delta_NPV).
    Expects a KeyError or similar error when the function tries to access the non-existent column.
    """
    df_data = {
        'loan_id': ['L001'],
        'NPV_orig': [100000],
        'NPV_new': [110000],
        'Delta_NPV': [10000]
    }
    # Create a copy and drop the specified column
    results_df = pd.DataFrame(df_data)
    results_df = results_df.drop(columns=[missing_col])

    with pytest.raises(KeyError):
        plot_delta_npv_waterfall(results_df)


@pytest.mark.parametrize("invalid_input", [
    None,
    123,
    "not a dataframe",
    [1, 2, 3],
    {'a': 1, 'b': 2}
])
def test_plot_delta_npv_waterfall_invalid_input_type(invalid_input):
    """
    Test case 4: Non-DataFrame input.
    Ensures the function raises a TypeError if the input is not a pandas DataFrame.
    """
    with pytest.raises(TypeError):
        plot_delta_npv_waterfall(invalid_input)


@pytest.mark.parametrize("col_to_corrupt", ["NPV_orig", "NPV_new", "Delta_NPV"])
def test_plot_delta_npv_waterfall_non_numeric_npv_data(col_to_corrupt):
    """
    Test case 5: DataFrame with non-numeric data in NPV columns.
    Ensures the function raises a ValueError or TypeError if required numerical columns
    contain non-numeric data, which would typically cause issues during plotting or calculations.
    """
    results_df = pd.DataFrame({
        'loan_id': ['L001'],
        'NPV_orig': [100000],
        'NPV_new': [110000],
        'Delta_NPV': [10000]
    })
    # Corrupt one column with a string
    results_df.loc[0, col_to_corrupt] = 'invalid_data'
    
    with pytest.raises((ValueError, TypeError)):
        plot_delta_npv_waterfall(results_df)