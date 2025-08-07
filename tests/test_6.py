import pytest
import pandas as pd
from unittest.mock import MagicMock

# Keep the definition_73db4dcc58ed483abeeaf4a4ed83b1b8 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_73db4dcc58ed483abeeaf4a4ed83b1b8 import run_sensitivity

# Helper function to mock calculate_npv as described in the notebook specification.
# This function is not part of the module being tested, but a dependency of run_sensitivity.
# Its purpose is to simulate the behavior of the real calculate_npv for testing run_sensitivity.
def _mock_calculate_npv_implementation(cashflows, discount_rate):
    """
    A simplified mock implementation for the calculate_npv function.
    Calculates NPV using the formula SUM(CF_t / (1+r)^t) for simplicity.
    """
    if not isinstance(cashflows, (pd.Series, list)):
        raise TypeError("Cashflows must be a pandas Series or list.")
    if not isinstance(discount_rate, (int, float)):
        raise TypeError("Discount rate must be a numeric value.")

    npv = 0.0
    # Assuming cashflows are ordered by time, starting from period 1
    for t, cf in enumerate(cashflows):
        npv += cf / ((1 + discount_rate)**(t + 1))
    return npv

@pytest.fixture
def sample_df_master():
    """
    Provides a sample pandas DataFrame mimicking df_master for testing.
    Includes original and new cash flows and discount rates for two loans.
    """
    data = {
        'loan_id': ['L001', 'L002'],
        'cashflow_orig': [[100, 100], [200, 200, 200]],
        'cashflow_new': [[90, 90], [180, 180, 180]],
        'discount_rate_orig': [0.05, 0.06],
        'discount_rate_new': [0.055, 0.065],
    }
    df = pd.DataFrame(data)
    # Convert cashflow lists to pandas Series to better simulate actual data structures
    # and allow for easier comparison with pd.testing.assert_series_equal
    df['cashflow_orig'] = df['cashflow_orig'].apply(lambda x: pd.Series(x))
    df['cashflow_new'] = df['cashflow_new'].apply(lambda x: pd.Series(x))
    return df

@pytest.fixture
def mock_calculate_npv(monkeypatch):
    """
    Mocks the calculate_npv function within the module being tested.
    Uses _mock_calculate_npv_implementation as the side effect to simulate behavior.
    """
    mock = MagicMock(side_effect=_mock_calculate_npv_implementation)
    monkeypatch.setattr(definition_73db4dcc58ed483abeeaf4a4ed83b1b8, 'calculate_npv', mock)
    return mock

@pytest.mark.parametrize("rate_shift_bp", [
    50,   # Positive shift: +50 basis points (0.005)
    -25,  # Negative shift: -25 basis points (-0.0025)
    0     # Zero shift: 0 basis points (no change)
])
def test_run_sensitivity_rate_shifts(sample_df_master, mock_calculate_npv, rate_shift_bp):
    """
    Tests run_sensitivity with positive, negative, and zero basis point shifts.
    Verifies that discount rates are correctly adjusted and passed to calculate_npv,
    and that the output DataFrame contains accurate shifted NPVs.
    """
    rate_shift_decimal = rate_shift_bp / 10000.0

    shifted_df = run_sensitivity(sample_df_master, rate_shift_bp)

    # Assertions on the output DataFrame structure
    assert isinstance(shifted_df, pd.DataFrame)
    assert 'loan_id' in shifted_df.columns
    assert 'NPV_orig_shifted' in shifted_df.columns
    assert 'NPV_new_shifted' in shifted_df.columns
    assert 'Delta_NPV_shifted' in shifted_df.columns
    assert len(shifted_df) == len(sample_df_master)

    # Verify calculate_npv was called the correct number of times
    assert mock_calculate_npv.call_count == 2 * len(sample_df_master) # For each loan: orig and new NPV

    calls = mock_calculate_npv.call_args_list
    
    # Iterate through sample_df_master to verify calls and calculated values
    for i, row in sample_df_master.iterrows():
        loan_id = row['loan_id']
        orig_cf = row['cashflow_orig']
        new_cf = row['cashflow_new']
        orig_rate = row['discount_rate_orig']
        new_rate = row['discount_rate_new']

        expected_orig_rate_shifted = orig_rate + rate_shift_decimal
        expected_new_rate_shifted = new_rate + rate_shift_decimal

        # Verify calls to mock_calculate_npv for original and new cashflows
        # The calls are expected to be in order for each loan (orig then new)
        orig_call_args = calls[2 * i].args
        new_call_args = calls[2 * i + 1].args

        pd.testing.assert_series_equal(orig_call_args[0], orig_cf, check_dtype=False, check_names=False)
        assert orig_call_args[1] == pytest.approx(expected_orig_rate_shifted)

        pd.testing.assert_series_equal(new_call_args[0], new_cf, check_dtype=False, check_names=False)
        assert new_call_args[1] == pytest.approx(expected_new_rate_shifted)

        # Verify the recomputed NPVs in the output DataFrame
        # We use the _mock_calculate_npv_implementation to predict the expected NPVs
        predicted_npv_orig_shifted = _mock_calculate_npv_implementation(orig_cf, expected_orig_rate_shifted)
        predicted_npv_new_shifted = _mock_calculate_npv_implementation(new_cf, expected_new_rate_shifted)

        result_row = shifted_df[shifted_df['loan_id'] == loan_id].iloc[0]
        assert result_row['NPV_orig_shifted'] == pytest.approx(predicted_npv_orig_shifted)
        assert result_row['NPV_new_shifted'] == pytest.approx(predicted_npv_new_shifted)
        assert result_row['Delta_NPV_shifted'] == pytest.approx(predicted_npv_new_shifted - predicted_npv_orig_shifted)

def test_run_sensitivity_empty_df_master(mock_calculate_npv):
    """
    Tests run_sensitivity with an empty input DataFrame.
    Should return an empty DataFrame with expected columns and not call calculate_npv.
    """
    empty_df = pd.DataFrame(columns=[
        'loan_id', 'cashflow_orig', 'cashflow_new',
        'discount_rate_orig', 'discount_rate_new'
    ])
    rate_shift_bp = 100

    shifted_df = run_sensitivity(empty_df, rate_shift_bp)

    assert isinstance(shifted_df, pd.DataFrame)
    assert shifted_df.empty
    assert 'loan_id' in shifted_df.columns
    assert 'NPV_orig_shifted' in shifted_df.columns
    assert 'NPV_new_shifted' in shifted_df.columns
    assert 'Delta_NPV_shifted' in shifted_df.columns
    assert mock_calculate_npv.call_count == 0 # No NPV calculations should occur for an empty DataFrame

@pytest.mark.parametrize("invalid_shift", [
    "abc",      # String
    10.5,       # Float
    None,       # NoneType
    [100],      # List
    {'shift': 50} # Dictionary
])
def test_run_sensitivity_invalid_rate_shift_bp_type(sample_df_master, invalid_shift):
    """
    Tests run_sensitivity with invalid types for rate_shift_bp, expecting a TypeError.
    """
    with pytest.raises(TypeError):
        run_sensitivity(sample_df_master, invalid_shift)

@pytest.mark.parametrize("missing_col", [
    'cashflow_orig',
    'cashflow_new',
    'discount_rate_orig',
    'discount_rate_new',
])
def test_run_sensitivity_missing_df_columns(sample_df_master, missing_col):
    """
    Tests run_sensitivity when essential columns are missing from df_master, expecting a KeyError.
    """
    df_missing_col = sample_df_master.drop(columns=[missing_col])
    rate_shift_bp = 100
    with pytest.raises(KeyError, match=f"'{missing_col}'"):
        run_sensitivity(df_missing_col, rate_shift_bp)