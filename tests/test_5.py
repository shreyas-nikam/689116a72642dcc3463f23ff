import pytest
import pandas as pd
from unittest.mock import patch

# Placeholder for the module import
from definition_a4b58d7fd1ae4d9fad449e463405440c import run_npv_analysis

# A mock function for calculate_npv to simplify testing run_npv_analysis.
# This mock simply returns the sum of cashflows.
# In a real scenario, this would be a more complex NPV calculation.
def _mock_calculate_npv(cashflows, discount_rate):
    """
    Mock calculate_npv to return sum of cashflows for controlled testing.
    The discount_rate argument is ignored in this mock, but passed through.
    """
    if cashflows.empty:
        return 0.0
    return float(cashflows.sum())

@pytest.fixture(autouse=True)
def mock_npv_calculator():
    """
    Fixture to patch calculate_npv for all tests in this module.
    Assumes calculate_npv is directly accessible in your_module.
    The patch target must be where run_npv_analysis looks for calculate_npv.
    """
    with patch('definition_a4b58d7fd1ae4d9fad449e463405440c.calculate_npv', new=_mock_calculate_npv) as mock_func:
        yield mock_func

# Define the materiality threshold as per notebook specification
MATERIALITY_THRESHOLD = 50000.0

@pytest.mark.parametrize(
    "loan_data, expected_results",
    [
        (   # Scenario 1: Material positive delta
            [
                {'loan_id': 'L001', 'date': '2023-01-01', 'cashflow_orig': 10000, 'cashflow_new': 40000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
                {'loan_id': 'L001', 'date': '2023-02-01', 'cashflow_orig': 10000, 'cashflow_new': 40000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
                {'loan_id': 'L001', 'date': '2023-03-01', 'cashflow_orig': 10000, 'cashflow_new': 40000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
            ],
            pd.DataFrame([{'loan_id': 'L001', 'NPV_orig': 30000.0, 'NPV_new': 120000.0, 'Delta_NPV': 90000.0, 'material': True}])
        ),
        (   # Scenario 2: Material negative delta
            [
                {'loan_id': 'L002', 'date': '2023-01-01', 'cashflow_orig': 50000, 'cashflow_new': 10000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
                {'loan_id': 'L002', 'date': '2023-02-01', 'cashflow_orig': 50000, 'cashflow_new': 10000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
                {'loan_id': 'L002', 'date': '2023-03-01', 'cashflow_orig': 50000, 'cashflow_new': 10000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
            ],
            pd.DataFrame([{'loan_id': 'L002', 'NPV_orig': 150000.0, 'NPV_new': 30000.0, 'Delta_NPV': -120000.0, 'material': True}])
        ),
        (   # Scenario 3: Non-material delta (positive)
            [
                {'loan_id': 'L003', 'date': '2023-01-01', 'cashflow_orig': 10000, 'cashflow_new': 15000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
                {'loan_id': 'L003', 'date': '2023-02-01', 'cashflow_orig': 10000, 'cashflow_new': 15000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.06},
            ],
            pd.DataFrame([{'loan_id': 'L003', 'NPV_orig': 20000.0, 'NPV_new': 30000.0, 'Delta_NPV': 10000.0, 'material': False}])
        ),
        (   # Scenario 4: No restructuring (Delta = 0)
            [
                {'loan_id': 'L004', 'date': '2023-01-01', 'cashflow_orig': 10000, 'cashflow_new': 10000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.05},
                {'loan_id': 'L004', 'date': '2023-02-01', 'cashflow_orig': 10000, 'cashflow_new': 10000, 'discount_rate_orig': 0.05, 'discount_rate_new': 0.05},
            ],
            pd.DataFrame([{'loan_id': 'L004', 'NPV_orig': 20000.0, 'NPV_new': 20000.0, 'Delta_NPV': 0.0, 'material': False}])
        ),
        (   # Scenario 5: Delta exactly at positive threshold boundary (non-material)
            [
                {'loan_id': 'L005', 'date': '2023-01-01', 'cashflow_orig': 0, 'cashflow_new': MATERIALITY_THRESHOLD, 'discount_rate_orig': 0.01, 'discount_rate_new': 0.01},
            ],
            pd.DataFrame([{'loan_id': 'L005', 'NPV_orig': 0.0, 'NPV_new': MATERIALITY_THRESHOLD, 'Delta_NPV': MATERIALITY_THRESHOLD, 'material': False}])
        ),
        (   # Scenario 6: Delta just above positive threshold boundary (material)
            [
                {'loan_id': 'L006', 'date': '2023-01-01', 'cashflow_orig': 0, 'cashflow_new': MATERIALITY_THRESHOLD + 0.01, 'discount_rate_orig': 0.01, 'discount_rate_new': 0.01},
            ],
            pd.DataFrame([{'loan_id': 'L006', 'NPV_orig': 0.0, 'NPV_new': MATERIALITY_THRESHOLD + 0.01, 'Delta_NPV': MATERIALITY_THRESHOLD + 0.01, 'material': True}])
        ),
    ]
)
def test_run_npv_analysis_various_scenarios(loan_data, expected_results):
    """
    Tests run_npv_analysis with various loan scenarios, covering material flag logic
    and positive/negative Delta_NPV, including boundary conditions.
    """
    df_master = pd.DataFrame(loan_data)
    result_df = run_npv_analysis(df_master)

    # Ensure consistent data types and order for robust comparison
    expected_results['loan_id'] = expected_results['loan_id'].astype(str)
    result_df['loan_id'] = result_df['loan_id'].astype(str)
    
    # Sort both DataFrames by loan_id to ensure consistent order
    result_df = result_df.sort_values(by='loan_id').reset_index(drop=True)
    expected_results = expected_results.sort_values(by='loan_id').reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, expected_results, check_dtype=True, check_exact=False, atol=1e-9)

def test_run_npv_analysis_empty_dataframe():
    """
    Tests run_npv_analysis with an empty input DataFrame.
    Should return an empty DataFrame with the correct columns and dtypes.
    """
    df_master = pd.DataFrame(columns=[
        'loan_id', 'date', 'cashflow_orig', 'cashflow_new',
        'discount_rate_orig', 'discount_rate_new'
    ])
    
    # Define expected empty DataFrame with specific dtypes for robust comparison
    expected_df = pd.DataFrame(columns=[
        'loan_id', 'NPV_orig', 'NPV_new', 'Delta_NPV', 'material'
    ]).astype({
        'loan_id': str,
        'NPV_orig': float,
        'NPV_new': float,
        'Delta_NPV': float,
        'material': bool
    })

    result_df = run_npv_analysis(df_master)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=True)

@pytest.mark.parametrize("invalid_input", [
    None,
    [],
    "not a dataframe",
    123,
    {'key': 'value'}
])
def test_run_npv_analysis_invalid_input_type(invalid_input):
    """
    Tests run_npv_analysis with non-DataFrame inputs, expecting a TypeError.
    """
    with pytest.raises(TypeError):
        run_npv_analysis(invalid_input)

@pytest.mark.parametrize("missing_column_df", [
    pd.DataFrame({'loan_id': ['L001'], 'date': ['2023-01-01'], 'cashflow_new': [100], 'discount_rate_orig': [0.05], 'discount_rate_new': [0.06]}), # Missing cashflow_orig
    pd.DataFrame({'loan_id': ['L001'], 'date': ['2023-01-01'], 'cashflow_orig': [100], 'cashflow_new': [100], 'discount_rate_new': [0.06]}), # Missing discount_rate_orig
    pd.DataFrame({'loan_id': ['L001'], 'date': ['2023-01-01'], 'cashflow_orig': [100], 'discount_rate_orig': [0.05], 'discount_rate_new': [0.06]}), # Missing cashflow_new
    pd.DataFrame({'loan_id': ['L001'], 'date': ['2023-01-01'], 'cashflow_orig': [100], 'cashflow_new': [100], 'discount_rate_orig': [0.05]}), # Missing discount_rate_new
    pd.DataFrame({'cashflow_orig': [100], 'cashflow_new': [100], 'discount_rate_orig': [0.05], 'discount_rate_new': [0.06]}), # Missing loan_id
])
def test_run_npv_analysis_missing_columns(missing_column_df):
    """
    Tests run_npv_analysis with input DataFrames missing required columns,
    expecting a KeyError or AttributeError depending on the exact pandas operation path.
    """
    with pytest.raises((KeyError, AttributeError)):
        run_npv_analysis(missing_column_df)