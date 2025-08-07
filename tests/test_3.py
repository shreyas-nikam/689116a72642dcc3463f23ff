import pytest
import pandas as pd
from definition_4d9aad16271249aa819c3048524c07b1 import tidy_merge

# Helper function to create dummy cash flow DataFrames
def create_cf_df(loan_ids, dates, cf_type='orig', missing_col=None):
    """
    Creates a dummy cash flow DataFrame (cf_orig or cf_new).
    Args:
        loan_ids (list): List of loan IDs.
        dates (list): List of pandas Timestamps for dates.
        cf_type (str): 'orig' for original, 'new' for new cash flows.
        missing_col (str, optional): If specified, this column will be dropped from the DataFrame.
    Returns:
        pd.DataFrame: A dummy cash flow DataFrame.
    """
    data = []
    for loan_id in loan_ids:
        for date in dates:
            data.append({
                'loan_id': loan_id,
                'date': date,
                f'cashflow_{cf_type}': float(loan_id * 100 + date.day) # Dummy cash flow value
            })
    df = pd.DataFrame(data)
    if missing_col and missing_col in df.columns:
        df = df.drop(columns=[missing_col])
    return df

# Helper function to create dummy loan metadata DataFrame
def create_loan_df(loan_ids, disc_rates_orig, disc_rates_new, missing_col=None):
    """
    Creates a dummy loan metadata DataFrame (df_loans).
    Args:
        loan_ids (list): List of loan IDs.
        disc_rates_orig (list): List of original discount rates.
        disc_rates_new (list): List of new discount rates.
        missing_col (str, optional): If specified, this column will be dropped from the DataFrame.
    Returns:
        pd.DataFrame: A dummy loan metadata DataFrame.
    """
    data = []
    for i, loan_id in enumerate(loan_ids):
        data.append({
            'loan_id': loan_id,
            'discount_rate_orig': disc_rates_orig[i],
            'discount_rate_new': disc_rates_new[i]
        })
    df = pd.DataFrame(data)
    if missing_col and missing_col in df.columns:
        df = df.drop(columns=[missing_col])
    return df

@pytest.mark.parametrize("test_name, cf_orig_params, cf_new_params, df_loans_params, expected_behavior", [
    (
        "standard_merge_successful",
        # cf_orig parameters
        {'loan_ids': [1, 2], 'dates': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]},
        # cf_new parameters
        {'loan_ids': [1, 2], 'dates': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]},
        # df_loans parameters
        {'loan_ids': [1, 2], 'disc_rates_orig': [0.05, 0.06], 'disc_rates_new': [0.055, 0.065]},
        "success"
    ),
    (
        "non_restructured_loan_handling",
        # cf_orig: Loan 1 & 2 have cashflows
        {'loan_ids': [1, 2], 'dates': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]},
        # cf_new: Only Loan 1 has cashflows (Loan 2 represents a non-restructured loan)
        {'loan_ids': [1], 'dates': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]},
        # df_loans: Both loans exist with their respective discount rates
        {'loan_ids': [1, 2], 'disc_rates_orig': [0.05, 0.06], 'disc_rates_new': [0.055, 0.06]}, # Loan 2 new rate might be same as orig
        "success_non_restructured"
    ),
    (
        "empty_cashflow_dataframes",
        # cf_orig: empty
        {'loan_ids': [], 'dates': []},
        # cf_new: empty
        {'loan_ids': [], 'dates': []},
        # df_loans: contains loans metadata
        {'loan_ids': [1, 2], 'disc_rates_orig': [0.05, 0.06], 'disc_rates_new': [0.055, 0.065]},
        "empty_result"
    ),
    (
        "missing_required_column",
        # cf_orig: missing 'loan_id' column
        {'loan_ids': [1], 'dates': [pd.Timestamp('2023-01-01')], 'missing_col': 'loan_id'},
        # cf_new: normal
        {'loan_ids': [1], 'dates': [pd.Timestamp('2023-01-01')]},
        # df_loans: normal
        {'loan_ids': [1], 'disc_rates_orig': [0.05], 'disc_rates_new': [0.055]},
        KeyError # Expected exception type
    ),
    (
        "invalid_input_type",
        # cf_orig as a string (not a DataFrame)
        "not_a_dataframe",
        # cf_new: empty DataFrame
        {'loan_ids': [], 'dates': []},
        # df_loans: empty DataFrame
        {'loan_ids': [], 'disc_rates_orig': [], 'disc_rates_new': []},
        TypeError # Expected exception type
    )
])
def test_tidy_merge(test_name, cf_orig_params, cf_new_params, df_loans_params, expected_behavior):
    """
    Tests the tidy_merge function for various scenarios including standard merges,
    handling of non-restructured loans, empty inputs, and error cases.
    """

    # --- Prepare input DataFrames based on parameters ---
    cf_orig = (create_cf_df(cf_orig_params['loan_ids'], cf_orig_params['dates'], 'orig', cf_orig_params.get('missing_col'))
               if isinstance(cf_orig_params, dict) else cf_orig_params)

    cf_new = (create_cf_df(cf_new_params['loan_ids'], cf_new_params['dates'], 'new', cf_new_params.get('missing_col'))
              if isinstance(cf_new_params, dict) else cf_new_params)
    
    df_loans = (create_loan_df(df_loans_params['loan_ids'], df_loans_params['disc_rates_orig'], 
                               df_loans_params['disc_rates_new'], df_loans_params.get('missing_col'))
                if isinstance(df_loans_params, dict) else df_loans_params)

    # --- Execute the test based on expected behavior ---
    if expected_behavior in ["success", "success_non_restructured", "empty_result"]:
        # Expected successful execution
        loan_cf_master = tidy_merge(cf_orig, cf_new, df_loans)

        assert isinstance(loan_cf_master, pd.DataFrame)
        expected_cols = {'loan_id', 'date', 'cashflow_orig', 'cashflow_new', 'discount_rate_orig', 'discount_rate_new'}
        assert expected_cols.issubset(loan_cf_master.columns), f"Missing expected columns in output: {expected_cols - set(loan_cf_master.columns)}"

        if expected_behavior == "success":
            assert not loan_cf_master.empty, "Output DataFrame should not be empty for standard merge."
            expected_rows = len(cf_orig_params['loan_ids']) * len(cf_orig_params['dates'])
            assert len(loan_cf_master) == expected_rows, f"Expected {expected_rows} rows, got {len(loan_cf_master)}"

            # Basic value verification (sample check)
            # Check cashflow_orig values
            for _, row in create_cf_df(cf_orig_params['loan_ids'], cf_orig_params['dates'], 'orig').iterrows():
                matched_row = loan_cf_master[(loan_cf_master['loan_id'] == row['loan_id']) & 
                                             (loan_cf_master['date'] == row['date'])]
                assert not matched_row.empty
                pd.testing.assert_series_equal(matched_row['cashflow_orig'].reset_index(drop=True), 
                                                pd.Series([row['cashflow_orig']]), check_dtype=False, check_names=False)
            # Check cashflow_new values
            for _, row in create_cf_df(cf_new_params['loan_ids'], cf_new_params['dates'], 'new').iterrows():
                matched_row = loan_cf_master[(loan_cf_master['loan_id'] == row['loan_id']) & 
                                             (loan_cf_master['date'] == row['date'])]
                assert not matched_row.empty
                pd.testing.assert_series_equal(matched_row['cashflow_new'].reset_index(drop=True), 
                                                pd.Series([row['cashflow_new']]), check_dtype=False, check_names=False)
            # Check discount rates
            for _, row in create_loan_df(df_loans_params['loan_ids'], df_loans_params['disc_rates_orig'], df_loans_params['disc_rates_new']).iterrows():
                matched_rows = loan_cf_master[loan_cf_master['loan_id'] == row['loan_id']]
                assert not matched_rows.empty
                assert (matched_rows['discount_rate_orig'] == row['discount_rate_orig']).all()
                assert (matched_rows['discount_rate_new'] == row['discount_rate_new']).all()

        elif expected_behavior == "success_non_restructured":
            assert not loan_cf_master.empty, "Output DataFrame should not be empty."
            
            # Verify Loan 1 (restructured) data
            loan1_data = loan_cf_master[loan_cf_master['loan_id'] == 1]
            assert not loan1_data.empty
            assert not loan1_data['cashflow_orig'].isnull().any()
            assert not loan1_data['cashflow_new'].isnull().any()
            assert not loan1_data['discount_rate_orig'].isnull().any()
            assert not loan1_data['discount_rate_new'].isnull().any()

            # Verify Loan 2 (non-restructured) data
            loan2_data = loan_cf_master[loan_cf_master['loan_id'] == 2]
            assert not loan2_data.empty
            assert not loan2_data['cashflow_orig'].isnull().any(), "Original cash flow for non-restructured loan should not be null."
            assert loan2_data['cashflow_new'].isnull().all(), "New cash flow for non-restructured loan should be null."
            assert not loan2_data['discount_rate_orig'].isnull().any()
            # As per spec, discount_rate_new from df_loans should always be present (can be same as orig)
            assert not loan2_data['discount_rate_new'].isnull().any() 

        elif expected_behavior == "empty_result":
            assert loan_cf_master.empty, "Output DataFrame should be empty."
            assert expected_cols.issubset(loan_cf_master.columns), "Empty DataFrame should still have expected columns."

    else:
        # Expected an exception
        with pytest.raises(expected_behavior):
            tidy_merge(cf_orig, cf_new, df_loans)