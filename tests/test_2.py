import pytest
import pandas as pd
from definition_23618abb73434dd89af902421d6f6586 import calc_discount_rate

# Define the credit spread as per the notebook specification (100 basis points or 0.01)
CREDIT_SPREAD = 0.01

@pytest.mark.parametrize("input_df, expected_output", [
    # Test Case 1: Standard scenario - No worsening in credit rating
    # (rating_after <= rating_before). r_new should be equal to orig_rate.
    (
        pd.DataFrame({
            'loan_id': [1, 2],
            'orig_rate': [0.05, 0.06],
            'rating_before': [3, 2],
            'rating_after': [2, 2] # 3->2 (improved), 2->2 (same)
        }),
        pd.DataFrame({
            'loan_id': [1, 2],
            'orig_rate': [0.05, 0.06],
            'rating_before': [3, 2],
            'rating_after': [2, 2],
            'r_orig': [0.05, 0.06],
            'r_new': [0.05, 0.06]
        })
    ),
    # Test Case 2: Standard scenario - Worsening in credit rating
    # (rating_after > rating_before). r_new should be orig_rate + CREDIT_SPREAD.
    (
        pd.DataFrame({
            'loan_id': [3, 4],
            'orig_rate': [0.04, 0.07],
            'rating_before': [1, 3],
            'rating_after': [2, 4] # 1->2 (worsened), 3->4 (worsened)
        }),
        pd.DataFrame({
            'loan_id': [3, 4],
            'orig_rate': [0.04, 0.07],
            'rating_before': [1, 3],
            'rating_after': [2, 4],
            'r_orig': [0.04, 0.07],
            'r_new': [0.04 + CREDIT_SPREAD, 0.07 + CREDIT_SPREAD]
        })
    ),
    # Test Case 3: Mixed ratings - some loans worsen, others do not.
    # Verifies conditional logic applies correctly per row.
    (
        pd.DataFrame({
            'loan_id': [5, 6, 7],
            'orig_rate': [0.03, 0.05, 0.08],
            'rating_before': [2, 3, 1],
            'rating_after': [1, 4, 1] # 2->1 (improved), 3->4 (worsened), 1->1 (same)
        }),
        pd.DataFrame({
            'loan_id': [5, 6, 7],
            'orig_rate': [0.03, 0.05, 0.08],
            'rating_before': [2, 3, 1],
            'rating_after': [1, 4, 1],
            'r_orig': [0.03, 0.05, 0.08],
            'r_new': [0.03, 0.05 + CREDIT_SPREAD, 0.08]
        })
    ),
    # Test Case 4: Edge case - Empty DataFrame input.
    # The function should return an empty DataFrame with the new columns added.
    (
        pd.DataFrame(columns=['loan_id', 'orig_rate', 'rating_before', 'rating_after']),
        pd.DataFrame(columns=['loan_id', 'orig_rate', 'rating_before', 'rating_after', 'r_orig', 'r_new'], dtype='float64')
    ),
    # Test Case 5: Edge case - Missing a required column ('orig_rate').
    # This should raise a KeyError, as the function relies on this column.
    (
        pd.DataFrame({
            'loan_id': [8],
            'rating_before': [3],
            'rating_after': [4]
        }),
        KeyError
    ),
])
def test_calc_discount_rate(input_df, expected_output):
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        # If an exception is expected, assert that it is raised
        with pytest.raises(expected_output):
            calc_discount_rate(input_df)
    else:
        # Otherwise, compare the actual output DataFrame with the expected one
        actual_output = calc_discount_rate(input_df)
        pd.testing.assert_frame_equal(actual_output, expected_output, check_dtype=True)