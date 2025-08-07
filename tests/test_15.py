import pytest
from definition_24ec0680ed0f4e0384268a31f9db4a58 import _generate_restructured_terms

@pytest.mark.parametrize("input_tuple, expected", [
    # Test 1: Standard positive inputs - Expect new_rate > orig_rate and new_term_mths > orig_term_mths
    ((0.05, 60), 
     lambda result, orig_rate, orig_term_mths: (
        isinstance(result, tuple) and
        len(result) == 2 and
        isinstance(result[0], float) and   # new_rate should be a float
        isinstance(result[1], int) and     # new_term_mths should be an int
        result[0] > orig_rate and          # Restructured rate should be strictly higher
        result[1] > orig_term_mths and     # Restructured term should be strictly longer
        result[0] > 0 and                  # New rate must be positive
        result[1] > 0                      # New term must be positive
     )),
    # Test 2: Edge case - Zero original rate - Expect new_rate > 0 and new_term_mths > orig_term_mths
    ((0.0, 120),
     lambda result, orig_rate, orig_term_mths: (
        isinstance(result, tuple) and
        len(result) == 2 and
        isinstance(result[0], float) and
        isinstance(result[1], int) and
        result[0] > orig_rate and          # New rate must be strictly greater than 0.0
        result[1] > orig_term_mths and     # New term must be strictly longer
        result[0] > 0 and
        result[1] > 0
     )),
    # Test 3: Edge case - Very short original term (e.g., 1 month) - Expect longer term and higher rate
    ((0.03, 1),
     lambda result, orig_rate, orig_term_mths: (
        isinstance(result, tuple) and
        len(result) == 2 and
        isinstance(result[0], float) and
        isinstance(result[1], int) and
        result[0] > orig_rate and          # New rate should be strictly higher
        result[1] > orig_term_mths and     # New term should be strictly longer (e.g., > 1)
        result[0] > 0 and
        result[1] > 0
     )),
    # Test 4: Invalid input type for orig_rate (string) - Expect TypeError
    (("invalid", 60), TypeError),
    # Test 5: Invalid input value for orig_term_mths (negative) - Expect ValueError
    ((0.05, -24), ValueError),
])
def test_generate_restructured_terms(input_tuple, expected):
    orig_rate, orig_term_mths = input_tuple
    try:
        new_rate, new_term_mths = _generate_restructured_terms(orig_rate, orig_term_mths)
        # If no exception, 'expected' should be a callable (lambda) for detailed assertions
        assert callable(expected)
        # Pass the actual result and original inputs to the lambda for verification
        assert expected((new_rate, new_term_mths), orig_rate, orig_term_mths)
    except Exception as e:
        # If an exception occurred, 'expected' should be the expected exception type
        assert isinstance(e, expected)