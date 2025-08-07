import pytest
from definition_8b419deac3f34f509aee963e07a23584 import _generate_loan_id

@pytest.mark.parametrize("num_loans, expected_outcome", [
    # Test Case 1: Positive - a typical number of loans (as hinted by "approximately 10 loans" in spec)
    (10, "success_list"),
    # Test Case 2: Edge Case - zero loans, should return an empty list
    (0, []),
    # Test Case 3: Edge Case - negative number of loans, should raise a ValueError
    (-5, ValueError),
    # Test Case 4: Edge Case - non-integer input for num_loans, should raise a TypeError
    ("invalid_input", TypeError),
    # Test Case 5: Positive - a single loan, checks uniqueness and type for minimal case
    (1, "success_list"),
])
def test_generate_loan_id(num_loans, expected_outcome):
    try:
        result = _generate_loan_id(num_loans)
        
        # Handle successful execution scenarios
        if expected_outcome == "success_list":
            # Assert that the result is a list of the correct length
            assert isinstance(result, list)
            assert len(result) == num_loans
            # Assert that all generated IDs are unique
            assert len(set(result)) == num_loans
            # Assert that all elements are strings or integers (as per docstring)
            assert all(isinstance(item, (str, int)) for item in result)
        elif expected_outcome == []:
            # Assert for the specific case of an empty list
            assert result == []
        else:
            # This branch implies an unexpected success where an exception was expected
            pytest.fail(f"Unexpected successful return for input {num_loans}. Result: {result}")

    except Exception as e:
        # Handle exception scenarios
        # Check if the raised exception type matches the expected exception type
        if isinstance(expected_outcome, type) and issubclass(expected_outcome, Exception):
            assert isinstance(e, expected_outcome)
        else:
            # This branch implies an unexpected exception was raised
            pytest.fail(f"Unexpected exception {type(e).__name__} for input {num_loans}. Expected successful outcome.")