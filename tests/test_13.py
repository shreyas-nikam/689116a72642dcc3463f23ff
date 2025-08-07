import pytest
from definition_75060cdebf7542ee8e78210c96e98fd9 import _generate_loan_attributes

@pytest.mark.parametrize(
    "num_loans, attribute_type, expected_type, expected_length, expected_min_val, expected_max_val, expected_exception",
    [
        # Test Case 1: Valid generation of 'principal' attributes
        (5, 'principal', list, 5, 1000.0, 1_000_000.0, None),
        # Test Case 2: Valid generation of 'rate' attributes
        (3, 'rate', list, 3, 0.01, 0.20, None),
        # Test Case 3: Valid generation of 'term_mths' attributes
        (7, 'term_mths', list, 7, 6, 360, None),
        # Test Case 4: Edge case: num_loans = 0 should return an empty list
        (0, 'principal', list, 0, None, None, None),
        # Test Case 5: Error case: Invalid attribute_type should raise ValueError
        (2, 'unsupported_attribute', None, None, None, None, ValueError),
    ]
)
def test_generate_loan_attributes(
    num_loans, attribute_type, expected_type, expected_length, expected_min_val, expected_max_val, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            _generate_loan_attributes(num_loans, attribute_type)
    else:
        result = _generate_loan_attributes(num_loans, attribute_type)

        assert isinstance(result, expected_type)
        assert len(result) == expected_length

        if expected_length > 0:
            for item in result:
                # Check element type based on the attribute_type
                if attribute_type == 'principal':
                    assert isinstance(item, (int, float)), f"Expected int or float for principal, got {type(item)}"
                elif attribute_type == 'rate':
                    assert isinstance(item, float), f"Expected float for rate, got {type(item)}"
                elif attribute_type == 'term_mths':
                    assert isinstance(item, int), f"Expected int for term_mths, got {type(item)}"
                else:
                    pytest.fail(f"Unhandled attribute_type in test: {attribute_type}") # Should not be reached with current params

                # Check if values fall within realistic financial ranges
                if expected_min_val is not None and expected_max_val is not None:
                    assert expected_min_val <= item <= expected_max_val, \
                        f"Value {item} for {attribute_type} is out of expected range [{expected_min_val}, {expected_max_val}]"