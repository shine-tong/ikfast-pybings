"""
Property tests for exception translation.

Feature: ikfast-python-binding, Property 10: Exception Translation

Tests that C++ exceptions are properly translated to appropriate Python
exception types with preserved error messages.

Validates: Requirements 6.1, 6.4
"""
import pytest
from hypothesis import given, strategies as st


def test_invalid_argument_to_value_error():
    """
    Test that std::invalid_argument is translated to ValueError.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    test_message = "test invalid argument"
    
    with pytest.raises(ValueError) as exc_info:
        ikfast._test_throw_invalid_argument(test_message)
    
    # Check that the error message is preserved
    error_message = str(exc_info.value)
    assert test_message in error_message, \
        f"Error message should contain '{test_message}', got: {error_message}"
    
    # Check that context is added
    assert "_test_throw_invalid_argument" in error_message, \
        f"Error message should contain function context, got: {error_message}"


def test_out_of_range_to_index_error():
    """
    Test that std::out_of_range is translated to IndexError.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    test_message = "index out of bounds"
    
    with pytest.raises(IndexError) as exc_info:
        ikfast._test_throw_out_of_range(test_message)
    
    # Check that the error message is preserved
    error_message = str(exc_info.value)
    assert test_message in error_message, \
        f"Error message should contain '{test_message}', got: {error_message}"
    
    # Check that context is added
    assert "_test_throw_out_of_range" in error_message, \
        f"Error message should contain function context, got: {error_message}"


def test_runtime_error_to_runtime_error():
    """
    Test that std::runtime_error is translated to RuntimeError.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    test_message = "runtime error occurred"
    
    with pytest.raises(RuntimeError) as exc_info:
        ikfast._test_throw_runtime_error(test_message)
    
    # Check that the error message is preserved
    error_message = str(exc_info.value)
    assert test_message in error_message, \
        f"Error message should contain '{test_message}', got: {error_message}"
    
    # Check that context is added
    assert "_test_throw_runtime_error" in error_message, \
        f"Error message should contain function context, got: {error_message}"


def test_generic_exception_to_runtime_error():
    """
    Test that generic std::exception is translated to RuntimeError with context.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    test_message = "generic exception"
    
    with pytest.raises(RuntimeError) as exc_info:
        ikfast._test_throw_generic_exception(test_message)
    
    # Check that the error message is preserved
    error_message = str(exc_info.value)
    assert test_message in error_message, \
        f"Error message should contain '{test_message}', got: {error_message}"
    
    # Check that context is added
    assert "_test_throw_generic_exception" in error_message, \
        f"Error message should contain function context, got: {error_message}"
    
    # Check that "Unexpected error" context is added for generic exceptions
    assert "Unexpected error" in error_message, \
        f"Error message should contain 'Unexpected error' for generic exceptions, got: {error_message}"


@given(error_message=st.text(min_size=1, max_size=100).filter(lambda s: '\x00' not in s))
def test_property_error_message_preservation(error_message):
    """
    Property test: Error messages are preserved across exception translation.
    
    For any error message string, when a C++ exception is thrown with that message,
    the translated Python exception should contain the original message.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Test with invalid_argument -> ValueError
    try:
        ikfast._test_throw_invalid_argument(error_message)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert error_message in str(e), \
            f"ValueError should contain original message '{error_message}', got: {str(e)}"
    
    # Test with out_of_range -> IndexError
    try:
        ikfast._test_throw_out_of_range(error_message)
        assert False, "Should have raised IndexError"
    except IndexError as e:
        assert error_message in str(e), \
            f"IndexError should contain original message '{error_message}', got: {str(e)}"
    
    # Test with runtime_error -> RuntimeError
    try:
        ikfast._test_throw_runtime_error(error_message)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert error_message in str(e), \
            f"RuntimeError should contain original message '{error_message}', got: {str(e)}"


@given(error_message=st.text(min_size=1, max_size=100).filter(lambda s: '\x00' not in s))
def test_property_exception_context_added(error_message):
    """
    Property test: Function context is added to exception messages.
    
    For any error message, the translated exception should include the
    function name as context to help with debugging.
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Test that function name is in the error message
    exception_tests = [
        (ikfast._test_throw_invalid_argument, ValueError, "_test_throw_invalid_argument"),
        (ikfast._test_throw_out_of_range, IndexError, "_test_throw_out_of_range"),
        (ikfast._test_throw_runtime_error, RuntimeError, "_test_throw_runtime_error"),
        (ikfast._test_throw_generic_exception, RuntimeError, "_test_throw_generic_exception"),
    ]
    
    for func, expected_exception, expected_context in exception_tests:
        try:
            func(error_message)
            assert False, f"Should have raised {expected_exception.__name__}"
        except expected_exception as e:
            error_str = str(e)
            assert expected_context in error_str, \
                f"Exception message should contain context '{expected_context}', got: {error_str}"


def test_exception_type_mapping():
    """
    Test that all exception types are mapped correctly.
    
    Verifies the complete exception mapping:
    - std::invalid_argument → ValueError
    - std::out_of_range → IndexError
    - std::runtime_error → RuntimeError
    - std::exception → RuntimeError
    
    Feature: ikfast-python-binding, Property 10: Exception Translation
    Validates: Requirements 6.1, 6.4
    """
    import ikfast_pybind._ikfast_pybind as ikfast
    
    exception_mapping = [
        (ikfast._test_throw_invalid_argument, ValueError),
        (ikfast._test_throw_out_of_range, IndexError),
        (ikfast._test_throw_runtime_error, RuntimeError),
        (ikfast._test_throw_generic_exception, RuntimeError),
    ]
    
    for func, expected_exception in exception_mapping:
        with pytest.raises(expected_exception):
            func("test message")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
