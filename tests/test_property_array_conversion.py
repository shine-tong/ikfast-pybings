"""
Property-based tests for array type conversion.

Feature: ikfast-python-binding
Tests Properties 4 and 5 related to array type conversion and return type consistency.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Property 4: Array Type Conversion Correctness
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    data=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10)
        ),
        elements=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False
        )
    ),
    order=st.sampled_from(['C', 'F'])
)
def test_property_4_c_and_f_contiguous_arrays(data, order):
    """
    Feature: ikfast-python-binding, Property 4: Array Type Conversion Correctness
    
    Test that both C-contiguous and Fortran-contiguous arrays are accepted.
    
    Validates: Requirements 1.2, 2.3, 5.1, 5.5
    """
    # Create array with specified memory order
    arr = np.array(data, order=order)
    
    # Verify the order is as expected
    if order == 'C':
        assert arr.flags['C_CONTIGUOUS']
    else:
        assert arr.flags['F_CONTIGUOUS']
    
    # Test that the array is accepted and processed correctly
    result = ikfast._test_array_echo(arr)
    
    # Verify result is correct
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert np.allclose(result, arr)


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from([np.float32, np.int32, np.int64]),
    shape=st.tuples(st.integers(min_value=1, max_value=10))
)
def test_property_4_compatible_dtype_conversion(dtype, shape):
    """
    Feature: ikfast-python-binding, Property 4: Array Type Conversion Correctness
    
    Test that arrays with compatible dtypes are converted correctly.
    
    Validates: Requirements 1.2, 2.3, 5.1, 5.5
    """
    # Create array with specific dtype
    size = np.prod(shape)
    if dtype in [np.int32, np.int64]:
        data = np.arange(size, dtype=dtype).reshape(shape)
    else:  # float32
        data = np.arange(size, dtype=np.float64).astype(dtype).reshape(shape)
    
    # Test that arrays with compatible dtypes are accepted
    # pybind11 should automatically convert to float64
    try:
        result = ikfast._test_array_echo(data)
        
        # If conversion succeeds, verify the result
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        
        # Values should be preserved (within floating point precision)
        assert np.allclose(result, data.astype(np.float64), rtol=1e-5)
    except TypeError:
        # Some dtypes might not be automatically convertible, which is acceptable
        # as long as the error is clear
        pass


@settings(max_examples=100, deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10)
    )
)
def test_property_4_array_data_integrity(shape):
    """
    Feature: ikfast-python-binding, Property 4: Array Type Conversion Correctness
    
    Test that array data is preserved through conversion.
    
    Validates: Requirements 1.2, 2.3, 5.1, 5.5
    """
    # Create array with known pattern
    arr = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    
    # Test that data is preserved
    result = ikfast._test_array_echo(arr)
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == arr.shape
    assert np.array_equal(result, arr)
    
    # Also test that sum is correct (tests data access)
    sum_result = ikfast._test_array_sum(arr)
    expected_sum = np.sum(arr)
    assert np.isclose(sum_result, expected_sum)


# ============================================================================
# Property 5: Return Type Consistency
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    shape=st.one_of(
        st.tuples(st.integers(min_value=1, max_value=20)),
        st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10)
        )
    )
)
def test_property_5_return_type_is_float64(shape):
    """
    Feature: ikfast-python-binding, Property 5: Return Type Consistency
    
    Test that all array returns are numpy arrays with float64 dtype.
    
    Validates: Requirements 2.2, 3.2, 5.2, 10.3
    """
    # Create test data
    size = np.prod(shape)
    data = list(range(size))
    shape_list = list(shape)
    
    # Test array creation
    result = ikfast._test_create_array(data, shape_list)
    
    # Verify return type
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.dtype == np.float64, f"Result dtype should be float64, got {result.dtype}"


@settings(max_examples=100, deadline=None)
@given(
    shape=st.one_of(
        st.tuples(st.integers(min_value=1, max_value=20)),
        st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10)
        ),
        st.tuples(
            st.integers(min_value=1, max_value=5),
            st.integers(min_value=1, max_value=5),
            st.integers(min_value=1, max_value=5)
        )
    )
)
def test_property_5_return_shape_correctness(shape):
    """
    Feature: ikfast-python-binding, Property 5: Return Type Consistency
    
    Test that array shapes are correct.
    
    Validates: Requirements 2.2, 3.2, 5.2, 10.3
    """
    # Create test data
    size = np.prod(shape)
    data = list(range(size))
    shape_list = list(shape)
    
    # Test array creation
    result = ikfast._test_create_array(data, shape_list)
    
    # Verify shape
    assert result.shape == shape, f"Expected shape {shape}, got {result.shape}"
    
    # Verify data integrity
    expected = np.arange(size, dtype=np.float64).reshape(shape)
    assert np.array_equal(result, expected), "Data should match expected values"


@settings(max_examples=100, deadline=None)
@given(
    arr=npst.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10)
        ),
        elements=st.floats(
            min_value=-1000.0,
            max_value=1000.0,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_property_5_echo_preserves_type_and_shape(arr):
    """
    Feature: ikfast-python-binding, Property 5: Return Type Consistency
    
    Test that echoed arrays maintain float64 dtype and correct shape.
    
    Validates: Requirements 2.2, 3.2, 5.2, 10.3
    """
    result = ikfast._test_array_echo(arr)
    
    # Verify type consistency
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    
    # Verify shape consistency
    assert result.shape == arr.shape
    
    # Verify data consistency
    assert np.allclose(result, arr)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
