"""
Property-based tests for type conversion handling.

Feature: ikfast-python-binding
Tests Property 12: Type Conversion or Error
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Property 12: Type Conversion or Error
# ============================================================================

# List of numeric dtypes to test
NUMERIC_DTYPES = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
]

# List of non-numeric dtypes that should fail
# Note: np.bool_ is treated as numeric by pybind11 (True=1.0, False=0.0)
NON_NUMERIC_DTYPES = [
    np.object_,
    np.str_,
    np.unicode_,
]


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from(NUMERIC_DTYPES),
    joints=npst.arrays(
        dtype=np.float64,
        shape=(6,),
        elements=st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_property_12_fk_numeric_dtype_conversion(dtype, joints):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_fk accepts numeric dtypes and converts them to float64.
    
    Validates: Requirements 5.4
    """
    # Convert joints to the test dtype
    joints_converted = joints.astype(dtype)
    
    # Should either succeed or raise TypeError (not ValueError)
    try:
        translation, rotation = ikfast.compute_fk_raw(joints_converted)
        
        # If successful, verify results are valid
        assert isinstance(translation, np.ndarray)
        assert isinstance(rotation, np.ndarray)
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64
        assert translation.shape == (3,)
        assert rotation.shape == (9,)
        
    except TypeError as e:
        # TypeError is acceptable for incompatible types
        error_msg = str(e)
        assert len(error_msg) > 0, "TypeError should have descriptive message"
    except ValueError:
        # ValueError should not be raised for dtype issues
        pytest.fail("ValueError should not be raised for dtype conversion issues")


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from(NUMERIC_DTYPES)
)
def test_property_12_ik_translation_numeric_dtype_conversion(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_ik accepts numeric dtypes for translation.
    
    Validates: Requirements 5.4
    """
    # Create valid translation and rotation
    translation = np.array([0.5, 0.5, 0.5], dtype=dtype)
    rotation = np.eye(3, dtype=np.float64).flatten()
    
    # Should either succeed or raise TypeError
    try:
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # If successful, verify result is valid
        assert isinstance(solutions, ikfast.IkSolutionList)
        
    except TypeError as e:
        # TypeError is acceptable
        error_msg = str(e)
        assert len(error_msg) > 0
    except ValueError:
        # ValueError should not be raised for dtype issues
        pytest.fail("ValueError should not be raised for dtype conversion issues")


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from(NUMERIC_DTYPES)
)
def test_property_12_ik_rotation_numeric_dtype_conversion(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_ik accepts numeric dtypes for rotation.
    
    Validates: Requirements 5.4
    """
    # Create valid translation and rotation
    translation = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=dtype).flatten()
    
    # Should either succeed or raise TypeError
    try:
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # If successful, verify result is valid
        assert isinstance(solutions, ikfast.IkSolutionList)
        
    except TypeError as e:
        # TypeError is acceptable
        error_msg = str(e)
        assert len(error_msg) > 0
    except ValueError:
        # ValueError should not be raised for dtype issues
        pytest.fail("ValueError should not be raised for dtype conversion issues")


@settings(max_examples=50, deadline=None)
@given(
    dtype=st.sampled_from(NON_NUMERIC_DTYPES)
)
def test_property_12_fk_non_numeric_dtype_rejected(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_fk rejects non-numeric dtypes with TypeError.
    
    Validates: Requirements 5.4
    """
    # Create array with non-numeric dtype
    try:
        if dtype == np.object_:
            joints = np.array([object()] * 6, dtype=dtype)
        else:
            joints = np.array(['a', 'b', 'c', 'd', 'e', 'f'], dtype=dtype)
        
        # Should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            ikfast.compute_fk_raw(joints)
            
    except (TypeError, ValueError):
        # Creating the array itself might fail, which is fine
        pass


@settings(max_examples=50, deadline=None)
@given(
    dtype=st.sampled_from(NON_NUMERIC_DTYPES)
)
def test_property_12_ik_non_numeric_dtype_rejected(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_ik rejects non-numeric dtypes with TypeError.
    
    Validates: Requirements 5.4
    """
    # Create arrays with non-numeric dtype
    try:
        if dtype == np.object_:
            translation = np.array([object()] * 3, dtype=dtype)
        else:
            translation = np.array(['a', 'b', 'c'], dtype=dtype)
        
        rotation = np.eye(3, dtype=np.float64).flatten()
        
        # Should raise TypeError
        with pytest.raises((TypeError, ValueError)):
            ikfast.compute_ik_raw(translation, rotation)
            
    except (TypeError, ValueError):
        # Creating the array itself might fail, which is fine
        pass


@settings(max_examples=100, deadline=None)
@given(
    int_dtype=st.sampled_from([np.int32, np.int64]),
    joints=npst.arrays(
        dtype=np.float64,
        shape=(6,),
        elements=st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_property_12_integer_dtype_handling(int_dtype, joints):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that integer dtypes are handled appropriately.
    
    Validates: Requirements 5.4
    """
    # Convert to integer (will lose precision but should work or fail gracefully)
    joints_int = (joints * 1000).astype(int_dtype) / 1000.0
    joints_int = joints_int.astype(int_dtype)
    
    # Should either convert and succeed, or raise TypeError
    try:
        translation, rotation = ikfast.compute_fk_raw(joints_int)
        
        # If successful, results should be float64
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64
        
    except TypeError:
        # TypeError is acceptable for integer types
        pass


@settings(max_examples=100, deadline=None)
@given(
    float_dtype=st.sampled_from([np.float16, np.float32, np.float64]),
    joints=npst.arrays(
        dtype=np.float64,
        shape=(6,),
        elements=st.floats(
            min_value=-np.pi,
            max_value=np.pi,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_property_12_float_dtype_handling(float_dtype, joints):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that different float dtypes are handled appropriately.
    
    Validates: Requirements 5.4
    """
    # Convert to the test float dtype
    joints_converted = joints.astype(float_dtype)
    
    # Should either convert and succeed, or raise TypeError
    try:
        translation, rotation = ikfast.compute_fk_raw(joints_converted)
        
        # If successful, results should be float64
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64
        
    except TypeError:
        # TypeError is acceptable
        pass


@settings(max_examples=100, deadline=None)
@given(
    dtype1=st.sampled_from(NUMERIC_DTYPES),
    dtype2=st.sampled_from(NUMERIC_DTYPES)
)
def test_property_12_mixed_dtypes_in_ik(dtype1, dtype2):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that compute_ik handles mixed dtypes for translation and rotation.
    
    Validates: Requirements 5.4
    """
    # Create arrays with different dtypes
    translation = np.array([0.5, 0.5, 0.5], dtype=dtype1)
    rotation = np.eye(3, dtype=dtype2).flatten()
    
    # Should either succeed or raise TypeError
    try:
        solutions = ikfast.compute_ik_raw(translation, rotation)
        assert isinstance(solutions, ikfast.IkSolutionList)
    except TypeError:
        # TypeError is acceptable for incompatible types
        pass
    except ValueError:
        # ValueError should not be raised for dtype issues
        pytest.fail("ValueError should not be raised for dtype conversion issues")


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from(NUMERIC_DTYPES)
)
def test_property_12_dtype_error_messages_clear(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that dtype-related errors have clear messages.
    
    Validates: Requirements 5.4
    """
    # Create array with test dtype
    joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype)
    
    try:
        translation, rotation = ikfast.compute_fk_raw(joints)
        # Success is fine
        assert True
    except TypeError as e:
        # If TypeError is raised, message should be descriptive
        error_msg = str(e)
        assert len(error_msg) > 5, \
            f"TypeError message should be descriptive, got: {error_msg}"
    except ValueError:
        # ValueError should not be raised for dtype issues
        pytest.fail("ValueError should not be raised for dtype conversion issues")


@settings(max_examples=100, deadline=None)
@given(
    dtype=st.sampled_from([np.float32, np.float64])
)
def test_property_12_common_float_dtypes_accepted(dtype):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that common float dtypes (float32, float64) are accepted.
    
    Validates: Requirements 5.4
    """
    # Create valid input with common float dtype
    joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=dtype)
    
    # Should succeed (these are the most common dtypes)
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Results should be float64
    assert translation.dtype == np.float64
    assert rotation.dtype == np.float64
    assert translation.shape == (3,)
    assert rotation.shape == (9,)


@settings(max_examples=100, deadline=None)
@given(
    bool_values=st.lists(st.booleans(), min_size=6, max_size=6)
)
def test_property_12_bool_dtype_converted_to_numeric(bool_values):
    """
    Feature: ikfast-python-binding, Property 12: Type Conversion or Error
    
    Test that bool dtype is accepted and converted to numeric (True=1.0, False=0.0).
    
    Validates: Requirements 5.4
    """
    # Create bool array
    joints = np.array(bool_values, dtype=np.bool_)
    
    # Should succeed - pybind11 converts bool to numeric
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Results should be float64
    assert translation.dtype == np.float64
    assert rotation.dtype == np.float64
    assert translation.shape == (3,)
    assert rotation.shape == (9,)
    
    # Verify no NaN or Inf
    assert not np.any(np.isnan(translation))
    assert not np.any(np.isnan(rotation))
    assert not np.any(np.isinf(translation))
    assert not np.any(np.isinf(rotation))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
