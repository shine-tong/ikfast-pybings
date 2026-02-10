"""
Property-based tests for input validation error handling.

Feature: ikfast-python-binding
Tests Property 6: Input Validation Error Handling
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Property 6: Input Validation Error Handling
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    wrong_size=st.integers(min_value=0, max_value=10).filter(lambda x: x != 3)
)
def test_property_6_compute_ik_invalid_translation_shape(wrong_size):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_ik raises ValueError for incorrect translation shape.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create translation with wrong shape
    translation = np.zeros(wrong_size, dtype=np.float64)
    rotation = np.eye(3).flatten()
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    # Error message should contain expected and actual dimensions
    error_msg = str(exc_info.value)
    assert "3" in error_msg or "shape" in error_msg.lower(), \
        f"Error message should mention expected shape: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    wrong_size=st.integers(min_value=0, max_value=15).filter(lambda x: x != 9)
)
def test_property_6_compute_ik_invalid_rotation_shape(wrong_size):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_ik raises ValueError for incorrect rotation shape.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create rotation with wrong shape
    translation = np.zeros(3, dtype=np.float64)
    rotation = np.zeros(wrong_size, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    # Error message should contain expected and actual dimensions
    error_msg = str(exc_info.value)
    assert "9" in error_msg or "shape" in error_msg.lower(), \
        f"Error message should mention expected shape: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    wrong_size=st.integers(min_value=0, max_value=10).filter(lambda x: x != 6)
)
def test_property_6_compute_fk_invalid_joints_shape(wrong_size):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_fk raises ValueError for incorrect joint angles shape.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create joints with wrong shape
    joints = np.zeros(wrong_size, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    # Error message should contain expected and actual dimensions
    error_msg = str(exc_info.value)
    assert "6" in error_msg or "shape" in error_msg.lower(), \
        f"Error message should mention expected shape: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    ndim=st.integers(min_value=2, max_value=4)
)
def test_property_6_compute_ik_multidimensional_translation(ndim):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_ik raises ValueError for multidimensional translation arrays.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create multidimensional array
    shape = tuple([3] * ndim)
    translation = np.zeros(shape, dtype=np.float64)
    rotation = np.eye(3).flatten()
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    # Error message should indicate dimension issue
    error_msg = str(exc_info.value)
    assert "dimension" in error_msg.lower() or "shape" in error_msg.lower(), \
        f"Error message should mention dimension issue: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    ndim=st.integers(min_value=2, max_value=4)
)
def test_property_6_compute_ik_multidimensional_rotation(ndim):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_ik raises ValueError for multidimensional rotation arrays
    (except 3x3 which is valid).
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Skip 2D case with shape (3,3) as it's valid
    if ndim == 2:
        return
    
    # Create multidimensional array
    shape = tuple([3] * ndim)
    translation = np.zeros(3, dtype=np.float64)
    rotation = np.zeros(shape, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    # Error message should indicate dimension issue
    error_msg = str(exc_info.value)
    assert "dimension" in error_msg.lower() or "shape" in error_msg.lower(), \
        f"Error message should mention dimension issue: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    ndim=st.integers(min_value=2, max_value=4)
)
def test_property_6_compute_fk_multidimensional_joints(ndim):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_fk raises ValueError for multidimensional joint arrays.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create multidimensional array
    shape = tuple([6] + [2] * (ndim - 1))
    joints = np.zeros(shape, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    # Error message should indicate dimension issue
    error_msg = str(exc_info.value)
    assert "dimension" in error_msg.lower() or "shape" in error_msg.lower(), \
        f"Error message should mention dimension issue: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    translation_size=st.integers(min_value=0, max_value=10).filter(lambda x: x != 3),
    rotation_size=st.integers(min_value=0, max_value=15).filter(lambda x: x != 9)
)
def test_property_6_compute_ik_both_invalid_shapes(translation_size, rotation_size):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that compute_ik raises ValueError when both inputs have incorrect shapes.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create arrays with wrong shapes
    translation = np.zeros(translation_size, dtype=np.float64)
    rotation = np.zeros(rotation_size, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    # Error message should be descriptive
    error_msg = str(exc_info.value)
    assert len(error_msg) > 0, "Error message should not be empty"


@settings(max_examples=100, deadline=None)
@given(
    extra_dims=st.integers(min_value=1, max_value=3)
)
def test_property_6_error_messages_contain_shape_info(extra_dims):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that error messages contain information about expected and actual shapes.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create translation with wrong shape
    wrong_size = 3 + extra_dims
    translation = np.zeros(wrong_size, dtype=np.float64)
    rotation = np.eye(3).flatten()
    
    # Should raise ValueError with descriptive message
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_ik_raw(translation, rotation)
    
    error_msg = str(exc_info.value).lower()
    
    # Error message should contain shape-related information
    # (either "shape", "size", "dimension", or specific numbers)
    has_shape_info = (
        "shape" in error_msg or
        "size" in error_msg or
        "dimension" in error_msg or
        "3" in error_msg or
        str(wrong_size) in error_msg
    )
    
    assert has_shape_info, \
        f"Error message should contain shape information: {exc_info.value}"


@settings(max_examples=100, deadline=None)
@given(
    joints_size=st.integers(min_value=0, max_value=10).filter(lambda x: x != 6)
)
def test_property_6_fk_error_message_quality(joints_size):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that FK error messages are descriptive and helpful.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create joints with wrong shape
    joints = np.zeros(joints_size, dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    error_msg = str(exc_info.value)
    
    # Error message should be non-empty and descriptive
    assert len(error_msg) > 10, \
        f"Error message should be descriptive, got: {error_msg}"
    
    # Should mention the expected size (6)
    assert "6" in error_msg or "shape" in error_msg.lower(), \
        f"Error message should mention expected size: {error_msg}"


@settings(max_examples=100, deadline=None)
@given(
    shape1=st.integers(min_value=1, max_value=5),
    shape2=st.integers(min_value=1, max_value=5)
)
def test_property_6_ik_2d_translation_rejected(shape1, shape2):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that 2D translation arrays are rejected.
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Create 2D translation array
    translation = np.zeros((shape1, shape2), dtype=np.float64)
    rotation = np.eye(3).flatten()
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        ikfast.compute_ik_raw(translation, rotation)


@settings(max_examples=100, deadline=None)
@given(
    shape1=st.integers(min_value=1, max_value=5),
    shape2=st.integers(min_value=1, max_value=5)
)
def test_property_6_fk_2d_joints_rejected(shape1, shape2):
    """
    Feature: ikfast-python-binding, Property 6: Input Validation Error Handling
    
    Test that 2D joint arrays are rejected (except valid shapes).
    
    Validates: Requirements 2.4, 5.3, 6.2, 6.3
    """
    # Skip if this would create a valid 1D array when flattened
    if shape1 == 6 and shape2 == 1:
        return
    if shape1 == 1 and shape2 == 6:
        return
    
    # Create 2D joints array
    joints = np.zeros((shape1, shape2), dtype=np.float64)
    
    # Should raise ValueError
    with pytest.raises(ValueError):
        ikfast.compute_fk_raw(joints)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
