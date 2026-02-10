"""
Unit tests for compute_fk_raw function.

Tests the forward kinematics computation binding, including:
- Known joint configuration tests
- Return type validation
- Return shape validation
- Error handling for incorrect input shapes
"""
import pytest
import numpy as np


def test_compute_fk_with_known_configuration():
    """Test compute_fk_raw with a known joint configuration."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Use zero configuration as a simple test case
    joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Check that we got results
    assert translation is not None, "Translation should not be None"
    assert rotation is not None, "Rotation should not be None"
    
    # Check types
    assert isinstance(translation, np.ndarray), "Translation should be numpy array"
    assert isinstance(rotation, np.ndarray), "Rotation should be numpy array"
    
    # Check shapes
    assert translation.shape == (3,), f"Translation shape should be (3,), got {translation.shape}"
    assert rotation.shape == (9,), f"Rotation shape should be (9,), got {rotation.shape}"
    
    # Check dtypes
    assert translation.dtype == np.float64, f"Translation dtype should be float64, got {translation.dtype}"
    assert rotation.dtype == np.float64, f"Rotation dtype should be float64, got {rotation.dtype}"


def test_compute_fk_return_types():
    """Test that compute_fk_raw returns numpy arrays."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Random joint configuration
    joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    
    result = ikfast.compute_fk_raw(joints)
    
    # Check that result is a tuple
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, f"Result should have 2 elements, got {len(result)}"
    
    translation, rotation = result
    
    # Check types
    assert isinstance(translation, np.ndarray), "Translation should be numpy array"
    assert isinstance(rotation, np.ndarray), "Rotation should be numpy array"


def test_compute_fk_return_shapes():
    """Test that compute_fk_raw returns correct shapes."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Test with various joint configurations
    test_configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.5, -0.5, 1.0, -1.0, 0.3, -0.3], dtype=np.float64),
        np.array([1.57, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=np.float64),
    ]
    
    for joints in test_configs:
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Check shapes
        assert translation.shape == (3,), \
            f"Translation shape should be (3,), got {translation.shape} for joints {joints}"
        assert rotation.shape == (9,), \
            f"Rotation shape should be (9,), got {rotation.shape} for joints {joints}"
        
        # Check dtypes
        assert translation.dtype == np.float64, \
            f"Translation dtype should be float64, got {translation.dtype}"
        assert rotation.dtype == np.float64, \
            f"Rotation dtype should be float64, got {rotation.dtype}"


def test_compute_fk_invalid_shape_too_few_joints():
    """Test that compute_fk_raw raises ValueError for incorrect input shape (too few joints)."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Only 5 joints instead of 6
    joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    # Check error message contains expected and actual shapes
    error_msg = str(exc_info.value)
    assert "joints" in error_msg.lower(), f"Error message should mention 'joints': {error_msg}"
    assert "6" in error_msg, f"Error message should mention expected size 6: {error_msg}"
    assert "5" in error_msg, f"Error message should mention actual size 5: {error_msg}"


def test_compute_fk_invalid_shape_too_many_joints():
    """Test that compute_fk_raw raises ValueError for incorrect input shape (too many joints)."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # 7 joints instead of 6
    joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    # Check error message
    error_msg = str(exc_info.value)
    assert "joints" in error_msg.lower(), f"Error message should mention 'joints': {error_msg}"
    assert "6" in error_msg, f"Error message should mention expected size 6: {error_msg}"
    assert "7" in error_msg, f"Error message should mention actual size 7: {error_msg}"


def test_compute_fk_invalid_shape_2d_array():
    """Test that compute_fk_raw raises ValueError for 2D array input."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # 2D array instead of 1D
    joints = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    
    with pytest.raises(ValueError) as exc_info:
        ikfast.compute_fk_raw(joints)
    
    # Check error message mentions dimensions
    error_msg = str(exc_info.value)
    assert "dimension" in error_msg.lower(), f"Error message should mention dimensions: {error_msg}"


def test_compute_fk_accepts_different_dtypes():
    """Test that compute_fk_raw accepts arrays with different dtypes (should convert)."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Test with float32
    joints_f32 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    translation, rotation = ikfast.compute_fk_raw(joints_f32)
    
    assert translation.dtype == np.float64, "Output should be float64"
    assert rotation.dtype == np.float64, "Output should be float64"
    
    # Test with int (should convert)
    joints_int = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    translation, rotation = ikfast.compute_fk_raw(joints_int)
    
    assert translation.dtype == np.float64, "Output should be float64"
    assert rotation.dtype == np.float64, "Output should be float64"


def test_compute_fk_rotation_is_valid():
    """Test that compute_fk_raw returns a valid rotation matrix."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Reshape rotation to 3x3 matrix
    rot_matrix = rotation.reshape(3, 3)
    
    # Check that it's a valid rotation matrix (determinant should be ≈ 1)
    det = np.linalg.det(rot_matrix)
    assert np.abs(det - 1.0) < 1e-6, \
        f"Rotation matrix determinant should be ≈ 1, got {det}"
    
    # Check that it's orthogonal (R^T * R should be identity)
    identity = np.dot(rot_matrix.T, rot_matrix)
    assert np.allclose(identity, np.eye(3), atol=1e-6), \
        "Rotation matrix should be orthogonal"


def test_compute_fk_different_configurations():
    """Test compute_fk_raw with various joint configurations."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    # Test several different configurations
    configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    ]
    
    for joints in configs:
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Basic sanity checks
        assert translation.shape == (3,), f"Translation shape incorrect for {joints}"
        assert rotation.shape == (9,), f"Rotation shape incorrect for {joints}"
        assert not np.any(np.isnan(translation)), f"Translation contains NaN for {joints}"
        assert not np.any(np.isnan(rotation)), f"Rotation contains NaN for {joints}"
        assert not np.any(np.isinf(translation)), f"Translation contains Inf for {joints}"
        assert not np.any(np.isinf(rotation)), f"Rotation contains Inf for {joints}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
