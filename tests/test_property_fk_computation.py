"""
Property-based tests for FK computation.

Feature: ikfast-python-binding
Tests Property 2 (partial) related to FK computation producing valid poses.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Property 2 (partial): FK computation produces valid poses
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_fk_returns_valid_translation(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that FK always returns valid translation.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Verify translation is valid
    assert isinstance(translation, np.ndarray), "Translation should be numpy array"
    assert translation.shape == (3,), f"Translation shape should be (3,), got {translation.shape}"
    assert translation.dtype == np.float64, f"Translation dtype should be float64, got {translation.dtype}"
    
    # Check that translation contains no NaN or Inf values
    assert not np.any(np.isnan(translation)), f"Translation contains NaN for joints {joints}"
    assert not np.any(np.isinf(translation)), f"Translation contains Inf for joints {joints}"
    
    # Translation values should be reasonable (within robot workspace)
    # For a typical 6-DOF manipulator, workspace is usually within a few meters
    assert np.all(np.abs(translation) < 10.0), \
        f"Translation values seem unreasonable: {translation} for joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_fk_returns_valid_rotation(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that FK always returns valid rotation.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Verify rotation is valid
    assert isinstance(rotation, np.ndarray), "Rotation should be numpy array"
    assert rotation.shape == (9,), f"Rotation shape should be (9,), got {rotation.shape}"
    assert rotation.dtype == np.float64, f"Rotation dtype should be float64, got {rotation.dtype}"
    
    # Check that rotation contains no NaN or Inf values
    assert not np.any(np.isnan(rotation)), f"Rotation contains NaN for joints {joints}"
    assert not np.any(np.isinf(rotation)), f"Rotation contains Inf for joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_rotation_matrix_determinant(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that rotation matrices have determinant ≈ 1.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Reshape rotation to 3x3 matrix
    rot_matrix = rotation.reshape(3, 3)
    
    # Check that determinant is approximately 1
    det = np.linalg.det(rot_matrix)
    assert np.abs(det - 1.0) < 1e-6, \
        f"Rotation matrix determinant should be ≈ 1, got {det} for joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_rotation_matrix_orthogonality(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that rotation matrices are orthogonal (R^T * R = I).
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Reshape rotation to 3x3 matrix
    rot_matrix = rotation.reshape(3, 3)
    
    # Check that R^T * R is approximately identity
    identity = np.dot(rot_matrix.T, rot_matrix)
    expected_identity = np.eye(3)
    
    assert np.allclose(identity, expected_identity, atol=1e-6), \
        f"Rotation matrix should be orthogonal (R^T * R = I), got:\n{identity}\nfor joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
    joints=npst.arrays(
        dtype=np.float64,
        shape=(6,),
        elements=st.floats(
            min_value=-2.0 * np.pi,
            max_value=2.0 * np.pi,
            allow_nan=False,
            allow_infinity=False
        )
    )
)
def test_property_2_fk_handles_extended_range(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that FK handles joint angles beyond [-π, π] range.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Verify basic validity
    assert isinstance(translation, np.ndarray)
    assert isinstance(rotation, np.ndarray)
    assert translation.shape == (3,)
    assert rotation.shape == (9,)
    
    # Check no NaN or Inf
    assert not np.any(np.isnan(translation))
    assert not np.any(np.isnan(rotation))
    assert not np.any(np.isinf(translation))
    assert not np.any(np.isinf(rotation))
    
    # Check rotation matrix validity
    rot_matrix = rotation.reshape(3, 3)
    det = np.linalg.det(rot_matrix)
    assert np.abs(det - 1.0) < 1e-6, \
        f"Rotation matrix determinant should be ≈ 1 even for extended range, got {det}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_fk_deterministic(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that FK is deterministic (same input produces same output).
    
    Validates: Requirements 2.1, 2.2
    """
    # Compute FK twice with same input
    translation1, rotation1 = ikfast.compute_fk_raw(joints)
    translation2, rotation2 = ikfast.compute_fk_raw(joints)
    
    # Results should be identical
    assert np.array_equal(translation1, translation2), \
        f"FK should be deterministic, got different translations for same joints {joints}"
    assert np.array_equal(rotation1, rotation2), \
        f"FK should be deterministic, got different rotations for same joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_rotation_matrix_rows_unit_length(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that rotation matrix rows have unit length.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Reshape rotation to 3x3 matrix
    rot_matrix = rotation.reshape(3, 3)
    
    # Check that each row has unit length
    for i in range(3):
        row = rot_matrix[i, :]
        length = np.linalg.norm(row)
        assert np.abs(length - 1.0) < 1e-6, \
            f"Rotation matrix row {i} should have unit length, got {length} for joints {joints}"


@settings(max_examples=100, deadline=None)
@given(
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
def test_property_2_rotation_matrix_columns_unit_length(joints):
    """
    Feature: ikfast-python-binding, Property 2 (partial): FK computation produces valid poses
    
    Test that rotation matrix columns have unit length.
    
    Validates: Requirements 2.1, 2.2
    """
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Reshape rotation to 3x3 matrix
    rot_matrix = rotation.reshape(3, 3)
    
    # Check that each column has unit length
    for i in range(3):
        col = rot_matrix[:, i]
        length = np.linalg.norm(col)
        assert np.abs(length - 1.0) < 1e-6, \
            f"Rotation matrix column {i} should have unit length, got {length} for joints {joints}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
