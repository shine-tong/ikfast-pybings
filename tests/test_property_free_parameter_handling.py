"""
Property-based test for free parameter handling.

Feature: ikfast-python-binding, Property 8: Free Parameter Handling

**Validates: Requirements 1.5, 3.3, 3.4**

Test that solutions with free parameters produce valid joint configurations
that satisfy the original end effector pose.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import hypothesis.extra.numpy as npst
import ikfast_pybind._ikfast_pybind as ikfast


# Strategy for generating valid joint configurations
joint_angles_strategy = npst.arrays(
    dtype=np.float64,
    shape=(6,),
    elements=st.floats(
        min_value=-np.pi,
        max_value=np.pi,
        allow_nan=False,
        allow_infinity=False
    )
)


@given(joint_angles=joint_angles_strategy)
@settings(max_examples=20, deadline=None)
def test_property_8_free_parameter_handling(joint_angles):
    """
    Feature: ikfast-python-binding, Property 8: Free Parameter Handling
    
    For any solution with free parameters and any valid free parameter values,
    calling get_solution with those values should produce a complete joint
    configuration that satisfies the original end effector pose.
    
    Since this solver has no free parameters (GetNumFreeParameters() returns 0),
    this test verifies that:
    1. Solutions report no free parameters
    2. get_solution() works without free_values parameter
    3. The returned joint configuration produces the same pose via FK
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution for a reachable pose
    assume(len(solutions) > 0)
    
    # Test each solution
    for sol in solutions:
        # Check that solution has no free parameters (for this solver)
        free_indices = sol.get_free_indices()
        num_free = len(free_indices)
        
        # For this specific solver, there should be no free parameters
        assert num_free == 0, f"Expected 0 free parameters, got {num_free}"
        
        # Get solution without free values (should work since no free parameters)
        joint_solution = sol.get_solution()
        
        # Verify it's a valid numpy array
        assert isinstance(joint_solution, np.ndarray)
        assert joint_solution.dtype == np.float64
        assert joint_solution.shape == (6,)
        
        # All values should be finite
        assert np.all(np.isfinite(joint_solution)), "Joint solution contains non-finite values"
        
        # Verify the solution produces the same pose via FK
        trans_check, rot_check = ikfast.compute_fk_raw(joint_solution)
        
        # Check translation matches (within tolerance)
        trans_diff = np.linalg.norm(translation - trans_check)
        assert trans_diff < 1e-6, f"Translation mismatch: {trans_diff}"
        
        # Check rotation matches (within tolerance)
        # Reshape to 3x3 for easier comparison
        rot_orig = rotation.reshape(3, 3)
        rot_result = rot_check.reshape(3, 3)
        rot_diff = np.linalg.norm(rot_orig - rot_result)
        assert rot_diff < 1e-6, f"Rotation mismatch: {rot_diff}"


@given(
    translation=npst.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ),
    rotation=npst.arrays(
        dtype=np.float64,
        shape=(3, 3),
        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_property_8_free_parameter_handling_arbitrary_pose(translation, rotation):
    """
    Feature: ikfast-python-binding, Property 8: Free Parameter Handling
    
    Test with arbitrary poses (may or may not be reachable).
    Verifies that when solutions exist, they correctly handle free parameters.
    """
    # Filter out invalid rotation matrices (determinant should be close to 1)
    det = np.linalg.det(rotation)
    assume(0.9 < abs(det) < 1.1)
    
    # Normalize to make it a proper rotation matrix
    U, _, Vt = np.linalg.svd(rotation)
    rotation_normalized = U @ Vt
    
    # Ensure determinant is +1 (not -1, which would be a reflection)
    if np.linalg.det(rotation_normalized) < 0:
        rotation_normalized[:, 0] *= -1
    
    try:
        # Compute IK
        solutions = ikfast.compute_ik_raw(translation, rotation_normalized.flatten())
        
        # If no solutions, that's OK (pose might be unreachable)
        if len(solutions) == 0:
            return
        
        # Test each solution
        for sol in solutions:
            # Get free indices
            free_indices = sol.get_free_indices()
            num_free = len(free_indices)
            
            # For this solver, should have no free parameters
            assert num_free == 0
            
            # Get solution
            joint_solution = sol.get_solution()
            
            # Verify it's valid
            assert isinstance(joint_solution, np.ndarray)
            assert joint_solution.dtype == np.float64
            assert joint_solution.shape == (6,)
            assert np.all(np.isfinite(joint_solution))
            
            # Verify FK produces the same pose
            trans_check, rot_check = ikfast.compute_fk_raw(joint_solution)
            
            # Check translation
            trans_diff = np.linalg.norm(translation - trans_check)
            assert trans_diff < 1e-5, f"Translation mismatch: {trans_diff}"
            
            # Check rotation
            rot_result = rot_check.reshape(3, 3)
            rot_diff = np.linalg.norm(rotation_normalized - rot_result)
            assert rot_diff < 1e-5, f"Rotation mismatch: {rot_diff}"
            
    except ValueError:
        # Invalid input is acceptable
        pass


def test_free_parameter_handling_with_known_pose():
    """
    Test free parameter handling with a known reachable pose.
    
    This is a concrete example test to complement the property tests.
    """
    # Use a simple reachable pose
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64).flatten()
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution
    assert len(solutions) > 0
    
    # Test first solution
    sol = solutions[0]
    
    # Check free parameters
    free_indices = sol.get_free_indices()
    assert isinstance(free_indices, list)
    assert len(free_indices) == 0  # This solver has no free parameters
    
    # Get solution
    joint_solution = sol.get_solution()
    
    # Verify it's valid
    assert isinstance(joint_solution, np.ndarray)
    assert joint_solution.shape == (6,)
    assert np.all(np.isfinite(joint_solution))
    
    # Verify FK
    trans_check, rot_check = ikfast.compute_fk_raw(joint_solution)
    
    # Check translation
    assert np.allclose(translation, trans_check, atol=1e-6)
    
    # Check rotation
    rot_check_matrix = rot_check.reshape(3, 3)
    rot_orig_matrix = rotation.reshape(3, 3)
    assert np.allclose(rot_orig_matrix, rot_check_matrix, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
