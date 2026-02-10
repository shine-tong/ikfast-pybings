"""
Unit tests for IkSolution class.

Tests the IkSolution wrapper class functionality including:
- get_solution() returns numpy array of correct shape
- get_free_indices() returns list of integers
- get_dof() returns correct value
- get_solution() with free parameters
"""

import pytest
import numpy as np
import ikfast_pybind._ikfast_pybind as ikfast


def test_ik_solution_get_solution_returns_numpy_array():
    """Test that get_solution() returns a numpy array of correct shape."""
    # Use a known reachable pose to get solutions
    # This is a simple forward position
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution for a reachable pose
    assert len(solutions) > 0, "Expected at least one solution for reachable pose"
    
    # Get first solution
    sol = solutions[0]
    joint_angles = sol.get_solution()
    
    # Check return type
    assert isinstance(joint_angles, np.ndarray), "get_solution() should return numpy array"
    
    # Check dtype
    assert joint_angles.dtype == np.float64, "get_solution() should return float64 array"
    
    # Check shape
    num_joints = ikfast.get_num_joints()
    assert joint_angles.shape == (num_joints,), f"get_solution() should return array of shape ({num_joints},)"


def test_ik_solution_get_free_indices_returns_list():
    """Test that get_free_indices() returns a list of integers."""
    # Use a known reachable pose to get solutions
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    assert len(solutions) > 0, "Expected at least one solution"
    
    # Get first solution
    sol = solutions[0]
    free_indices = sol.get_free_indices()
    
    # Check return type
    assert isinstance(free_indices, list), "get_free_indices() should return a list"
    
    # Check that all elements are integers
    for idx in free_indices:
        assert isinstance(idx, int), "All free indices should be integers"
    
    # For this solver, there should be no free parameters
    num_free = ikfast.get_num_free_parameters()
    assert len(free_indices) == num_free, f"Number of free indices should match get_num_free_parameters() = {num_free}"


def test_ik_solution_get_dof_returns_correct_value():
    """Test that get_dof() returns the correct number of degrees of freedom."""
    # Use a known reachable pose to get solutions
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    assert len(solutions) > 0, "Expected at least one solution"
    
    # Get first solution
    sol = solutions[0]
    dof = sol.get_dof()
    
    # Check return type
    assert isinstance(dof, int), "get_dof() should return an integer"
    
    # Check value matches get_num_joints()
    num_joints = ikfast.get_num_joints()
    assert dof == num_joints, f"get_dof() should return {num_joints}"


def test_ik_solution_with_no_free_parameters():
    """Test get_solution() when there are no free parameters."""
    # Use a known reachable pose to get solutions
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    assert len(solutions) > 0, "Expected at least one solution"
    
    # Get first solution
    sol = solutions[0]
    
    # Call get_solution() without free_values (should work if no free parameters)
    joint_angles = sol.get_solution()
    
    # Should return valid joint angles
    assert joint_angles is not None
    assert len(joint_angles) == ikfast.get_num_joints()
    
    # All values should be finite
    assert np.all(np.isfinite(joint_angles)), "All joint angles should be finite"


def test_ik_solution_multiple_solutions():
    """Test that multiple solutions can be accessed independently."""
    # Use a known reachable pose to get solutions
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # If there are multiple solutions, test each one
    if len(solutions) > 1:
        for i, sol in enumerate(solutions):
            joint_angles = sol.get_solution()
            
            # Each solution should be valid
            assert isinstance(joint_angles, np.ndarray)
            assert joint_angles.shape == (ikfast.get_num_joints(),)
            assert np.all(np.isfinite(joint_angles))
            
            # Each solution should have the same DOF
            assert sol.get_dof() == ikfast.get_num_joints()


def test_ik_solution_invalid_free_values_size():
    """Test that providing wrong size free_values raises ValueError."""
    # Use a known reachable pose to get solutions
    translation = np.array([0.5, 0.0, 0.5], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    
    # Get solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    assert len(solutions) > 0, "Expected at least one solution"
    
    # Get first solution
    sol = solutions[0]
    num_free = len(sol.get_free_indices())
    
    # If there are free parameters, test with wrong size
    if num_free > 0:
        # Provide wrong number of free values
        wrong_free_values = np.array([0.0] * (num_free + 1), dtype=np.float64)
        
        with pytest.raises(ValueError, match="Invalid free_values size"):
            sol.get_solution(wrong_free_values)
    else:
        # If no free parameters, providing free values should still work (they'll be ignored)
        # or raise an error - either is acceptable
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
