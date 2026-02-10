"""
Unit tests for compute_ik_raw function.

Tests the low-level compute_ik_raw binding that wraps the C++ ComputeIk function.
"""

import pytest
import numpy as np
import ikfast_pybind._ikfast_pybind as ikfast


class TestComputeIkRaw:
    """Test suite for compute_ik_raw function."""
    
    def test_compute_ik_with_reachable_pose(self):
        """Test compute_ik_raw with a known reachable pose."""
        # Use a known joint configuration
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Compute FK to get a reachable pose
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Compute IK for this pose
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Should return an IkSolutionList
        assert hasattr(solutions, '__len__')
        assert hasattr(solutions, '__getitem__')
        
        # Should have at least one solution
        assert len(solutions) > 0
        
        # Verify at least one solution produces the same pose
        found_match = False
        for sol in solutions:
            sol_joints = sol.get_solution()
            trans_check, rot_check = ikfast.compute_fk_raw(sol_joints)
            
            if (np.allclose(translation, trans_check, atol=1e-6) and
                np.allclose(rotation, rot_check, atol=1e-6)):
                found_match = True
                break
        
        assert found_match, "No IK solution produced the original pose"
    
    def test_compute_ik_return_type(self):
        """Test that compute_ik_raw returns IkSolutionList."""
        # Use a simple reachable pose
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Compute IK
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Check type
        assert type(solutions).__name__ == 'IkSolutionList'
        
        # Check it supports list operations
        assert hasattr(solutions, '__len__')
        assert hasattr(solutions, '__getitem__')
        assert hasattr(solutions, '__iter__')
    
    def test_compute_ik_with_flat_rotation(self):
        """Test compute_ik_raw with flat rotation array [9]."""
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # rotation is already flat [9]
        assert rotation.shape == (9,)
        
        # Should work with flat rotation
        solutions = ikfast.compute_ik_raw(translation, rotation)
        assert len(solutions) >= 0  # May be empty or have solutions
    
    def test_compute_ik_with_matrix_rotation(self):
        """Test compute_ik_raw with 3x3 rotation matrix."""
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Reshape to 3x3 matrix
        rotation_matrix = rotation.reshape(3, 3)
        
        # Should work with matrix rotation
        solutions = ikfast.compute_ik_raw(translation, rotation_matrix)
        assert len(solutions) >= 0  # May be empty or have solutions
    
    def test_compute_ik_invalid_translation_shape(self):
        """Test that invalid translation shape raises ValueError."""
        # Wrong shape for translation
        translation = np.array([1.0, 2.0], dtype=np.float64)  # Should be [3]
        rotation = np.eye(3, dtype=np.float64)
        
        with pytest.raises(ValueError) as exc_info:
            ikfast.compute_ik_raw(translation, rotation)
        
        # Check error message contains expected and actual shapes
        assert "eetrans" in str(exc_info.value)
        assert "Expected" in str(exc_info.value) or "expected" in str(exc_info.value).lower()
    
    def test_compute_ik_invalid_rotation_shape(self):
        """Test that invalid rotation shape raises ValueError."""
        translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        # Wrong shape for rotation
        rotation = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)  # Should be [9] or [3,3]
        
        with pytest.raises(ValueError) as exc_info:
            ikfast.compute_ik_raw(translation, rotation)
        
        # Check error message contains expected and actual shapes
        assert "eerot" in str(exc_info.value)
        assert "Expected" in str(exc_info.value) or "expected" in str(exc_info.value).lower()
    
    def test_compute_ik_empty_solution_for_unreachable_pose(self):
        """Test that unreachable pose returns empty solution list without exception."""
        # Use a pose that is likely unreachable (very far away)
        translation = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64).flatten()
        
        # Should not raise exception
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Should return empty list
        assert len(solutions) == 0
    
    def test_compute_ik_with_optional_free_parameters(self):
        """Test compute_ik_raw with optional free parameters."""
        # Get a reachable pose
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Check if solver has free parameters
        num_free = ikfast.get_num_free_parameters()
        
        if num_free > 0:
            # Test with free parameters
            free_params = np.zeros(num_free, dtype=np.float64)
            solutions = ikfast.compute_ik_raw(translation, rotation, free_params)
            assert len(solutions) >= 0
        else:
            # Test without free parameters (should work with None)
            solutions = ikfast.compute_ik_raw(translation, rotation)
            assert len(solutions) >= 0
            
            # Test with None explicitly
            solutions = ikfast.compute_ik_raw(translation, rotation, None)
            assert len(solutions) >= 0
    
    def test_compute_ik_solutions_are_valid(self):
        """Test that all returned solutions produce valid joint configurations."""
        # Use a known reachable pose
        joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        # Compute IK
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Check each solution
        for i, sol in enumerate(solutions):
            # Should be able to get solution
            sol_joints = sol.get_solution()
            
            # Should be numpy array
            assert isinstance(sol_joints, np.ndarray)
            
            # Should have correct shape
            assert sol_joints.shape == (6,)
            
            # Should have correct dtype
            assert sol_joints.dtype == np.float64
            
            # Should not contain NaN or inf
            assert not np.any(np.isnan(sol_joints))
            assert not np.any(np.isinf(sol_joints))
