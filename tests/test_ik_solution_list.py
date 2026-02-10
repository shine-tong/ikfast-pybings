"""
Unit tests for IkSolutionList class.

Tests the Python list-like interface of IkSolutionList including:
- len() support
- Indexing with valid indices
- Iteration with for loop
- IndexError on out-of-bounds access
"""

import pytest
import numpy as np
import ikfast_pybind._ikfast_pybind as ikfast


class TestIkSolutionList:
    """Test suite for IkSolutionList class."""
    
    @pytest.fixture
    def sample_pose(self):
        """Provide a sample reachable pose for testing."""
        # Use a known reachable pose from forward kinematics
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        translation, rotation = ikfast.compute_fk_raw(joints)
        return translation, rotation
    
    @pytest.fixture
    def solution_list(self, sample_pose):
        """Provide a solution list with at least one solution."""
        translation, rotation = sample_pose
        solutions = ikfast.compute_ik_raw(translation, rotation)
        return solutions
    
    def test_len_works_correctly(self, solution_list):
        """Test that len() returns the correct number of solutions."""
        # Get number of solutions using both methods
        num_solutions_len = len(solution_list)
        num_solutions_method = solution_list.get_num_solutions()
        
        # They should match
        assert num_solutions_len == num_solutions_method
        
        # Should be at least one solution for a reachable pose
        assert num_solutions_len > 0
        
        # Should be a reasonable number (typically 1-8 for 6-DOF)
        assert num_solutions_len <= 16
    
    def test_indexing_with_valid_indices(self, solution_list):
        """Test that indexing with valid indices returns IkSolution objects."""
        num_solutions = len(solution_list)
        
        # Test positive indices
        for i in range(num_solutions):
            solution = solution_list[i]
            
            # Should be an IkSolution object
            assert hasattr(solution, 'get_solution')
            assert hasattr(solution, 'get_free_indices')
            assert hasattr(solution, 'get_dof')
            
            # Should be able to get joint angles
            joint_angles = solution.get_solution()
            assert isinstance(joint_angles, np.ndarray)
            assert joint_angles.shape == (6,)
            assert joint_angles.dtype == np.float64
    
    def test_negative_indexing(self, solution_list):
        """Test that negative indexing works correctly."""
        num_solutions = len(solution_list)
        
        if num_solutions > 0:
            # Test last element
            last_solution = solution_list[-1]
            assert hasattr(last_solution, 'get_solution')
            
            # Should be same as positive index
            last_solution_positive = solution_list[num_solutions - 1]
            joints_negative = last_solution.get_solution()
            joints_positive = last_solution_positive.get_solution()
            np.testing.assert_array_almost_equal(joints_negative, joints_positive)
        
        if num_solutions > 1:
            # Test second-to-last element
            second_last = solution_list[-2]
            assert hasattr(second_last, 'get_solution')
    
    def test_iteration_with_for_loop(self, solution_list):
        """Test that iteration with for loop works correctly."""
        num_solutions = len(solution_list)
        
        # Iterate and count
        count = 0
        for solution in solution_list:
            count += 1
            
            # Each solution should be valid
            assert hasattr(solution, 'get_solution')
            assert hasattr(solution, 'get_free_indices')
            assert hasattr(solution, 'get_dof')
            
            # Should be able to get joint angles
            joint_angles = solution.get_solution()
            assert isinstance(joint_angles, np.ndarray)
            assert joint_angles.shape == (6,)
        
        # Should have iterated through all solutions
        assert count == num_solutions
    
    def test_index_error_on_out_of_bounds_positive(self, solution_list):
        """Test that out-of-bounds positive index raises IndexError."""
        num_solutions = len(solution_list)
        
        # Try to access beyond the last element
        with pytest.raises(IndexError) as exc_info:
            _ = solution_list[num_solutions]
        
        # Error message should mention the index
        error_msg = str(exc_info.value)
        assert "Index out of range" in error_msg or "out of range" in error_msg.lower()
        assert str(num_solutions) in error_msg
    
    def test_index_error_on_out_of_bounds_negative(self, solution_list):
        """Test that out-of-bounds negative index raises IndexError."""
        num_solutions = len(solution_list)
        
        # Try to access before the first element
        with pytest.raises(IndexError) as exc_info:
            _ = solution_list[-(num_solutions + 1)]
        
        # Error message should mention index out of range
        error_msg = str(exc_info.value)
        assert "Index out of range" in error_msg or "out of range" in error_msg.lower()
    
    def test_empty_solution_list(self):
        """Test that empty solution list behaves correctly."""
        # Use an unreachable pose (very far away)
        translation = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64).flatten()
        
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Should have zero solutions
        assert len(solutions) == 0
        assert solutions.get_num_solutions() == 0
        
        # Iteration should work but yield nothing
        count = 0
        for _ in solutions:
            count += 1
        assert count == 0
        
        # Indexing should raise IndexError
        with pytest.raises(IndexError):
            _ = solutions[0]
    
    def test_clear_method(self, solution_list):
        """Test that clear() method removes all solutions."""
        # Verify we have solutions initially
        initial_count = len(solution_list)
        assert initial_count > 0
        
        # Clear the list
        solution_list.clear()
        
        # Should now be empty
        assert len(solution_list) == 0
        assert solution_list.get_num_solutions() == 0
        
        # Indexing should raise IndexError
        with pytest.raises(IndexError):
            _ = solution_list[0]
    
    def test_get_num_solutions_method(self, solution_list):
        """Test that get_num_solutions() method works correctly."""
        num_solutions = solution_list.get_num_solutions()
        
        # Should be a non-negative integer
        assert isinstance(num_solutions, int)
        assert num_solutions >= 0
        
        # Should match len()
        assert num_solutions == len(solution_list)
    
    def test_solution_list_consistency(self, solution_list):
        """Test that accessing the same index multiple times returns consistent results."""
        if len(solution_list) > 0:
            # Access first solution multiple times
            solution1 = solution_list[0]
            solution2 = solution_list[0]
            
            # Get joint angles from both
            joints1 = solution1.get_solution()
            joints2 = solution2.get_solution()
            
            # Should be identical
            np.testing.assert_array_equal(joints1, joints2)
    
    def test_multiple_poses(self):
        """Test solution list with multiple different poses."""
        # Test several different joint configurations
        test_configs = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
        
        for joints in test_configs:
            # Compute FK
            translation, rotation = ikfast.compute_fk_raw(joints)
            
            # Compute IK
            solutions = ikfast.compute_ik_raw(translation, rotation)
            
            # Should have at least one solution
            assert len(solutions) > 0
            
            # All solutions should be valid
            for solution in solutions:
                joint_angles = solution.get_solution()
                assert joint_angles.shape == (6,)
                assert joint_angles.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
