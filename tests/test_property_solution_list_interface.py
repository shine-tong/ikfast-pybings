"""
Property-based tests for solution list interface compliance.

Feature: ikfast-python-binding, Property 7: Solution List Interface Compliance

Tests that solution lists support Python iteration protocols and each solution
is accessible as a numpy array.

Validates: Requirements 3.1, 3.2, 3.5
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
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


@settings(max_examples=100, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_7_solution_list_interface_compliance(joint_angles):
    """
    Feature: ikfast-python-binding, Property 7: Solution List Interface Compliance
    
    For any valid joint configuration, the solution list returned by IK
    should support Python iteration protocols (len, indexing, iteration)
    and each solution should be accessible as a numpy array.
    
    Validates: Requirements 3.1, 3.2, 3.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    # Property 1: Solution list should support len()
    num_solutions = len(solution_list)
    assert isinstance(num_solutions, int)
    assert num_solutions >= 0
    
    # Property 2: len() should match get_num_solutions()
    assert num_solutions == solution_list.get_num_solutions()
    
    # For reachable poses, we should have at least one solution
    assume(num_solutions > 0)
    
    # Property 3: Solution list should support indexing
    for i in range(num_solutions):
        solution = solution_list[i]
        
        # Each solution should have the required methods
        assert hasattr(solution, 'get_solution')
        assert hasattr(solution, 'get_free_indices')
        assert hasattr(solution, 'get_dof')
        
        # Property 4: Each solution should be accessible as numpy array
        joint_angles_result = solution.get_solution()
        assert isinstance(joint_angles_result, np.ndarray)
        assert joint_angles_result.dtype == np.float64
        assert joint_angles_result.shape == (6,)
        
        # Values should be finite
        assert np.all(np.isfinite(joint_angles_result))
    
    # Property 5: Solution list should support iteration
    solutions_from_iteration = []
    for solution in solution_list:
        assert hasattr(solution, 'get_solution')
        solutions_from_iteration.append(solution.get_solution())
    
    # Number of solutions from iteration should match len()
    assert len(solutions_from_iteration) == num_solutions
    
    # Property 6: Solutions from iteration should match solutions from indexing
    for i in range(num_solutions):
        solution_from_index = solution_list[i].get_solution()
        solution_from_iter = solutions_from_iteration[i]
        np.testing.assert_array_equal(solution_from_index, solution_from_iter)
    
    # Property 7: Negative indexing should work
    if num_solutions > 0:
        last_solution = solution_list[-1]
        last_solution_positive = solution_list[num_solutions - 1]
        np.testing.assert_array_equal(
            last_solution.get_solution(),
            last_solution_positive.get_solution()
        )
    
    # Property 8: Out-of-bounds access should raise IndexError
    with pytest.raises(IndexError):
        _ = solution_list[num_solutions]
    
    with pytest.raises(IndexError):
        _ = solution_list[-(num_solutions + 1)]


@settings(max_examples=50, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_7_solution_list_consistency(joint_angles):
    """
    Feature: ikfast-python-binding, Property 7: Solution List Interface Compliance
    
    For any valid joint configuration, accessing the same solution multiple times
    should return consistent results.
    
    Validates: Requirements 3.1, 3.2
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    assume(num_solutions > 0)
    
    # Access each solution multiple times and verify consistency
    for i in range(num_solutions):
        solution1 = solution_list[i]
        solution2 = solution_list[i]
        solution3 = solution_list[i]
        
        joints1 = solution1.get_solution()
        joints2 = solution2.get_solution()
        joints3 = solution3.get_solution()
        
        # All accesses should return the same values
        np.testing.assert_array_equal(joints1, joints2)
        np.testing.assert_array_equal(joints2, joints3)


@settings(max_examples=50, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_7_solution_list_immutability(joint_angles):
    """
    Feature: ikfast-python-binding, Property 7: Solution List Interface Compliance
    
    For any valid joint configuration, the solution list should remain unchanged
    during iteration and indexing operations.
    
    Validates: Requirements 3.1, 3.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions_initial = len(solution_list)
    assume(num_solutions_initial > 0)
    
    # Iterate through solutions
    for _ in solution_list:
        pass
    
    # Length should remain the same
    assert len(solution_list) == num_solutions_initial
    
    # Access all solutions by index
    for i in range(num_solutions_initial):
        _ = solution_list[i]
    
    # Length should still be the same
    assert len(solution_list) == num_solutions_initial
    
    # Iterate again
    count = 0
    for _ in solution_list:
        count += 1
    
    # Should iterate through the same number of solutions
    assert count == num_solutions_initial


@settings(max_examples=50, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_7_empty_solution_list_interface(joint_angles):
    """
    Feature: ikfast-python-binding, Property 7: Solution List Interface Compliance
    
    Empty solution lists (from unreachable poses) should still support
    all iteration protocols correctly.
    
    Validates: Requirements 3.1, 3.5
    """
    # Create an unreachable pose by using a very large translation
    translation = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64).flatten()
    
    # Compute IK (should return empty list)
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have zero length
    assert len(solution_list) == 0
    assert solution_list.get_num_solutions() == 0
    
    # Iteration should work but yield nothing
    count = 0
    for _ in solution_list:
        count += 1
    assert count == 0
    
    # Indexing should raise IndexError
    with pytest.raises(IndexError):
        _ = solution_list[0]
    
    with pytest.raises(IndexError):
        _ = solution_list[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
