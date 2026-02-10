"""
Property-based tests for index bounds checking.

Feature: ikfast-python-binding, Property 11: Index Bounds Checking

Tests that out-of-bounds access to solution lists raises IndexError.

Validates: Requirements 6.5
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
@given(
    joint_angles=joint_angles_strategy,
    offset=st.integers(min_value=1, max_value=100)
)
def test_property_11_positive_index_out_of_bounds(joint_angles, offset):
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    For any solution list, accessing an index >= len(list) should raise IndexError.
    
    Validates: Requirements 6.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    
    # Try to access beyond the last element
    out_of_bounds_index = num_solutions + offset
    
    with pytest.raises(IndexError) as exc_info:
        _ = solution_list[out_of_bounds_index]
    
    # Error message should be descriptive
    error_msg = str(exc_info.value)
    assert "Index out of range" in error_msg or "out of range" in error_msg.lower()


@settings(max_examples=100, deadline=None)
@given(
    joint_angles=joint_angles_strategy,
    offset=st.integers(min_value=1, max_value=100)
)
def test_property_11_negative_index_out_of_bounds(joint_angles, offset):
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    For any solution list, accessing an index < -len(list) should raise IndexError.
    
    Validates: Requirements 6.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    
    # Try to access before the first element
    out_of_bounds_index = -(num_solutions + offset)
    
    with pytest.raises(IndexError) as exc_info:
        _ = solution_list[out_of_bounds_index]
    
    # Error message should be descriptive
    error_msg = str(exc_info.value)
    assert "Index out of range" in error_msg or "out of range" in error_msg.lower()


@settings(max_examples=100, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_11_valid_indices_no_error(joint_angles):
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    For any solution list, accessing valid indices [0, len-1] and [-len, -1]
    should NOT raise IndexError.
    
    Validates: Requirements 6.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    assume(num_solutions > 0)
    
    # All positive indices should work
    for i in range(num_solutions):
        solution = solution_list[i]
        assert solution is not None
        assert hasattr(solution, 'get_solution')
    
    # All negative indices should work
    for i in range(-num_solutions, 0):
        solution = solution_list[i]
        assert solution is not None
        assert hasattr(solution, 'get_solution')


def test_property_11_empty_list_bounds_checking():
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    For empty solution lists, any index access should raise IndexError.
    
    Validates: Requirements 6.5
    """
    # Create an unreachable pose
    translation = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64).flatten()
    
    # Compute IK (should return empty list)
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    assert len(solution_list) == 0
    
    # Any index should raise IndexError
    test_indices = [0, 1, -1, -2, 10, -10, 100, -100]
    
    for index in test_indices:
        with pytest.raises(IndexError):
            _ = solution_list[index]


@settings(max_examples=100, deadline=None)
@given(
    joint_angles=joint_angles_strategy,
    index=st.integers(min_value=-1000, max_value=1000)
)
def test_property_11_arbitrary_index_bounds_checking(joint_angles, index):
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    For any solution list and any arbitrary index, the behavior should be:
    - If index is in valid range: return solution
    - If index is out of bounds: raise IndexError
    
    Validates: Requirements 6.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    
    # Determine if index is valid
    is_valid = False
    if num_solutions > 0:
        if 0 <= index < num_solutions:
            is_valid = True
        elif -num_solutions <= index < 0:
            is_valid = True
    
    if is_valid:
        # Should not raise error
        solution = solution_list[index]
        assert solution is not None
        assert hasattr(solution, 'get_solution')
    else:
        # Should raise IndexError
        with pytest.raises(IndexError):
            _ = solution_list[index]


@settings(max_examples=50, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_property_11_boundary_indices(joint_angles):
    """
    Feature: ikfast-python-binding, Property 11: Index Bounds Checking
    
    Test boundary conditions: first, last, and just-out-of-bounds indices.
    
    Validates: Requirements 6.5
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK to get solution list
    solution_list = ikfast.compute_ik_raw(translation, rotation)
    
    num_solutions = len(solution_list)
    assume(num_solutions > 0)
    
    # First element (positive index) - should work
    first_solution = solution_list[0]
    assert first_solution is not None
    
    # Last element (positive index) - should work
    last_solution = solution_list[num_solutions - 1]
    assert last_solution is not None
    
    # First element (negative index) - should work
    first_solution_neg = solution_list[-num_solutions]
    assert first_solution_neg is not None
    
    # Last element (negative index) - should work
    last_solution_neg = solution_list[-1]
    assert last_solution_neg is not None
    
    # Just beyond last element - should raise IndexError
    with pytest.raises(IndexError):
        _ = solution_list[num_solutions]
    
    # Just before first element - should raise IndexError
    with pytest.raises(IndexError):
        _ = solution_list[-(num_solutions + 1)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
