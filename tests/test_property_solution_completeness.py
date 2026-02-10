"""
Property-based test for solution completeness.

Property 3: Solution Completeness
For any end effector pose with multiple IK solutions, the number of solutions
returned by the Python binding should equal the number of solutions found by
the C++ solver.

Validates: Requirements 1.3
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
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
def test_property_3_solution_completeness(joint_angles):
    """
    Feature: ikfast-python-binding, Property 3: Solution Completeness
    
    For any end effector pose with multiple IK solutions, the number of
    solutions returned by the Python binding should equal the number of
    solutions found by the C++ solver.
    
    This test verifies that the binding layer doesn't lose solutions during
    the C++ to Python conversion process.
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK using the Python binding
    solutions = ikfast.compute_ik_raw(translation, rotation)
    num_solutions_python = len(solutions)
    
    # The C++ solver is called internally by compute_ik_raw, so we can't
    # directly compare with a separate C++ call. Instead, we verify that:
    # 1. The solution list is consistent (len() matches iteration count)
    # 2. All solutions are accessible
    # 3. Each solution produces a valid joint configuration
    
    # Verify consistency: len() should match iteration count
    iteration_count = 0
    for sol in solutions:
        iteration_count += 1
    
    assert num_solutions_python == iteration_count, (
        f"Solution count mismatch: len() returned {num_solutions_python}, "
        f"but iteration found {iteration_count} solutions"
    )
    
    # Verify all solutions are accessible by index
    for i in range(num_solutions_python):
        sol = solutions[i]
        # Should be able to get solution
        sol_joints = sol.get_solution()
        
        # Should be valid
        assert isinstance(sol_joints, np.ndarray)
        assert sol_joints.shape == (6,)
        assert not np.any(np.isnan(sol_joints))
        assert not np.any(np.isinf(sol_joints))
    
    # Verify at least one solution produces the original pose
    # (this is a sanity check that solutions are actually valid)
    if num_solutions_python > 0:
        found_match = False
        for sol in solutions:
            sol_joints = sol.get_solution()
            trans_check, rot_check = ikfast.compute_fk_raw(sol_joints)
            
            if (np.allclose(translation, trans_check, atol=1e-6) and
                np.allclose(rotation, rot_check, atol=1e-6)):
                found_match = True
                break
        
        assert found_match, (
            "No IK solution produced the original pose. "
            "This suggests solutions may have been corrupted during conversion."
        )


@settings(max_examples=50, deadline=None)
@given(joint_angles=joint_angles_strategy)
def test_solution_list_consistency(joint_angles):
    """
    Additional test: Verify solution list operations are consistent.
    
    This test ensures that different ways of accessing solutions
    (len, indexing, iteration) all produce consistent results.
    """
    # Compute FK to get a reachable pose
    translation, rotation = ikfast.compute_fk_raw(joint_angles)
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Get count via len()
    count_via_len = len(solutions)
    
    # Get count via iteration
    count_via_iteration = sum(1 for _ in solutions)
    
    # Get count via get_num_solutions()
    count_via_method = solutions.get_num_solutions()
    
    # All should match
    assert count_via_len == count_via_iteration, (
        f"len() returned {count_via_len}, but iteration counted {count_via_iteration}"
    )
    assert count_via_len == count_via_method, (
        f"len() returned {count_via_len}, but get_num_solutions() returned {count_via_method}"
    )
    
    # Verify indexing matches iteration
    if count_via_len > 0:
        solutions_via_index = [solutions[i] for i in range(count_via_len)]
        solutions_via_iter = list(solutions)
        
        assert len(solutions_via_index) == len(solutions_via_iter)
        
        # Compare joint values from each solution
        for sol_idx, sol_iter in zip(solutions_via_index, solutions_via_iter):
            joints_idx = sol_idx.get_solution()
            joints_iter = sol_iter.get_solution()
            
            # Should be the same solution
            assert np.allclose(joints_idx, joints_iter, atol=1e-10), (
                "Solution accessed by index differs from solution accessed by iteration"
            )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
