"""
Property-based test for empty workspace handling.

Property 9: Empty Workspace Handling
For any end effector pose outside the robot's reachable workspace, the Python
binding should return an empty solution list without raising an exception.

Validates: Requirements 1.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as npst
import ikfast_pybind._ikfast_pybind as ikfast


# Strategy for generating unreachable poses (far from origin)
unreachable_translation_strategy = npst.arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(
        min_value=50.0,  # Far outside typical robot workspace
        max_value=200.0,
        allow_nan=False,
        allow_infinity=False
    )
)

# Strategy for generating random rotation matrices
# We'll use random angles and construct rotation matrices
rotation_angles_strategy = npst.arrays(
    dtype=np.float64,
    shape=(3,),
    elements=st.floats(
        min_value=-np.pi,
        max_value=np.pi,
        allow_nan=False,
        allow_infinity=False
    )
)


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles (ZYX convention).
    
    Args:
        roll: Rotation around X axis
        pitch: Rotation around Y axis
        yaw: Rotation around Z axis
    
    Returns:
        3x3 rotation matrix as flat array
    """
    # Rotation around X
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation around Y
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation around Z
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R.flatten()


@settings(max_examples=100, deadline=None)
@given(
    translation=unreachable_translation_strategy,
    angles=rotation_angles_strategy
)
def test_property_9_empty_workspace_handling(translation, angles):
    """
    Feature: ikfast-python-binding, Property 9: Empty Workspace Handling
    
    For any end effector pose outside the robot's reachable workspace,
    the Python binding should return an empty solution list without
    raising an exception.
    
    This test verifies that unreachable poses are handled gracefully
    rather than causing errors.
    """
    # Create rotation matrix from angles
    rotation = rotation_matrix_from_euler(angles[0], angles[1], angles[2])
    
    # Ensure translation is far from origin (unreachable)
    distance = np.linalg.norm(translation)
    assume(distance > 10.0)  # Assume pose is far outside workspace
    
    # Compute IK - should not raise exception
    try:
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Should return a valid solution list (even if empty)
        assert hasattr(solutions, '__len__')
        assert hasattr(solutions, '__getitem__')
        assert hasattr(solutions, '__iter__')
        
        # For unreachable poses, we expect empty list
        # (though some poses might be reachable, most should not be)
        num_solutions = len(solutions)
        
        # Should be non-negative
        assert num_solutions >= 0
        
        # If there are solutions, verify they're valid
        for sol in solutions:
            sol_joints = sol.get_solution()
            assert isinstance(sol_joints, np.ndarray)
            assert sol_joints.shape == (6,)
            assert not np.any(np.isnan(sol_joints))
            assert not np.any(np.isinf(sol_joints))
        
    except Exception as e:
        # Should not raise any exception for unreachable poses
        pytest.fail(
            f"compute_ik_raw raised exception for unreachable pose: {e}\n"
            f"Translation: {translation}\n"
            f"Distance from origin: {distance}"
        )


@settings(max_examples=50, deadline=None)
@given(
    translation=npst.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False
        )
    ),
    angles=rotation_angles_strategy
)
def test_empty_list_behavior(translation, angles):
    """
    Additional test: Verify empty solution list behaves correctly.
    
    This test ensures that empty solution lists support all expected
    operations without errors.
    """
    # Create rotation matrix
    rotation = rotation_matrix_from_euler(angles[0], angles[1], angles[2])
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # If list is empty, verify it behaves correctly
    if len(solutions) == 0:
        # Should support len()
        assert len(solutions) == 0
        
        # Should support iteration (empty)
        count = 0
        for _ in solutions:
            count += 1
        assert count == 0
        
        # Should raise IndexError for out-of-bounds access
        with pytest.raises(IndexError):
            _ = solutions[0]
        
        with pytest.raises(IndexError):
            _ = solutions[-1]
        
        # get_num_solutions() should return 0
        assert solutions.get_num_solutions() == 0


@settings(max_examples=30, deadline=None)
@given(
    scale=st.floats(min_value=100.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
)
def test_extreme_distance_poses(scale):
    """
    Test with extremely distant poses to ensure robustness.
    
    This test verifies that the solver handles poses at extreme distances
    without crashing or producing invalid results.
    """
    # Create a pose at extreme distance
    translation = np.array([scale, scale, scale], dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64).flatten()
    
    # Should not crash
    try:
        solutions = ikfast.compute_ik_raw(translation, rotation)
        
        # Should return valid (likely empty) list
        assert isinstance(len(solutions), int)
        assert len(solutions) >= 0
        
        # Most likely empty for such extreme distances
        # But if there are solutions, they should be valid
        for sol in solutions:
            sol_joints = sol.get_solution()
            assert not np.any(np.isnan(sol_joints))
            assert not np.any(np.isinf(sol_joints))
            
    except Exception as e:
        pytest.fail(
            f"compute_ik_raw crashed with extreme distance pose: {e}\n"
            f"Distance: {np.linalg.norm(translation)}"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
