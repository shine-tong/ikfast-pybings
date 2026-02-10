"""
Property-based tests for FK-IK round trip consistency.

Feature: ikfast-python-binding
Tests Property 2: FK-IK Round Trip Consistency
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Helper strategies for generating valid poses
# ============================================================================

def valid_rotation_matrix_strategy():
    """
    Generate valid rotation matrices using axis-angle representation.
    This ensures we get proper rotation matrices with det=1.
    """
    @st.composite
    def rotation_from_axis_angle(draw):
        # Generate random axis (unit vector)
        axis = draw(npst.arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ))
        
        # Normalize axis (skip if zero vector)
        norm = np.linalg.norm(axis)
        assume(norm > 1e-6)
        axis = axis / norm
        
        # Generate random angle
        angle = draw(st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False))
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R.flatten()
    
    return rotation_from_axis_angle()


def reachable_pose_strategy():
    """
    Generate poses that are likely to be reachable by first computing FK
    from random joint configurations.
    """
    @st.composite
    def pose_from_fk(draw):
        # Generate random joint configuration
        joints = draw(npst.arrays(
            dtype=np.float64,
            shape=(6,),
            elements=st.floats(
                min_value=-np.pi,
                max_value=np.pi,
                allow_nan=False,
                allow_infinity=False
            )
        ))
        
        # Compute FK to get a reachable pose
        translation, rotation = ikfast.compute_fk_raw(joints)
        
        return translation, rotation
    
    return pose_from_fk()


# ============================================================================
# Property 2: FK-IK Round Trip Consistency
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(pose=reachable_pose_strategy())
def test_property_2_fk_ik_round_trip(pose):
    """
    Feature: ikfast-python-binding, Property 2: FK-IK Round Trip Consistency
    
    For any reachable end effector pose, IK then FK should produce
    the same end effector pose.
    
    Validates: Requirements 1.1, 2.1
    """
    translation, rotation = pose
    
    # Compute IK to get joint solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution for a reachable pose
    assert len(solutions) > 0, \
        f"IK should find solutions for reachable pose\n" \
        f"Translation: {translation}\n" \
        f"Rotation: {rotation}"
    
    # For each solution, verify that FK produces the same pose
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        
        # Compute FK for this solution
        trans_check, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        # The pose should match the original
        assert np.allclose(translation, trans_check, atol=1e-6), \
            f"Translation mismatch for solution {i}\n" \
            f"Original: {translation}\n" \
            f"FK result: {trans_check}\n" \
            f"Joint angles: {joint_angles}"
        
        assert np.allclose(rotation, rot_check, atol=1e-6), \
            f"Rotation mismatch for solution {i}\n" \
            f"Original: {rotation}\n" \
            f"FK result: {rot_check}\n" \
            f"Joint angles: {joint_angles}"


@settings(max_examples=100, deadline=None)
@given(pose=reachable_pose_strategy())
def test_property_2_fk_ik_round_trip_translation_preserved(pose):
    """
    Feature: ikfast-python-binding, Property 2: FK-IK Round Trip Consistency
    
    Test that translation is preserved in IK→FK round trip.
    
    Validates: Requirements 1.1, 2.1
    """
    translation, rotation = pose
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution
    assert len(solutions) > 0
    
    # Check all solutions preserve translation
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        trans_check, _ = ikfast.compute_fk_raw(joint_angles)
        
        assert np.allclose(translation, trans_check, atol=1e-6), \
            f"Translation not preserved for solution {i}"


@settings(max_examples=100, deadline=None)
@given(pose=reachable_pose_strategy())
def test_property_2_fk_ik_round_trip_rotation_preserved(pose):
    """
    Feature: ikfast-python-binding, Property 2: FK-IK Round Trip Consistency
    
    Test that rotation is preserved in IK→FK round trip.
    
    Validates: Requirements 1.1, 2.1
    """
    translation, rotation = pose
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution
    assert len(solutions) > 0
    
    # Check all solutions preserve rotation
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        _, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        assert np.allclose(rotation, rot_check, atol=1e-6), \
            f"Rotation not preserved for solution {i}"


@settings(max_examples=100, deadline=None)
@given(pose=reachable_pose_strategy())
def test_property_2_all_solutions_valid(pose):
    """
    Feature: ikfast-python-binding, Property 2: FK-IK Round Trip Consistency
    
    Test that all IK solutions produce valid poses when verified with FK.
    
    Validates: Requirements 1.1, 2.1
    """
    translation, rotation = pose
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # Should have at least one solution
    assert len(solutions) > 0
    
    # Verify all solutions
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        
        # Verify joint angles are valid (no NaN or Inf)
        assert not np.any(np.isnan(joint_angles)), \
            f"Solution {i} contains NaN values"
        assert not np.any(np.isinf(joint_angles)), \
            f"Solution {i} contains Inf values"
        
        # Compute FK
        trans_check, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        # Verify FK results match original pose
        assert np.allclose(translation, trans_check, atol=1e-6), \
            f"Solution {i} does not produce correct translation"
        assert np.allclose(rotation, rot_check, atol=1e-6), \
            f"Solution {i} does not produce correct rotation"


@settings(max_examples=50, deadline=None)
@given(
    translation=npst.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ),
    rotation=valid_rotation_matrix_strategy()
)
def test_property_2_fk_ik_round_trip_arbitrary_poses(translation, rotation):
    """
    Feature: ikfast-python-binding, Property 2: FK-IK Round Trip Consistency
    
    Test round trip with arbitrary (possibly unreachable) poses.
    If IK finds solutions, they should produce the same pose via FK.
    
    Validates: Requirements 1.1, 2.1
    """
    # Compute IK (may return empty list for unreachable poses)
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # If solutions exist, verify them
    if len(solutions) > 0:
        for i in range(len(solutions)):
            solution = solutions[i]
            joint_angles = solution.get_solution()
            
            # Compute FK
            trans_check, rot_check = ikfast.compute_fk_raw(joint_angles)
            
            # Verify pose matches
            assert np.allclose(translation, trans_check, atol=1e-6), \
                f"Solution {i} translation mismatch"
            assert np.allclose(rotation, rot_check, atol=1e-6), \
                f"Solution {i} rotation mismatch"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
