"""
Property-based tests for IK-FK round trip consistency.

Feature: ikfast-python-binding
Tests Property 1: IK-FK Round Trip Consistency
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst


# Import the module
import ikfast_pybind._ikfast_pybind as ikfast


# ============================================================================
# Property 1: IK-FK Round Trip Consistency
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
def test_property_1_ik_fk_round_trip(joints):
    """
    Feature: ikfast-python-binding, Property 1: IK-FK Round Trip Consistency
    
    For any valid joint configuration, FK then IK should produce
    a configuration that yields the same end effector pose.
    
    Validates: Requirements 1.1, 2.1
    """
    # Compute FK to get end effector pose
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Compute IK to get joint solutions
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # At least one solution should exist for a reachable pose
    assert len(solutions) > 0, \
        f"IK should find at least one solution for FK-computed pose from joints {joints}"
    
    # Check that at least one solution produces the same pose
    found_match = False
    for i in range(len(solutions)):
        solution = solutions[i]
        
        # Get joint angles from solution
        joint_angles = solution.get_solution()
        
        # Compute FK for this solution
        trans_check, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        # Check if this solution produces the same pose
        translation_match = np.allclose(translation, trans_check, atol=1e-6)
        rotation_match = np.allclose(rotation, rot_check, atol=1e-6)
        
        if translation_match and rotation_match:
            found_match = True
            break
    
    assert found_match, \
        f"No IK solution produced the original pose.\n" \
        f"Original joints: {joints}\n" \
        f"Original translation: {translation}\n" \
        f"Original rotation: {rotation}\n" \
        f"Number of IK solutions: {len(solutions)}"


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
def test_property_1_ik_fk_round_trip_translation_only(joints):
    """
    Feature: ikfast-python-binding, Property 1: IK-FK Round Trip Consistency
    
    Test that translation is preserved in FK→IK round trip.
    
    Validates: Requirements 1.1, 2.1
    """
    # Compute FK
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # At least one solution should exist
    assert len(solutions) > 0
    
    # Check that at least one solution preserves translation
    found_match = False
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        trans_check, _ = ikfast.compute_fk_raw(joint_angles)
        
        if np.allclose(translation, trans_check, atol=1e-6):
            found_match = True
            break
    
    assert found_match, \
        f"No IK solution preserved the original translation {translation}"


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
def test_property_1_ik_fk_round_trip_rotation_only(joints):
    """
    Feature: ikfast-python-binding, Property 1: IK-FK Round Trip Consistency
    
    Test that rotation is preserved in FK→IK round trip.
    
    Validates: Requirements 1.1, 2.1
    """
    # Compute FK
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # At least one solution should exist
    assert len(solutions) > 0
    
    # Check that at least one solution preserves rotation
    found_match = False
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        _, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        if np.allclose(rotation, rot_check, atol=1e-6):
            found_match = True
            break
    
    assert found_match, \
        f"No IK solution preserved the original rotation {rotation}"


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
def test_property_1_ik_fk_round_trip_extended_range(joints):
    """
    Feature: ikfast-python-binding, Property 1: IK-FK Round Trip Consistency
    
    Test round trip consistency with extended joint angle range.
    
    Validates: Requirements 1.1, 2.1
    """
    # Compute FK
    translation, rotation = ikfast.compute_fk_raw(joints)
    
    # Compute IK
    solutions = ikfast.compute_ik_raw(translation, rotation)
    
    # At least one solution should exist
    assert len(solutions) > 0
    
    # Check that at least one solution produces the same pose
    found_match = False
    for i in range(len(solutions)):
        solution = solutions[i]
        joint_angles = solution.get_solution()
        trans_check, rot_check = ikfast.compute_fk_raw(joint_angles)
        
        if (np.allclose(translation, trans_check, atol=1e-6) and
            np.allclose(rotation, rot_check, atol=1e-6)):
            found_match = True
            break
    
    assert found_match, \
        f"Round trip failed for extended range joints {joints}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
