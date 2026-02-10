#!/usr/bin/env python3
"""
Basic Forward Kinematics Example

This example demonstrates:
1. Computing FK for a joint configuration
2. Understanding the FK output format
3. Verifying IK solutions using FK
4. Working with rotation matrices
"""

import numpy as np
import ikfast_pybind as ik


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to Euler angles (ZYX convention).
    
    Returns angles in radians as (roll, pitch, yaw).
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return roll, pitch, yaw


def verify_rotation_matrix(R):
    """
    Verify that a matrix is a valid rotation matrix.
    
    A valid rotation matrix must be orthonormal (R^T * R = I) and have determinant 1.
    """
    # Check orthonormality
    should_be_identity = R.T @ R
    is_orthonormal = np.allclose(should_be_identity, np.eye(3), atol=1e-6)
    
    # Check determinant
    det = np.linalg.det(R)
    has_unit_det = np.isclose(det, 1.0, atol=1e-6)
    
    return is_orthonormal and has_unit_det


def main():
    """Main example demonstrating FK computation."""
    
    print("=" * 60)
    print("IKFast Python Bindings - Basic FK Example")
    print("=" * 60)
    
    # Example 1: FK for zero configuration
    print("\n" + "-" * 60)
    print("Example 1: FK for zero joint configuration")
    print("-" * 60)
    
    joints_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"\nJoint angles: {joints_zero}")
    
    translation, rotation = ik.compute_fk(joints_zero)
    
    print(f"\nEnd effector pose:")
    print(f"  Translation: {translation}")
    print(f"  Rotation matrix:")
    for row in rotation:
        print(f"    {np.round(row, 6)}")
    
    # Verify rotation matrix
    if verify_rotation_matrix(rotation):
        print("\n✓ Rotation matrix is valid (orthonormal with det=1)")
    else:
        print("\n✗ Warning: Rotation matrix may be invalid")
    
    # Convert to Euler angles
    roll, pitch, yaw = rotation_matrix_to_euler(rotation)
    print(f"\nEuler angles (roll, pitch, yaw):")
    print(f"  {np.degrees(roll):.2f}°, {np.degrees(pitch):.2f}°, {np.degrees(yaw):.2f}°")
    
    # Example 2: FK for various configurations
    print("\n" + "-" * 60)
    print("Example 2: FK for different joint configurations")
    print("-" * 60)
    
    test_configurations = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ]
    
    print("\nComputing FK for multiple configurations:")
    for i, joints in enumerate(test_configurations):
        translation, rotation = ik.compute_fk(joints)
        print(f"\nConfiguration {i+1}: {np.round(joints, 3)}")
        print(f"  Position: {np.round(translation, 4)}")
        print(f"  Distance from origin: {np.linalg.norm(translation):.4f}")
    
    # Example 3: Verifying IK solutions with FK
    print("\n" + "-" * 60)
    print("Example 3: Verifying IK solutions with FK")
    print("-" * 60)
    
    # Define a target pose
    target_translation = np.array([0.5, 0.0, 0.5])
    target_rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    print(f"\nTarget pose:")
    print(f"  Translation: {target_translation}")
    print(f"  Rotation: Identity matrix")
    
    # Compute IK
    print("\nComputing IK solutions...")
    solutions = ik.compute_ik(target_translation, target_rotation)
    print(f"Found {len(solutions)} solution(s)")
    
    if not solutions:
        print("No solutions found - trying a different pose")
        # Try a pose we know is reachable (from zero config)
        target_translation, target_rotation = ik.compute_fk(joints_zero)
        solutions = ik.compute_ik(target_translation, target_rotation)
        print(f"Found {len(solutions)} solution(s) for reachable pose")
    
    # Verify each solution
    if solutions:
        print("\nVerifying each solution with FK:")
        for i, solution in enumerate(solutions):
            print(f"\n  Solution {i+1}: {np.round(solution, 4)}")
            
            # Compute FK for this solution
            computed_trans, computed_rot = ik.compute_fk(solution)
            
            # Check translation error
            trans_error = np.linalg.norm(target_translation - computed_trans)
            print(f"    Translation error: {trans_error:.6e}")
            
            # Check rotation error
            rot_error = np.linalg.norm(target_rotation - computed_rot)
            print(f"    Rotation error: {rot_error:.6e}")
            
            # Verify
            if trans_error < 1e-6 and rot_error < 1e-6:
                print(f"    ✓ Solution verified")
            else:
                print(f"    ✗ Solution verification failed")
    
    # Example 4: Understanding rotation matrices
    print("\n" + "-" * 60)
    print("Example 4: Working with rotation matrices")
    print("-" * 60)
    
    # Create a rotation about Z-axis by 90 degrees
    angle = np.pi / 2
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    print(f"\nRotation matrix (90° about Z-axis):")
    for row in rotation_z:
        print(f"  {np.round(row, 4)}")
    
    # Try to find IK for this orientation
    test_translation = np.array([0.5, 0.0, 0.5])
    print(f"\nTrying IK with translation {test_translation}")
    print(f"and 90° rotation about Z-axis...")
    
    solutions = ik.compute_ik(test_translation, rotation_z)
    print(f"Found {len(solutions)} solution(s)")
    
    if solutions:
        # Verify the first solution
        test_solution = solutions[0]
        print(f"\nFirst solution: {np.round(test_solution, 4)}")
        
        computed_trans, computed_rot = ik.compute_fk(test_solution)
        
        print(f"\nVerification:")
        print(f"  Target translation:   {np.round(test_translation, 4)}")
        print(f"  Computed translation: {np.round(computed_trans, 4)}")
        print(f"  Error: {np.linalg.norm(test_translation - computed_trans):.6e}")
        
        print(f"\n  Target rotation determinant:   {np.linalg.det(rotation_z):.6f}")
        print(f"  Computed rotation determinant: {np.linalg.det(computed_rot):.6f}")
    
    print("\n" + "=" * 60)
    print("Example complete")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • FK converts joint angles to end effector pose")
    print("  • Rotation matrices must be orthonormal with determinant 1")
    print("  • FK can verify IK solutions")
    print("  • All IK solutions should produce the same end effector pose")


if __name__ == "__main__":
    main()
