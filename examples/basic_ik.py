#!/usr/bin/env python3
"""
Basic Inverse Kinematics Example

This example demonstrates:
1. Computing IK solutions for a target pose
2. Iterating through multiple solutions
3. Selecting the best solution based on criteria
"""

import numpy as np
import ikfast_pybind as ik


def compute_joint_distance(joints1, joints2):
    """
    Compute the distance between two joint configurations.
    
    This is useful for selecting solutions close to a current configuration.
    """
    return np.linalg.norm(joints1 - joints2)


def select_closest_solution(solutions, current_joints):
    """
    Select the solution closest to the current joint configuration.
    
    This minimizes joint motion, which is often desirable for smooth trajectories.
    """
    if not solutions:
        return None
    
    min_distance = float('inf')
    best_solution = None
    
    for solution in solutions:
        distance = compute_joint_distance(solution, current_joints)
        if distance < min_distance:
            min_distance = distance
            best_solution = solution
    
    return best_solution


def select_solution_avoiding_limits(solutions, joint_limits):
    """
    Select a solution that stays away from joint limits.
    
    Args:
        solutions: List of joint angle arrays
        joint_limits: List of (min, max) tuples for each joint
    
    Returns:
        Solution with maximum distance from limits, or None if no solutions
    """
    if not solutions:
        return None
    
    best_solution = None
    best_margin = -float('inf')
    
    for solution in solutions:
        # Calculate minimum margin to any limit
        margins = []
        for joint_angle, (min_limit, max_limit) in zip(solution, joint_limits):
            margin_to_min = joint_angle - min_limit
            margin_to_max = max_limit - joint_angle
            margins.append(min(margin_to_min, margin_to_max))
        
        min_margin = min(margins)
        if min_margin > best_margin:
            best_margin = min_margin
            best_solution = solution
    
    return best_solution


def main():
    """Main example demonstrating IK computation and solution selection."""
    
    print("=" * 60)
    print("IKFast Python Bindings - Basic IK Example")
    print("=" * 60)
    
    # Get solver information
    info = ik.get_solver_info()
    print(f"\nSolver Information:")
    print(f"  Number of joints: {info['num_joints']}")
    print(f"  Free parameters: {info['num_free_parameters']}")
    print(f"  IK type: {hex(info['ik_type'])}")
    print(f"  Kinematics hash: {info['kinematics_hash']}")
    
    # Define a target pose
    print("\n" + "-" * 60)
    print("Example 1: Computing IK for a target pose")
    print("-" * 60)
    
    target_translation = np.array([0.5, 0.0, 0.5])
    target_rotation = np.eye(3)  # Identity rotation (no rotation)
    
    print(f"\nTarget pose:")
    print(f"  Translation: {target_translation}")
    print(f"  Rotation:\n{target_rotation}")
    
    # Compute IK solutions
    print("\nComputing IK solutions...")
    solutions = ik.compute_ik(target_translation, target_rotation)
    
    print(f"Found {len(solutions)} solution(s)")
    
    if solutions:
        print("\nAll solutions:")
        for i, solution in enumerate(solutions):
            print(f"  Solution {i+1}: {np.round(solution, 4)}")
    else:
        print("No solutions found - pose may be unreachable")
        return
    
    # Example 2: Select solution closest to current configuration
    print("\n" + "-" * 60)
    print("Example 2: Selecting closest solution to current pose")
    print("-" * 60)
    
    current_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"\nCurrent joint configuration: {current_joints}")
    
    closest_solution = select_closest_solution(solutions, current_joints)
    if closest_solution is not None:
        distance = compute_joint_distance(closest_solution, current_joints)
        print(f"Closest solution: {np.round(closest_solution, 4)}")
        print(f"Distance from current: {distance:.4f} radians")
    
    # Example 3: Select solution avoiding joint limits
    print("\n" + "-" * 60)
    print("Example 3: Selecting solution away from joint limits")
    print("-" * 60)
    
    # Define typical joint limits (example values)
    joint_limits = [
        (-np.pi, np.pi),      # Joint 1
        (-np.pi/2, np.pi/2),  # Joint 2
        (-np.pi, np.pi),      # Joint 3
        (-np.pi, np.pi),      # Joint 4
        (-np.pi/2, np.pi/2),  # Joint 5
        (-np.pi, np.pi),      # Joint 6
    ]
    
    safe_solution = select_solution_avoiding_limits(solutions, joint_limits)
    if safe_solution is not None:
        print(f"Safest solution: {np.round(safe_solution, 4)}")
        
        # Show margins for each joint
        print("\nMargins to joint limits:")
        for i, (angle, (min_lim, max_lim)) in enumerate(zip(safe_solution, joint_limits)):
            margin_min = angle - min_lim
            margin_max = max_lim - angle
            print(f"  Joint {i+1}: {margin_min:.3f} rad from min, {margin_max:.3f} rad from max")
    
    # Example 4: Verify solution with FK
    print("\n" + "-" * 60)
    print("Example 4: Verifying IK solution with FK")
    print("-" * 60)
    
    # Use the first solution
    test_solution = solutions[0]
    print(f"\nTesting solution: {np.round(test_solution, 4)}")
    
    # Compute FK for this solution
    computed_translation, computed_rotation = ik.compute_fk(test_solution)
    
    print(f"\nOriginal target translation: {target_translation}")
    print(f"Computed translation:        {computed_translation}")
    print(f"Difference: {np.linalg.norm(target_translation - computed_translation):.6e}")
    
    print(f"\nRotation matrix difference (Frobenius norm):")
    rotation_diff = np.linalg.norm(target_rotation - computed_rotation)
    print(f"  {rotation_diff:.6e}")
    
    if np.allclose(target_translation, computed_translation, atol=1e-6) and \
       np.allclose(target_rotation, computed_rotation, atol=1e-6):
        print("\n✓ Solution verified successfully!")
    else:
        print("\n✗ Solution verification failed")
    
    print("\n" + "=" * 60)
    print("Example complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
