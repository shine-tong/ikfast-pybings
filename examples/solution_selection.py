#!/usr/bin/env python3
"""
Advanced Solution Selection Example

This example demonstrates:
1. Advanced solution selection criteria
2. Handling free parameters for redundant robots
3. Dealing with workspace boundary cases
4. Trajectory planning considerations
"""

import numpy as np
import ikfast_pybind as ik


class SolutionSelector:
    """
    Advanced solution selector with multiple criteria.
    """
    
    def __init__(self, joint_limits=None, joint_weights=None):
        """
        Initialize the solution selector.
        
        Args:
            joint_limits: List of (min, max) tuples for each joint
            joint_weights: Weights for each joint when computing distances
        """
        self.joint_limits = joint_limits
        self.joint_weights = joint_weights or np.ones(6)
    
    def select_by_distance(self, solutions, reference_joints):
        """
        Select solution closest to reference configuration.
        
        Uses weighted Euclidean distance in joint space.
        """
        if not solutions:
            return None
        
        min_distance = float('inf')
        best_solution = None
        
        for solution in solutions:
            weighted_diff = self.joint_weights * (solution - reference_joints)
            distance = np.linalg.norm(weighted_diff)
            
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
        
        return best_solution, min_distance
    
    def select_by_manipulability(self, solutions):
        """
        Select solution with best manipulability (simplified heuristic).
        
        This is a simplified version - a real implementation would compute
        the manipulability ellipsoid from the Jacobian.
        """
        if not solutions:
            return None
        
        # Heuristic: prefer configurations away from singularities
        # For this example, we prefer solutions where joints are not near 0 or ±π
        best_solution = None
        best_score = -float('inf')
        
        for solution in solutions:
            # Simple heuristic: sum of distances from singular positions
            score = 0
            for angle in solution:
                # Distance from 0
                dist_from_zero = abs(angle)
                # Distance from ±π
                dist_from_pi = min(abs(angle - np.pi), abs(angle + np.pi))
                # Prefer angles away from these positions
                score += min(dist_from_zero, dist_from_pi)
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution, best_score
    
    def select_by_joint_limits(self, solutions):
        """
        Select solution with maximum margin from joint limits.
        """
        if not solutions or self.joint_limits is None:
            return None
        
        best_solution = None
        best_min_margin = -float('inf')
        
        for solution in solutions:
            margins = []
            for angle, (min_lim, max_lim) in zip(solution, self.joint_limits):
                margin_to_min = angle - min_lim
                margin_to_max = max_lim - angle
                margins.append(min(margin_to_min, margin_to_max))
            
            min_margin = min(margins)
            if min_margin > best_min_margin:
                best_min_margin = min_margin
                best_solution = solution
        
        return best_solution, best_min_margin
    
    def select_by_energy(self, solutions):
        """
        Select solution with minimum "energy" (sum of squared joint angles).
        
        This tends to prefer configurations closer to zero position.
        """
        if not solutions:
            return None
        
        min_energy = float('inf')
        best_solution = None
        
        for solution in solutions:
            energy = np.sum(solution**2)
            if energy < min_energy:
                min_energy = energy
                best_solution = solution
        
        return best_solution, min_energy
    
    def filter_by_limits(self, solutions):
        """
        Filter out solutions that violate joint limits.
        """
        if not solutions or self.joint_limits is None:
            return solutions
        
        valid_solutions = []
        for solution in solutions:
            is_valid = True
            for angle, (min_lim, max_lim) in zip(solution, self.joint_limits):
                if angle < min_lim or angle > max_lim:
                    is_valid = False
                    break
            
            if is_valid:
                valid_solutions.append(solution)
        
        return valid_solutions


def handle_free_parameters():
    """
    Demonstrate handling of free parameters for redundant robots.
    """
    print("\n" + "=" * 60)
    print("Free Parameter Handling")
    print("=" * 60)
    
    info = ik.get_solver_info()
    num_free = info['num_free_parameters']
    
    print(f"\nSolver has {num_free} free parameter(s)")
    
    if num_free > 0:
        print(f"Free parameter indices: {info['free_parameters']}")
        
        # Example: compute IK with different free parameter values
        target_trans = np.array([0.5, 0.0, 0.5])
        target_rot = np.eye(3)
        
        print("\nComputing IK with different free parameter values:")
        
        for free_value in [0.0, 0.5, 1.0]:
            free_params = np.array([free_value] * num_free)
            solutions = ik.compute_ik(target_trans, target_rot, free_params)
            print(f"  Free param = {free_value}: {len(solutions)} solution(s)")
            
            if solutions:
                print(f"    First solution: {np.round(solutions[0], 4)}")
    else:
        print("This robot has no free parameters (non-redundant)")


def handle_workspace_boundaries():
    """
    Demonstrate handling of workspace boundary cases.
    """
    print("\n" + "=" * 60)
    print("Workspace Boundary Cases")
    print("=" * 60)
    
    # Test various poses to understand workspace
    test_poses = [
        # (translation, description)
        (np.array([0.3, 0.0, 0.3]), "Close to robot base"),
        (np.array([0.5, 0.0, 0.5]), "Mid-range"),
        (np.array([0.8, 0.0, 0.8]), "Far from base"),
        (np.array([1.5, 0.0, 1.5]), "Very far (likely unreachable)"),
        (np.array([0.0, 0.0, 0.0]), "At origin (likely unreachable)"),
    ]
    
    rotation = np.eye(3)
    
    print("\nTesting reachability of various poses:")
    for translation, description in test_poses:
        solutions = ik.compute_ik(translation, rotation)
        reachable = "✓ Reachable" if solutions else "✗ Unreachable"
        distance = np.linalg.norm(translation)
        print(f"\n  {description}")
        print(f"    Position: {translation}")
        print(f"    Distance from origin: {distance:.3f}")
        print(f"    {reachable} ({len(solutions)} solution(s))")


def trajectory_planning_example():
    """
    Demonstrate solution selection for trajectory planning.
    """
    print("\n" + "=" * 60)
    print("Trajectory Planning Example")
    print("=" * 60)
    
    # Define a simple trajectory (linear path in Cartesian space)
    start_pos = np.array([0.4, 0.0, 0.4])
    end_pos = np.array([0.6, 0.0, 0.6])
    num_waypoints = 5
    
    print(f"\nPlanning trajectory from {start_pos} to {end_pos}")
    print(f"Number of waypoints: {num_waypoints}")
    
    # Generate waypoints
    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        pos = start_pos + t * (end_pos - start_pos)
        waypoints.append(pos)
    
    # Compute IK for each waypoint, selecting solutions for smooth motion
    rotation = np.eye(3)
    joint_limits = [(-np.pi, np.pi)] * 6
    selector = SolutionSelector(joint_limits=joint_limits)
    
    trajectory = []
    current_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("\nComputing joint trajectory:")
    for i, waypoint in enumerate(waypoints):
        solutions = ik.compute_ik(waypoint, rotation)
        
        if not solutions:
            print(f"  Waypoint {i+1}: ✗ Unreachable")
            break
        
        # Select solution closest to current configuration
        best_solution, distance = selector.select_by_distance(solutions, current_joints)
        
        trajectory.append(best_solution)
        current_joints = best_solution
        
        print(f"  Waypoint {i+1}: ✓ {len(solutions)} solution(s), "
              f"selected with distance {distance:.4f}")
        print(f"    Joints: {np.round(best_solution, 4)}")
    
    if len(trajectory) == num_waypoints:
        print("\n✓ Trajectory planning successful")
        
        # Compute total joint motion
        total_motion = 0
        for i in range(1, len(trajectory)):
            motion = np.linalg.norm(trajectory[i] - trajectory[i-1])
            total_motion += motion
        
        print(f"Total joint space motion: {total_motion:.4f} radians")
    else:
        print("\n✗ Trajectory planning failed - some waypoints unreachable")


def main():
    """Main example demonstrating advanced solution selection."""
    
    print("=" * 60)
    print("IKFast Python Bindings - Advanced Solution Selection")
    print("=" * 60)
    
    # Example 1: Multiple selection criteria
    print("\n" + "-" * 60)
    print("Example 1: Comparing selection criteria")
    print("-" * 60)
    
    target_trans = np.array([0.5, 0.0, 0.5])
    target_rot = np.eye(3)
    reference_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print(f"\nTarget pose:")
    print(f"  Translation: {target_trans}")
    print(f"  Rotation: Identity")
    print(f"\nReference joints: {reference_joints}")
    
    solutions = ik.compute_ik(target_trans, target_rot)
    print(f"\nFound {len(solutions)} solution(s)")
    
    if solutions:
        joint_limits = [(-np.pi, np.pi)] * 6
        selector = SolutionSelector(joint_limits=joint_limits)
        
        # Try different selection criteria
        print("\nSelection by different criteria:")
        
        # Distance criterion
        sol_dist, dist = selector.select_by_distance(solutions, reference_joints)
        print(f"\n  1. By distance from reference:")
        print(f"     Solution: {np.round(sol_dist, 4)}")
        print(f"     Distance: {dist:.4f}")
        
        # Energy criterion
        sol_energy, energy = selector.select_by_energy(solutions)
        print(f"\n  2. By minimum energy:")
        print(f"     Solution: {np.round(sol_energy, 4)}")
        print(f"     Energy: {energy:.4f}")
        
        # Joint limits criterion
        sol_limits, margin = selector.select_by_joint_limits(solutions)
        print(f"\n  3. By joint limit margins:")
        print(f"     Solution: {np.round(sol_limits, 4)}")
        print(f"     Min margin: {margin:.4f}")
        
        # Manipulability criterion
        sol_manip, score = selector.select_by_manipulability(solutions)
        print(f"\n  4. By manipulability (heuristic):")
        print(f"     Solution: {np.round(sol_manip, 4)}")
        print(f"     Score: {score:.4f}")
    
    # Example 2: Free parameters
    handle_free_parameters()
    
    # Example 3: Workspace boundaries
    handle_workspace_boundaries()
    
    # Example 4: Trajectory planning
    trajectory_planning_example()
    
    print("\n" + "=" * 60)
    print("Example complete")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  • Different selection criteria suit different applications")
    print("  • Distance-based selection creates smooth trajectories")
    print("  • Joint limit awareness prevents hardware damage")
    print("  • Workspace boundaries must be handled gracefully")
    print("  • Free parameters provide additional flexibility for redundant robots")


if __name__ == "__main__":
    main()
