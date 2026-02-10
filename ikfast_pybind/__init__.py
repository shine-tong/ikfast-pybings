"""
IKFast Python Bindings

Python bindings for the IKFast inverse kinematics solver.

This module provides a high-level Pythonic interface to the IKFast C++ solver,
with seamless numpy array integration and comprehensive error handling.

Example:
    >>> import ikfast_pybind as ik
    >>> import numpy as np
    >>> 
    >>> # Compute forward kinematics
    >>> joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> translation, rotation = ik.compute_fk(joints)
    >>> 
    >>> # Compute inverse kinematics
    >>> solutions = ik.compute_ik(translation, rotation)
    >>> for sol in solutions:
    >>>     print(sol)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Import the low-level C++ binding module
from . import _ikfast_pybind

__version__ = "0.1.0"

# Re-export low-level classes for advanced users
IkSolution = _ikfast_pybind.IkSolution
IkSolutionList = _ikfast_pybind.IkSolutionList


def compute_ik(
    translation: np.ndarray,
    rotation: np.ndarray,
    free_params: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Compute inverse kinematics solutions.
    
    Given a desired end effector pose (translation and rotation), this function
    computes all possible joint configurations that achieve that pose.
    
    Args:
        translation: End effector translation [x, y, z] as numpy array with shape (3,)
        rotation: End effector rotation matrix as numpy array with shape (3, 3) or 
                 flattened (9,). The rotation matrix should be orthonormal.
        free_params: Optional free parameter values as numpy array. Required if the
                    robot has redundant degrees of freedom. Default is None.
    
    Returns:
        List of joint angle arrays, each with shape (num_joints,) and dtype float64.
        Returns an empty list if no solutions exist (pose is unreachable).
    
    Raises:
        ValueError: If input arrays have incorrect shape or invalid values
        TypeError: If inputs are not array-like or cannot be converted to numpy arrays
        RuntimeError: If the solver encounters numerical issues
    
    Example:
        >>> translation = np.array([0.5, 0.0, 0.5])
        >>> rotation = np.eye(3)
        >>> solutions = compute_ik(translation, rotation)
        >>> print(f"Found {len(solutions)} solutions")
        >>> if solutions:
        >>>     print(f"First solution: {solutions[0]}")
    """
    # Input validation and conversion
    translation = np.asarray(translation, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    
    # Validate translation shape
    if translation.shape != (3,):
        raise ValueError(
            f"compute_ik: Invalid translation shape. Expected: (3,), Got: {translation.shape}"
        )
    
    # Handle both flat [9] and matrix [3,3] rotation inputs
    if rotation.shape == (3, 3):
        # Matrix format - flatten it
        rotation = rotation.flatten()
    elif rotation.shape != (9,):
        raise ValueError(
            f"compute_ik: Invalid rotation shape. Expected: (3, 3) or (9,), Got: {rotation.shape}"
        )
    
    # Ensure arrays are contiguous
    translation = np.ascontiguousarray(translation)
    rotation = np.ascontiguousarray(rotation)
    
    # Handle optional free parameters
    pfree = None
    if free_params is not None:
        pfree = np.asarray(free_params, dtype=np.float64)
        pfree = np.ascontiguousarray(pfree)
    
    # Call low-level C++ function
    solution_list = _ikfast_pybind.compute_ik_raw(translation, rotation, pfree)
    
    # Convert IkSolutionList to list of numpy arrays
    solutions = []
    for i in range(len(solution_list)):
        solution = solution_list[i]
        # Get concrete joint angles (with free parameters if provided)
        joint_angles = solution.get_solution(pfree)
        solutions.append(joint_angles)
    
    return solutions


def compute_fk(
    joint_angles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward kinematics.
    
    Given a joint configuration, this function computes the resulting end effector
    pose (translation and rotation).
    
    Args:
        joint_angles: Array of joint angles with shape (num_joints,). For a 6-DOF
                     manipulator, this should be shape (6,).
    
    Returns:
        Tuple of (translation, rotation_matrix) where:
            - translation: numpy array with shape (3,) and dtype float64
            - rotation_matrix: numpy array with shape (3, 3) and dtype float64
    
    Raises:
        ValueError: If joint_angles has incorrect shape
        TypeError: If joint_angles is not array-like or cannot be converted
    
    Example:
        >>> joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> translation, rotation = compute_fk(joints)
        >>> print(f"End effector position: {translation}")
        >>> print(f"End effector orientation:\\n{rotation}")
    """
    # Input validation and conversion
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    
    # Validate shape
    num_joints = _ikfast_pybind.get_num_joints()
    if joint_angles.shape != (num_joints,):
        raise ValueError(
            f"compute_fk: Invalid joint_angles shape. Expected: ({num_joints},), Got: {joint_angles.shape}"
        )
    
    # Ensure array is contiguous
    joint_angles = np.ascontiguousarray(joint_angles)
    
    # Call low-level C++ function
    translation, rotation_flat = _ikfast_pybind.compute_fk_raw(joint_angles)
    
    # Reshape rotation from flat [9] to matrix [3,3]
    rotation_matrix = rotation_flat.reshape(3, 3)
    
    return translation, rotation_matrix


def get_solver_info() -> Dict[str, Any]:
    """
    Get solver information and properties.
    
    Returns a dictionary containing information about the IK solver configuration,
    including the number of joints, free parameters, solver type, and version.
    
    Returns:
        Dictionary with the following keys:
            - num_joints (int): Number of joints in the robot
            - num_free_parameters (int): Number of free parameters (0 if no redundancy)
            - free_parameters (List[int]): Indices of free parameter joints
            - ik_type (int): IK solver type identifier constant
            - ik_type_name (str): Human-readable name of the IK solver type
            - kinematics_hash (str): Hash identifying the robot kinematics configuration
            - ikfast_version (str): IKFast version used to generate the solver
    
    Example:
        >>> info = get_solver_info()
        >>> print(f"Robot has {info['num_joints']} joints")
        >>> print(f"Solver type: {info['ik_type_name']} ({hex(info['ik_type'])})")
        >>> print(f"Kinematics hash: {info['kinematics_hash']}")
    """
    ik_type = _ikfast_pybind.get_ik_type()
    return {
        'num_joints': _ikfast_pybind.get_num_joints(),
        'num_free_parameters': _ikfast_pybind.get_num_free_parameters(),
        'free_parameters': _ikfast_pybind.get_free_parameters(),
        'ik_type': ik_type,
        'ik_type_name': get_ik_type_name(ik_type),
        'kinematics_hash': _ikfast_pybind.get_kinematics_hash(),
        'ikfast_version': _ikfast_pybind.get_ikfast_version(),
    }


def get_ik_type_name(ik_type: int) -> str:
    """
    Get human-readable name for IK solver type.
    
    Converts the IK type constant to a descriptive string that explains
    what kind of inverse kinematics problem the solver addresses.
    
    Args:
        ik_type: IK type constant (integer)
    
    Returns:
        Human-readable description of the IK type
    
    IK Type Constants:
        - 0x67000001: Transform6D - Complete 6D pose (position + orientation)
        - 0x34000002: Translation3D - 3D position only, no orientation constraint
        - 0x34000003: Direction3D - 3D direction vector
        - 0x34000004: Ray4D - Ray (origin + direction)
        - 0x34000005: Lookat3D - Look-at point
        - 0x34000006: TranslationDirection5D - Position + direction
        - 0x34000007: TranslationXY2D - XY plane position + orientation
        - 0x34000008: TranslationXYOrientation3D - XY plane position + Z-axis orientation
        - 0x34000009: TranslationLocalGlobal6D - Position with local/global frame
        - 0x3400000a: TranslationXAxisAngle4D - Position + rotation around X-axis
        - 0x3400000b: TranslationYAxisAngle4D - Position + rotation around Y-axis
        - 0x3400000c: TranslationZAxisAngle4D - Position + rotation around Z-axis
        - 0x3400000d: TranslationXAxisAngleZNorm4D - Position + X-axis rotation + Z normalization
        - 0x3400000e: TranslationYAxisAngleXNorm4D - Position + Y-axis rotation + X normalization
        - 0x3400000f: TranslationZAxisAngleYNorm4D - Position + Z-axis rotation + Y normalization
    
    Example:
        >>> ik_type = 0x67000001
        >>> print(get_ik_type_name(ik_type))
        Transform6D (默认) - 完整的位置和姿态
    """
    # IK type mapping based on IKFast documentation
    ik_types = {
        0x67000001: "Transform6D (默认) - 完整的位置和姿态",
        0x34000002: "Translation3D - 仅位置，无姿态约束",
        0x34000003: "Direction3D - 方向向量",
        0x34000004: "Ray4D - 射线（原点+方向）",
        0x34000005: "Lookat3D - 注视点",
        0x34000006: "TranslationDirection5D - 位置+方向",
        0x34000007: "TranslationXY2D - XY平面位置+姿态",
        0x34000008: "TranslationXYOrientation3D - XY平面位置+Z轴姿态",
        0x34000009: "TranslationLocalGlobal6D - 局部/全局坐标系位置",
        0x3400000a: "TranslationXAxisAngle4D - 位置+绕X轴旋转",
        0x3400000b: "TranslationYAxisAngle4D - 位置+绕Y轴旋转",
        0x3400000c: "TranslationZAxisAngle4D - 位置+绕Z轴旋转",
        0x3400000d: "TranslationXAxisAngleZNorm4D - 位置+X轴旋转+Z归一化",
        0x3400000e: "TranslationYAxisAngleXNorm4D - 位置+Y轴旋转+X归一化",
        0x3400000f: "TranslationZAxisAngleYNorm4D - 位置+Z轴旋转+Y归一化",
    }
    
    return ik_types.get(ik_type, f"Unknown IK Type (0x{ik_type:08x})")


# Export public API
__all__ = [
    'compute_ik',
    'compute_fk',
    'get_solver_info',
    'get_ik_type_name',
    'IkSolution',
    'IkSolutionList',
]
