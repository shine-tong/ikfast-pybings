"""
Unit tests for high-level Python API.

Tests the high-level Pythonic interface functions:
- compute_ik() wrapper function
- compute_fk() wrapper function
- get_solver_info() function
- Type hints validation
"""
import pytest
import numpy as np
import ikfast_pybind as ik


class TestComputeIk:
    """Test suite for high-level compute_ik() function."""
    
    def test_compute_ik_with_matrix_rotation(self):
        """Test compute_ik() accepts 3x3 rotation matrix."""
        # Get a reachable pose using FK
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        translation, rotation = ik.compute_fk(joints)
        
        # rotation is already 3x3 from compute_fk
        assert rotation.shape == (3, 3)
        
        # Compute IK with matrix rotation
        solutions = ik.compute_ik(translation, rotation)
        
        # Should return a list
        assert isinstance(solutions, list)
        
        # Should have at least one solution
        assert len(solutions) > 0
        
        # Each solution should be a numpy array
        for sol in solutions:
            assert isinstance(sol, np.ndarray)
            assert sol.shape == (6,)
            assert sol.dtype == np.float64
    
    def test_compute_ik_with_flat_rotation(self):
        """Test compute_ik() accepts flat [9] rotation array."""
        # Get a reachable pose
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        translation, rotation = ik.compute_fk(joints)
        
        # Flatten rotation to [9]
        rotation_flat = rotation.flatten()
        assert rotation_flat.shape == (9,)
        
        # Compute IK with flat rotation
        solutions = ik.compute_ik(translation, rotation_flat)
        
        # Should return a list
        assert isinstance(solutions, list)
        assert len(solutions) > 0
    
    def test_compute_ik_returns_list_of_arrays(self):
        """Test that compute_ik() returns list of numpy arrays."""
        # Get a reachable pose
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        translation, rotation = ik.compute_fk(joints)
        
        # Compute IK
        solutions = ik.compute_ik(translation, rotation)
        
        # Should be a list
        assert isinstance(solutions, list)
        
        # Each element should be numpy array with correct properties
        for sol in solutions:
            assert isinstance(sol, np.ndarray)
            assert sol.shape == (6,)
            assert sol.dtype == np.float64
            assert not np.any(np.isnan(sol))
            assert not np.any(np.isinf(sol))
    
    def test_compute_ik_empty_list_for_unreachable_pose(self):
        """Test that compute_ik() returns empty list for unreachable pose."""
        # Use a pose that is very far away (unreachable)
        translation = np.array([100.0, 100.0, 100.0])
        rotation = np.eye(3)
        
        # Should not raise exception
        solutions = ik.compute_ik(translation, rotation)
        
        # Should return empty list
        assert isinstance(solutions, list)
        assert len(solutions) == 0
    
    def test_compute_ik_invalid_translation_shape(self):
        """Test that compute_ik() raises ValueError for invalid translation shape."""
        translation = np.array([1.0, 2.0])  # Wrong shape
        rotation = np.eye(3)
        
        with pytest.raises(ValueError) as exc_info:
            ik.compute_ik(translation, rotation)
        
        error_msg = str(exc_info.value)
        assert "translation" in error_msg.lower()
        assert "(3,)" in error_msg
        assert "(2,)" in error_msg
    
    def test_compute_ik_invalid_rotation_shape(self):
        """Test that compute_ik() raises ValueError for invalid rotation shape."""
        translation = np.array([1.0, 2.0, 3.0])
        rotation = np.array([1.0, 2.0, 3.0, 4.0])  # Wrong shape
        
        with pytest.raises(ValueError) as exc_info:
            ik.compute_ik(translation, rotation)
        
        error_msg = str(exc_info.value)
        assert "rotation" in error_msg.lower()
    
    def test_compute_ik_accepts_list_inputs(self):
        """Test that compute_ik() accepts list inputs (converts to numpy)."""
        # Get a reachable pose
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        translation, rotation = ik.compute_fk(joints)
        
        # Convert to lists
        translation_list = translation.tolist()
        rotation_list = rotation.tolist()
        
        # Should work with list inputs
        solutions = ik.compute_ik(translation_list, rotation_list)
        
        assert isinstance(solutions, list)
        assert len(solutions) > 0
    
    def test_compute_ik_with_free_parameters(self):
        """Test compute_ik() with optional free parameters."""
        # Get a reachable pose
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        translation, rotation = ik.compute_fk(joints)
        
        # Check if solver has free parameters
        info = ik.get_solver_info()
        num_free = info['num_free_parameters']
        
        if num_free > 0:
            # Test with free parameters
            free_params = np.zeros(num_free)
            solutions = ik.compute_ik(translation, rotation, free_params)
            assert isinstance(solutions, list)
        else:
            # Test without free parameters
            solutions = ik.compute_ik(translation, rotation)
            assert isinstance(solutions, list)


class TestComputeFk:
    """Test suite for high-level compute_fk() function."""
    
    def test_compute_fk_returns_tuple(self):
        """Test that compute_fk() returns tuple of (translation, rotation)."""
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        result = ik.compute_fk(joints)
        
        # Should be a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        translation, rotation = result
        
        # Check types
        assert isinstance(translation, np.ndarray)
        assert isinstance(rotation, np.ndarray)
    
    def test_compute_fk_returns_correct_shapes(self):
        """Test that compute_fk() returns properly shaped outputs."""
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        translation, rotation = ik.compute_fk(joints)
        
        # Check shapes
        assert translation.shape == (3,), f"Expected (3,), got {translation.shape}"
        assert rotation.shape == (3, 3), f"Expected (3, 3), got {rotation.shape}"
        
        # Check dtypes
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64
    
    def test_compute_fk_rotation_is_matrix(self):
        """Test that compute_fk() returns rotation as 3x3 matrix (not flat)."""
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        translation, rotation = ik.compute_fk(joints)
        
        # Rotation should be 3x3 matrix
        assert rotation.shape == (3, 3)
        assert rotation.ndim == 2
    
    def test_compute_fk_rotation_is_valid(self):
        """Test that compute_fk() returns a valid rotation matrix."""
        joints = np.array([0.5, -0.5, 1.0, -1.0, 0.3, -0.3])
        
        translation, rotation = ik.compute_fk(joints)
        
        # Check determinant is ≈ 1
        det = np.linalg.det(rotation)
        assert np.abs(det - 1.0) < 1e-6, f"Determinant should be ≈ 1, got {det}"
        
        # Check orthogonality
        identity = np.dot(rotation.T, rotation)
        assert np.allclose(identity, np.eye(3), atol=1e-6)
    
    def test_compute_fk_invalid_shape(self):
        """Test that compute_fk() raises ValueError for invalid joint shape."""
        # Wrong number of joints
        joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Only 5 joints
        
        with pytest.raises(ValueError) as exc_info:
            ik.compute_fk(joints)
        
        error_msg = str(exc_info.value)
        assert "joint" in error_msg.lower()
        assert "6" in error_msg
        assert "5" in error_msg
    
    def test_compute_fk_accepts_list_input(self):
        """Test that compute_fk() accepts list input (converts to numpy)."""
        joints_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        translation, rotation = ik.compute_fk(joints_list)
        
        # Should work and return numpy arrays
        assert isinstance(translation, np.ndarray)
        assert isinstance(rotation, np.ndarray)
        assert translation.shape == (3,)
        assert rotation.shape == (3, 3)
    
    def test_compute_fk_accepts_different_dtypes(self):
        """Test that compute_fk() accepts different dtypes and converts."""
        # Test with float32
        joints_f32 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        translation, rotation = ik.compute_fk(joints_f32)
        
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64
        
        # Test with int
        joints_int = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        translation, rotation = ik.compute_fk(joints_int)
        
        assert translation.dtype == np.float64
        assert rotation.dtype == np.float64


class TestGetSolverInfo:
    """Test suite for get_solver_info() function."""
    
    def test_get_solver_info_returns_dict(self):
        """Test that get_solver_info() returns a dictionary."""
        info = ik.get_solver_info()
        
        assert isinstance(info, dict)
    
    def test_get_solver_info_has_required_keys(self):
        """Test that get_solver_info() returns complete dictionary."""
        info = ik.get_solver_info()
        
        # Check all required keys are present
        required_keys = [
            'num_joints',
            'num_free_parameters',
            'free_parameters',
            'ik_type',
            'kinematics_hash',
            'ikfast_version',
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_get_solver_info_correct_types(self):
        """Test that get_solver_info() returns correct value types."""
        info = ik.get_solver_info()
        
        # Check types
        assert isinstance(info['num_joints'], int)
        assert isinstance(info['num_free_parameters'], int)
        assert isinstance(info['free_parameters'], list)
        assert isinstance(info['ik_type'], int)
        assert isinstance(info['kinematics_hash'], str)
        assert isinstance(info['ikfast_version'], str)
    
    def test_get_solver_info_correct_values(self):
        """Test that get_solver_info() returns expected values."""
        info = ik.get_solver_info()
        
        # Check expected values
        assert info['num_joints'] == 6
        assert info['num_free_parameters'] >= 0
        assert len(info['free_parameters']) == info['num_free_parameters']
        assert info['ik_type'] == 0x67000001
        assert len(info['kinematics_hash']) > 0
        assert len(info['ikfast_version']) > 0
    
    def test_get_solver_info_free_parameters_are_ints(self):
        """Test that free_parameters list contains integers."""
        info = ik.get_solver_info()
        
        for param in info['free_parameters']:
            assert isinstance(param, int)


class TestTypeHints:
    """Test suite for type hints validation."""
    
    def test_compute_ik_has_type_hints(self):
        """Test that compute_ik() has type hints."""
        import inspect
        
        sig = inspect.signature(ik.compute_ik)
        
        # Check that parameters have annotations
        assert 'translation' in sig.parameters
        assert 'rotation' in sig.parameters
        assert 'free_params' in sig.parameters
        
        # Check return annotation exists
        assert sig.return_annotation != inspect.Signature.empty
    
    def test_compute_fk_has_type_hints(self):
        """Test that compute_fk() has type hints."""
        import inspect
        
        sig = inspect.signature(ik.compute_fk)
        
        # Check that parameters have annotations
        assert 'joint_angles' in sig.parameters
        
        # Check return annotation exists
        assert sig.return_annotation != inspect.Signature.empty
    
    def test_get_solver_info_has_type_hints(self):
        """Test that get_solver_info() has type hints."""
        import inspect
        
        sig = inspect.signature(ik.get_solver_info)
        
        # Check return annotation exists
        assert sig.return_annotation != inspect.Signature.empty


class TestModuleInterface:
    """Test suite for module-level interface."""
    
    def test_module_has_version(self):
        """Test that module has __version__ attribute."""
        assert hasattr(ik, '__version__')
        assert isinstance(ik.__version__, str)
        assert len(ik.__version__) > 0
    
    def test_module_has_docstring(self):
        """Test that module has docstring."""
        assert ik.__doc__ is not None
        assert len(ik.__doc__) > 0
        assert "IKFast" in ik.__doc__
    
    def test_module_exports_expected_functions(self):
        """Test that module exports expected public API."""
        # Check main functions are available
        assert hasattr(ik, 'compute_ik')
        assert hasattr(ik, 'compute_fk')
        assert hasattr(ik, 'get_solver_info')
        
        # Check classes are available
        assert hasattr(ik, 'IkSolution')
        assert hasattr(ik, 'IkSolutionList')
    
    def test_module_has_all_attribute(self):
        """Test that module has __all__ attribute."""
        assert hasattr(ik, '__all__')
        assert isinstance(ik.__all__, list)
        
        # Check expected exports
        expected = ['compute_ik', 'compute_fk', 'get_solver_info', 'IkSolution', 'IkSolutionList']
        for name in expected:
            assert name in ik.__all__, f"Missing from __all__: {name}"


class TestRoundTrip:
    """Test suite for IK-FK round trip consistency."""
    
    def test_fk_ik_round_trip(self):
        """Test that FK→IK produces solutions that yield the same pose."""
        # Start with a joint configuration
        joints = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        
        # Compute FK
        translation, rotation = ik.compute_fk(joints)
        
        # Compute IK
        solutions = ik.compute_ik(translation, rotation)
        
        # Should have at least one solution
        assert len(solutions) > 0
        
        # At least one solution should produce the same pose
        found_match = False
        for sol in solutions:
            trans_check, rot_check = ik.compute_fk(sol)
            
            if (np.allclose(translation, trans_check, atol=1e-6) and
                np.allclose(rotation, rot_check, atol=1e-6)):
                found_match = True
                break
        
        assert found_match, "No IK solution produced the original pose"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
