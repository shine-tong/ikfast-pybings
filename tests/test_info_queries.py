"""
Unit tests for IKFast information query functions.

Tests the basic information query functions that provide metadata
about the solver configuration.
"""
import pytest


def test_module_import():
    """Test that the module can be imported."""
    try:
        import ikfast_pybind._ikfast_pybind as ikfast
        assert ikfast is not None
    except ImportError as e:
        pytest.fail(f"Failed to import ikfast_pybind._ikfast_pybind: {e}")


def test_get_num_joints():
    """Test that get_num_joints() returns 6."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    num_joints = ikfast.get_num_joints()
    assert isinstance(num_joints, int), "get_num_joints() should return an integer"
    assert num_joints == 6, f"Expected 6 joints, got {num_joints}"


def test_get_num_free_parameters():
    """Test that get_num_free_parameters() returns a non-negative integer."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    num_free = ikfast.get_num_free_parameters()
    assert isinstance(num_free, int), "get_num_free_parameters() should return an integer"
    assert num_free >= 0, f"Number of free parameters should be non-negative, got {num_free}"


def test_get_free_parameters():
    """Test that get_free_parameters() returns a list."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    free_params = ikfast.get_free_parameters()
    assert isinstance(free_params, list), "get_free_parameters() should return a list"
    
    # Length should match get_num_free_parameters()
    num_free = ikfast.get_num_free_parameters()
    assert len(free_params) == num_free, \
        f"Length of free parameters list ({len(free_params)}) should match get_num_free_parameters() ({num_free})"
    
    # All elements should be integers
    for param in free_params:
        assert isinstance(param, int), f"Free parameter {param} should be an integer"


def test_get_ik_type():
    """Test that get_ik_type() returns expected constant."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    ik_type = ikfast.get_ik_type()
    assert isinstance(ik_type, int), "get_ik_type() should return an integer"
    # The expected value is 0x67000001 based on the solver file
    assert ik_type == 0x67000001, f"Expected IK type 0x67000001, got {hex(ik_type)}"


def test_get_kinematics_hash():
    """Test that get_kinematics_hash() returns a non-empty string."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    hash_str = ikfast.get_kinematics_hash()
    assert isinstance(hash_str, str), "get_kinematics_hash() should return a string"
    assert len(hash_str) > 0, "Kinematics hash should not be empty"
    # Should contain robot identifier
    assert "robot" in hash_str.lower() or "GenericRobot" in hash_str, \
        f"Hash should contain robot identifier, got: {hash_str}"


def test_get_ikfast_version():
    """Test that get_ikfast_version() returns a version string."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    version = ikfast.get_ikfast_version()
    assert isinstance(version, str), "get_ikfast_version() should return a string"
    assert len(version) > 0, "IKFast version should not be empty"
    # Version should be in hex format
    assert version.startswith("0x"), f"Version should start with '0x', got: {version}"


def test_module_has_version():
    """Test that the module has a __version__ attribute."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    assert hasattr(ikfast, '__version__'), "Module should have __version__ attribute"
    version = ikfast.__version__
    assert isinstance(version, str), "__version__ should be a string"
    assert len(version) > 0, "__version__ should not be empty"


def test_module_has_docstring():
    """Test that the module has a docstring."""
    import ikfast_pybind._ikfast_pybind as ikfast
    
    assert ikfast.__doc__ is not None, "Module should have a docstring"
    assert len(ikfast.__doc__) > 0, "Module docstring should not be empty"
    assert "IKFast" in ikfast.__doc__, "Docstring should mention IKFast"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
