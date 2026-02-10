"""
Test to verify project structure and build configuration.
This test checks that all necessary files exist and are properly configured.
"""
import os
import sys


def test_project_structure():
    """Verify that all required project files exist."""
    required_files = [
        'pyproject.toml',
        'setup.py',
        'README.md',
        'ikfast_pybind/__init__.py',
        'ikfast_pybind/_ikfast_pybind.cpp',
        'include/ikfast.h',
        'src/sa0521_manipulator_ikfast_solver.cpp',
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Required file missing: {file_path}"


def test_pyproject_toml():
    """Verify pyproject.toml has correct build dependencies."""
    with open('pyproject.toml', 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert 'pybind11>=2.6.0' in content, "pybind11 dependency missing"
    assert 'numpy>=1.20.0' in content, "numpy dependency missing"
    assert 'setuptools>=45' in content, "setuptools dependency missing"


def test_setup_py():
    """Verify setup.py has correct configuration."""
    with open('setup.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '_ikfast_pybind.cpp' in content, "C++ source file not in setup.py"
    assert 'sa0521_manipulator_ikfast_solver.cpp' in content, "Solver source not in setup.py"
    assert 'IKFAST_HAS_LIBRARY' in content, "IKFAST_HAS_LIBRARY flag missing"
    assert 'IKFAST_NO_MAIN' in content, "IKFAST_NO_MAIN flag missing"


def test_cpp_module():
    """Verify C++ module file has basic structure."""
    with open('ikfast_pybind/_ikfast_pybind.cpp', 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '#include <pybind11/pybind11.h>' in content, "pybind11 header missing"
    assert '#include <pybind11/numpy.h>' in content, "numpy header missing"
    assert 'PYBIND11_MODULE' in content, "PYBIND11_MODULE macro missing"
    assert '_ikfast_pybind' in content, "Module name missing"


if __name__ == '__main__':
    test_project_structure()
    test_pyproject_toml()
    test_setup_py()
    test_cpp_module()
    print("All project structure tests passed!")
