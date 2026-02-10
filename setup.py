import os
import sys
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# Fix for MinGW compatibility
import distutils.cygwinccompiler
distutils.cygwinccompiler.get_msvcr = lambda: []


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()


def find_ikfast_solver():
    """
    Find IKFast solver source file in src/ directory.
    
    Looks for files matching *_ikfast_solver.cpp pattern.
    If multiple files found, uses the one specified by IKFAST_SOLVER_FILE
    environment variable, or the first one found.
    
    Returns:
        str: Path to the IKFast solver source file
        
    Raises:
        FileNotFoundError: If no IKFast solver file is found
    """
    # Check for environment variable override
    env_solver = os.environ.get('IKFAST_SOLVER_FILE')
    if env_solver and os.path.exists(env_solver):
        print(f"Using IKFast solver from environment: {env_solver}")
        return env_solver
    
    # Search for solver files in src/ directory
    solver_pattern = os.path.join('src', '*_ikfast_solver.cpp')
    solver_files = glob.glob(solver_pattern)
    
    if not solver_files:
        raise FileNotFoundError(
            f"No IKFast solver file found matching pattern: {solver_pattern}\n"
            "Please ensure your IKFast solver .cpp file is in the src/ directory "
            "and follows the naming pattern: *_ikfast_solver.cpp\n"
            "Alternatively, set the IKFAST_SOLVER_FILE environment variable to "
            "specify the solver file path."
        )
    
    if len(solver_files) > 1:
        print(f"Warning: Multiple IKFast solver files found: {solver_files}")
        print(f"Using: {solver_files[0]}")
        print("To specify a different solver, set IKFAST_SOLVER_FILE environment variable")
    
    solver_file = solver_files[0]
    print(f"Using IKFast solver: {solver_file}")
    return solver_file


# Find the IKFast solver source file
ikfast_solver_source = find_ikfast_solver()

ext_modules = [
    Extension(
        'ikfast_pybind._ikfast_pybind',
        sources=[
            'ikfast_pybind/_ikfast_pybind.cpp',
            ikfast_solver_source,
        ],
        include_dirs=[
            get_pybind_include(),
            'include',
        ],
        language='c++',
        extra_compile_args=[
            '-std=c++14',
            '-O3',
            '-DIKFAST_HAS_LIBRARY',
            '-DIKFAST_NO_MAIN',
        ],
        extra_link_args=[
            '-static-libgcc',
            '-static-libstdc++',
        ],
    ),
]


class BuildExt(build_ext):
    """Custom build extension to add compiler-specific options"""
    def build_extensions(self):
        # Add numpy include directory
        import numpy
        for ext in self.extensions:
            ext.include_dirs.append(numpy.get_include())
        
        build_ext.build_extensions(self)


setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)
