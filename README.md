# IKFast Python Bindings

[中文文档](README_CN.md) | English

High-performance Python bindings for the IKFast inverse kinematics solver using pybind11. This package provides a clean, Pythonic interface to analytical IK solutions for 6-DOF robotic manipulators with seamless NumPy integration.

## Features

- **⚡ Fast Analytical Solutions**: Leverage IKFast's analytical inverse kinematics for real-time performance
- **🔢 NumPy Integration**: Seamless conversion between NumPy arrays and C++ data structures with zero-copy where possible
- **🎯 Multiple Solutions**: Access all valid IK solutions for a given pose
- **🐍 Pythonic API**: Clean, intuitive interface following Python conventions
- **🔒 Type Safety**: Full type hints and comprehensive input validation
- **⚠️ Error Handling**: Descriptive error messages with proper exception types
- **📊 Property-Based Testing**: Validated with 146 tests including property-based tests (100+ iterations each)
- **🌐 Cross-Platform**: Supports Windows, Linux, and macOS

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using Custom Solvers](#using-custom-solvers)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Before installing, ensure you have:
- **Python**: 3.8 or later
- **C++ Compiler**: 
  - Windows: MSVC 14.0+ (Visual Studio 2015 or later)
  - Linux: GCC 7.0+ or Clang 5.0+
  - macOS: Xcode Command Line Tools
- **NumPy**: 1.20.0 or later
- **pybind11**: 2.6.0 or later

See [BUILD.md](BUILD.md) for detailed build instructions and troubleshooting.

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd ikfast_pybind

# Install build dependencies
pip install pybind11 numpy

# Build and install
pip install .
```

### Development Installation

For development with editable installation and testing tools:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Verify Installation

```python
import ikfast_pybind as ik
print(f"IKFast Python Bindings v{ik.__version__}")
print(f"Solver has {ik.get_solver_info()['num_joints']} joints")
```

## Using Custom Solvers

**The bindings work with any IKFast-generated solver!** When you change your robot model, simply:

1. Use [ikfast-online](https://github.com/shine-tong/ikfast-online) to generate an IKFast solver `.cpp` file for your robot
2. Replace the solver file in the `src/` directory (must end with `_ikfast_solver.cpp`)
3. Rebuild: `pip install . --force-reinstall`

The build system automatically detects and uses your solver file.

### Quick Example

```bash
# 1. Generate IKFast solver using ikfast-online
For detailed steps, see the repository (https://github.com/shine-tong/ikfast-online)

# 2. Copy generated solver to project
cd /path/to/ikfast_pybind
cp /path/to/your_robot_ikfast_solver.cpp src/   # cpp file directory
rm src/sa0521_manipulator_ikfast_solver.cpp     # Remove old solver

# 3. Rebuild
pip install . --force-reinstall
```

**For detailed instructions on using custom solvers, see [CUSTOM_SOLVER.md](CUSTOM_SOLVER.md)** (or [中文版](CUSTOM_SOLVER_CN.md)).

## Quick Start

```python
import ikfast_pybind as ik
import numpy as np

# Get solver information
info = ik.get_solver_info()
print(f"Robot has {info['num_joints']} joints")

# Compute forward kinematics
joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
translation, rotation = ik.compute_fk(joints)
print(f"End effector position: {translation}")

# Compute inverse kinematics
target_translation = np.array([0.5, 0.0, 0.5])
target_rotation = np.eye(3)
solutions = ik.compute_ik(target_translation, target_rotation)

print(f"Found {len(solutions)} IK solutions")
for i, solution in enumerate(solutions):
    print(f"Solution {i+1}: {solution}")
```

## API Reference

### High-Level Functions

#### `compute_ik(translation, rotation, free_params=None)`

Compute inverse kinematics solutions for a target end effector pose.

**Parameters:**
- `translation` (np.ndarray): End effector position [x, y, z], shape (3,)
- `rotation` (np.ndarray): End effector orientation as rotation matrix, shape (3, 3) or flattened (9,)
- `free_params` (np.ndarray, optional): Free parameter values for redundant joints

**Returns:**
- `List[np.ndarray]`: List of joint angle solutions, each with shape (num_joints,). Returns empty list if no solutions exist.

**Raises:**
- `ValueError`: Invalid input shapes or values
- `TypeError`: Inputs cannot be converted to numpy arrays
- `RuntimeError`: Solver numerical issues

**Example:**
```python
translation = np.array([0.5, 0.0, 0.5])
rotation = np.eye(3)
solutions = ik.compute_ik(translation, rotation)
```

#### `compute_fk(joint_angles)`

Compute forward kinematics for a given joint configuration.

**Parameters:**
- `joint_angles` (np.ndarray): Joint angles, shape (num_joints,)

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (translation, rotation_matrix)
  - `translation`: shape (3,)
  - `rotation_matrix`: shape (3, 3)

**Raises:**
- `ValueError`: Invalid joint_angles shape
- `TypeError`: Input cannot be converted to numpy array

**Example:**
```python
joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
translation, rotation = ik.compute_fk(joints)
```

#### `get_solver_info()`

Get information about the IK solver configuration.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `num_joints` (int): Number of joints
  - `num_free_parameters` (int): Number of free parameters
  - `free_parameters` (List[int]): Free parameter indices
  - `ik_type` (int): Solver type identifier
  - `kinematics_hash` (str): Kinematics configuration hash
  - `ikfast_version` (str): IKFast version

**Example:**
```python
info = ik.get_solver_info()
print(f"Solver type: {hex(info['ik_type'])}")
```

### Low-Level Classes

For advanced usage, the following classes are also available:

- `IkSolution`: Individual IK solution with free parameter support
- `IkSolutionList`: Container for multiple IK solutions with Python iteration support

See the [examples](examples/) directory for more detailed usage patterns.

## Examples

The `examples/` directory contains comprehensive example scripts demonstrating various use cases:

### Basic Examples

#### 1. **basic_ik.py** - Computing IK Solutions
Demonstrates:
- Computing IK for a target pose
- Iterating through multiple solutions
- Selecting the closest solution to current configuration
- Selecting solutions away from joint limits
- Verifying solutions with FK

```bash
python examples/basic_ik.py
```

#### 2. **basic_fk.py** - Forward Kinematics
Demonstrates:
- Computing FK for joint configurations
- Validating rotation matrices
- Converting between rotation representations
- Verifying IK solutions with FK round-trip

```bash
python examples/basic_fk.py
```

#### 3. **solution_selection.py** - Advanced Selection
Demonstrates:
- Multiple selection criteria (distance, energy, manipulability)
- Handling free parameters for redundant robots
- Workspace boundary detection
- Trajectory planning with smooth joint motion

```bash
python examples/solution_selection.py
```

### Code Snippets

**Select closest solution to current pose:**
```python
import numpy as np
import ikfast_pybind as ik

def select_closest_solution(solutions, current_joints):
    """Select IK solution closest to current joint configuration."""
    if not solutions:
        return None
    
    distances = [np.linalg.norm(sol - current_joints) for sol in solutions]
    return solutions[np.argmin(distances)]

# Usage
current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
solutions = ik.compute_ik(translation, rotation)
best = select_closest_solution(solutions, current)
```

**Verify solution with FK:**
```python
def verify_ik_solution(solution, target_trans, target_rot, tol=1e-6):
    """Verify that an IK solution produces the target pose."""
    computed_trans, computed_rot = ik.compute_fk(solution)
    
    trans_error = np.linalg.norm(target_trans - computed_trans)
    rot_error = np.linalg.norm(target_rot - computed_rot)
    
    return trans_error < tol and rot_error < tol
```

**Handle unreachable poses:**
```python
def safe_compute_ik(translation, rotation):
    """Compute IK with graceful handling of unreachable poses."""
    try:
        solutions = ik.compute_ik(translation, rotation)
        if not solutions:
            print("Warning: Pose is outside robot workspace")
            return None
        return solutions
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None
    except RuntimeError as e:
        print(f"Solver error: {e}")
        return None
```

## Testing

The project includes comprehensive test coverage with both unit tests and property-based tests.

### Test Statistics

- **Total Tests**: 146
- **Test Coverage**: 95%
- **Property-Based Tests**: 71 tests with 100+ iterations each
- **Unit Tests**: 75 tests covering specific examples and edge cases

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run only unit tests
pytest tests/test_*.py -k "not property"

# Run only property tests
pytest tests/test_property_*.py

# Run with coverage report
pytest tests/ --cov=ikfast_pybind --cov-report=html

# Run specific test file
pytest tests/test_compute_ik.py
```

### Test Categories

1. **Build Tests** (`test_build.py`)
   - Project structure validation
   - Build configuration verification

2. **Unit Tests** (`test_*.py`)
   - Specific input/output examples
   - Edge cases and boundary conditions
   - Error handling validation

3. **Property-Based Tests** (`test_property_*.py`)
   - IK-FK round-trip consistency
   - FK-IK round-trip consistency
   - Array type conversion correctness
   - Input validation
   - Exception translation
   - Solution completeness
   - Free parameter handling
   - Index bounds checking

### Continuous Integration

Tests are validated across:
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Operating systems: Windows, Linux, macOS
- NumPy versions: 1.20.0+

## Troubleshooting

### Build Issues

**Problem**: `error: Microsoft Visual C++ 14.0 or greater is required`

**Solution**: Install Visual Studio Build Tools or Visual Studio with C++ support. See [BUILD.md](BUILD.md) for details.

---

**Problem**: `fatal error: pybind11/pybind11.h: No such file or directory`

**Solution**: Install pybind11 before building:
```bash
pip install pybind11
```

---

**Problem**: `ImportError: DLL load failed while importing _ikfast_pybind`

**Solution**: Ensure the C++ runtime libraries are installed. On Windows, install the Visual C++ Redistributable.

### Runtime Issues

**Problem**: `ValueError: compute_ik: Invalid translation shape`

**Solution**: Ensure translation is a 1D array with 3 elements:
```python
translation = np.array([x, y, z])  # Correct
# Not: translation = [[x, y, z]]  # Wrong - 2D array
```

---

**Problem**: Empty solution list returned

**Solution**: The target pose may be outside the robot's workspace. Verify the pose is reachable:
```python
solutions = ik.compute_ik(translation, rotation)
if not solutions:
    print("Pose is unreachable")
```

---

**Problem**: `RuntimeError: IKFast solver error`

**Solution**: The rotation matrix may be invalid. Ensure it's orthonormal:
```python
# Check if rotation is valid
det = np.linalg.det(rotation)
assert np.isclose(det, 1.0), "Rotation matrix must have determinant 1"
```

### Performance Issues

**Problem**: Slow performance with repeated IK calls

**Solution**: Ensure arrays are contiguous and use appropriate dtypes:
```python
# Good - contiguous float64
translation = np.ascontiguousarray(translation, dtype=np.float64)

# Avoid creating new arrays in loops
for pose in poses:
    solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))
```

### Getting Help

If you encounter issues not covered here:

1. Check the [BUILD.md](BUILD.md) for detailed build instructions
2. Review the [examples](examples/) for usage patterns
3. Ensure your inputs match the expected shapes and types
4. Verify your C++ compiler and Python environment are properly configured
5. Check that NumPy and pybind11 are correctly installed

## Performance

### Benchmarks

The Python bindings add minimal overhead compared to direct C++ calls:

- **IK computation**: < 5% overhead
- **FK computation**: < 3% overhead
- **Array conversion**: Zero-copy where possible
- **GIL release**: Enabled during C++ computations for multi-threading

### Optimization Tips

1. **Use contiguous arrays:**
```python
# Good - contiguous array
translation = np.ascontiguousarray(translation, dtype=np.float64)

# Avoid - non-contiguous slices may require copying
translation = some_array[::2, :]  # May not be contiguous
```

2. **Reuse arrays when possible:**
```python
# Good - reuse array
joints = np.zeros(6, dtype=np.float64)
for i, config in enumerate(configs):
    joints[:] = config
    trans, rot = ik.compute_fk(joints)

# Avoid - creating new arrays in loop
for config in configs:
    trans, rot = ik.compute_fk(np.array(config))
```

3. **Batch processing:**
```python
# Process multiple poses efficiently
results = []
for pose in poses:
    solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))
    if solutions:
        results.append(solutions[0])
```

### Memory Management

- **Automatic**: pybind11 handles reference counting automatically
- **No memory leaks**: RAII ensures proper cleanup
- **Efficient**: Minimal allocations for array operations

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 for Python code
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Type Hints**: Include type annotations for public APIs

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ikfast_pybind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=ikfast_pybind --cov-report=html
```

### Running Property-Based Tests

Property-based tests use Hypothesis for randomized testing:

```bash
# Run with default iterations (100)
pytest tests/test_property_*.py

# Run with more iterations for thorough testing
pytest tests/test_property_*.py --hypothesis-iterations=1000

# Run with specific seed for reproducibility
pytest tests/test_property_*.py --hypothesis-seed=12345
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Application                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Python API calls
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Binding Layer (pybind11)                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  High-Level API (ikfast_pybind/__init__.py)          │   │
│  │  - compute_ik() → List[np.ndarray]                   │   │
│  │  - compute_fk() → Tuple[np.ndarray, np.ndarray]      │   │
│  │  - get_solver_info() → Dict                          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Low-Level API (_ikfast_pybind.cpp)                  │   │
│  │  - IkSolutionList wrapper                            │   │
│  │  - IkSolution wrapper                                │   │
│  │  - Direct function bindings                          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Type Conversion Layer                               │   │
│  │  - numpy ↔ C++ array conversion                      │   │
│  │  - Exception translation                             │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ C++ function calls
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              IKFast C++ Solver                               │
│  - ComputeIk()                                               │
│  - ComputeFk()                                               │
│  - GetNumJoints(), GetNumFreeParameters(), etc.              │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
ikfast_pybind/
├── ikfast_pybind/
│   ├── __init__.py              # High-level Python API
│   └── _ikfast_pybind.cpp       # pybind11 binding code
├── src/
│   └── sa0521_manipulator_ikfast_solver.cpp  # IKFast solver
├── include/
│   └── ikfast.h                 # IKFast header
├── examples/
│   ├── basic_ik.py              # Basic IK example
│   ├── basic_fk.py              # Basic FK example
│   └── solution_selection.py   # Advanced selection
├── tests/
│   ├── test_*.py                # Unit tests
│   └── test_property_*.py       # Property-based tests
├── setup.py                     # Build configuration
├── pyproject.toml               # Package metadata
├── MANIFEST.in                  # Package data files
├── README.md                    # This file
└── BUILD.md                     # Build instructions
```

## Requirements

- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **NumPy**: 1.20.0 or later
- **pybind11**: 2.6.0 or later
- **C++ Compiler**: C++11 compatible
  - Windows: MSVC 14.0+ (Visual Studio 2015+)
  - Linux: GCC 7.0+ or Clang 5.0+
  - macOS: Xcode Command Line Tools

### Optional Dependencies

- **pytest**: 6.0+ (for running tests)
- **hypothesis**: 6.0+ (for property-based tests)
- **pytest-cov**: 5.0+ (for coverage reports)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- **IKFast**: Part of the OpenRAVE project
- **pybind11**: Seamless operability between C++11 and Python
- **NumPy**: Fundamental package for scientific computing

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ikfast_pybind,
  title = {IKFast Python Bindings},
  author = {IKFast Python Bindings Contributors},
  year = {2026},
  url = {<repository-url>}
}
```
