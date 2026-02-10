# Using Custom IKFast Solvers

English | [中文文档](CUSTOM_SOLVER_CN.md)

This guide explains how to use the IKFast Python bindings with different robot models and custom IKFast solver files.

## Overview

The IKFast Python bindings are designed to work with any IKFast-generated solver. When you change your robot model, you need to:

1. Generate a new IKFast solver `.cpp` file for your robot
2. Replace or add the solver file to the `src/` directory
3. Rebuild the Python bindings

The build system automatically detects and uses IKFast solver files in the `src/` directory.

## Quick Start

### Step 1: Generate IKFast Solver Using ikfast-online

[ikfast-online](https://github.com/shine-tong/ikfast-online) is a web-based tool that simplifies IKFast solver generation. It provides a convenient web interface to generate solver files directly from URDF files.

**For detailed usage instructions, please refer to the [ikfast-online README](https://github.com/shine-tong/ikfast-online).**

> **⚠️ Important**: The solver filename must follow the pattern `*_ikfast_solver.cpp`.

### Step 2: Replace Solver File

Replace the existing solver in the `src/` directory:

```bash
# Remove old solver (optional if you want to keep multiple solvers)
rm src/*_ikfast_solver.cpp

# Copy your new solver
cp your_robot_ikfast_solver.cpp src/
```

### Step 3: Rebuild

Rebuild the Python bindings:

```bash
# Clean previous build
pip uninstall ikfast-pybind -y
rm -rf build/ dist/ *.egg-info/

# Rebuild and install
pip install .
```

### Step 4: Verify

Test the new solver:

```python
import ikfast_pybind as ik

# Check solver information
info = ik.get_solver_info()
print(f"Number of joints: {info['num_joints']}")
print(f"Kinematics hash: {info['kinematics_hash']}")

# Test FK/IK
import numpy as np
joints = np.zeros(info['num_joints'])
translation, rotation = ik.compute_fk(joints)
print(f"End effector position: {translation}")
```

## Solver File Requirements

Your IKFast solver file must:

1. **Follow naming convention**: `*_ikfast_solver.cpp`
2. **Include required functions**: `ComputeIk()`, `ComputeFk()`, `GetNumJoints()`, etc.
3. **Be compatible with ikfast.h**: Use the same IKFast version as `include/ikfast.h`

## Troubleshooting

### Issue: "No IKFast solver file found"

**Error:**
```
FileNotFoundError: No IKFast solver file found matching pattern: src/*_ikfast_solver.cpp
```

**Solution:**
1. Ensure your solver file is in the `src/` directory
2. Check that the filename ends with `_ikfast_solver.cpp`
3. Verify the file exists: `ls src/*_ikfast_solver.cpp`

### Issue: "Multiple IKFast solver files found"

**Warning:**
```
Warning: Multiple IKFast solver files found: ['src/robot_a_ikfast_solver.cpp', 'src/robot_b_ikfast_solver.cpp']
Using: src/robot_a_ikfast_solver.cpp
```

**Solution:**
Specify which solver to use:
```bash
export IKFAST_SOLVER_FILE=src/robot_b_ikfast_solver.cpp
pip install . --force-reinstall
```

Or remove unused solvers:
```bash
rm src/robot_a_ikfast_solver.cpp
```

### Issue: Compilation errors with new solver

**Error:**
```
error: 'ComputeIk' was not declared in this scope
```

**Solution:**
1. Verify your solver file is valid IKFast output
2. Check that it includes all required functions
3. Ensure IKFast version compatibility:
   ```bash
   grep "IKFAST_VERSION" src/your_solver.cpp
   grep "IKFAST_VERSION" include/ikfast.h
   ```

### Issue: Wrong number of joints

**Error:**
```python
>>> info = ik.get_solver_info()
>>> print(info['num_joints'])
6  # Expected 7 for your robot
```

**Solution:**
1. Verify you're using the correct solver file
2. Regenerate the IKFast solver with correct robot model
3. Rebuild the bindings

### Issue: IK solutions don't match expected results

**Possible causes:**
1. Wrong solver file (for different robot)
2. Different coordinate frame conventions
3. Joint angle units (radians vs degrees)

**Solution:**
1. Verify solver kinematics hash matches your robot:
   ```python
   info = ik.get_solver_info()
   print(info['kinematics_hash'])
   ```
2. Test with known joint configurations
3. Compare with OpenRAVE results

## Example: Using ikfast-online with 6-DOF Robot

```bash
# 1. Follow ikfast-online setup instructions from:
https://github.com/shine-tong/ikfast-online

# 2. Copy solver to ikfast_pybind project
cp ur5_ikfast_solver.cpp /path/to/ikfast_pybind/src/

# 3. Rebuild
cd /path/to/ikfast_pybind
pip install . --force-reinstall

# 4. Test
python -c "
import ikfast_pybind as ik
import numpy as np

info = ik.get_solver_info()
print(f'Joints: {info[\"num_joints\"]}')
print(f'Hash: {info[\"kinematics_hash\"]}')

# Test FK
joints = np.zeros(info['num_joints'])
trans, rot = ik.compute_fk(joints)
print(f'FK at zero config: {trans}')
"
```

## Resources

- **ikfast-online**: https://github.com/shine-tong/ikfast-online - Web-based tool for generating IKFast solvers
- **IKFast Documentation**: http://openrave.org/docs/latest_stable/openravepy/ikfast/

## Support

If you encounter issues with custom solvers:

1. Verify your IKFast solver is correctly generated using ikfast-online
2. Check the troubleshooting section above
3. Consult the ikfast-online README for solver generation issues
