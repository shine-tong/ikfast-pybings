Troubleshooting
===============

Build Issues
------------

**Problem**: ``error: Microsoft Visual C++ 14.0 or greater is required``

**Solution**: Install Visual Studio Build Tools or Visual Studio with C++ support. See :doc:`guides/building` for details.

----

**Problem**: ``fatal error: pybind11/pybind11.h: No such file or directory``

**Solution**: Install pybind11 before building:

.. code-block:: bash

   pip install pybind11

----

**Problem**: ``ImportError: DLL load failed while importing _ikfast_pybind``

**Solution**: Ensure the C++ runtime libraries are installed. On Windows, install the Visual C++ Redistributable.

Runtime Issues
--------------

**Problem**: ``ValueError: compute_ik: Invalid translation shape``

**Solution**: Ensure translation is a 1D array with 3 elements:

.. code-block:: python

   translation = np.array([x, y, z])  # Correct
   # Not: translation = [[x, y, z]]  # Wrong - 2D array

----

**Problem**: Empty solution list returned

**Solution**: The target pose may be outside the robot's workspace. Verify the pose is reachable:

.. code-block:: python

   solutions = ik.compute_ik(translation, rotation)
   if not solutions:
       print("Pose is unreachable")

----

**Problem**: ``RuntimeError: IKFast solver error``

**Solution**: The rotation matrix may be invalid. Ensure it's orthonormal:

.. code-block:: python

   # Check if rotation is valid
   det = np.linalg.det(rotation)
   assert np.isclose(det, 1.0), "Rotation matrix must have determinant 1"

Performance Issues
------------------

**Problem**: Slow performance with repeated IK calls

**Solution**: Ensure arrays are contiguous and use appropriate dtypes:

.. code-block:: python

   # Good - contiguous float64
   translation = np.ascontiguousarray(translation, dtype=np.float64)

   # Avoid creating new arrays in loops
   for pose in poses:
       solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))

Getting Help
------------

If you encounter issues not covered here:

1. Check the :doc:`guides/building` for detailed build instructions
2. Review the :doc:`examples/index` for usage patterns
3. Ensure your inputs match the expected shapes and types
4. Verify your C++ compiler and Python environment are properly configured
5. Check that NumPy and pybind11 are correctly installed
