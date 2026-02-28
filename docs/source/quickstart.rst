Quick Start
===========

.. code-block:: python

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