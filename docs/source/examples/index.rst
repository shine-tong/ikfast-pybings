Examples
========

The ``examples/`` directory contains comprehensive example scripts demonstrating various use cases:

Basic Examples
--------------

1. basic_ik.py - Computing IK Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates:

- Computing IK for a target pose
- Iterating through multiple solutions
- Selecting the closest solution to current configuration
- Selecting solutions away from joint limits
- Verifying solutions with FK

.. code-block:: bash

   python examples/basic_ik.py

2. basic_fk.py - Forward Kinematics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates:

- Computing FK for joint configurations
- Validating rotation matrices
- Converting between rotation representations
- Verifying IK solutions with FK round-trip

.. code-block:: bash

   python examples/basic_fk.py

3. solution_selection.py - Advanced Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrates:

- Multiple selection criteria (distance, energy, manipulability)
- Handling free parameters for redundant robots
- Workspace boundary detection
- Trajectory planning with smooth joint motion

.. code-block:: bash

   python examples/solution_selection.py

Code Snippets
-------------

Select closest solution to current pose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Verify solution with FK
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def verify_ik_solution(solution, target_trans, target_rot, tol=1e-6):
       """Verify that an IK solution produces the target pose."""
       computed_trans, computed_rot = ik.compute_fk(solution)
       
       trans_error = np.linalg.norm(target_trans - computed_trans)
       rot_error = np.linalg.norm(target_rot - computed_rot)
       
       return trans_error < tol and rot_error < tol

Handle unreachable poses
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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
