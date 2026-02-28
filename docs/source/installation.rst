Installation
============

Prerequisites
-------------

Before installing, ensure you have:

- **Python**: 3.8 or later
- **C++ Compiler**:

  - Windows: MSVC 14.0+ (Visual Studio 2015 or later)
  - Linux: GCC 7.0+ or Clang 5.0+
  - macOS: Xcode Command Line Tools

- **NumPy**: 1.20.0 or later
- **pybind11**: 2.6.0 or later

See :doc:`Building <guides/building>` for detailed build instructions and troubleshooting.

From Source
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/shine-tong/ikfast-pybings.git
   cd ikfast_pybind

   # Install build dependencies
   pip install pybind11 numpy

   # Build and install
   pip install .

Development Installation
------------------------

For development with editable installation and testing tools:

.. code-block:: bash

   # Install with development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest tests/

Verify Installation
-------------------

.. code-block:: python

   import ikfast_pybind as ik
   print(f"IKFast Python Bindings v{ik.__version__}")
   print(f"Solver has {ik.get_solver_info()['num_joints']} joints")