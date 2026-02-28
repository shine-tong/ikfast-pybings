Build Instructions
==================

English | :doc:`中文文档 <building_cn>`

This document provides detailed build instructions for IKFast Python Bindings, including prerequisites, build steps, troubleshooting, and cross-platform support.

.. contents:: Table of Contents
   :local:
   :depth: 2


.. _prerequisites:

Prerequisites
-------------

Required Software
~~~~~~~~~~~~~~~~~

**1. Python 3.8 or Later**

Verify Python installation:

.. code-block:: bash

   python --version

If not installed:

- **Windows**: https://www.python.org/downloads/
- **Linux**: Use package manager (e.g., ``apt``, ``yum``)
- **macOS**: Use Homebrew or download from python.org


**2. C++ Compiler**

Choose based on your operating system:

**Windows: Microsoft Visual C++ 14.0 or Greater**

Option A: Install Visual Studio Build Tools (Recommended)

1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select "Desktop development with C++"
4. Ensure:
   - MSVC v142 or later
   - Windows 10 SDK
   - C++ CMake tools (optional)

Option B: Install Full Visual Studio

1. Download: https://visualstudio.microsoft.com/downloads/
2. Install Visual Studio Community (free) or higher
3. Select "Desktop development with C++"

Verify installation:

.. code-block:: doscon

   cl


**Linux: GCC 7.0+ or Clang 5.0+**

Ubuntu/Debian:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install build-essential python3-dev

CentOS/RHEL:

.. code-block:: bash

   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel

Fedora:

.. code-block:: bash

   sudo dnf groupinstall "Development Tools"
   sudo dnf install python3-devel

Verify:

.. code-block:: bash

   gcc --version
   clang --version


**macOS: Xcode Command Line Tools**

Install:

.. code-block:: bash

   xcode-select --install

Verify:

.. code-block:: bash

   clang --version


**3. Python Build Dependencies**

.. code-block:: bash

   pip install --upgrade pip setuptools wheel
   pip install pybind11>=2.6.0 numpy>=1.20.0

Verify:

.. code-block:: python

   import pybind11, numpy
   print(pybind11.__version__)
   print(numpy.__version__)


.. _build_steps:

Build Steps
-----------

Method 1: Standard Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd ikfast_pybind
   pip install .

Test import:

.. code-block:: python

   import ikfast_pybind as ik


Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd ikfast_pybind
   pip install -e ".[dev]"

Rebuild after modifying C++:

.. code-block:: bash

   pip install -e ".[dev]" --force-reinstall --no-deps


Method 3: In-Place Build
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py build_ext --inplace


Method 4: Create Distribution Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py sdist
   python setup.py bdist_wheel


.. _verification:

Verification
------------

1. Test Project Structure

.. code-block:: bash

   python tests/test_build.py

2. Test Module Import

.. code-block:: python

   import ikfast_pybind as ik
   print(ik.__version__)

3. Run Tests

.. code-block:: bash

   pytest tests/


.. _troubleshooting:

Troubleshooting
---------------

Windows Common Issues
~~~~~~~~~~~~~~~~~~~~~

Issue 1: Missing Microsoft Visual C++ 14.0

Error:

::

   error: Microsoft Visual C++ 14.0 or greater is required.

Solution:

1. Install Build Tools
2. Restart terminal
3. Retry installation


Issue 2: DLL Load Failed

Error:

::

   ImportError: DLL load failed while importing _ikfast_pybind

Solution:

1. Install Visual C++ Redistributable
2. Reinstall dependencies:

   .. code-block:: bash

      pip install numpy pybind11

3. Ensure Python version matches build version


Linux Common Issues
~~~~~~~~~~~~~~~~~~~

Issue 1: Missing Compiler

::

   error: command 'gcc' failed

Solution:

.. code-block:: bash

   sudo apt-get install build-essential python3-dev


macOS Common Issues
~~~~~~~~~~~~~~~~~~~

Issue: Invalid Developer Path

::

   xcrun: error: invalid active developer path

Solution:

.. code-block:: bash

   xcode-select --install


General Issues
~~~~~~~~~~~~~~

Issue: pybind11 Not Found

::

   fatal error: pybind11/pybind11.h

Solution:

.. code-block:: bash

   pip install pybind11


.. _build_configuration:

Build Configuration
-------------------

Configuration Files
~~~~~~~~~~~~~~~~~~~

**pyproject.toml**

.. code-block:: toml

   [build-system]
   requires = ["setuptools>=45", "wheel", "pybind11>=2.6.0", "numpy>=1.20.0"]
   build-backend = "setuptools.build_meta"

**setup.py**

.. code-block:: python

   Extension(
       'ikfast_pybind._ikfast_pybind',
       sources=['ikfast_pybind/_ikfast_pybind.cpp'],
       language='c++'
   )


.. _cross_platform_support:

Cross-Platform Support
----------------------

Supported Platforms
~~~~~~~~~~~~~~~~~~~

+--------------+-------------+---------------+------------------+
| Platform     | Architecture| Python Vers.  | Status           |
+==============+=============+===============+==================+
| Windows 10/11| x64         | 3.8–3.12      | Fully Supported  |
+--------------+-------------+---------------+------------------+
| Ubuntu 20.04+| x64         | 3.8–3.12      | Fully Supported  |
+--------------+-------------+---------------+------------------+
| macOS 11+    | ARM/x64     | 3.8–3.12      | Fully Supported  |
+--------------+-------------+---------------+------------------+


.. _advanced_build_options:

Advanced Build Options
----------------------

Custom Compiler
~~~~~~~~~~~~~~~

.. code-block:: bash

   export CXX=/usr/bin/g++-9
   pip install .


Debug Build
~~~~~~~~~~~

.. code-block:: bash

   export CXXFLAGS="-g -O0"
   pip install .


Parallel Build
~~~~~~~~~~~~~~

.. code-block:: bash

   export MAX_JOBS=4
   pip install .


Clean Build
~~~~~~~~~~~

.. code-block:: bash

   rm -rf build/ dist/ *.egg-info/


Getting Help
------------

1. Check prerequisites
2. Use verbose mode:

   .. code-block:: bash

      pip install . -v

3. Clean and rebuild
4. See :doc:`README <README>` for usage instructions


Version History
---------------

v0.1.0
~~~~~~

- Initial release
- 6-DOF support
- Cross-platform support