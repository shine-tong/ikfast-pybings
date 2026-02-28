安装
====

前置要求
--------

安装前，请确保您有：

- **Python**：3.8 或更高版本
- **C++ 编译器**：

  - Windows：MSVC 14.0+（Visual Studio 2015 或更高版本）
  - Linux：GCC 7.0+ 或 Clang 5.0+
  - macOS：Xcode 命令行工具

- **NumPy**：1.20.0 或更高版本
- **pybind11**：2.6.0 或更高版本

详细的构建说明和故障排除请参见 :doc:`guides/building_cn`。

从源码安装
----------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/shine-tong/ikfast-pybings.git
   cd ikfast_pybind

   # 安装构建依赖
   pip install pybind11 numpy

   # 构建并安装
   pip install .

开发安装
--------

用于开发的可编辑安装和测试工具：

.. code-block:: bash

   # 安装开发依赖
   pip install -e ".[dev]"

   # 运行测试
   pytest tests/

验证安装
--------

.. code-block:: python

   import ikfast_pybind as ik
   print(f"IKFast Python Bindings v{ik.__version__}")
   print(f"求解器有 {ik.get_solver_info()['num_joints']} 个关节")