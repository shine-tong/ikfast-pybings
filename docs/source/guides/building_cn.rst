构建说明
========

:doc:`English <building>` | 中文文档

本文档提供 IKFast Python 绑定的详细构建说明，包括前置要求、构建步骤、故障排除和跨平台支持。

.. contents:: 目录
   :local:
   :depth: 2


前置要求
--------

必需软件
~~~~~~~~

**1. Python 3.8 或更高版本**

验证 Python 安装：

.. code-block:: bash

   python --version

如果未安装：

- **Windows**: https://www.python.org/downloads/
- **Linux**: 使用包管理器（如 ``apt``、``yum``）
- **macOS**: 使用 Homebrew 或从 python.org 下载


**2. C++ 编译器**

根据您的操作系统选择：

**Windows：Microsoft Visual C++ 14.0 或更高版本**

选项 A：安装 Visual Studio Build Tools（推荐）

1. 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 运行安装程序
3. 选择“使用 C++ 的桌面开发”工作负载
4. 确保选中以下组件：
   - MSVC v142 或更高版本
   - Windows 10 SDK
   - C++ CMake 工具（可选）

选项 B：安装完整的 Visual Studio

1. 下载：https://visualstudio.microsoft.com/downloads/
2. 安装 Visual Studio Community（免费）或更高版本
3. 安装期间选择“使用 C++ 的桌面开发”

验证安装：

.. code-block:: doscon

   cl

应显示 Microsoft C/C++ 编译器版本信息。


**Linux：GCC 7.0+ 或 Clang 5.0+**

Ubuntu/Debian：

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install build-essential python3-dev

CentOS/RHEL：

.. code-block:: bash

   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel

Fedora：

.. code-block:: bash

   sudo dnf groupinstall "Development Tools"
   sudo dnf install python3-devel

验证：

.. code-block:: bash

   gcc --version
   clang --version


**macOS：Xcode 命令行工具**

安装：

.. code-block:: bash

   xcode-select --install

验证：

.. code-block:: bash

   clang --version


**3. Python 构建依赖**

.. code-block:: bash

   pip install --upgrade pip setuptools wheel
   pip install pybind11>=2.6.0 numpy>=1.20.0

验证：

.. code-block:: python

   import pybind11, numpy
   print(pybind11.__version__)
   print(numpy.__version__)


构建步骤
--------

方法 1：标准安装（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd ikfast_pybind
   pip install .

测试导入：

.. code-block:: python

   import ikfast_pybind as ik


方法 2：开发安装（可编辑模式）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd ikfast_pybind
   pip install -e ".[dev]"

修改 C++ 代码后重新构建：

.. code-block:: bash

   pip install -e ".[dev]" --force-reinstall --no-deps


方法 3：就地构建
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py build_ext --inplace


方法 4：创建分发包
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python setup.py sdist
   python setup.py bdist_wheel


验证安装
--------

1. 测试项目结构

.. code-block:: bash

   python tests/test_build.py

2. 测试模块导入

.. code-block:: python

   import ikfast_pybind as ik
   print(ik.__version__)

3. 运行测试

.. code-block:: bash

   pytest tests/


故障排除
--------

Windows 常见问题
~~~~~~~~~~~~~~~~

问题 1：缺少 Microsoft Visual C++ 14.0

::

   error: Microsoft Visual C++ 14.0 or greater is required.

解决方案：

1. 安装 Build Tools
2. 重启终端
3. 重试安装


问题 2：DLL 加载失败

::

   ImportError: DLL load failed while importing _ikfast_pybind

解决方案：

1. 安装 Visual C++ Redistributable
2. 重新安装依赖：

   .. code-block:: bash

      pip install numpy pybind11

3. 确保 Python 版本与构建时一致


Linux 常见问题
~~~~~~~~~~~~~~

问题：缺少编译器

::

   error: command 'gcc' failed

解决方案：

.. code-block:: bash

   sudo apt-get install build-essential python3-dev


macOS 常见问题
~~~~~~~~~~~~~~

问题：invalid active developer path

::

   xcrun: error: invalid active developer path

解决方案：

.. code-block:: bash

   xcode-select --install


通用问题
~~~~~~~~

问题：找不到 pybind11

::

   fatal error: pybind11/pybind11.h

解决方案：

.. code-block:: bash

   pip install pybind11


构建配置
--------

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


跨平台支持
----------

支持的平台
~~~~~~~~~~

+--------------+-------------+------------+----------+
| 平台         | 架构        | Python 版本| 状态     |
+==============+=============+============+==========+
| Windows 10/11| x64         | 3.8–3.12   | 完全支持 |
+--------------+-------------+------------+----------+
| Ubuntu 20.04+| x64         | 3.8–3.12   | 完全支持 |
+--------------+-------------+------------+----------+
| macOS 11+    | ARM/x64     | 3.8–3.12   | 完全支持 |
+--------------+-------------+------------+----------+


高级构建选项
------------

自定义编译器
~~~~~~~~~~~~

.. code-block:: bash

   export CXX=/usr/bin/g++-9
   pip install .


调试构建
~~~~~~~~

.. code-block:: bash

   export CXXFLAGS="-g -O0"
   pip install .


清理构建
~~~~~~~~

.. code-block:: bash

   rm -rf build/ dist/ *.egg-info/


获取帮助
--------

1. 检查前置要求
2. 使用详细模式：

   .. code-block:: bash

      pip install . -v

3. 清理并重新构建
4. 查看 :doc:`项目简介 </introduction_cn>` 获取使用说明


版本历史
--------

v0.1.0
~~~~~~

- 初始版本
- 支持 6-DOF 机械臂
- 跨平台支持（Windows、Linux、macOS）