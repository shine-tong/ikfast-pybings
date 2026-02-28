故障排除
========

构建问题
--------

**问题**：``error: Microsoft Visual C++ 14.0 or greater is required``

**解决方案**：安装 Visual Studio Build Tools 或带有 C++ 支持的 Visual Studio。详见 :doc:`构建指南 <guides/building_cn>`。

----

**问题**：``fatal error: pybind11/pybind11.h: No such file or directory``

**解决方案**：构建前安装 pybind11：

.. code-block:: bash

   pip install pybind11

----

**问题**：``ImportError: DLL load failed while importing _ikfast_pybind``

**解决方案**：确保已安装 C++ 运行时库。在 Windows 上，安装 Visual C++ Redistributable。

运行时问题
----------

**问题**：``ValueError: compute_ik: Invalid translation shape``

**解决方案**：确保 translation 是包含 3 个元素的一维数组：

.. code-block:: python

   translation = np.array([x, y, z])  # 正确
   # 不是：translation = [[x, y, z]]  # 错误 - 二维数组

----

**问题**：返回空解列表

**解决方案**：目标位姿可能在机器人工作空间之外。验证位姿是否可达：

.. code-block:: python

   solutions = ik.compute_ik(translation, rotation)
   if not solutions:
       print("位姿不可达")

----

**问题**：``RuntimeError: IKFast solver error``

**解决方案**：旋转矩阵可能无效。确保它是正交归一化的：

.. code-block:: python

   # 检查旋转是否有效
   det = np.linalg.det(rotation)
   assert np.isclose(det, 1.0), "旋转矩阵的行列式必须为 1"

性能问题
--------

**问题**：重复调用逆运动学时性能慢

**解决方案**：确保数组是连续的并使用适当的数据类型：

.. code-block:: python

   # 好 - 连续的 float64
   translation = np.ascontiguousarray(translation, dtype=np.float64)

   # 避免在循环中创建新数组
   for pose in poses:
       solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))

获取帮助
--------

如果遇到此处未涵盖的问题：

1. 查看 :doc:`guides/building_cn` 了解详细的构建说明
2. 查看 :doc:`examples/index_cn` 了解使用模式
3. 确保您的输入符合预期的形状和类型
4. 验证您的 C++ 编译器和 Python 环境配置正确
5. 检查 NumPy 和 pybind11 是否正确安装
