示例
====

``examples/`` 目录包含演示各种用例的综合示例脚本：

基础示例
--------

1. basic_ik.py - 计算逆运动学解
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

演示：

- 计算目标位姿的逆运动学
- 遍历多个解
- 选择最接近当前配置的解
- 选择远离关节限位的解
- 用正运动学验证解

.. code-block:: bash

   python examples/basic_ik.py

2. basic_fk.py - 正运动学
~~~~~~~~~~~~~~~~~~~~~~~~~~

演示：

- 计算关节配置的正运动学
- 验证旋转矩阵
- 旋转表示之间的转换
- 用正运动学往返验证逆运动学解

.. code-block:: bash

   python examples/basic_fk.py

3. solution_selection.py - 高级选择
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

演示：

- 多种选择标准（距离、能量、可操作性）
- 处理冗余机器人的自由参数
- 工作空间边界检测
- 平滑关节运动的轨迹规划

.. code-block:: bash

   python examples/solution_selection.py

代码片段
--------

选择最接近当前位姿的解
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import ikfast_pybind as ik

   def select_closest_solution(solutions, current_joints):
       """选择最接近当前关节配置的逆运动学解。"""
       if not solutions:
           return None
       
       distances = [np.linalg.norm(sol - current_joints) for sol in solutions]
       return solutions[np.argmin(distances)]

   # 使用
   current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
   solutions = ik.compute_ik(translation, rotation)
   best = select_closest_solution(solutions, current)

用正运动学验证解
~~~~~~~~~~~~~~~~

.. code-block:: python

   def verify_ik_solution(solution, target_trans, target_rot, tol=1e-6):
       """验证逆运动学解是否产生目标位姿。"""
       computed_trans, computed_rot = ik.compute_fk(solution)
       
       trans_error = np.linalg.norm(target_trans - computed_trans)
       rot_error = np.linalg.norm(target_rot - computed_rot)
       
       return trans_error < tol and rot_error < tol

处理不可达位姿
~~~~~~~~~~~~~~

.. code-block:: python

   def safe_compute_ik(translation, rotation):
       """计算逆运动学并优雅地处理不可达位姿。"""
       try:
           solutions = ik.compute_ik(translation, rotation)
           if not solutions:
               print("警告：位姿在机器人工作空间之外")
               return None
           return solutions
       except ValueError as e:
           print(f"输入无效：{e}")
           return None
       except RuntimeError as e:
           print(f"求解器错误：{e}")
           return None
