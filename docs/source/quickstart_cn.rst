快速开始
========

.. code-block:: python

   import ikfast_pybind as ik
   import numpy as np

   # 获取求解器信息
   info = ik.get_solver_info()
   print(f"机器人有 {info['num_joints']} 个关节")

   # 计算正运动学
   joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
   translation, rotation = ik.compute_fk(joints)
   print(f"末端执行器位置：{translation}")

   # 计算逆运动学
   target_translation = np.array([0.5, 0.0, 0.5])
   target_rotation = np.eye(3)
   solutions = ik.compute_ik(target_translation, target_rotation)

   print(f"找到 {len(solutions)} 个逆运动学解")
   for i, solution in enumerate(solutions):
       print(f"解 {i+1}：{solution}")