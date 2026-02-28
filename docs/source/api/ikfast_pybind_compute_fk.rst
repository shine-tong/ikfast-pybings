ikfast_pybind.compute_fk
========================

.. py:function:: ikfast_pybind. compute_fk ( joint_angles : np.ndarray ) → Tuple [ np.ndarray , np.ndarray ] ¶

   计算给定关节配置的正运动学。
   
   给定关节配置，此函数计算末端执行器的位姿（平移和旋转）。

   :returns: translation(np.ndarray) – 末端执行器位置，形状为 (3,)，数据类型为 float64rotation_matrix(np.ndarray) – 末端执行器旋转矩阵，形状为 (3, 3)，数据类型为 float64
   :rtype: Tuple[np.ndarray, np.ndarray]

   :raises ValueError: 如果 joint_angles 形状不正确
   :raises TypeError: 如果输入不是类数组或无法转换为 numpy 数组

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      import numpy as np
      
      # 基本用法 - 零位配置
      joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      translation, rotation = ik.compute_fk(joints)
      
      print(f"末端执行器位置：{translation}")
      print(f"末端执行器姿态：\n{rotation}")
      
      # 验证旋转矩阵的正交性
      det = np.linalg.det(rotation)
      assert np.isclose(det, 1.0), "旋转矩阵应该是正交的"
      
      # 验证逆运动学解
      target_pos = np.array([0.5, 0.0, 0.5])
      target_rot = np.eye(3)
      
      # 计算逆运动学
      solutions = ik.compute_ik(target_pos, target_rot)
      
      # 验证每个解
      for i, sol in enumerate(solutions):
          computed_pos, computed_rot = ik.compute_fk(sol)
          
          pos_error = np.linalg.norm(target_pos - computed_pos)
          rot_error = np.linalg.norm(target_rot - computed_rot)
          
          print(f"解 {i+1} - 位置误差: {pos_error:.6f}, 姿态误差: {rot_error:.6f}")
          assert pos_error < 1e-6, "FK-IK 往返误差过大"
      
      # 批量计算正运动学
      joint_configs = [
          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
          np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
          np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
      ]
      
      poses = []
      for joints in joint_configs:
          pos, rot = ik.compute_fk(joints)
          poses.append((pos, rot))
      
      print(f"计算了 {len(poses)} 个位姿")
      


   .. note::

      算法说明
      正运动学通过连续应用齐次变换矩阵来计算末端执行器位姿。对于串联机械臂，变换链从基座到末端执行器依次应用每个关节的变换。

   .. note::

      输入验证
      joint_angles 必须是形状为 (num_joints,) 的一维数组关节数量必须与求解器配置匹配（使用 get_solver_info() 查询）输入会自动转换为连续的 float64 数组以获得最佳性能

   .. note::

      旋转矩阵格式
      返回的旋转矩阵是标准的 3x3 正交矩阵，表示末端执行器相对于基座坐标系的方向：
      其中每一列代表末端执行器坐标系的一个轴在基座坐标系中的方向。

   .. note::

      性能提示
      正运动学计算非常快速（通常 < 1 微秒）使用连续的 float64 数组以获得最佳性能对于批量计算，可以在循环中重用数组GIL 在 C++ 计算期间释放，支持多线程


   **See Also:**

   - :py:func:`compute_ik- 逆运动学计算`
   - :py:func:`get_solver_info- 获取求解器配置信息`
