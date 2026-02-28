ikfast_pybind.compute_ik
========================

.. py:function:: ikfast_pybind. compute_ik ( translation : np.ndarray , rotation : np.ndarray , free_params : Optional [ np.ndarray ] = None ) → List [ np.ndarray ]

   计算目标末端执行器位姿的逆运动学解。
   
   给定期望的末端执行器位姿（平移和旋转），此函数计算所有可能达到该位姿的关节配置。

   :param translation: 末端执行器位置 [x, y, z]，形状为 (3,) 的 numpy 数组
   :type translation: np.ndarray
   :param rotation: 末端执行器旋转矩阵，形状为 (3, 3) 或展平的 (9,) 的 numpy 数组。旋转矩阵应该是正交归一化的
   :type rotation: np.ndarray
   :param free_params: 可选的自由参数值 numpy 数组。如果机器人有冗余自由度则需要。默认为 None
   :type free_params: Optional[np.ndarray]

   :returns: 关节角度数组的列表，每个数组形状为 (num_joints,)，数据类型为 float64。如果无解则返回空列表（位姿不可达）
   :rtype: List[np.ndarray]

   :raises ValueError: 如果输入数组形状不正确或值无效
   :raises TypeError: 如果输入不是类数组或无法转换为 numpy 数组
   :raises RuntimeError: 如果求解器遇到数值问题

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      import numpy as np
      
      # 基本用法
      translation = np.array([0.5, 0.0, 0.5])
      rotation = np.eye(3)
      solutions = ik.compute_ik(translation, rotation)
      
      print(f"找到 {len(solutions)} 个解")
      if solutions:
          print(f"第一个解：{solutions[0]}")
      
      # 使用展平的旋转矩阵
      rotation_flat = np.array([1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0])
      solutions = ik.compute_ik(translation, rotation_flat)
      
      # 处理不可达位姿
      unreachable_pos = np.array([10.0, 10.0, 10.0])
      solutions = ik.compute_ik(unreachable_pos, rotation)
      if not solutions:
          print("位姿不可达")
      
      # 选择最接近当前配置的解
      current_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      solutions = ik.compute_ik(translation, rotation)
      if solutions:
          distances = [np.linalg.norm(sol - current_joints) for sol in solutions]
          best_solution = solutions[np.argmin(distances)]
          print(f"最佳解：{best_solution}")
      


   .. note::

      算法说明
      该函数使用 IKFast 生成的解析式逆运动学求解器。对于 6 自由度机械臂，通常会返回多个解（最多 8 个），每个解对应不同的关节配置。

   .. note::

      输入验证
      translation 必须是形状为 (3,) 的一维数组rotation 可以是形状为 (3, 3) 的矩阵或形状为 (9,) 的展平数组rotation 矩阵应该是正交归一化的（行列式为 1）所有输入会自动转换为连续的 float64 数组以获得最佳性能

   .. note::

      性能提示
      使用连续的 float64 数组以获得最佳性能避免在循环中创建新数组，尽可能重用数组对于批量计算，考虑使用多线程（GIL 在 C++ 计算期间释放）


   **See Also:**

   - :py:func:`compute_fk- 正运动学计算`
   - :py:func:`IkSolution- 低级解对象`
