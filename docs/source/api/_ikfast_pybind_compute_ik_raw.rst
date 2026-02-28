_ikfast_pybind.compute_ik_raw
=============================

.. py:function:: _ikfast_pybind. compute_ik_raw ( translation : np.ndarray , rotation : np.ndarray , free_params : Optional [ np.ndarray ] = None ) → IkSolutionList

   底层逆运动学计算函数（C++ 绑定）。
   
   此函数直接调用 IKFast C++ 求解器，返回原始的 IkSolutionList 对象。与高级 APIcompute_ik()不同，此函数不会自动将解转换为 numpy 数组列表。

   :param translation: 末端执行器位置 [x, y, z]，形状为 (3,)，必须是展平的一维数组
   :type translation: np.ndarray
   :param rotation: 末端执行器旋转矩阵，形状为 (9,)，必须是展平的一维数组（按行优先顺序）
   :type rotation: np.ndarray
   :param free_params: 可选的自由参数值。默认为 None
   :type free_params: Optional[np.ndarray]

   :returns: 包含所有逆运动学解的 IkSolutionList 对象
   :rtype: IkSolutionList

   :raises ValueError: 如果输入数组形状不正确
   :raises TypeError: 如果输入类型无效

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      import numpy as np
      
      # 使用底层 API
      translation = np.array([0.5, 0.0, 0.5])
      rotation = np.eye(3).flatten()  # 必须展平
      
      # 调用底层函数
      solution_list = ik._ikfast_pybind.compute_ik_raw(translation, rotation)
      
      # 手动处理解
      print(f"找到 {len(solution_list)} 个解")
      for i in range(len(solution_list)):
          solution = solution_list[i]
          joints = solution.get_solution()
          print(f"解 {i+1}: {joints}")
      
      # 推荐：使用高级 API 更简单
      solutions = ik.compute_ik(translation, np.eye(3))  # 可以直接传矩阵
      for i, sol in enumerate(solutions):
          print(f"解 {i+1}: {sol}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口，供内部实现使用。推荐使用高级 APIcompute_ik()代替。

   .. note::

      使用场景
      需要直接访问 IkSolutionList 对象的高级用户性能关键代码，需要避免额外的数组转换内部实现和测试代码


   **See Also:**

   - :py:func:`compute_ik- 推荐的高级 API`
   - :py:func:`IkSolutionList- 返回的解列表对象`
