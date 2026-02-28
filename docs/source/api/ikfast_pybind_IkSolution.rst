ikfast_pybind.IkSolution
========================

.. py:class:: IkSolution

   单个逆运动学解，支持自由参数。
   
   此类表示 IKFast 求解器返回的单个逆运动学解。对于冗余机器人，解可能包含自由参数，需要在获取具体关节角度时指定。
   
   注意大多数用户应该使用高级compute_ik()函数，它会自动处理解的转换。此类适用于需要直接访问 IKFast 求解器的高级用户。

   .. note::

      注意大多数用户应该使用高级compute_ik()函数，它会自动处理解的转换。此类适用于需要直接访问 IKFast 求解器的高级用户。

   .. note::

      使用场景
      此类主要用于以下高级场景：
      需要直接访问 IKFast 求解器的内部表示处理冗余机器人时需要精细控制自由参数实现自定义的解选择策略调试和分析求解器行为


   **Methods:**

   .. py:method:: get_solution ( free_params : Optional [ np.ndarray ] = None ) → np.ndarray

      获取具体的关节角度。

      :returns: 关节角度数组，形状为 (num_joints,)，数据类型为 float64
      :rtype: np.ndarray

      **Example:**
   
      .. code-block:: python
   
         import ikfast_pybind as ik
         import numpy as np
         
         # 使用低级 API
         translation = np.array([0.5, 0.0, 0.5])
         rotation = np.eye(3).flatten()
         
         # 获取解列表
         solution_list = ik._ikfast_pybind.compute_ik_raw(translation, rotation, None)
         
         # 遍历每个解
         for i in range(len(solution_list)):
             solution = solution_list[i]
             joints = solution.get_solution()
             print(f"解 {i+1}: {joints}")
         
   


   .. py:method:: get_free_dof_indices ( ) → List [ int ]

      获取自由参数的关节索引。

      :returns: 自由参数关节的索引列表
      :rtype: List[int]

      **Example:**
   
      .. code-block:: python
   
         # 检查解是否有自由参数
         free_indices = solution.get_free_dof_indices()
         if free_indices:
             print(f"自由参数关节索引: {free_indices}")
         else:
             print("此解没有自由参数")
         
   


   **See Also:**

   - :py:func:`compute_ik- 推荐的高级 API`
   - :py:func:`IkSolutionList- 解的容器类`
