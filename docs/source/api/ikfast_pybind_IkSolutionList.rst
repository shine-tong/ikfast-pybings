ikfast_pybind.IkSolutionList
============================

.. py:class:: IkSolutionList

   多个逆运动学解的容器，支持 Python 迭代。
   
   此类包含 IKFast 求解器返回的所有逆运动学解。它支持 Python 的标准容器操作，如索引、迭代和长度查询。

   **Methods:**

   .. py:method:: __len__ ( ) → int ¶

      返回解的数量。

      :returns: 解的数量
      :rtype: int

      **Example:**
   
      .. code-block:: python
   
         num_solutions = len(solution_list)
         print(f"找到 {num_solutions} 个解")
         
   


   .. py:method:: __getitem__ ( index : int ) → IkSolution ¶

      通过索引访问解。

      :returns: 指定索引的解对象
      :rtype: IkSolution

      **Example:**
   
      .. code-block:: python
   
         # 访问第一个解
         first_solution = solution_list[0]
         
         # 访问最后一个解
         last_solution = solution_list[len(solution_list) - 1]
         
   


   **See Also:**

   - :py:func:`IkSolution- 单个解对象`
   - :py:func:`compute_ik- 推荐的高级 API`
