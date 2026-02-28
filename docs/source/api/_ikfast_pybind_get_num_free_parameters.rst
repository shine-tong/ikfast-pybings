_ikfast_pybind.get_num_free_parameters
======================================

.. py:function:: _ikfast_pybind. get_num_free_parameters ( ) → int

   获取自由参数数量。

   :returns: 自由参数的数量，如果无冗余则为 0
   :rtype: int

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 使用底层 API
      num_free = ik._ikfast_pybind.get_num_free_parameters()
      print(f"自由参数数量: {num_free}")
      
      # 推荐：使用高级 API
      info = ik.get_solver_info()
      print(f"自由参数数量: {info['num_free_parameters']}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口。推荐使用get_solver_info()获取所有求解器信息。


   **See Also:**

   - :py:func:`get_solver_info- 推荐的高级 API`
