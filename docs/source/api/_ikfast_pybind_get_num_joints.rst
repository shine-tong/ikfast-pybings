_ikfast_pybind.get_num_joints
=============================

.. py:function:: _ikfast_pybind. get_num_joints ( ) → int

   获取机器人关节数量。

   :returns: 机器人的关节数量
   :rtype: int

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 使用底层 API
      num_joints = ik._ikfast_pybind.get_num_joints()
      print(f"关节数量: {num_joints}")
      
      # 推荐：使用高级 API
      info = ik.get_solver_info()
      print(f"关节数量: {info['num_joints']}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口。推荐使用get_solver_info()获取所有求解器信息。


   **See Also:**

   - :py:func:`get_solver_info- 推荐的高级 API`
