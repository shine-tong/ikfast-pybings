_ikfast_pybind.get_ik_type
==========================

.. py:function:: _ikfast_pybind. get_ik_type ( ) → int

   获取 IK 求解器类型标识符。

   :returns: IK 求解器类型常量（如 0x67000001 表示 Transform6D）
   :rtype: int

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 使用底层 API
      ik_type = ik._ikfast_pybind.get_ik_type()
      print(f"IK 类型: {hex(ik_type)}")
      
      # 推荐：使用高级 API
      info = ik.get_solver_info()
      print(f"IK 类型: {hex(info['ik_type'])}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口。推荐使用get_solver_info()获取所有求解器信息。


   **See Also:**

   - :py:func:`get_solver_info- 推荐的高级 API`
