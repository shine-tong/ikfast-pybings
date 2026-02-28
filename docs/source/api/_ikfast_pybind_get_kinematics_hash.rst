_ikfast_pybind.get_kinematics_hash
==================================

.. py:function:: _ikfast_pybind. get_kinematics_hash ( ) → str

   获取运动学配置哈希。

   :returns: 标识机器人运动学配置的唯一哈希字符串
   :rtype: str

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 使用底层 API
      hash_str = ik._ikfast_pybind.get_kinematics_hash()
      print(f"运动学哈希: {hash_str}")
      
      # 推荐：使用高级 API
      info = ik.get_solver_info()
      print(f"运动学哈希: {info['kinematics_hash']}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口。推荐使用get_solver_info()获取所有求解器信息。


   **See Also:**

   - :py:func:`get_solver_info- 推荐的高级 API`
