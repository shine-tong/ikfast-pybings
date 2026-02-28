_ikfast_pybind.get_ikfast_version
=================================

.. py:function:: _ikfast_pybind. get_ikfast_version ( ) → str

   获取 IKFast 版本。

   :returns: 用于生成求解器的 IKFast 版本字符串
   :rtype: str

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 使用底层 API
      version = ik._ikfast_pybind.get_ikfast_version()
      print(f"IKFast 版本: {version}")
      
      # 推荐：使用高级 API
      info = ik.get_solver_info()
      print(f"IKFast 版本: {info['ikfast_version']}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口。推荐使用get_solver_info()获取所有求解器信息。


   **See Also:**

   - :py:func:`get_solver_info- 推荐的高级 API`
