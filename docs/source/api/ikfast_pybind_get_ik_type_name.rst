ikfast_pybind.get_ik_type_name
==============================

.. py:function:: ikfast_pybind. get_ik_type_name ( ik_type : int ) → str

   获取 IK 求解器类型的人类可读名称。
   
   将 IK 类型常量转换为描述性的中文字符串，解释求解器处理的逆运动学问题类型。

   :returns: IK 类型的人类可读描述（中文）
   :rtype: str

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 方法1：从 get_solver_info 获取
      info = ik.get_solver_info()
      print(f"IK 类型: {info['ik_type_name']}")
      print(f"类型代码: {hex(info['ik_type'])}")
      
      # 方法2：直接查询特定类型
      type_name = ik.get_ik_type_name(0x67000001)
      print(type_name)
      # 输出: Transform6D (默认) - 完整的位置和姿态
      
      # 方法3：列出所有支持的类型
      known_types = [
          0x67000001,  # Transform6D
          0x34000002,  # Translation3D
          0x34000003,  # Direction3D
          0x34000004,  # Ray4D
      ]
      
      for ik_type in known_types:
          name = ik.get_ik_type_name(ik_type)
          print(f"{hex(ik_type)}: {name}")
      


   .. note::

      使用场景
      在用户界面中显示求解器类型生成求解器配置报告调试和日志记录验证求解器类型是否符合预期

   .. note::

      如果传入未知的 IK 类型代码，函数会返回 "Unknown IK Type (0xXXXXXXXX)" 格式的字符串最常见的类型是0x67000001(Transform6D)，适用于标准 6 自由度机械臂不同的 IK 类型需要不同的输入参数格式，请参考compute_ik文档


   **See Also:**

   - :py:func:`get_solver_info- 获取完整的求解器信息（包含 ik_type_name）`
   - :py:func:`get_ik_type- 获取原始 IK 类型常量`
