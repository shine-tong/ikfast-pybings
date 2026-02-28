ikfast_pybind.get_solver_info
=============================

.. py:function:: ikfast_pybind. get_solver_info ( ) → Dict [ str , Any ]

   获取求解器信息和属性。
   
   返回包含 IK 求解器配置信息的字典，包括关节数量、自由参数、求解器类型和版本。

   :returns: 包含以下键的字典：num_joints(int) - 机器人的关节数量num_free_parameters(int) - 自由参数数量（如果无冗余则为 0）free_parameters(List[int]) - 自由参数关节的索引ik_type(int) - IK 求解器类型标识符常量ik_type_name(str) - IK 求解器类型的人类可读名称（中文描述）kinematics_hash(str) - 标识机器人运动学配置的哈希值ikfast_version(str) - 用于生成求解器的 IKFast 版本
   :rtype: Dict[str, Any]

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      
      # 获取求解器信息
      info = ik.get_solver_info()
      
      # 打印所有信息
      print("求解器信息：")
      for key, value in info.items():
          print(f"  {key}: {value}")
      
      # 访问特定信息
      print(f"\n机器人有 {info['num_joints']} 个关节")
      print(f"求解器类型：{info['ik_type_name']} ({hex(info['ik_type'])})")
      print(f"运动学哈希：{info['kinematics_hash']}")
      print(f"IKFast 版本：{info['ikfast_version']}")
      
      # 检查是否有冗余自由度
      if info['num_free_parameters'] > 0:
          print(f"\n机器人有 {info['num_free_parameters']} 个自由参数")
          print(f"自由参数索引：{info['free_parameters']}")
      else:
          print("\n机器人没有冗余自由度")
      


   .. note::

      IK 类型说明
      ik_type 字段是一个整数常量，标识求解器的类型。常见的类型包括：
      0x67000001 - Transform6D（6D 位姿）0x34000002 - Translation3D（3D 位置）0x34000003 - Direction3D（3D 方向）0x34000004 - Ray4D（4D 射线）

   .. note::

      运动学哈希
      kinematics_hash 是一个唯一标识符，用于验证求解器是否与特定的机器人模型匹配。如果更换了机器人模型或 DH 参数，此哈希值会改变。


   **See Also:**

   - :py:func:`compute_ik- 逆运动学计算`
   - :py:func:`compute_fk- 正运动学计算`
