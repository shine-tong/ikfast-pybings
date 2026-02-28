_ikfast_pybind.compute_fk_raw
=============================

.. py:function:: _ikfast_pybind. compute_fk_raw ( joint_angles : np.ndarray ) → Tuple [ np.ndarray , np.ndarray ]

   底层正运动学计算函数（C++ 绑定）。
   
   此函数直接调用 IKFast C++ 求解器计算正运动学。与高级 APIcompute_fk()不同，此函数返回展平的旋转矩阵而不是 3x3 矩阵。

   :returns: translation(np.ndarray) – 末端执行器位置，形状为 (3,)rotation_flat(np.ndarray) – 末端执行器旋转矩阵（展平），形状为 (9,)
   :rtype: Tuple[np.ndarray, np.ndarray]

   **Example:**

   .. code-block:: python

      import ikfast_pybind as ik
      import numpy as np
      
      # 使用底层 API
      joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      translation, rotation_flat = ik._ikfast_pybind.compute_fk_raw(joints)
      
      print(f"位置: {translation}")
      print(f"旋转（展平）: {rotation_flat}")
      
      # 需要手动重塑为矩阵
      rotation_matrix = rotation_flat.reshape(3, 3)
      print(f"旋转矩阵:\n{rotation_matrix}")
      
      # 推荐：使用高级 API 更简单
      translation, rotation = ik.compute_fk(joints)  # 直接返回矩阵
      print(f"旋转矩阵:\n{rotation}")
      


   .. note::

      内部 API此函数是底层 C++ 绑定接口，供内部实现使用。推荐使用高级 APIcompute_fk()代替。


   **See Also:**

   - :py:func:`compute_fk- 推荐的高级 API`
