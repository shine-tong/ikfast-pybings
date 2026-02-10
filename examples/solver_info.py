#!/usr/bin/env python3
"""
示例：获取IKFast求解器信息

演示如何获取和显示IKFast求解器的详细信息，包括IK类型的人类可读描述。
"""

import ikfast_pybind as ik


def main():
    """获取并显示求解器信息"""
    print("=" * 60)
    print("IKFast 求解器信息")
    print("=" * 60)
    
    # 获取求解器信息
    info = ik.get_solver_info()
    
    # 显示基本信息
    print(f"\n关节数量: {info['num_joints']}")
    print(f"自由参数数量: {info['num_free_parameters']}")
    
    if info['free_parameters']:
        print(f"自由参数索引: {info['free_parameters']}")
    else:
        print("自由参数索引: 无（非冗余机器人）")
    
    # 显示IK类型信息
    print(f"\nIK 求解器类型:")
    print(f"  类型代码: 0x{info['ik_type']:08x}")
    print(f"  类型名称: {info['ik_type_name']}")
    
    # 显示版本信息
    print(f"\nIKFast 版本: {info['ikfast_version']}")
    print(f"运动学哈希: {info['kinematics_hash']}")
    
    # 解释IK类型
    print("\n" + "=" * 60)
    print("IK 类型说明")
    print("=" * 60)
    
    ik_type = info['ik_type']
    
    if ik_type == 0x67000001:
        print("""
Transform6D (6D 变换)
- 这是最常见的IK类型
- 求解完整的6D位姿（3D位置 + 3D姿态）
- 输入：末端执行器的位置向量和旋转矩阵
- 适用于：标准6自由度机械臂
        """)
    elif ik_type == 0x34000002:
        print("""
Translation3D (3D 平移)
- 仅求解位置，不考虑姿态
- 输入：末端执行器的目标位置
- 适用于：只关心位置的应用（如点焊）
        """)
    elif ik_type == 0x34000003:
        print("""
Direction3D (3D 方向)
- 求解方向向量
- 输入：目标方向向量
- 适用于：需要指向特定方向的应用
        """)
    elif ik_type == 0x34000004:
        print("""
Ray4D (4D 射线)
- 求解射线（原点+方向）
- 输入：射线的起点和方向
- 适用于：激光切割、喷涂等应用
        """)
    else:
        print(f"""
其他IK类型 (0x{ik_type:08x})
- 请参考IKFast文档了解详细信息
        """)
    
    print("=" * 60)


def demonstrate_ik_type_function():
    """演示get_ik_type_name函数的使用"""
    print("\n" + "=" * 60)
    print("所有支持的IK类型")
    print("=" * 60)
    
    # 所有已知的IK类型
    known_types = [
        0x67000001,  # Transform6D
        0x34000002,  # Translation3D
        0x34000003,  # Direction3D
        0x34000004,  # Ray4D
        0x34000005,  # Lookat3D
        0x34000006,  # TranslationDirection5D
        0x34000007,  # TranslationXY2D
        0x34000008,  # TranslationXYOrientation3D
        0x34000009,  # TranslationLocalGlobal6D
        0x3400000a,  # TranslationXAxisAngle4D
        0x3400000b,  # TranslationYAxisAngle4D
        0x3400000c,  # TranslationZAxisAngle4D
        0x3400000d,  # TranslationXAxisAngleZNorm4D
        0x3400000e,  # TranslationYAxisAngleXNorm4D
        0x3400000f,  # TranslationZAxisAngleYNorm4D
    ]
    
    for ik_type in known_types:
        name = ik.get_ik_type_name(ik_type)
        print(f"0x{ik_type:08x}: {name}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
    demonstrate_ik_type_function()
