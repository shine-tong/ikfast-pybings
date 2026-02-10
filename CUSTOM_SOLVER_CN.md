# 使用自定义 IKFast 求解器

中文文档 | [English](CUSTOM_SOLVER.md)

本指南说明如何在不同机器人模型和自定义 IKFast 求解器文件中使用 IKFast Python 绑定。

## 概述

IKFast Python 绑定设计为可与任何 IKFast 生成的求解器配合使用。当您更改机器人模型时，需要：

1. 为您的机器人生成新的 IKFast 求解器 `.cpp` 文件
2. 替换或添加求解器文件到 `src/` 目录
3. 重新构建 Python 绑定

构建系统会自动检测并使用 `src/` 目录中的 IKFast 求解器文件。

## 快速开始

### 步骤 1：使用 ikfast-online 生成 IKFast 求解器

[ikfast-online](https://github.com/shine-tong/ikfast-online) 是一个基于 Web 的工具，可以简化 IKFast 求解器的生成。它提供了便捷的 Web 界面，可以直接从 URDF 文件生成求解器文件。

**详细使用方法请参阅 [ikfast-online README](https://github.com/shine-tong/ikfast-online)。**

> **⚠️ 重要**：求解器文件名必须遵循 `*_ikfast_solver.cpp` 模式。

### 步骤 2：替换求解器文件

替换 `src/` 目录中的现有求解器：

```bash
# 删除旧求解器（如果要保留多个求解器则可选）
rm src/*_ikfast_solver.cpp

# 复制新求解器
cp your_robot_ikfast_solver.cpp src/
```

### 步骤 3：重新构建

重新构建 Python 绑定：

```bash
# 清理之前的构建
pip uninstall ikfast-pybind -y
rm -rf build/ dist/ *.egg-info/

# 重新构建并安装
pip install .
```

### 步骤 4：验证

测试新求解器：

```python
import ikfast_pybind as ik

# 检查求解器信息
info = ik.get_solver_info()
print(f"关节数量：{info['num_joints']}")
print(f"运动学哈希：{info['kinematics_hash']}")

# 测试正逆运动学
import numpy as np
joints = np.zeros(info['num_joints'])
translation, rotation = ik.compute_fk(joints)
print(f"末端执行器位置：{translation}")
```

## 求解器文件要求

您的 IKFast 求解器文件必须：

1. **遵循命名约定**：`*_ikfast_solver.cpp`
2. **包含必需的函数**：`ComputeIk()`、`ComputeFk()`、`GetNumJoints()` 等
3. **与 ikfast.h 兼容**：使用与 `include/ikfast.h` 相同的 IKFast 版本

## 故障排除

### 问题："未找到 IKFast 求解器文件"

**错误：**
```
FileNotFoundError: No IKFast solver file found matching pattern: src/*_ikfast_solver.cpp
```

**解决方案：**
1. 确保求解器文件在 `src/` 目录中
2. 检查文件名是否以 `_ikfast_solver.cpp` 结尾
3. 验证文件存在：`ls src/*_ikfast_solver.cpp`

### 问题："找到多个 IKFast 求解器文件"

**警告：**
```
Warning: Multiple IKFast solver files found: ['src/robot_a_ikfast_solver.cpp', 'src/robot_b_ikfast_solver.cpp']
Using: src/robot_a_ikfast_solver.cpp
```

**解决方案：**
指定要使用的求解器：
```bash
export IKFAST_SOLVER_FILE=src/robot_b_ikfast_solver.cpp
pip install . --force-reinstall
```

或删除未使用的求解器：
```bash
rm src/robot_a_ikfast_solver.cpp
```

### 问题：新求解器编译错误

**错误：**
```
error: 'ComputeIk' was not declared in this scope
```

**解决方案：**
1. 验证求解器文件是有效的 IKFast 输出
2. 检查是否包含所有必需的函数
3. 确保 IKFast 版本兼容性：
   ```bash
   grep "IKFAST_VERSION" src/your_solver.cpp
   grep "IKFAST_VERSION" include/ikfast.h
   ```

### 问题：关节数量错误

**错误：**
```python
>>> info = ik.get_solver_info()
>>> print(info['num_joints'])
6  # 您的机器人应该是 7
```

**解决方案：**
1. 验证您使用的是正确的求解器文件
2. 使用正确的机器人模型重新生成 IKFast 求解器
3. 重新构建绑定

### 问题：逆运动学解不符合预期结果

**可能原因：**
1. 错误的求解器文件（用于不同的机器人）
2. 不同的坐标系约定
3. 关节角度单位（弧度 vs 度）

**解决方案：**
1. 验证求解器运动学哈希与您的机器人匹配：
   ```python
   info = ik.get_solver_info()
   print(info['kinematics_hash'])
   ```
2. 使用已知的关节配置进行测试
3. 与 OpenRAVE 结果比较

## 示例：使用 ikfast-online 处理 6 自由度机器人

```bash
# 1. 按照以下链接的 ikfast-online 设置说明操作：
https://github.com/shine-tong/ikfast-online

# 2. 复制求解器到 ikfast_pybind 项目
cp ur5_ikfast_solver.cpp /path/to/ikfast_pybind/src/

# 3. 重新构建
cd /path/to/ikfast_pybind
pip install . --force-reinstall

# 4. 测试
python -c "
import ikfast_pybind as ik
import numpy as np

info = ik.get_solver_info()
print(f'关节数：{info[\"num_joints\"]}')
print(f'哈希：{info[\"kinematics_hash\"]}')

# 测试正运动学
joints = np.zeros(info['num_joints'])
trans, rot = ik.compute_fk(joints)
print(f'零配置的正运动学：{trans}')
"
```

## 资源

- **ikfast-online**：https://github.com/shine-tong/ikfast-online - 基于 Web 的 IKFast 求解器生成工具
- **IKFast 文档**：http://openrave.org/docs/latest_stable/openravepy/ikfast/

## 支持

如果您在使用自定义求解器时遇到问题：

1. 使用 ikfast-online 验证您的 IKFast 求解器是否正确生成
2. 查看上面的故障排除部分
3. 查阅 ikfast-online README 了解求解器生成问题
