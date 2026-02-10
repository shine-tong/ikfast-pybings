# IKFast Python 绑定

中文文档 | [English](README.md)

使用 pybind11 为 IKFast 逆运动学求解器提供的高性能 Python 绑定。本包为 6 自由度机械臂的解析式逆运动学解提供了简洁的 Python 接口，并与 NumPy 无缝集成。

## 特性

- **⚡ 快速解析解**：利用 IKFast 的解析式逆运动学实现实时性能
- **🔢 NumPy 集成**：NumPy 数组与 C++ 数据结构之间的无缝转换，尽可能实现零拷贝
- **🎯 多解支持**：访问给定位姿的所有有效逆运动学解
- **🐍 Python 风格 API**：遵循 Python 约定的简洁直观接口
- **🔒 类型安全**：完整的类型提示和全面的输入验证
- **⚠️ 错误处理**：描述性错误消息和适当的异常类型
- **📊 基于属性的测试**：通过 146 个测试验证，包括基于属性的测试（每个测试 100+ 次迭代）
- **🌐 跨平台**：支持 Windows、Linux 和 macOS

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [使用自定义求解器](#使用自定义求解器)
- [API 参考](#api-参考)
- [示例](#示例)
- [测试](#测试)
- [故障排除](#故障排除)
- [性能](#性能)
- [贡献](#贡献)
- [许可证](#许可证)

## 安装

### 前置要求

安装前，请确保您有：
- **Python**：3.8 或更高版本
- **C++ 编译器**：
  - Windows：MSVC 14.0+（Visual Studio 2015 或更高版本）
  - Linux：GCC 7.0+ 或 Clang 5.0+
  - macOS：Xcode 命令行工具
- **NumPy**：1.20.0 或更高版本
- **pybind11**：2.6.0 或更高版本

详细的构建说明和故障排除请参见 [BUILD_CN.md](BUILD_CN.md)。

### 从源码安装

```bash
# 克隆仓库
git clone <repository-url>
cd ikfast_pybind

# 安装构建依赖
pip install pybind11 numpy

# 构建并安装
pip install .
```

### 开发安装

用于开发的可编辑安装和测试工具：

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
```

### 验证安装

```python
import ikfast_pybind as ik
print(f"IKFast Python Bindings v{ik.__version__}")
print(f"求解器有 {ik.get_solver_info()['num_joints']} 个关节")
```

## 使用自定义求解器

**该绑定可与任何 IKFast 生成的求解器配合使用！** 当您更改机器人模型时，只需：

1. 使用 [ikfast-online](https://github.com/shine-tong/ikfast-online) 为您的机器人生成 IKFast 求解器 `.cpp` 文件
2. 替换 `src/` 目录中的求解器文件（必须以 `_ikfast_solver.cpp` 结尾）
3. 重新构建：`pip install . --force-reinstall`

构建系统会自动检测并使用您的求解器文件。

### 快速示例

```bash
# 1. 使用 ikfast-online 生成 IKFast solver
详细步骤查看仓库(https://github.com/shine-tong/ikfast-online)

# 2. 复制生成的求解器到项目
cd /path/to/ikfast_pybind
cp /path/to/your_robot_ikfast_solver.cpp src/   # cpp 所在目录
rm src/sa0521_manipulator_ikfast_solver.cpp     # 删除旧求解器

# 3. 重新构建
pip install . --force-reinstall
```

**有关使用自定义求解器的详细说明，请参见 [CUSTOM_SOLVER_CN.md](CUSTOM_SOLVER_CN.md)**（或 [English](CUSTOM_SOLVER.md)）。

## 快速开始

```python
import ikfast_pybind as ik
import numpy as np

# 获取求解器信息
info = ik.get_solver_info()
print(f"机器人有 {info['num_joints']} 个关节")

# 计算正运动学
joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
translation, rotation = ik.compute_fk(joints)
print(f"末端执行器位置：{translation}")

# 计算逆运动学
target_translation = np.array([0.5, 0.0, 0.5])
target_rotation = np.eye(3)
solutions = ik.compute_ik(target_translation, target_rotation)

print(f"找到 {len(solutions)} 个逆运动学解")
for i, solution in enumerate(solutions):
    print(f"解 {i+1}：{solution}")
```

## API 参考

### 高级函数

#### `compute_ik(translation, rotation, free_params=None)`

计算目标末端执行器位姿的逆运动学解。

**参数：**
- `translation` (np.ndarray)：末端执行器位置 [x, y, z]，形状 (3,)
- `rotation` (np.ndarray)：末端执行器方向，旋转矩阵，形状 (3, 3) 或展平的 (9,)
- `free_params` (np.ndarray, 可选)：冗余关节的自由参数值

**返回：**
- `List[np.ndarray]`：关节角度解的列表，每个形状为 (num_joints,)。如果无解则返回空列表。

**异常：**
- `ValueError`：输入数组形状或值无效
- `TypeError`：输入无法转换为 numpy 数组
- `RuntimeError`：求解器数值问题

**示例：**
```python
translation = np.array([0.5, 0.0, 0.5])
rotation = np.eye(3)
solutions = ik.compute_ik(translation, rotation)
```

#### `compute_fk(joint_angles)`

计算给定关节配置的正运动学。

**参数：**
- `joint_angles` (np.ndarray)：关节角度，形状 (num_joints,)

**返回：**
- `Tuple[np.ndarray, np.ndarray]`：(translation, rotation_matrix)
  - `translation`：形状 (3,)
  - `rotation_matrix`：形状 (3, 3)

**异常：**
- `ValueError`：joint_angles 形状无效
- `TypeError`：输入无法转换为 numpy 数组

**示例：**
```python
joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
translation, rotation = ik.compute_fk(joints)
```

#### `get_solver_info()`

获取逆运动学求解器配置信息。

**返回：**
- `Dict[str, Any]`：包含以下键的字典：
  - `num_joints` (int)：关节数量
  - `num_free_parameters` (int)：自由参数数量
  - `free_parameters` (List[int])：自由参数索引
  - `ik_type` (int)：求解器类型标识符
  - `kinematics_hash` (str)：运动学配置哈希
  - `ikfast_version` (str)：IKFast 版本

**示例：**
```python
info = ik.get_solver_info()
print(f"求解器类型：{hex(info['ik_type'])}")
```

### 低级类

对于高级用法，还提供以下类：

- `IkSolution`：支持自由参数的单个逆运动学解
- `IkSolutionList`：支持 Python 迭代的多个逆运动学解容器

详细用法模式请参见 [examples](examples/) 目录。

## 示例

`examples/` 目录包含演示各种用例的综合示例脚本：

### 基础示例

#### 1. **basic_ik.py** - 计算逆运动学解
演示：
- 计算目标位姿的逆运动学
- 遍历多个解
- 选择最接近当前配置的解
- 选择远离关节限位的解
- 用正运动学验证解

```bash
python examples/basic_ik.py
```

#### 2. **basic_fk.py** - 正运动学
演示：
- 计算关节配置的正运动学
- 验证旋转矩阵
- 旋转表示之间的转换
- 用正运动学往返验证逆运动学解

```bash
python examples/basic_fk.py
```

#### 3. **solution_selection.py** - 高级选择
演示：
- 多种选择标准（距离、能量、可操作性）
- 处理冗余机器人的自由参数
- 工作空间边界检测
- 平滑关节运动的轨迹规划

```bash
python examples/solution_selection.py
```

### 代码片段

**选择最接近当前位姿的解：**
```python
import numpy as np
import ikfast_pybind as ik

def select_closest_solution(solutions, current_joints):
    """选择最接近当前关节配置的逆运动学解。"""
    if not solutions:
        return None
    
    distances = [np.linalg.norm(sol - current_joints) for sol in solutions]
    return solutions[np.argmin(distances)]

# 使用
current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
solutions = ik.compute_ik(translation, rotation)
best = select_closest_solution(solutions, current)
```

**用正运动学验证解：**
```python
def verify_ik_solution(solution, target_trans, target_rot, tol=1e-6):
    """验证逆运动学解是否产生目标位姿。"""
    computed_trans, computed_rot = ik.compute_fk(solution)
    
    trans_error = np.linalg.norm(target_trans - computed_trans)
    rot_error = np.linalg.norm(target_rot - computed_rot)
    
    return trans_error < tol and rot_error < tol
```

**处理不可达位姿：**
```python
def safe_compute_ik(translation, rotation):
    """计算逆运动学并优雅地处理不可达位姿。"""
    try:
        solutions = ik.compute_ik(translation, rotation)
        if not solutions:
            print("警告：位姿在机器人工作空间之外")
            return None
        return solutions
    except ValueError as e:
        print(f"输入无效：{e}")
        return None
    except RuntimeError as e:
        print(f"求解器错误：{e}")
        return None
```

## 测试

项目包含全面的测试覆盖，包括单元测试和基于属性的测试。

### 测试统计

- **总测试数**：146
- **测试覆盖率**：95%
- **基于属性的测试**：71 个测试，每个测试 100+ 次迭代
- **单元测试**：75 个测试，覆盖特定示例和边界情况

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 详细输出运行
pytest tests/ -v

# 仅运行单元测试
pytest tests/test_*.py -k "not property"

# 仅运行属性测试
pytest tests/test_property_*.py

# 运行并生成覆盖率报告
pytest tests/ --cov=ikfast_pybind --cov-report=html

# 运行特定测试文件
pytest tests/test_compute_ik.py
```

### 测试类别

1. **构建测试** (`test_build.py`)
   - 项目结构验证
   - 构建配置验证

2. **单元测试** (`test_*.py`)
   - 特定输入/输出示例
   - 边界情况和边界条件
   - 错误处理验证

3. **基于属性的测试** (`test_property_*.py`)
   - IK-FK 往返一致性
   - FK-IK 往返一致性
   - 数组类型转换正确性
   - 输入验证
   - 异常转换
   - 解完整性
   - 自由参数处理
   - 索引边界检查

### 持续集成

测试在以下环境中验证：
- Python 版本：3.8、3.9、3.10、3.11、3.12
- 操作系统：Windows、Linux、macOS
- NumPy 版本：1.20.0+

## 故障排除

### 构建问题

**问题**：`error: Microsoft Visual C++ 14.0 or greater is required`

**解决方案**：安装 Visual Studio Build Tools 或带有 C++ 支持的 Visual Studio。详见 [BUILD_CN.md](BUILD_CN.md)。

---

**问题**：`fatal error: pybind11/pybind11.h: No such file or directory`

**解决方案**：构建前安装 pybind11：
```bash
pip install pybind11
```

---

**问题**：`ImportError: DLL load failed while importing _ikfast_pybind`

**解决方案**：确保已安装 C++ 运行时库。在 Windows 上，安装 Visual C++ Redistributable。

### 运行时问题

**问题**：`ValueError: compute_ik: Invalid translation shape`

**解决方案**：确保 translation 是包含 3 个元素的一维数组：
```python
translation = np.array([x, y, z])  # 正确
# 不是：translation = [[x, y, z]]  # 错误 - 二维数组
```

---

**问题**：返回空解列表

**解决方案**：目标位姿可能在机器人工作空间之外。验证位姿是否可达：
```python
solutions = ik.compute_ik(translation, rotation)
if not solutions:
    print("位姿不可达")
```

---

**问题**：`RuntimeError: IKFast solver error`

**解决方案**：旋转矩阵可能无效。确保它是正交归一化的：
```python
# 检查旋转是否有效
det = np.linalg.det(rotation)
assert np.isclose(det, 1.0), "旋转矩阵的行列式必须为 1"
```

### 性能问题

**问题**：重复调用逆运动学时性能慢

**解决方案**：确保数组是连续的并使用适当的数据类型：
```python
# 好 - 连续的 float64
translation = np.ascontiguousarray(translation, dtype=np.float64)

# 避免在循环中创建新数组
for pose in poses:
    solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))
```

### 获取帮助

如果遇到此处未涵盖的问题：

1. 查看 [BUILD_CN.md](BUILD_CN.md) 了解详细的构建说明
2. 查看 [examples](examples/) 了解使用模式
3. 确保您的输入符合预期的形状和类型
4. 验证您的 C++ 编译器和 Python 环境配置正确
5. 检查 NumPy 和 pybind11 是否正确安装

## 性能

### 基准测试

Python 绑定相比直接 C++ 调用增加的开销很小：

- **逆运动学计算**：< 5% 开销
- **正运动学计算**：< 3% 开销
- **数组转换**：尽可能零拷贝
- **GIL 释放**：在 C++ 计算期间启用，支持多线程

### 优化技巧

1. **使用连续数组：**
```python
# 好 - 连续数组
translation = np.ascontiguousarray(translation, dtype=np.float64)

# 避免 - 非连续切片可能需要复制
translation = some_array[::2, :]  # 可能不连续
```

2. **尽可能重用数组：**
```python
# 好 - 重用数组
joints = np.zeros(6, dtype=np.float64)
for i, config in enumerate(configs):
    joints[:] = config
    trans, rot = ik.compute_fk(joints)

# 避免 - 在循环中创建新数组
for config in configs:
    trans, rot = ik.compute_fk(np.array(config))
```

3. **批处理：**
```python
# 高效处理多个位姿
results = []
for pose in poses:
    solutions = ik.compute_ik(pose[:3], pose[3:].reshape(3, 3))
    if solutions:
        results.append(solutions[0])
```

### 内存管理

- **自动**：pybind11 自动处理引用计数
- **无内存泄漏**：RAII 确保正确清理
- **高效**：数组操作的最小分配

## 贡献

欢迎贡献！请遵循以下指南：

1. **代码风格**：Python 代码遵循 PEP 8
2. **测试**：为新功能添加测试
3. **文档**：更新文档字符串和 README
4. **类型提示**：为公共 API 包含类型注释

### 开发设置

```bash
# 克隆仓库
git clone <repository-url>
cd ikfast_pybind

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows 上：venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 运行测试并生成覆盖率报告
pytest tests/ --cov=ikfast_pybind --cov-report=html
```

### 运行基于属性的测试

基于属性的测试使用 Hypothesis 进行随机测试：

```bash
# 使用默认迭代次数运行（100）
pytest tests/test_property_*.py

# 使用更多迭代次数进行彻底测试
pytest tests/test_property_*.py --hypothesis-iterations=1000

# 使用特定种子以实现可重现性
pytest tests/test_property_*.py --hypothesis-seed=12345
```

## 文件结构

```
ikfast_pybind/
├── ikfast_pybind/
│   ├── __init__.py              # 高级 Python API
│   └── _ikfast_pybind.cpp       # pybind11 绑定代码
├── src/
│   └── sa0521_manipulator_ikfast_solver.cpp  # IKFast 求解器
├── include/
│   └── ikfast.h                 # IKFast 头文件
├── examples/
│   ├── basic_ik.py              # 基础逆运动学示例
│   ├── basic_fk.py              # 基础正运动学示例
│   └── solution_selection.py   # 高级选择
├── tests/
│   ├── test_*.py                # 单元测试
│   └── test_property_*.py       # 基于属性的测试
├── setup.py                     # 构建配置
├── pyproject.toml               # 包元数据
├── MANIFEST.in                  # 包数据文件
├── README_CN.md                 # 本文件
└── BUILD_CN.md                  # 构建说明
```

## 系统要求

- **Python**：3.8、3.9、3.10、3.11 或 3.12
- **NumPy**：1.20.0 或更高版本
- **pybind11**：2.6.0 或更高版本
- **C++ 编译器**：支持 C++11
  - Windows：MSVC 14.0+（Visual Studio 2015+）
  - Linux：GCC 7.0+ 或 Clang 5.0+
  - macOS：Xcode 命令行工具

### 可选依赖

- **pytest**：6.0+（用于运行测试）
- **hypothesis**：6.0+（用于基于属性的测试）
- **pytest-cov**：5.0+（用于覆盖率报告）

## 许可证

根据 Apache License 2.0 许可。详见 [LICENSE](LICENSE) 文件。

## 致谢

- **IKFast**：OpenRAVE 项目的一部分
- **pybind11**：C++11 和 Python 之间的无缝互操作性
- **NumPy**：科学计算的基础包

## 引用

如果您在研究中使用此软件，请引用：

```bibtex
@software{ikfast_pybind,
  title = {IKFast Python Bindings},
  author = {IKFast Python Bindings Contributors},
  year = {2026},
  url = {<repository-url>}
}
```
