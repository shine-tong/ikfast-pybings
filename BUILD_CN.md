# 构建说明

[English](BUILD.md) | 中文文档

本文档提供 IKFast Python 绑定的详细构建说明，包括前置要求、构建步骤、故障排除和跨平台支持。

## 目录

- [前置要求](#前置要求)
- [构建步骤](#构建步骤)
- [验证安装](#验证安装)
- [故障排除](#故障排除)
- [构建配置](#构建配置)
- [跨平台支持](#跨平台支持)
- [高级构建选项](#高级构建选项)

## 前置要求

### 必需软件

#### 1. Python 3.8 或更高版本

验证 Python 安装：
```bash
python --version
```

如果未安装，请从以下位置下载：
- **Windows**: https://www.python.org/downloads/
- **Linux**: 使用包管理器（如 `apt`、`yum`）
- **macOS**: 使用 Homebrew 或从 python.org 下载

#### 2. C++ 编译器

根据您的操作系统选择：

**Windows: Microsoft Visual C++ 14.0 或更高版本**

选项 A：安装 Visual Studio Build Tools（推荐）
1. 下载：https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 运行安装程序
3. 选择"使用 C++ 的桌面开发"工作负载
4. 确保选中以下组件：
   - MSVC v142 或更高版本
   - Windows 10 SDK
   - C++ CMake 工具（可选）

选项 B：安装完整的 Visual Studio
1. 下载：https://visualstudio.microsoft.com/downloads/
2. 安装 Visual Studio Community（免费）或更高版本
3. 在安装期间选择"使用 C++ 的桌面开发"

验证安装：
```cmd
cl
```
应显示 Microsoft C/C++ 编译器版本信息。

**Linux: GCC 7.0+ 或 Clang 5.0+**

Ubuntu/Debian：
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

CentOS/RHEL：
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

Fedora：
```bash
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

验证安装：
```bash
gcc --version
# 或
clang --version
```

**macOS: Xcode 命令行工具**

安装：
```bash
xcode-select --install
```

如果已安装，更新：
```bash
softwareupdate --install -a
```

验证安装：
```bash
clang --version
```

#### 3. Python 构建依赖

安装必需的 Python 包：

```bash
# 升级 pip 和基础工具
pip install --upgrade pip setuptools wheel

# 安装构建依赖
pip install pybind11>=2.6.0 numpy>=1.20.0
```

验证安装：
```python
python -c "import pybind11; print(pybind11.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

## 构建步骤

### 方法 1：标准安装（推荐）

这是最简单的方法，适合大多数用户：

```bash
# 1. 克隆或下载仓库
git clone <repository-url>
cd ikfast_pybind

# 2. 安装包
pip install .
```

安装完成后，您可以在任何地方导入该包：
```python
import ikfast_pybind as ik
```

### 方法 2：开发安装（可编辑模式）

如果您计划修改代码，使用可编辑安装：

```bash
# 1. 克隆仓库
git clone <repository-url>
cd ikfast_pybind

# 2. 安装开发依赖
pip install -e ".[dev]"
```

这允许您修改 Python 代码而无需重新安装。但是，如果修改 C++ 代码，您需要重新构建：

```bash
# 修改 C++ 代码后重新构建
pip install -e ".[dev]" --force-reinstall --no-deps
```

### 方法 3：就地构建（用于测试）

仅构建扩展模块而不安装：

```bash
# 构建扩展模块
python setup.py build_ext --inplace
```

这会在当前目录中创建 `_ikfast_pybind.pyd`（Windows）或 `_ikfast_pybind.so`（Linux/macOS）。

### 方法 4：创建分发包

创建可分发的包：

```bash
# 创建源码分发
python setup.py sdist

# 创建二进制 wheel（推荐）
pip install wheel
python setup.py bdist_wheel
```

生成的文件将在 `dist/` 目录中。

## 验证安装

### 1. 测试项目结构

```bash
python tests/test_build.py
```

这将验证：
- 项目目录结构
- 配置文件（pyproject.toml、setup.py）
- C++ 源文件存在

### 2. 测试模块导入

```python
import ikfast_pybind as ik

# 打印版本信息
print(f"版本：{ik.__version__}")

# 获取求解器信息
info = ik.get_solver_info()
print(f"关节数：{info['num_joints']}")
print(f"求解器类型：{hex(info['ik_type'])}")
```

### 3. 运行基础测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_compute_ik.py
pytest tests/test_compute_fk.py
```

### 4. 运行示例

```bash
# 运行基础逆运动学示例
python examples/basic_ik.py

# 运行正运动学示例
python examples/basic_fk.py

# 运行高级选择示例
python examples/solution_selection.py
```

## 故障排除

### Windows 常见问题

#### 问题 1：缺少 Microsoft Visual C++ 14.0

**错误消息：**
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"
```

**解决方案：**
1. 从 https://visualstudio.microsoft.com/visual-cpp-build-tools/ 下载并安装 Build Tools
2. 重启终端
3. 重试安装

#### 问题 2：DLL 加载失败

**错误消息：**
```
ImportError: DLL load failed while importing _ikfast_pybind: 找不到指定的模块。
```

**解决方案：**
1. 安装 Visual C++ Redistributable：
   - 下载：https://aka.ms/vs/17/release/vc_redist.x64.exe
   - 运行安装程序
2. 确保所有依赖项已安装：
   ```bash
   pip install numpy pybind11
   ```
3. 检查 Python 版本是否与构建时使用的版本匹配

#### 问题 3：编码错误

**错误消息：**
```
UnicodeDecodeError: 'gbk' codec can't decode byte...
```

**解决方案：**
在命令提示符中设置环境变量：
```cmd
set PYTHONIOENCODING=utf-8
pip install .
```

### Linux 常见问题

#### 问题 1：缺少编译器

**错误消息：**
```
error: command 'gcc' failed: No such file or directory
```

**解决方案：**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install python3-devel
```

#### 问题 2：缺少 Python 头文件

**错误消息：**
```
fatal error: Python.h: No such file or directory
```

**解决方案：**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# CentOS/RHEL
sudo yum install python3-devel

# Fedora
sudo dnf install python3-devel
```

#### 问题 3：权限被拒绝

**错误消息：**
```
PermissionError: [Errno 13] Permission denied
```

**解决方案：**
使用用户安装或虚拟环境：
```bash
# 选项 1：用户安装
pip install --user .

# 选项 2：使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate
pip install .
```

### macOS 常见问题

#### 问题 1：缺少 Xcode 命令行工具

**错误消息：**
```
xcrun: error: invalid active developer path
```

**解决方案：**
```bash
xcode-select --install
```

#### 问题 2：部署目标无效

**错误消息：**
```
clang: error: invalid deployment target for -stdlib=libc++
```

**解决方案：**
更新 Xcode 命令行工具：
```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

#### 问题 3：架构不匹配

**错误消息：**
```
ImportError: dlopen(...): mach-o, but wrong architecture
```

**解决方案：**
确保 Python 和编译器架构匹配：
```bash
# 检查 Python 架构
python -c "import platform; print(platform.machine())"

# 对于 Apple Silicon (M1/M2)，使用 ARM Python
# 对于 Intel Mac，使用 x86_64 Python
```

### 通用问题

#### 问题 1：找不到 pybind11

**错误消息：**
```
fatal error: pybind11/pybind11.h: No such file or directory
```

**解决方案：**
```bash
pip install pybind11
```

#### 问题 2：找不到 NumPy

**错误消息：**
```
ModuleNotFoundError: No module named 'numpy'
```

**解决方案：**
```bash
pip install numpy
```

#### 问题 3：构建失败，没有明确错误

**解决方案：**
使用详细输出重新构建：
```bash
pip install . -v
```

这将显示详细的编译器输出，帮助识别问题。

#### 问题 4：测试失败

**解决方案：**
1. 确保所有依赖项已安装：
   ```bash
   pip install pytest hypothesis pytest-cov
   ```
2. 检查模块是否正确安装：
   ```python
   import ikfast_pybind
   print(ikfast_pybind.__file__)
   ```
3. 运行特定测试以隔离问题：
   ```bash
   pytest tests/test_compute_ik.py -v
   ```

## 构建配置

### 配置文件

构建系统在以下文件中配置：

#### 1. `pyproject.toml`

现代 Python 包元数据和构建要求：

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "pybind11>=2.6.0", "numpy>=1.20.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ikfast-pybind"
version = "0.1.0"
description = "Python bindings for IKFast inverse kinematics solver"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
]
```

#### 2. `setup.py`

扩展模块构建配置：

```python
ext_modules = [
    Extension(
        'ikfast_pybind._ikfast_pybind',
        sources=[
            'ikfast_pybind/_ikfast_pybind.cpp',
            'src/sa0521_manipulator_ikfast_solver.cpp',
        ],
        include_dirs=[
            get_pybind_include(),
            'include',
            numpy.get_include(),
        ],
        language='c++',
        extra_compile_args=[...],
        extra_link_args=[...],
    ),
]
```

#### 3. `MANIFEST.in`

指定要包含在源码分发中的额外文件：

```
include README.md
include LICENSE
include pyproject.toml
include setup.py
recursive-include include *.h
recursive-include src *.cpp
recursive-include ikfast_pybind *.cpp
```

### 编译器标志

#### Unix/Linux/macOS

```bash
-std=c++14              # C++14 标准
-O3                     # 优化级别 3（最大优化）
-DIKFAST_HAS_LIBRARY    # 启用 IKFast 库模式
-DIKFAST_NO_MAIN        # 排除 main 函数
-fPIC                   # 位置无关代码（共享库需要）
```

#### Windows (MSVC)

```bash
/std:c++14              # C++14 标准
/O2                     # 优化级别 2
/DIKFAST_HAS_LIBRARY    # 启用 IKFast 库模式
/DIKFAST_NO_MAIN        # 排除 main 函数
/EHsc                   # 异常处理模型
```

### 链接器标志

#### Unix/Linux

```bash
-static-libgcc          # 静态链接 GCC 运行时
-static-libstdc++       # 静态链接 C++ 标准库
```

这些标志确保二进制文件可以在没有特定 GCC 版本的系统上运行。

#### macOS

```bash
-undefined dynamic_lookup  # 允许未定义的符号（Python 扩展需要）
```

#### Windows

通常不需要特殊的链接器标志，因为 MSVC 会自动处理。

## 跨平台支持

### 支持的平台

| 平台 | 架构 | Python 版本 | 状态 |
|------|------|-------------|------|
| Windows 10/11 | x64 | 3.8-3.12 | ✅ 完全支持 |
| Ubuntu 20.04+ | x64 | 3.8-3.12 | ✅ 完全支持 |
| Ubuntu 20.04+ | ARM64 | 3.8-3.12 | ✅ 完全支持 |
| macOS 11+ | x64 (Intel) | 3.8-3.12 | ✅ 完全支持 |
| macOS 11+ | ARM64 (M1/M2) | 3.8-3.12 | ✅ 完全支持 |
| CentOS 7+ | x64 | 3.8-3.12 | ✅ 完全支持 |
| Debian 10+ | x64 | 3.8-3.12 | ✅ 完全支持 |

### 平台特定注意事项

#### Windows

- 需要 Visual Studio 2015 或更高版本
- 推荐使用 64 位 Python
- 可能需要 Visual C++ Redistributable

#### Linux

- 需要 GCC 7.0+ 或 Clang 5.0+
- 需要 python3-dev 包
- 某些发行版可能需要额外的开发包

#### macOS

- 需要 Xcode 命令行工具
- Apple Silicon (M1/M2) 完全支持
- 可能需要 Rosetta 2 用于某些工具

### 测试矩阵

持续集成在以下环境中测试：

```yaml
Python 版本: [3.8, 3.9, 3.10, 3.11, 3.12]
操作系统: [ubuntu-latest, windows-latest, macos-latest]
NumPy 版本: [1.20.0, 1.21.0, 1.22.0, 1.23.0, 1.24.0, latest]
```

## 高级构建选项

### 自定义编译器

指定自定义 C++ 编译器：

**Linux/macOS:**
```bash
export CXX=/usr/bin/g++-9
pip install .
```

**Windows:**
```cmd
set CXX=cl.exe
pip install .
```

### 调试构建

构建带有调试符号的版本：

**Linux/macOS:**
```bash
export CFLAGS="-g -O0"
export CXXFLAGS="-g -O0"
pip install .
```

**Windows:**
```cmd
set CFLAGS=/Zi /Od
set CXXFLAGS=/Zi /Od
pip install .
```

### 并行构建

加速编译（如果支持）：

```bash
pip install . --global-option="build_ext" --global-option="-j4"
```

或使用环境变量：
```bash
export MAX_JOBS=4
pip install .
```

### 清理构建

删除所有构建产物：

```bash
# 删除构建目录
rm -rf build/ dist/ *.egg-info/

# 删除编译的扩展
find . -name "*.so" -delete
find . -name "*.pyd" -delete

# 删除 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### 构建文档

如果项目包含文档：

```bash
# 安装文档依赖
pip install sphinx sphinx-rtd-theme

# 构建 HTML 文档
cd docs
make html

# 查看文档
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
start _build/html/index.html  # Windows
```

## 性能优化

### 编译器优化级别

不同的优化级别权衡编译时间和运行时性能：

| 级别 | GCC/Clang | MSVC | 描述 |
|------|-----------|------|------|
| 无优化 | `-O0` | `/Od` | 最快编译，最慢运行 |
| 基础 | `-O1` | `/O1` | 平衡 |
| 推荐 | `-O2` | `/O2` | 良好的性能，合理的编译时间 |
| 最大 | `-O3` | `/Ox` | 最佳性能，较慢编译 |

当前配置使用 `-O3`（Unix）和 `/O2`（Windows）以获得最佳性能。

### 链接时优化 (LTO)

启用链接时优化以获得更好的性能：

**GCC/Clang:**
```bash
export CXXFLAGS="-O3 -flto"
export LDFLAGS="-flto"
pip install .
```

**MSVC:**
```cmd
set CXXFLAGS=/O2 /GL
set LDFLAGS=/LTCG
pip install .
```

注意：LTO 会显著增加编译时间。

## 获取帮助

如果遇到构建问题：

1. **检查前置要求**：确保所有必需软件已安装
2. **查看错误消息**：仔细阅读编译器输出
3. **使用详细模式**：`pip install . -v` 获取更多信息
4. **检查环境**：验证 Python、编译器和依赖项版本
5. **清理并重试**：删除构建产物并重新构建
6. **查阅文档**：查看 [README_CN.md](README_CN.md) 了解使用说明
7. **搜索问题**：在项目问题跟踪器中搜索类似问题

## 相关资源

- **pybind11 文档**：https://pybind11.readthedocs.io/
- **NumPy 文档**：https://numpy.org/doc/
- **Python 打包指南**：https://packaging.python.org/
- **CMake 文档**：https://cmake.org/documentation/
- **IKFast 文档**：http://openrave.org/docs/latest_stable/openravepy/ikfast/

## 版本历史

### v0.1.0（当前）

- 初始版本
- 支持 6-DOF 机械臂
- 完整的 IK/FK 功能
- 146 个测试，95% 覆盖率
- 跨平台支持（Windows、Linux、macOS）