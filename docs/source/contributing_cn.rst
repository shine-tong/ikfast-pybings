贡献
====

欢迎贡献！请遵循以下指南：

1. **代码风格**：Python 代码遵循 PEP 8
2. **测试**：为新功能添加测试
3. **文档**：更新文档字符串和 README
4. **类型提示**：为公共 API 包含类型注释

开发设置
--------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/shine-tong/ikfast-pybings.git
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

运行基于属性的测试
------------------

基于属性的测试使用 Hypothesis 进行随机测试：

.. code-block:: bash

   # 使用默认迭代次数运行（100）
   pytest tests/test_property_*.py

   # 使用更多迭代次数进行彻底测试
   pytest tests/test_property_*.py --hypothesis-iterations=1000

   # 使用特定种子以实现可重现性
   pytest tests/test_property_*.py --hypothesis-seed=12345
