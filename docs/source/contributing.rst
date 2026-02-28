Contributing
============

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8 for Python code
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Type Hints**: Include type annotations for public APIs

Development Setup
-----------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/shine-tong/ikfast-pybings.git
   cd ikfast_pybind

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest tests/

   # Run tests with coverage
   pytest tests/ --cov=ikfast_pybind --cov-report=html

Running Property-Based Tests
-----------------------------

Property-based tests use Hypothesis for randomized testing:

.. code-block:: bash

   # Run with default iterations (100)
   pytest tests/test_property_*.py

   # Run with more iterations for thorough testing
   pytest tests/test_property_*.py --hypothesis-iterations=1000

   # Run with specific seed for reproducibility
   pytest tests/test_property_*.py --hypothesis-seed=12345
