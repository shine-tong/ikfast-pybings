"""
Unit tests for HTML API documentation extraction.

Tests extraction from sample HTML files, handling of missing elements,
and Chinese content preservation.

Requirements: 3.1, 3.2
"""

import pytest
from pathlib import Path
from migration_tool.html_extractor import HTMLAPIExtractor
from migration_tool.rst_generator import RSTAPIGenerator


def test_extract_from_real_html_file():
    """Test extraction from an actual HTML file in docs_old."""
    extractor = HTMLAPIExtractor()
    
    # Use the compute_ik.html file
    html_file = Path("docs_old/ikfast_pybind.compute_ik.html")
    
    if not html_file.exists():
        pytest.skip("HTML file not found")
    
    api_doc = extractor.extract_from_file(html_file)
    
    # Verify basic extraction
    assert api_doc.name == "compute_ik"
    assert api_doc.full_name == "ikfast_pybind.compute_ik"
    assert not api_doc.is_class
    assert not api_doc.is_internal
    
    # Verify parameters extracted
    assert len(api_doc.parameters) >= 2  # translation, rotation at minimum
    param_names = [p.name for p in api_doc.parameters]
    assert "translation" in param_names
    assert "rotation" in param_names
    
    # Verify return info
    assert api_doc.returns is not None
    
    # Verify examples extracted
    assert len(api_doc.examples) > 0
    
    # Verify description
    assert api_doc.description
    assert len(api_doc.description) > 10


def test_extract_class_from_real_html():
    """Test extraction of a class from actual HTML file."""
    extractor = HTMLAPIExtractor()
    
    html_file = Path("docs_old/ikfast_pybind.IkSolution.html")
    
    if not html_file.exists():
        pytest.skip("HTML file not found")
    
    api_doc = extractor.extract_from_file(html_file)
    
    # Verify it's recognized as a class
    assert api_doc.is_class
    assert api_doc.name == "IkSolution"
    assert api_doc.full_name == "ikfast_pybind.IkSolution"
    
    # Verify methods extracted
    assert len(api_doc.methods) > 0
    method_names = [m.name for m in api_doc.methods]
    assert "get_solution" in method_names or any("solution" in name.lower() for name in method_names)


def test_chinese_content_preservation():
    """Test that Chinese content is properly preserved."""
    extractor = HTMLAPIExtractor()
    
    # Create HTML with Chinese content
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <body>
        <section>
            <h1>ikfast_pybind.compute_ik</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">ikfast_pybind.</span></span>
                    <span class="sig-name descname"><span class="pre">compute_ik</span></span>
                    <span class="sig-paren">(</span>
                    <em class="sig-param"><span class="n"><span class="pre">translation</span></span></em>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>计算目标末端执行器位姿的逆运动学解。</p>
                    <dl class="field-list simple">
                        <dt class="field-odd">参数:</dt>
                        <dd class="field-odd">
                            <ul class="simple">
                                <li><p><strong>translation</strong> (<em>np.ndarray</em>) – 末端执行器位置</p></li>
                            </ul>
                        </dd>
                        <dt class="field-even">返回:</dt>
                        <dd class="field-even"><p>关节角度数组的列表</p></dd>
                    </dl>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    # Verify Chinese characters are preserved
    assert "计算" in api_doc.description or "逆运动学" in api_doc.description
    assert api_doc.parameters[0].name == "translation"
    assert "末端执行器" in api_doc.parameters[0].description or "位置" in api_doc.parameters[0].description
    assert api_doc.returns is not None
    assert "关节" in api_doc.returns.description or "列表" in api_doc.returns.description


def test_handle_missing_parameters():
    """Test handling of function with no parameters."""
    extractor = HTMLAPIExtractor()
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <section>
            <h1>module.simple_func</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">module.</span></span>
                    <span class="sig-name descname"><span class="pre">simple_func</span></span>
                    <span class="sig-paren">(</span>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>A simple function with no parameters.</p>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    assert api_doc.name == "simple_func"
    assert len(api_doc.parameters) == 0
    assert api_doc.description == "A simple function with no parameters."


def test_handle_missing_return_info():
    """Test handling of function with no return information."""
    extractor = HTMLAPIExtractor()
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <section>
            <h1>module.void_func</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">module.</span></span>
                    <span class="sig-name descname"><span class="pre">void_func</span></span>
                    <span class="sig-paren">(</span>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>A function that returns nothing.</p>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    assert api_doc.name == "void_func"
    assert api_doc.returns is None


def test_handle_missing_examples():
    """Test handling of documentation with no code examples."""
    extractor = HTMLAPIExtractor()
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <section>
            <h1>module.no_example_func</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">module.</span></span>
                    <span class="sig-name descname"><span class="pre">no_example_func</span></span>
                    <span class="sig-paren">(</span>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>A function without examples.</p>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    assert api_doc.name == "no_example_func"
    assert len(api_doc.examples) == 0


def test_rst_generation_from_extracted_data():
    """Test that RST can be generated from extracted API documentation."""
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    html_file = Path("docs_old/ikfast_pybind.compute_ik.html")
    
    if not html_file.exists():
        pytest.skip("HTML file not found")
    
    # Extract
    api_doc = extractor.extract_from_file(html_file)
    
    # Generate RST
    rst_content = generator.generate_rst(api_doc)
    
    # Verify RST structure
    assert "ikfast_pybind.compute_ik" in rst_content
    assert ".. py:function::" in rst_content
    assert ":param" in rst_content
    assert ":returns:" in rst_content
    assert "code-block" in rst_content or "Example" in rst_content


def test_internal_api_detection():
    """Test detection of internal API (starts with underscore)."""
    extractor = HTMLAPIExtractor()
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <section>
            <h1>ikfast_pybind._ikfast_pybind.compute_ik_raw</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">ikfast_pybind._ikfast_pybind.</span></span>
                    <span class="sig-name descname"><span class="pre">compute_ik_raw</span></span>
                    <span class="sig-paren">(</span>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>Internal raw IK computation.</p>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    assert api_doc.is_internal
    assert api_doc.name == "compute_ik_raw"


def test_exception_extraction():
    """Test extraction of exception information."""
    extractor = HTMLAPIExtractor()
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
        <section>
            <h1>module.risky_func</h1>
            <dl class="py function">
                <dt class="sig sig-object py">
                    <span class="sig-prename descclassname"><span class="pre">module.</span></span>
                    <span class="sig-name descname"><span class="pre">risky_func</span></span>
                    <span class="sig-paren">(</span>
                    <span class="sig-paren">)</span>
                </dt>
                <dd>
                    <p>A function that may raise exceptions.</p>
                    <dl class="field-list simple">
                        <dt class="field-odd">Raises:</dt>
                        <dd class="field-odd">
                            <ul class="simple">
                                <li><p><strong>ValueError</strong> – If input is invalid</p></li>
                                <li><p><strong>RuntimeError</strong> – If computation fails</p></li>
                            </ul>
                        </dd>
                    </dl>
                </dd>
            </dl>
        </section>
    </body>
    </html>
    """
    
    api_doc = extractor.extract_from_html(html_content)
    
    assert len(api_doc.exceptions) == 2
    exception_types = [e.exception_type for e in api_doc.exceptions]
    assert "ValueError" in exception_types
    assert "RuntimeError" in exception_types


def test_api_index_generation():
    """Test generation of API index with public/internal separation."""
    generator = RSTAPIGenerator()
    
    # Create mock API docs
    from migration_tool.html_extractor import APIDocumentation
    
    public_api = APIDocumentation(
        name="compute_ik",
        full_name="ikfast_pybind.compute_ik",
        signature="compute_ik(translation, rotation)",
        description="Public API function",
        parameters=[],
        returns=None,
        exceptions=[],
        examples=[],
        notes=[],
        see_also=[],
        is_class=False,
        is_method=False,
        is_internal=False
    )
    
    internal_api = APIDocumentation(
        name="compute_ik_raw",
        full_name="ikfast_pybind._ikfast_pybind.compute_ik_raw",
        signature="compute_ik_raw(translation, rotation)",
        description="Internal API function",
        parameters=[],
        returns=None,
        exceptions=[],
        examples=[],
        notes=[],
        see_also=[],
        is_class=False,
        is_method=False,
        is_internal=True
    )
    
    # Generate index
    index_rst = generator.generate_api_index([public_api, internal_api], "API Reference")
    
    # Verify structure
    assert "API Reference" in index_rst
    assert "Public API" in index_rst
    assert "Internal API" in index_rst
    assert ".. toctree::" in index_rst
    assert "ikfast_pybind_compute_ik" in index_rst
    assert "ikfast_pybind__ikfast_pybind_compute_ik_raw" in index_rst


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
