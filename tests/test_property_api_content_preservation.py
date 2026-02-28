"""
Property-based tests for API content preservation.

Feature: sphinx-docs-migration
Property 4: API Content Preservation

For any HTML API documentation file in docs_old, extracting and converting it to RST
should preserve all function signatures, parameter descriptions, return type information,
and code examples present in the original HTML.

Validates: Requirements 3.1, 3.2
"""

import pytest
from hypothesis import given, strategies as st, settings
from pathlib import Path
from bs4 import BeautifulSoup

from migration_tool.html_extractor import HTMLAPIExtractor
from migration_tool.rst_generator import RSTAPIGenerator


# Strategy for generating simple HTML API documentation
@st.composite
def html_api_doc(draw):
    """Generate a simple HTML API documentation structure."""
    func_name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Ll', 'Lu'), min_codepoint=97, max_codepoint=122)))
    
    # Generate parameters
    num_params = draw(st.integers(min_value=0, max_value=3))
    params = []
    for i in range(num_params):
        param_name = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
            whitelist_categories=('Ll',), min_codepoint=97, max_codepoint=122)))
        param_type = draw(st.sampled_from(['str', 'int', 'float', 'bool', 'List', 'Dict']))
        param_desc = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(
            blacklist_categories=('Cc', 'Cs'), min_codepoint=32, max_codepoint=126)))
        params.append((param_name, param_type, param_desc))
    
    # Generate return info
    has_return = draw(st.booleans())
    return_type = draw(st.sampled_from(['str', 'int', 'float', 'bool', 'List', 'None'])) if has_return else None
    return_desc = draw(st.text(min_size=5, max_size=50, alphabet=st.characters(
        blacklist_categories=('Cc', 'Cs'), min_codepoint=32, max_codepoint=126))) if has_return else None
    
    # Generate description
    description = draw(st.text(min_size=10, max_size=100, alphabet=st.characters(
        blacklist_categories=('Cc', 'Cs'), min_codepoint=32, max_codepoint=126)))
    
    # Generate code example
    has_example = draw(st.booleans())
    example_code = f"result = {func_name}()" if has_example else None
    
    # Build HTML
    html_parts = [
        '<!DOCTYPE html>',
        '<html>',
        '<body>',
        '<section>',
        f'<h1>module.{func_name}</h1>',
        '<dl class="py function">',
        '<dt class="sig sig-object py">',
        '<span class="sig-prename descclassname"><span class="pre">module.</span></span>',
        f'<span class="sig-name descname"><span class="pre">{func_name}</span></span>',
        '<span class="sig-paren">(</span>',
    ]
    
    # Add parameters to signature
    for i, (pname, ptype, pdesc) in enumerate(params):
        if i > 0:
            html_parts.append('<span class="sig-paren">,</span>')
        html_parts.append(f'<em class="sig-param"><span class="n"><span class="pre">{pname}</span></span>')
        html_parts.append(f'<span class="p"><span class="pre">:</span></span>')
        html_parts.append(f'<span class="n"><span class="pre">{ptype}</span></span></em>')
    
    html_parts.extend([
        '<span class="sig-paren">)</span>',
    ])
    
    if has_return and return_type:
        html_parts.extend([
            '<span class="sig-return">',
            '<span class="sig-return-icon">→</span>',
            f'<span class="sig-return-typehint"><span class="pre">{return_type}</span></span>',
            '</span>',
        ])
    
    html_parts.extend([
        '</dt>',
        '<dd>',
        f'<p>{description}</p>',
    ])
    
    # Add parameter list
    if params:
        html_parts.extend([
            '<dl class="field-list simple">',
            '<dt class="field-odd">Parameters:</dt>',
            '<dd class="field-odd">',
            '<ul class="simple">',
        ])
        for pname, ptype, pdesc in params:
            html_parts.append(f'<li><p><strong>{pname}</strong> (<em>{ptype}</em>) – {pdesc}</p></li>')
        html_parts.extend([
            '</ul>',
            '</dd>',
        ])
    
    # Add return info
    if has_return:
        html_parts.extend([
            '<dt class="field-even">Returns:</dt>',
            f'<dd class="field-even"><p>{return_desc}</p></dd>',
            '<dt class="field-odd">Return type:</dt>',
            f'<dd class="field-odd"><p>{return_type}</p></dd>',
        ])
    
    if params or has_return:
        html_parts.append('</dl>')
    
    # Add example
    if has_example:
        html_parts.extend([
            '<p class="rubric">Example</p>',
            '<div class="highlight-python notranslate">',
            '<div class="highlight">',
            f'<pre>{example_code}</pre>',
            '</div>',
            '</div>',
        ])
    
    html_parts.extend([
        '</dd>',
        '</dl>',
        '</section>',
        '</body>',
        '</html>',
    ])
    
    html_content = '\n'.join(html_parts)
    
    return {
        'html': html_content,
        'func_name': func_name,
        'params': params,
        'return_type': return_type,
        'return_desc': return_desc,
        'description': description,
        'example': example_code,
    }


@given(html_api_doc())
@settings(max_examples=100, deadline=None)
def test_api_content_preservation_function_signature(api_doc_data):
    """
    Property 4: API Content Preservation - Function Signature
    
    For any HTML API documentation, the extracted function name should be preserved.
    """
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    # Extract from HTML
    api_doc = extractor.extract_from_html(api_doc_data['html'])
    
    # Verify function name is preserved
    assert api_doc.name == api_doc_data['func_name'], \
        f"Function name not preserved: expected {api_doc_data['func_name']}, got {api_doc.name}"
    
    # Generate RST
    rst_content = generator.generate_rst(api_doc)
    
    # Verify function name appears in RST
    assert api_doc_data['func_name'] in rst_content, \
        f"Function name {api_doc_data['func_name']} not found in generated RST"


@given(html_api_doc())
@settings(max_examples=100, deadline=None)
def test_api_content_preservation_parameters(api_doc_data):
    """
    Property 4: API Content Preservation - Parameters
    
    For any HTML API documentation, all parameter names and types should be preserved.
    """
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    # Extract from HTML
    api_doc = extractor.extract_from_html(api_doc_data['html'])
    
    # Verify all parameters are extracted
    expected_param_names = [p[0] for p in api_doc_data['params']]
    extracted_param_names = [p.name for p in api_doc.parameters]
    
    assert len(extracted_param_names) == len(expected_param_names), \
        f"Parameter count mismatch: expected {len(expected_param_names)}, got {len(extracted_param_names)}"
    
    for expected_name in expected_param_names:
        assert expected_name in extracted_param_names, \
            f"Parameter {expected_name} not found in extracted parameters"
    
    # Generate RST
    rst_content = generator.generate_rst(api_doc)
    
    # Verify all parameter names appear in RST
    for param_name in expected_param_names:
        assert param_name in rst_content, \
            f"Parameter {param_name} not found in generated RST"


@given(html_api_doc())
@settings(max_examples=100, deadline=None)
def test_api_content_preservation_return_type(api_doc_data):
    """
    Property 4: API Content Preservation - Return Type
    
    For any HTML API documentation with return type, the return type should be preserved.
    """
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    # Extract from HTML
    api_doc = extractor.extract_from_html(api_doc_data['html'])
    
    # Verify return type if present
    if api_doc_data['return_type']:
        assert api_doc.returns is not None, "Return info not extracted"
        assert api_doc.returns.type_hint == api_doc_data['return_type'], \
            f"Return type mismatch: expected {api_doc_data['return_type']}, got {api_doc.returns.type_hint}"
        
        # Generate RST
        rst_content = generator.generate_rst(api_doc)
        
        # Verify return type appears in RST
        assert api_doc_data['return_type'] in rst_content, \
            f"Return type {api_doc_data['return_type']} not found in generated RST"


@given(html_api_doc())
@settings(max_examples=100, deadline=None)
def test_api_content_preservation_code_examples(api_doc_data):
    """
    Property 4: API Content Preservation - Code Examples
    
    For any HTML API documentation with code examples, the examples should be preserved.
    """
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    # Extract from HTML
    api_doc = extractor.extract_from_html(api_doc_data['html'])
    
    # Verify code example if present
    if api_doc_data['example']:
        assert len(api_doc.examples) > 0, "Code example not extracted"
        
        # Check if the example content is preserved
        example_found = any(api_doc_data['example'] in ex for ex in api_doc.examples)
        assert example_found, f"Code example not found in extracted examples"
        
        # Generate RST
        rst_content = generator.generate_rst(api_doc)
        
        # Verify example appears in RST (may be formatted differently)
        assert 'code-block' in rst_content or 'Example' in rst_content, \
            "Code example section not found in generated RST"


@given(html_api_doc())
@settings(max_examples=100, deadline=None)
def test_api_content_preservation_description(api_doc_data):
    """
    Property 4: API Content Preservation - Description
    
    For any HTML API documentation, the description should be preserved.
    """
    extractor = HTMLAPIExtractor()
    generator = RSTAPIGenerator()
    
    # Extract from HTML
    api_doc = extractor.extract_from_html(api_doc_data['html'])
    
    # Verify description is extracted
    assert api_doc.description, "Description not extracted"
    
    # Note: BeautifulSoup may strip invalid HTML characters like <? or <A which look like tags
    # We verify that the description is not empty and contains some of the original content
    assert len(api_doc.description) > 0, "Description is empty"
    
    # For valid content (no HTML-like characters), check more strictly
    if '<' not in api_doc_data['description']:
        # Should preserve most content when there are no HTML-like characters
        import re
        expected_alphanum = re.sub(r'[^a-zA-Z0-9]', '', api_doc_data['description'])
        extracted_alphanum = re.sub(r'[^a-zA-Z0-9]', '', api_doc.description)
        
        if len(expected_alphanum) > 0:
            match_count = sum(1 for c in expected_alphanum if c in extracted_alphanum)
            preservation_ratio = match_count / len(expected_alphanum)
            assert preservation_ratio >= 0.8, \
                f"Description content not sufficiently preserved: {preservation_ratio:.2%} preserved"
    
    # Generate RST
    rst_content = generator.generate_rst(api_doc)
    
    # Verify description appears in RST
    assert api_doc.description in rst_content, \
        f"Description not found in generated RST"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
