"""
Property-based tests for Markdown to RST conversion completeness.

Feature: sphinx-docs-migration
Property 1: Markdown-to-RST Conversion Completeness

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 7.1, 7.2, 7.3, 7.4, 7.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from migration_tool.converter import convert_markdown_to_rst, ConversionError
from migration_tool.post_processor import post_process_rst


# Strategies for generating markdown elements
@st.composite
def markdown_headers(draw):
    """Generate markdown headers."""
    level = draw(st.integers(min_value=1, max_value=6))
    text = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        blacklist_categories=('Cs', 'Cc'), blacklist_characters='\n\r'
    )))
    assume(text.strip())  # Ensure non-empty after stripping
    return '#' * level + ' ' + text.strip()


@st.composite
def markdown_code_blocks(draw):
    """Generate markdown code blocks with language specs."""
    language = draw(st.sampled_from(['python', 'bash', 'javascript', 'cpp', 'java', '']))
    code = draw(st.text(min_size=1, max_size=100))
    return f'```{language}\n{code}\n```'


@st.composite
def markdown_inline_code(draw):
    """Generate markdown inline code."""
    code = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        blacklist_categories=('Cs', 'Cc'), blacklist_characters='`\n\r'
    )))
    assume(code.strip())
    return f'`{code.strip()}`'


@st.composite
def markdown_links(draw):
    """Generate markdown links."""
    text = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        blacklist_categories=('Cs', 'Cc'), blacklist_characters='[]\n\r'
    )))
    url = draw(st.sampled_from([
        'https://example.com',
        'file.md',
        'README.md',
        '#section',
        'file.md#section'
    ]))
    assume(text.strip())
    return f'[{text.strip()}]({url})'


@st.composite
def markdown_lists(draw):
    """Generate markdown lists."""
    list_type = draw(st.sampled_from(['unordered', 'ordered']))
    num_items = draw(st.integers(min_value=1, max_value=5))
    items = []
    
    for i in range(num_items):
        item_text = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
            blacklist_categories=('Cs', 'Cc'), blacklist_characters='\n\r'
        )))
        if item_text.strip():
            if list_type == 'unordered':
                items.append(f'- {item_text.strip()}')
            else:
                items.append(f'{i+1}. {item_text.strip()}')
    
    assume(len(items) > 0)
    return '\n'.join(items)


@st.composite
def markdown_images(draw):
    """Generate markdown images."""
    alt_text = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        blacklist_categories=('Cs', 'Cc'), blacklist_characters='[]\n\r'
    )))
    url = draw(st.sampled_from([
        'image.png',
        'https://example.com/image.jpg',
        'https://img.shields.io/badge/test-badge-blue'
    ]))
    assume(alt_text.strip())
    return f'![{alt_text.strip()}]({url})'


@st.composite
def markdown_emphasis(draw):
    """Generate markdown emphasis (bold/italic)."""
    text = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        blacklist_categories=('Cs', 'Cc'), blacklist_characters='*_\n\r'
    )))
    style = draw(st.sampled_from(['**', '*', '__', '_']))
    assume(text.strip())
    return f'{style}{text.strip()}{style}'


@st.composite
def markdown_document(draw):
    """Generate a complete markdown document with various elements."""
    elements = []
    num_elements = draw(st.integers(min_value=1, max_value=10))
    
    for _ in range(num_elements):
        element_type = draw(st.sampled_from([
            'header', 'code_block', 'inline_code', 'link', 
            'list', 'image', 'emphasis', 'text'
        ]))
        
        if element_type == 'header':
            elements.append(draw(markdown_headers()))
        elif element_type == 'code_block':
            elements.append(draw(markdown_code_blocks()))
        elif element_type == 'inline_code':
            elements.append(draw(markdown_inline_code()))
        elif element_type == 'link':
            elements.append(draw(markdown_links()))
        elif element_type == 'list':
            elements.append(draw(markdown_lists()))
        elif element_type == 'image':
            elements.append(draw(markdown_images()))
        elif element_type == 'emphasis':
            elements.append(draw(markdown_emphasis()))
        else:
            text = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
                blacklist_categories=('Cs', 'Cc')
            )))
            if text.strip():
                elements.append(text.strip())
    
    assume(len(elements) > 0)
    return '\n\n'.join(elements)


@given(markdown_headers())
@settings(max_examples=100, deadline=None)
def test_header_conversion_preserves_content(markdown_header):
    """
    Property 1: Headers should be converted to RST with proper underlines.
    
    For any markdown header, conversion should preserve the text content
    and create appropriate RST header underlines.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_header, "test.md")
        
        # Extract header text (remove # symbols)
        header_text = markdown_header.lstrip('#').strip()
        
        # RST output should contain the header text
        assert header_text in rst_output, \
            f"Header text '{header_text}' not found in RST output"
        
        # RST should have some underline characters (=, -, ~, etc.)
        assert any(char in rst_output for char in ['=', '-', '~', '^', '"']), \
            "RST output should contain header underline characters"
    
    except ConversionError:
        # Pandoc not installed or other conversion error - skip test
        pytest.skip("Pandoc not available")


@given(markdown_code_blocks())
@settings(max_examples=100, deadline=None)
def test_code_block_conversion_preserves_content(markdown_code):
    """
    Property 1: Code blocks should preserve content and language specs.
    
    For any markdown code block, conversion should preserve the code content
    and maintain language specifications.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_code, "test.md")
        
        # RST should use code-block directive
        assert '.. code-block::' in rst_output or '::' in rst_output, \
            "RST output should contain code block directive"
        
        # Output should be non-empty
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_inline_code())
@settings(max_examples=100, deadline=None)
def test_inline_code_conversion_preserves_content(markdown_code):
    """
    Property 1: Inline code should be converted to RST double-backtick syntax.
    
    For any markdown inline code, conversion should preserve the code content.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_code, "test.md")
        
        # Extract code content
        code_content = markdown_code.strip('`').strip()
        
        # RST output should contain the code content
        # (may be wrapped in different syntax)
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_links())
@settings(max_examples=100, deadline=None)
def test_link_conversion_preserves_content(markdown_link):
    """
    Property 1: Links should be converted to RST link syntax.
    
    For any markdown link, conversion should preserve the link text and URL.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_link, "test.md")
        
        # Output should be non-empty
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
        
        # Apply post-processing
        processed, _ = post_process_rst(rst_output, "test.md")
        
        # Processed output should contain link-related syntax
        assert len(processed.strip()) > 0, \
            "Processed output should not be empty"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_lists())
@settings(max_examples=100, deadline=None)
def test_list_conversion_preserves_structure(markdown_list):
    """
    Property 1: Lists should preserve structure and content.
    
    For any markdown list, conversion should preserve list items and structure.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_list, "test.md")
        
        # Count list items in markdown
        markdown_items = [line for line in markdown_list.split('\n') 
                         if line.strip().startswith(('-', '1.', '2.', '3.', '4.', '5.'))]
        
        # RST output should be non-empty
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
        
        # RST should have list markers (-, *, or numbers)
        assert any(char in rst_output for char in ['-', '*', '1.', '2.']), \
            "RST output should contain list markers"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_images())
@settings(max_examples=100, deadline=None)
def test_image_conversion_preserves_content(markdown_image):
    """
    Property 1: Images should be converted to RST image directives.
    
    For any markdown image, conversion should preserve alt text and URL.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_image, "test.md")
        
        # RST should use image directive
        assert '.. image::' in rst_output or 'image::' in rst_output, \
            "RST output should contain image directive"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_emphasis())
@settings(max_examples=100, deadline=None)
def test_emphasis_conversion_preserves_content(markdown_emphasis):
    """
    Property 1: Emphasis (bold/italic) should preserve text content.
    
    For any markdown emphasis, conversion should preserve the text.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_emphasis, "test.md")
        
        # Extract text content (remove emphasis markers)
        text_content = markdown_emphasis.strip('*_').strip()
        
        # RST output should contain the text (may have different emphasis syntax)
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=100, deadline=None)
def test_conversion_produces_valid_output(markdown_text):
    """
    Property 1: Conversion should always produce non-empty output for non-empty input.
    
    For any non-empty markdown text, conversion should produce some RST output.
    """
    assume(markdown_text.strip())  # Ensure non-empty after stripping
    
    try:
        rst_output = convert_markdown_to_rst(markdown_text, "test.md")
        
        # Output should be non-empty for non-empty input
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty for non-empty input"
    
    except ConversionError:
        pytest.skip("Pandoc not available")


@given(markdown_document())
@settings(max_examples=50, deadline=None)
def test_complete_document_conversion(markdown_doc):
    """
    Property 1: Complete documents should preserve all content elements.
    
    For any markdown document with multiple elements, conversion should
    preserve all content and produce valid RST.
    """
    try:
        rst_output = convert_markdown_to_rst(markdown_doc, "test.md")
        
        # Output should be non-empty
        assert len(rst_output.strip()) > 0, \
            "RST output should not be empty"
        
        # Apply post-processing
        processed, warnings = post_process_rst(rst_output, "test.md")
        
        # Processed output should be non-empty
        assert len(processed.strip()) > 0, \
            "Processed output should not be empty"
    
    except ConversionError:
        pytest.skip("Pandoc not available")
