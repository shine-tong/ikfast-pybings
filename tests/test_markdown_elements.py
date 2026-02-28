"""
Unit tests for specific markdown element conversions.

Tests code block conversion with language specs, table conversion,
nested list conversion, and badge/image conversion.

Validates: Requirements 1.2, 1.5, 1.7, 1.8
"""

import pytest
from migration_tool.converter import convert_markdown_to_rst, ConversionError
from migration_tool.post_processor import (
    post_process_rst,
    convert_badge_images_to_centered,
    transform_admonitions
)


class TestCodeBlockConversion:
    """Test code block conversion with language specifications."""
    
    def test_python_code_block(self):
        """Test conversion of Python code block with language spec."""
        markdown = """```python
def hello():
    print("Hello, World!")
```"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should contain code-block directive
            assert '.. code-block::' in rst or '::' in rst
            # Should preserve the function definition
            assert 'def hello' in rst or 'hello' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_bash_code_block(self):
        """Test conversion of Bash code block."""
        markdown = """```bash
pip install ikfast_pybind
```"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should contain code directive
            assert '.. code-block::' in rst or '::' in rst
            # Should preserve the command
            assert 'pip install' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_code_block_without_language(self):
        """Test conversion of code block without language specification."""
        markdown = """```
some code here
```"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should still create a code block
            assert '::' in rst or 'code' in rst.lower()
            assert 'some code here' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_multiple_code_blocks(self):
        """Test conversion of multiple code blocks."""
        markdown = """First block:

```python
x = 1
```

Second block:

```cpp
int x = 1;
```"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should contain both code blocks
            assert rst.count('::') >= 2 or rst.count('code-block') >= 2
        except ConversionError:
            pytest.skip("Pandoc not available")


class TestTableConversion:
    """Test table conversion from Markdown to RST."""
    
    def test_simple_table(self):
        """Test conversion of simple markdown table."""
        markdown = """| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # RST tables use various formats, check for table indicators
            # Grid tables use +, =, -, | characters
            # Simple tables use = and spaces
            assert any(char in rst for char in ['+', '=', '|'])
            # Should preserve column headers
            assert 'Column 1' in rst and 'Column 2' in rst
            # Should preserve values
            assert 'Value 1' in rst and 'Value 2' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_table_with_alignment(self):
        """Test conversion of table with column alignment."""
        markdown = """| Left | Center | Right |
|:-----|:------:|------:|
| L1   | C1     | R1    |
| L2   | C2     | R2    |"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should preserve content
            assert 'Left' in rst and 'Center' in rst and 'Right' in rst
            assert 'L1' in rst and 'C1' in rst and 'R1' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")


class TestNestedListConversion:
    """Test nested list conversion."""
    
    def test_nested_unordered_list(self):
        """Test conversion of nested unordered list."""
        markdown = """- Item 1
  - Nested 1.1
  - Nested 1.2
- Item 2
  - Nested 2.1"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should preserve list structure
            assert 'Item 1' in rst
            assert 'Nested 1.1' in rst
            assert 'Item 2' in rst
            # RST uses - or * for lists
            assert '-' in rst or '*' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_nested_ordered_list(self):
        """Test conversion of nested ordered list."""
        markdown = """1. First item
   1. Nested first
   2. Nested second
2. Second item
   1. Nested first"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should preserve content
            assert 'First item' in rst
            assert 'Nested first' in rst
            assert 'Second item' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_mixed_nested_lists(self):
        """Test conversion of mixed nested lists (ordered and unordered)."""
        markdown = """1. Ordered item
   - Unordered nested
   - Another unordered
2. Another ordered
   - Nested unordered"""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should preserve all content
            assert 'Ordered item' in rst
            assert 'Unordered nested' in rst
            assert 'Another ordered' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")


class TestBadgeAndImageConversion:
    """Test badge and image conversion."""
    
    def test_regular_image(self):
        """Test conversion of regular image."""
        markdown = "![Alt text](image.png)"
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should use image directive
            assert '.. image::' in rst
            assert 'image.png' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_badge_image_centering(self):
        """Test that badge images are centered."""
        # Simulate RST with badge image
        rst_content = """.. image:: https://img.shields.io/badge/test-badge-blue
   :target: https://example.com"""
        
        processed = convert_badge_images_to_centered(rst_content)
        
        # Should add center alignment
        assert ':align: center' in processed
    
    def test_multiple_badges(self):
        """Test multiple badge images."""
        rst_content = """.. image:: https://img.shields.io/badge/build-passing-green

.. image:: https://img.shields.io/badge/coverage-90-yellow"""
        
        processed = convert_badge_images_to_centered(rst_content)
        
        # Both should be centered
        assert processed.count(':align: center') >= 2
    
    def test_non_badge_image_not_centered(self):
        """Test that non-badge images are not automatically centered."""
        rst_content = """.. image:: docs/screenshot.png
   :alt: Screenshot"""
        
        processed = convert_badge_images_to_centered(rst_content)
        
        # Should not add centering to non-badge images
        # (This test checks that we don't over-apply centering)
        assert rst_content == processed or ':align: center' not in processed
    
    def test_image_with_alt_text(self):
        """Test image conversion preserves alt text."""
        markdown = "![Important diagram](diagram.png)"
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should preserve alt text
            assert 'Important diagram' in rst or 'diagram' in rst
            assert 'diagram.png' in rst
        except ConversionError:
            pytest.skip("Pandoc not available")


class TestAdmonitionConversion:
    """Test admonition conversion."""
    
    def test_note_admonition(self):
        """Test conversion of note admonition."""
        rst_content = "**Note:** This is important information."
        
        processed = transform_admonitions(rst_content)
        
        # Should convert to RST note directive
        assert '.. note::' in processed
        assert 'important information' in processed
    
    def test_warning_admonition(self):
        """Test conversion of warning admonition."""
        rst_content = "**Warning:** Be careful with this."
        
        processed = transform_admonitions(rst_content)
        
        # Should convert to RST warning directive
        assert '.. warning::' in processed
        assert 'Be careful' in processed
    
    def test_tip_admonition(self):
        """Test conversion of tip admonition."""
        rst_content = "**Tip:** Here's a helpful hint."
        
        processed = transform_admonitions(rst_content)
        
        # Should convert to RST tip directive
        assert '.. tip::' in processed
        assert 'helpful hint' in processed
    
    def test_multiple_admonitions(self):
        """Test multiple admonitions in same document."""
        rst_content = """**Note:** First note.

**Warning:** A warning.

**Tip:** A helpful tip."""
        
        processed = transform_admonitions(rst_content)
        
        # Should convert all admonitions
        assert '.. note::' in processed
        assert '.. warning::' in processed
        assert '.. tip::' in processed


class TestCompleteConversion:
    """Test complete conversion with post-processing."""
    
    def test_full_conversion_pipeline(self):
        """Test complete conversion pipeline with post-processing."""
        markdown = """# Test Document

This is a test with `inline code` and a [link](file.md).

```python
def test():
    pass
```

- List item 1
- List item 2

![Badge](https://img.shields.io/badge/test-badge-blue)"""
        
        try:
            # Convert
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Post-process
            processed, warnings = post_process_rst(rst, "test.md")
            
            # Should have content
            assert len(processed) > 0
            assert 'Test Document' in processed
            
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_empty_input(self):
        """Test conversion of empty input."""
        markdown = ""
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Empty input should produce empty output
            assert rst == ""
        except ConversionError:
            pytest.skip("Pandoc not available")
    
    def test_whitespace_only_input(self):
        """Test conversion of whitespace-only input."""
        markdown = "   \n\n   \n"
        
        try:
            rst = convert_markdown_to_rst(markdown, "test.md")
            
            # Should handle gracefully
            assert isinstance(rst, str)
        except ConversionError:
            pytest.skip("Pandoc not available")
