"""
Property-based tests for link resolution correctness.

Feature: sphinx-docs-migration
Property 2: Link Resolution Correctness

For any markdown link (internal document link, internal section link, or external URL),
converting it to RST should produce the correct RST syntax (:doc: directive for internal
documents, :ref: directive for sections, or inline link syntax for external URLs).

Validates: Requirements 5.1, 5.3, 5.5
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from migration_tool.link_resolver import LinkResolver, LinkType
import re


# Strategy for generating markdown file names
@st.composite
def markdown_filename(draw):
    """Generate valid markdown filenames."""
    base_names = ['README', 'BUILD', 'CUSTOM_SOLVER', 'guide', 'tutorial', 'api']
    suffixes = ['', '_CN', '_cn']
    
    base = draw(st.sampled_from(base_names))
    suffix = draw(st.sampled_from(suffixes))
    
    return f"{base}{suffix}.md"


# Strategy for generating section names
@st.composite
def section_name(draw):
    """Generate valid section names."""
    words = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
            min_size=1,
            max_size=10
        ),
        min_size=1,
        max_size=3
    ))
    return '-'.join(words)


# Strategy for generating URLs
@st.composite
def external_url(draw):
    """Generate external URLs."""
    protocols = ['http://', 'https://']
    domains = ['example.com', 'github.com', 'docs.python.org']
    
    protocol = draw(st.sampled_from(protocols))
    domain = draw(st.sampled_from(domains))
    path = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='/-_'),
        min_size=0,
        max_size=20
    ))
    
    return f"{protocol}{domain}/{path}" if path else f"{protocol}{domain}"


@given(markdown_filename())
@settings(max_examples=100)
def test_internal_doc_link_produces_doc_directive(md_file):
    """
    Property 2: Link Resolution Correctness
    
    For any internal markdown document link, conversion should produce :doc: directive.
    
    **Validates: Requirements 5.1, 5.3, 5.5**
    """
    resolver = LinkResolver()
    
    # Create RST content with markdown-style link
    rst_content = f'`Link Text <{md_file}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # Should contain :doc: directive
    assert ':doc:' in transformed, \
        f"Internal doc link '{md_file}' should be transformed to :doc: directive"
    
    # Should not contain the original markdown link syntax
    assert f'<{md_file}>' not in transformed, \
        f"Original markdown link syntax should be removed"
    
    # Verify the link was classified correctly
    links = resolver.detect_links(rst_content)
    if links:
        _, target, link_type = links[0]
        assert link_type == LinkType.INTERNAL_DOC, \
            f"Markdown file link should be classified as INTERNAL_DOC"


@given(section_name())
@settings(max_examples=100)
def test_section_link_produces_ref_directive(section):
    """
    Property 2: Link Resolution Correctness
    
    For any internal section link, conversion should produce :ref: directive.
    
    **Validates: Requirements 5.1, 5.3, 5.5**
    """
    assume(len(section) > 0)
    
    resolver = LinkResolver()
    
    # Create RST content with section link
    rst_content = f'`Link Text <#{section}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # Should contain :ref: directive
    assert ':ref:' in transformed, \
        f"Section link '#{section}' should be transformed to :ref: directive"
    
    # Should not contain the original markdown link syntax with #
    assert f'<#{section}>' not in transformed, \
        f"Original section link syntax should be removed"
    
    # Verify the link was classified correctly
    links = resolver.detect_links(rst_content)
    if links:
        _, target, link_type = links[0]
        assert link_type == LinkType.INTERNAL_REF, \
            f"Section link should be classified as INTERNAL_REF"


@given(external_url())
@settings(max_examples=100)
def test_external_url_preserves_rst_syntax(url):
    """
    Property 2: Link Resolution Correctness
    
    For any external URL, conversion should preserve RST inline link syntax.
    
    **Validates: Requirements 5.1, 5.3, 5.5**
    """
    assume(len(url) > 10)  # Ensure valid URL
    
    resolver = LinkResolver()
    
    # Create RST content with external link
    rst_content = f'`Link Text <{url}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # External links should be preserved in RST inline syntax
    assert f'<{url}>' in transformed, \
        f"External URL should be preserved in RST inline link syntax"
    
    # Should NOT be transformed to :doc: or :ref:
    assert ':doc:' not in transformed, \
        f"External URL should not be transformed to :doc: directive"
    assert ':ref:' not in transformed, \
        f"External URL should not be transformed to :ref: directive"
    
    # Verify the link was classified correctly
    links = resolver.detect_links(rst_content)
    if links:
        _, target, link_type = links[0]
        assert link_type == LinkType.EXTERNAL, \
            f"External URL should be classified as EXTERNAL"


@given(markdown_filename(), section_name())
@settings(max_examples=100)
def test_doc_with_section_link_produces_ref_directive(md_file, section):
    """
    Property 2: Link Resolution Correctness
    
    For any link to a section in another document, conversion should produce :ref: directive.
    
    **Validates: Requirements 5.1, 5.3, 5.5**
    """
    assume(len(section) > 0)
    
    resolver = LinkResolver()
    
    # Create RST content with doc+section link
    rst_content = f'`Link Text <{md_file}#{section}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # Should contain :ref: directive (doc with section is treated as ref)
    assert ':ref:' in transformed, \
        f"Document+section link '{md_file}#{section}' should be transformed to :ref: directive"
    
    # Should not contain the original markdown link syntax
    assert f'<{md_file}#{section}>' not in transformed, \
        f"Original link syntax should be removed"


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=100)
def test_link_detection_finds_all_link_types(link_target):
    """
    Property 2: Link Resolution Correctness
    
    For any link target, detection should correctly classify it.
    
    **Validates: Requirements 5.1, 5.3**
    """
    resolver = LinkResolver()
    
    # Create RST content with the link
    rst_content = f'`Text <{link_target}>`_'
    
    # Detect links
    links = resolver.detect_links(rst_content)
    
    # Should detect at least one link
    assert len(links) >= 1, \
        f"Link detection should find the link in content"
    
    # Verify the detected link has the correct target
    found = False
    for text, target, link_type in links:
        if target == link_target:
            found = True
            # Verify link_type is one of the valid types
            assert isinstance(link_type, LinkType), \
                f"Detected link should have a valid LinkType"
            break
    
    assert found, f"Link target '{link_target}' should be detected"


@given(markdown_filename())
@settings(max_examples=100)
def test_doc_directive_format_is_valid(md_file):
    """
    Property 2: Link Resolution Correctness
    
    For any internal doc link, the generated :doc: directive should have valid format.
    
    **Validates: Requirements 5.1, 5.5**
    """
    resolver = LinkResolver()
    
    # Create RST content with markdown-style link
    rst_content = f'`Link Text <{md_file}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # Extract :doc: directive
    doc_pattern = r':doc:`([^`]+)`'
    matches = re.findall(doc_pattern, transformed)
    
    assert len(matches) > 0, \
        f"Should generate at least one :doc: directive"
    
    for match in matches:
        # Valid format is either "text <path>" or just "path"
        if '<' in match and '>' in match:
            # Format: "text <path>"
            assert match.count('<') == 1 and match.count('>') == 1, \
                f":doc: directive should have valid 'text <path>' format"
        else:
            # Format: just path
            assert len(match) > 0, \
                f":doc: directive should have non-empty path"


@given(section_name())
@settings(max_examples=100)
def test_ref_directive_format_is_valid(section):
    """
    Property 2: Link Resolution Correctness
    
    For any section link, the generated :ref: directive should have valid format.
    
    **Validates: Requirements 5.1, 5.5**
    """
    assume(len(section) > 0)
    
    resolver = LinkResolver()
    
    # Create RST content with section link
    rst_content = f'`Link Text <#{section}>`_'
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
    
    # Extract :ref: directive
    ref_pattern = r':ref:`([^`]+)`'
    matches = re.findall(ref_pattern, transformed)
    
    assert len(matches) > 0, \
        f"Should generate at least one :ref: directive"
    
    for match in matches:
        # Valid format is either "text <label>" or just "label"
        if '<' in match and '>' in match:
            # Format: "text <label>"
            assert match.count('<') == 1 and match.count('>') == 1, \
                f":ref: directive should have valid 'text <label>' format"
            # Extract label
            label_match = re.search(r'<([^>]+)>', match)
            if label_match:
                label = label_match.group(1)
                # Label should be lowercase and use hyphens
                assert label.islower() or '-' in label, \
                    f":ref: label should be lowercase with hyphens"
        else:
            # Format: just label
            assert len(match) > 0, \
                f":ref: directive should have non-empty label"


@given(st.lists(markdown_filename(), min_size=1, max_size=5))
@settings(max_examples=50)
def test_multiple_links_all_transformed(md_files):
    """
    Property 2: Link Resolution Correctness
    
    For any content with multiple links, all should be transformed correctly.
    
    **Validates: Requirements 5.1, 5.3, 5.5**
    """
    resolver = LinkResolver()
    
    # Create RST content with multiple links
    links_content = '\n'.join([f'`Link {i} <{f}>`_' for i, f in enumerate(md_files)])
    
    # Resolve links
    transformed, warnings = resolver.resolve_links(links_content, "test.rst")
    
    # Count :doc: directives in output
    doc_count = transformed.count(':doc:')
    
    # Should have transformed all markdown file links
    assert doc_count == len(md_files), \
        f"All {len(md_files)} internal doc links should be transformed to :doc: directives"
    
    # Original markdown syntax should be removed
    for md_file in md_files:
        assert f'<{md_file}>' not in transformed, \
            f"Original markdown link syntax for '{md_file}' should be removed"
