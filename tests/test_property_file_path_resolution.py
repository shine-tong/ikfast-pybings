"""
Property-based tests for file path resolution.

Feature: sphinx-docs-migration
Property 5: File Path Resolution

For any file path reference in the documentation (to examples, source files, or other resources),
the converted RST should contain a path that correctly resolves to the target file from the new
location in docs/source.

Validates: Requirements 5.2
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from migration_tool.link_resolver import LinkResolver
from pathlib import Path
import re


# Strategy for generating file paths
@st.composite
def file_path(draw):
    """Generate valid file paths."""
    directories = ['examples', 'src', 'tests', 'docs', 'migration_tool']
    filenames = ['example.py', 'test.py', 'config.json', 'data.csv', 'README.md']
    
    # Sometimes include subdirectory
    use_subdir = draw(st.booleans())
    if use_subdir:
        dir1 = draw(st.sampled_from(directories))
        dir2 = draw(st.sampled_from(['subdir', 'nested', 'utils']))
        filename = draw(st.sampled_from(filenames))
        return f"{dir1}/{dir2}/{filename}"
    else:
        directory = draw(st.sampled_from(directories))
        filename = draw(st.sampled_from(filenames))
        return f"{directory}/{filename}"


# Strategy for generating RST file locations in docs/source
@st.composite
def rst_file_location(draw):
    """Generate RST file locations in docs/source."""
    locations = [
        'docs/source/introduction.rst',
        'docs/source/guides/building.rst',
        'docs/source/guides/custom_solver.rst',
        'docs/source/api/high_level.rst',
        'docs/source/examples/basic_ik.rst'
    ]
    return draw(st.sampled_from(locations))


@given(file_path(), rst_file_location())
@settings(max_examples=100)
def test_file_path_resolution_adds_correct_prefix(original_path, rst_location):
    """
    Property 5: File Path Resolution
    
    For any file path reference, the resolved path should include correct ../ prefix
    to navigate from docs/source to project root.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(original_path, rst_location)
    
    # Resolved path should start with ../ to go back to root
    assert resolved_path.startswith('../'), \
        f"Resolved path should start with ../ to navigate from docs/source"
    
    # Should contain the original path
    assert original_path in resolved_path, \
        f"Resolved path should contain the original path '{original_path}'"


@given(file_path(), rst_file_location())
@settings(max_examples=100)
def test_file_path_depth_matches_location(original_path, rst_location):
    """
    Property 5: File Path Resolution
    
    For any file path, the number of ../ prefixes should match the depth of the RST file
    location in docs/source.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(original_path, rst_location)
    
    # Count ../ in resolved path
    parent_count = resolved_path.count('../')
    
    # Calculate expected depth
    # Files in docs/source need 2 levels (docs, source)
    # Files in docs/source/guides need 3 levels (docs, source, guides)
    if rst_location.startswith('docs/source/'):
        relative_part = rst_location[len('docs/source/'):]
        expected_depth = len(Path(relative_part).parent.parts) + 2
    else:
        expected_depth = 2
    
    assert parent_count == expected_depth, \
        f"Path depth {parent_count} should match location depth {expected_depth} for {rst_location}"


@given(file_path())
@settings(max_examples=100)
def test_absolute_paths_unchanged(original_path):
    """
    Property 5: File Path Resolution
    
    For any absolute path, it should remain unchanged.
    
    **Validates: Requirements 5.2**
    """
    # Make it an absolute path
    absolute_path = '/' + original_path
    
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(absolute_path, "docs/source/test.rst")
    
    # Absolute paths should not be modified
    assert resolved_path == absolute_path, \
        f"Absolute path should remain unchanged"


@given(st.sampled_from(['http://example.com/file.py', 'https://github.com/repo/file.py']))
@settings(max_examples=50)
def test_url_paths_unchanged(url_path):
    """
    Property 5: File Path Resolution
    
    For any URL path, it should remain unchanged.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(url_path, "docs/source/test.rst")
    
    # URLs should not be modified
    assert resolved_path == url_path, \
        f"URL path should remain unchanged"


@given(file_path(), rst_file_location())
@settings(max_examples=100)
def test_resolved_path_is_relative(original_path, rst_location):
    """
    Property 5: File Path Resolution
    
    For any relative file path, the resolved path should also be relative (not absolute).
    
    **Validates: Requirements 5.2**
    """
    # Ensure original path is relative
    assume(not original_path.startswith('/'))
    assume(not original_path.startswith('http'))
    
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(original_path, rst_location)
    
    # Resolved path should be relative
    assert not resolved_path.startswith('/'), \
        f"Resolved path should be relative, not absolute"
    
    # Should not be a URL
    assert '://' not in resolved_path, \
        f"Resolved path should not be a URL"


@given(file_path())
@settings(max_examples=100)
def test_path_resolution_preserves_filename(original_path):
    """
    Property 5: File Path Resolution
    
    For any file path, the resolved path should preserve the original filename.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Get the filename from original path
    original_filename = Path(original_path).name
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(original_path, "docs/source/test.rst")
    
    # Resolved path should end with the same filename
    assert resolved_path.endswith(original_filename), \
        f"Resolved path should preserve filename '{original_filename}'"


@given(file_path(), rst_file_location())
@settings(max_examples=100)
def test_path_resolution_uses_forward_slashes(original_path, rst_location):
    """
    Property 5: File Path Resolution
    
    For any file path, the resolved path should use forward slashes (POSIX style).
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Transform the file path
    resolved_path = resolver.transform_file_path(original_path, rst_location)
    
    # Should not contain backslashes
    assert '\\' not in resolved_path, \
        f"Resolved path should use forward slashes, not backslashes"


@given(st.lists(file_path(), min_size=1, max_size=5), rst_file_location())
@settings(max_examples=50)
def test_multiple_paths_all_resolved_consistently(file_paths, rst_location):
    """
    Property 5: File Path Resolution
    
    For any set of file paths from the same RST location, all should be resolved
    with the same depth prefix.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Transform all file paths
    resolved_paths = [
        resolver.transform_file_path(path, rst_location)
        for path in file_paths
    ]
    
    # Count ../ in each resolved path
    parent_counts = [path.count('../') for path in resolved_paths]
    
    # All should have the same depth
    assert len(set(parent_counts)) == 1, \
        f"All paths from same location should have same depth prefix"


@given(file_path())
@settings(max_examples=100)
def test_download_directive_path_resolution(original_path):
    """
    Property 5: File Path Resolution
    
    For any file path in a :download: directive, resolution should work correctly.
    
    **Validates: Requirements 5.2**
    """
    resolver = LinkResolver()
    
    # Create RST content with download directive
    rst_content = f':download:`File <{original_path}>`'
    rst_location = "docs/source/guides/test.rst"
    
    # Resolve links (which includes path resolution)
    transformed, warnings = resolver.resolve_links(rst_content, rst_location)
    
    # Should still contain :download: directive
    assert ':download:' in transformed, \
        f"Download directive should be preserved"
    
    # Should contain a resolved path with ../
    assert '../' in transformed, \
        f"Download directive should contain resolved path with ../"
    
    # Original path should not appear unchanged (unless it was absolute/URL)
    if not original_path.startswith(('/', 'http')):
        # The path should have been modified
        assert f'<{original_path}>' not in transformed, \
            f"Original relative path should be transformed"
