"""
Unit tests for link edge cases in link resolution.

Tests broken link detection, circular reference handling, and external link preservation.

Validates: Requirements 5.4
"""

import pytest
from migration_tool.link_resolver import LinkResolver, LinkType, LinkMapping
from pathlib import Path


class TestBrokenLinkDetection:
    """Test detection of broken links."""
    
    def test_detect_broken_internal_doc_link(self):
        """Test detection of broken internal document link."""
        resolver = LinkResolver()
        
        # Create a link to non-existent document
        rst_content = '`Link <nonexistent.md>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should transform to :doc: directive
        assert ':doc:' in transformed
        
        # Validate links - should report broken link
        all_files = {'test.rst': 'docs/source/test.rst'}
        validation_warnings = resolver.validate_links(all_files)
        
        # Should have at least one warning about broken link
        assert len(validation_warnings) > 0
        assert any('not found' in w.lower() for w in validation_warnings)
    
    def test_detect_broken_download_link(self):
        """Test detection of broken download file link."""
        resolver = LinkResolver()
        
        # Create a download link to non-existent file
        rst_content = ':download:`File <nonexistent_file.py>`'
        transformed, warnings = resolver.resolve_links(rst_content, "docs/source/test.rst")
        
        # Validate links - should report broken download
        all_files = {}
        validation_warnings = resolver.validate_links(all_files)
        
        # Should have warning about missing file
        assert len(validation_warnings) > 0
        assert any('not found' in w.lower() for w in validation_warnings)
    
    def test_valid_link_no_warning(self, tmp_path):
        """Test that valid links don't generate warnings."""
        resolver = LinkResolver(docs_root=str(tmp_path))
        
        # Create a valid target file
        target_file = tmp_path / "introduction.rst"
        target_file.write_text("# Introduction")
        
        # Create link to existing document
        rst_content = '`Link <README.md>`_'
        resolver.build_path_mapping({'README.md': str(target_file)})
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Validate links
        all_files = {'README.md': str(target_file)}
        validation_warnings = resolver.validate_links(all_files)
        
        # Should have no warnings for valid link
        assert len(validation_warnings) == 0
    
    def test_multiple_broken_links_all_detected(self):
        """Test that multiple broken links are all detected."""
        resolver = LinkResolver()
        
        # Create content with multiple broken links
        rst_content = '''
        `Link 1 <file1.md>`_
        `Link 2 <file2.md>`_
        `Link 3 <file3.md>`_
        '''
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Validate links
        all_files = {}
        validation_warnings = resolver.validate_links(all_files)
        
        # Should detect all three broken links
        assert len(validation_warnings) >= 3


class TestCircularReferenceHandling:
    """Test handling of circular references."""
    
    def test_self_reference_allowed(self):
        """Test that a document can reference itself (for sections)."""
        resolver = LinkResolver()
        
        # Create self-reference to section
        rst_content = '`Section <#introduction>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should transform to :ref: directive
        assert ':ref:' in transformed
        
        # Should not generate warnings
        assert len(warnings) == 0
    
    def test_bidirectional_references_allowed(self):
        """Test that bidirectional references between documents are allowed."""
        resolver = LinkResolver()
        
        # Build path mapping for two documents
        resolver.build_path_mapping({
            'doc1.md': 'docs/source/doc1.rst',
            'doc2.md': 'docs/source/doc2.rst'
        })
        
        # Doc1 references Doc2
        rst_content1 = '`Link to Doc2 <doc2.md>`_'
        transformed1, warnings1 = resolver.resolve_links(rst_content1, "docs/source/doc1.rst")
        
        # Doc2 references Doc1
        rst_content2 = '`Link to Doc1 <doc1.md>`_'
        transformed2, warnings2 = resolver.resolve_links(rst_content2, "docs/source/doc2.rst")
        
        # Both should transform successfully
        assert ':doc:' in transformed1
        assert ':doc:' in transformed2
        
        # No warnings for bidirectional references
        assert len(warnings1) == 0
        assert len(warnings2) == 0
    
    def test_circular_chain_references_allowed(self):
        """Test that circular chains (A->B->C->A) are allowed."""
        resolver = LinkResolver()
        
        # Build path mapping for three documents
        resolver.build_path_mapping({
            'doc1.md': 'docs/source/doc1.rst',
            'doc2.md': 'docs/source/doc2.rst',
            'doc3.md': 'docs/source/doc3.rst'
        })
        
        # Create circular chain: doc1 -> doc2 -> doc3 -> doc1
        rst_content1 = '`Link <doc2.md>`_'
        rst_content2 = '`Link <doc3.md>`_'
        rst_content3 = '`Link <doc1.md>`_'
        
        transformed1, warnings1 = resolver.resolve_links(rst_content1, "docs/source/doc1.rst")
        transformed2, warnings2 = resolver.resolve_links(rst_content2, "docs/source/doc2.rst")
        transformed3, warnings3 = resolver.resolve_links(rst_content3, "docs/source/doc3.rst")
        
        # All should transform successfully
        assert ':doc:' in transformed1
        assert ':doc:' in transformed2
        assert ':doc:' in transformed3
        
        # No warnings - circular references are allowed in documentation
        assert len(warnings1) == 0
        assert len(warnings2) == 0
        assert len(warnings3) == 0


class TestExternalLinkPreservation:
    """Test preservation of external links."""
    
    def test_http_link_preserved(self):
        """Test that HTTP links are preserved."""
        resolver = LinkResolver()
        
        rst_content = '`Example <http://example.com>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the link as-is
        assert 'http://example.com' in transformed
        
        # Should not transform to :doc: or :ref:
        assert ':doc:' not in transformed
        assert ':ref:' not in transformed
    
    def test_https_link_preserved(self):
        """Test that HTTPS links are preserved."""
        resolver = LinkResolver()
        
        rst_content = '`GitHub <https://github.com/user/repo>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the link
        assert 'https://github.com/user/repo' in transformed
        
        # Should not transform to directives
        assert ':doc:' not in transformed
        assert ':ref:' not in transformed
    
    def test_mailto_link_preserved(self):
        """Test that mailto links are preserved."""
        resolver = LinkResolver()
        
        rst_content = '`Email <mailto:user@example.com>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the mailto link
        assert 'mailto:user@example.com' in transformed
    
    def test_ftp_link_preserved(self):
        """Test that FTP links are preserved."""
        resolver = LinkResolver()
        
        rst_content = '`FTP <ftp://ftp.example.com/file.txt>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the FTP link
        assert 'ftp://ftp.example.com/file.txt' in transformed
    
    def test_mixed_internal_external_links(self):
        """Test content with both internal and external links."""
        resolver = LinkResolver()
        
        rst_content = '''
        `Internal <README.md>`_
        `External <https://example.com>`_
        `Section <#intro>`_
        '''
        
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Internal doc link should be transformed
        assert ':doc:' in transformed
        
        # Section link should be transformed
        assert ':ref:' in transformed
        
        # External link should be preserved
        assert 'https://example.com' in transformed
    
    def test_external_link_with_query_params(self):
        """Test external links with query parameters are preserved."""
        resolver = LinkResolver()
        
        url = 'https://example.com/search?q=test&lang=en'
        rst_content = f'`Search <{url}>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the full URL with query params
        assert url in transformed
    
    def test_external_link_with_fragment(self):
        """Test external links with fragments are preserved."""
        resolver = LinkResolver()
        
        url = 'https://example.com/page#section'
        rst_content = f'`Link <{url}>`_'
        transformed, warnings = resolver.resolve_links(rst_content, "test.rst")
        
        # Should preserve the full URL with fragment
        assert url in transformed


class TestLinkClassification:
    """Test link classification logic."""
    
    def test_classify_markdown_file_as_internal_doc(self):
        """Test that .md files are classified as internal doc."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('README.md')
        assert link_type == LinkType.INTERNAL_DOC
    
    def test_classify_rst_file_as_internal_doc(self):
        """Test that .rst files are classified as internal doc."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('guide.rst')
        assert link_type == LinkType.INTERNAL_DOC
    
    def test_classify_section_as_internal_ref(self):
        """Test that #section is classified as internal ref."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('#introduction')
        assert link_type == LinkType.INTERNAL_REF
    
    def test_classify_http_as_external(self):
        """Test that HTTP URLs are classified as external."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('http://example.com')
        assert link_type == LinkType.EXTERNAL
    
    def test_classify_https_as_external(self):
        """Test that HTTPS URLs are classified as external."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('https://example.com')
        assert link_type == LinkType.EXTERNAL
    
    def test_classify_python_file_as_download(self):
        """Test that .py files are classified as download."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('examples/example.py')
        assert link_type == LinkType.DOWNLOAD
    
    def test_classify_json_file_as_download(self):
        """Test that .json files are classified as download."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('config.json')
        assert link_type == LinkType.DOWNLOAD
    
    def test_classify_relative_path_as_internal_doc(self):
        """Test that relative paths without extension are classified as internal doc."""
        resolver = LinkResolver()
        
        link_type = resolver._classify_link('guides/building')
        assert link_type == LinkType.INTERNAL_DOC


class TestLinkDetection:
    """Test link detection in RST content."""
    
    def test_detect_inline_link(self):
        """Test detection of inline RST link."""
        resolver = LinkResolver()
        
        rst_content = '`Link Text <target.md>`_'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 1
        text, target, link_type = links[0]
        assert text == 'Link Text'
        assert target == 'target.md'
    
    def test_detect_doc_directive(self):
        """Test detection of :doc: directive."""
        resolver = LinkResolver()
        
        rst_content = ':doc:`introduction`'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 1
        text, target, link_type = links[0]
        assert target == 'introduction'
        assert link_type == LinkType.INTERNAL_DOC
    
    def test_detect_ref_directive(self):
        """Test detection of :ref: directive."""
        resolver = LinkResolver()
        
        rst_content = ':ref:`section-label`'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 1
        text, target, link_type = links[0]
        assert target == 'section-label'
        assert link_type == LinkType.INTERNAL_REF
    
    def test_detect_download_directive(self):
        """Test detection of :download: directive."""
        resolver = LinkResolver()
        
        rst_content = ':download:`File <example.py>`'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 1
        text, target, link_type = links[0]
        assert text == 'File'
        assert target == 'example.py'
        assert link_type == LinkType.DOWNLOAD
    
    def test_detect_image_directive(self):
        """Test detection of image directive."""
        resolver = LinkResolver()
        
        rst_content = '.. image:: images/logo.png'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 1
        text, target, link_type = links[0]
        assert target == 'images/logo.png'
    
    def test_detect_multiple_links(self):
        """Test detection of multiple links in content."""
        resolver = LinkResolver()
        
        rst_content = '''
        `Link 1 <file1.md>`_
        :doc:`file2`
        :ref:`section`
        `External <https://example.com>`_
        '''
        links = resolver.detect_links(rst_content)
        
        # Should detect all 4 links
        assert len(links) == 4
    
    def test_detect_no_links_in_plain_text(self):
        """Test that plain text without links returns empty list."""
        resolver = LinkResolver()
        
        rst_content = 'This is plain text without any links.'
        links = resolver.detect_links(rst_content)
        
        assert len(links) == 0


class TestPathMapping:
    """Test path mapping functionality."""
    
    def test_build_path_mapping(self):
        """Test building path mapping from source to target."""
        resolver = LinkResolver()
        
        source_files = {
            'README.md': 'docs/source/introduction.rst',
            'BUILD.md': 'docs/source/guides/building.rst'
        }
        
        resolver.build_path_mapping(source_files)
        
        # Should have mappings
        assert len(resolver.path_mappings) > 0
        
        # Should include both with and without extensions
        assert 'README.md' in resolver.path_mappings
        assert 'README' in resolver.path_mappings
    
    def test_path_mapping_normalization(self):
        """Test that paths are normalized in mapping."""
        resolver = LinkResolver()
        
        source_files = {
            'README.md': 'docs/source/introduction.rst'
        }
        
        resolver.build_path_mapping(source_files)
        
        # Paths should use forward slashes
        for path in resolver.path_mappings.values():
            assert '\\' not in path
    
    def test_get_link_mappings(self):
        """Test retrieving link mappings."""
        resolver = LinkResolver()
        
        rst_content = '`Link <README.md>`_'
        resolver.resolve_links(rst_content, "test.rst")
        
        mappings = resolver.get_link_mappings()
        
        # Should have at least one mapping
        assert len(mappings) > 0
        
        # Mappings should be LinkMapping objects
        assert all(isinstance(m, LinkMapping) for m in mappings)
