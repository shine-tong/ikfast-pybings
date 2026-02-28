"""
Unit tests for file organization manager.

Tests directory creation, file naming conventions, and bilingual file pairing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from migration_tool.file_organizer import (
    FileOrganizer,
    FileMapping,
    OrganizationError
)


@pytest.fixture
def temp_docs_root():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def organizer(temp_docs_root):
    """Create a FileOrganizer instance with temporary root."""
    return FileOrganizer(docs_root=temp_docs_root)


class TestDirectoryCreation:
    """Test directory structure creation."""
    
    def test_create_directory_structure(self, organizer, temp_docs_root):
        """Test that all required directories are created."""
        created_dirs = organizer.create_directory_structure()
        
        # Check that directories were created
        assert len(created_dirs) > 0
        
        # Verify each required directory exists
        root = Path(temp_docs_root)
        assert root.exists()
        assert (root / "guides").exists()
        assert (root / "api").exists()
        assert (root / "examples").exists()
        assert (root / "_static").exists()
        assert (root / "_templates").exists()
    
    def test_create_directory_structure_idempotent(self, organizer):
        """Test that creating directories multiple times doesn't fail."""
        # Create once
        organizer.create_directory_structure()
        
        # Create again - should not raise error
        created_dirs = organizer.create_directory_structure()
        assert len(created_dirs) > 0
    
    def test_directory_structure_returns_paths(self, organizer, temp_docs_root):
        """Test that created directory paths are returned."""
        created_dirs = organizer.create_directory_structure()
        
        # Check that returned paths are Path objects or can be converted
        for dir_path in created_dirs:
            assert Path(dir_path).exists()


class TestFileNamingConventions:
    """Test file naming conventions and mapping."""
    
    def test_readme_mapping(self, organizer, temp_docs_root):
        """Test README.md maps to introduction.rst."""
        mapping = organizer.map_source_to_target("README.md")
        
        assert mapping.source_file == "README.md"
        assert mapping.target_file.endswith("introduction.rst")
        assert not mapping.is_bilingual
        assert mapping.language == "en"
    
    def test_readme_cn_mapping(self, organizer, temp_docs_root):
        """Test README_CN.md maps to introduction_cn.rst."""
        mapping = organizer.map_source_to_target("README_CN.md")
        
        assert mapping.source_file == "README_CN.md"
        assert mapping.target_file.endswith("introduction_cn.rst")
        assert mapping.is_bilingual
        assert mapping.language == "cn"
    
    def test_build_mapping(self, organizer, temp_docs_root):
        """Test BUILD.md maps to guides/building.rst."""
        mapping = organizer.map_source_to_target("BUILD.md")
        
        assert mapping.source_file == "BUILD.md"
        assert "guides" in mapping.target_file
        assert mapping.target_file.endswith("building.rst")
        assert not mapping.is_bilingual
        assert mapping.language == "en"
    
    def test_build_cn_mapping(self, organizer, temp_docs_root):
        """Test BUILD_CN.md maps to guides/building_cn.rst."""
        mapping = organizer.map_source_to_target("BUILD_CN.md")
        
        assert mapping.source_file == "BUILD_CN.md"
        assert "guides" in mapping.target_file
        assert mapping.target_file.endswith("building_cn.rst")
        assert mapping.is_bilingual
        assert mapping.language == "cn"
    
    def test_custom_solver_mapping(self, organizer, temp_docs_root):
        """Test CUSTOM_SOLVER.md maps to guides/custom_solver.rst."""
        mapping = organizer.map_source_to_target("CUSTOM_SOLVER.md")
        
        assert mapping.source_file == "CUSTOM_SOLVER.md"
        assert "guides" in mapping.target_file
        assert mapping.target_file.endswith("custom_solver.rst")
        assert not mapping.is_bilingual
        assert mapping.language == "en"
    
    def test_custom_solver_cn_mapping(self, organizer, temp_docs_root):
        """Test CUSTOM_SOLVER_CN.md maps to guides/custom_solver_cn.rst."""
        mapping = organizer.map_source_to_target("CUSTOM_SOLVER_CN.md")
        
        assert mapping.source_file == "CUSTOM_SOLVER_CN.md"
        assert "guides" in mapping.target_file
        assert mapping.target_file.endswith("custom_solver_cn.rst")
        assert mapping.is_bilingual
        assert mapping.language == "cn"
    
    def test_unknown_file_raises_error(self, organizer):
        """Test that unknown source files raise OrganizationError."""
        with pytest.raises(OrganizationError) as exc_info:
            organizer.map_source_to_target("UNKNOWN.md")
        
        assert "Unknown source file" in str(exc_info.value)
    
    def test_case_insensitive_mapping(self, organizer, temp_docs_root):
        """Test that file mapping is case-insensitive."""
        # Test lowercase
        mapping_lower = organizer.map_source_to_target("readme.md")
        assert mapping_lower.target_file.endswith("introduction.rst")
        
        # Test mixed case
        mapping_mixed = organizer.map_source_to_target("ReAdMe.md")
        assert mapping_mixed.target_file.endswith("introduction.rst")


class TestBilingualFilePairing:
    """Test bilingual file pairing functionality."""
    
    def test_english_file_has_chinese_pair(self, organizer):
        """Test that English files return their Chinese pair."""
        en_mapping = organizer.map_source_to_target("README.md")
        cn_pair = organizer.get_bilingual_pair(en_mapping)
        
        assert cn_pair is not None
        assert cn_pair.endswith("introduction_cn.rst")
    
    def test_chinese_file_has_english_pair(self, organizer):
        """Test that Chinese files return their English pair."""
        cn_mapping = organizer.map_source_to_target("README_CN.md")
        en_pair = organizer.get_bilingual_pair(cn_mapping)
        
        assert en_pair is not None
        assert en_pair.endswith("introduction.rst")
        assert "_cn" not in en_pair
    
    def test_guides_bilingual_pairing(self, organizer):
        """Test bilingual pairing for guide files."""
        # English BUILD.md
        en_mapping = organizer.map_source_to_target("BUILD.md")
        cn_pair = organizer.get_bilingual_pair(en_mapping)
        assert cn_pair is not None
        assert "building_cn.rst" in cn_pair
        
        # Chinese BUILD_CN.md
        cn_mapping = organizer.map_source_to_target("BUILD_CN.md")
        en_pair = organizer.get_bilingual_pair(cn_mapping)
        assert en_pair is not None
        assert "building.rst" in en_pair
        assert "_cn" not in en_pair


class TestSectionIndexCreation:
    """Test section index file creation."""
    
    def test_create_guides_index(self, organizer):
        """Test creation of guides section index."""
        files = ["building", "building_cn", "custom_solver", "custom_solver_cn"]
        index_content = organizer.create_section_index(
            "guides",
            files,
            "User Guides"
        )
        
        assert "User Guides" in index_content
        assert ".. toctree::" in index_content
        assert ":maxdepth:" in index_content
        assert "building" in index_content
        assert "custom_solver" in index_content
    
    def test_create_api_index(self, organizer):
        """Test creation of API section index."""
        files = ["high_level", "low_level", "classes"]
        index_content = organizer.create_section_index(
            "api",
            files,
            "API Reference"
        )
        
        assert "API Reference" in index_content
        assert ".. toctree::" in index_content
        assert "high_level" in index_content
        assert "low_level" in index_content
        assert "classes" in index_content
    
    def test_create_empty_index(self, organizer):
        """Test creation of index with no files."""
        index_content = organizer.create_section_index(
            "examples",
            [],
            "Examples"
        )
        
        assert "Examples" in index_content
        assert ".. toctree::" in index_content
    
    def test_index_title_underline_length(self, organizer):
        """Test that title underline matches title length."""
        title = "User Guides"
        index_content = organizer.create_section_index("guides", [], title)
        
        lines = index_content.split("\n")
        assert lines[0] == title
        assert lines[1] == "=" * len(title)


class TestFileOrganization:
    """Test complete file organization workflow."""
    
    def test_organize_single_file(self, organizer, temp_docs_root):
        """Test organizing a single converted file."""
        converted_files = {
            "README.md": "Introduction\n============\n\nContent here."
        }
        
        organized = organizer.organize_files(converted_files, create_indexes=False)
        
        assert len(organized) == 1
        target_file = list(organized.keys())[0]
        assert target_file.endswith("introduction.rst")
        assert organized[target_file] == converted_files["README.md"]
    
    def test_organize_multiple_files(self, organizer, temp_docs_root):
        """Test organizing multiple converted files."""
        converted_files = {
            "README.md": "Introduction content",
            "README_CN.md": "介绍内容",
            "BUILD.md": "Build instructions",
            "BUILD_CN.md": "构建说明"
        }
        
        organized = organizer.organize_files(converted_files, create_indexes=False)
        
        # Should have 4 files
        assert len(organized) == 4
        
        # Check that all files are mapped
        target_files = list(organized.keys())
        assert any("introduction.rst" in f for f in target_files)
        assert any("introduction_cn.rst" in f for f in target_files)
        assert any("building.rst" in f for f in target_files)
        assert any("building_cn.rst" in f for f in target_files)
    
    def test_organize_with_indexes(self, organizer, temp_docs_root):
        """Test organizing files with automatic index creation."""
        converted_files = {
            "BUILD.md": "Build instructions",
            "CUSTOM_SOLVER.md": "Custom solver guide"
        }
        
        organized = organizer.organize_files(converted_files, create_indexes=True)
        
        # Should have original files plus index files
        assert len(organized) > 2
        
        # Check for index files
        target_files = list(organized.keys())
        assert any("guides/index.rst" in f for f in target_files)
        assert any("api/index.rst" in f for f in target_files)
        assert any("examples/index.rst" in f for f in target_files)
    
    def test_organize_creates_directories(self, organizer, temp_docs_root):
        """Test that organize_files creates directory structure."""
        converted_files = {
            "README.md": "Content"
        }
        
        organizer.organize_files(converted_files)
        
        # Verify directories were created
        root = Path(temp_docs_root)
        assert (root / "guides").exists()
        assert (root / "api").exists()
        assert (root / "examples").exists()
    
    def test_organize_preserves_content(self, organizer, temp_docs_root):
        """Test that file content is preserved during organization."""
        test_content = "Test Content\n============\n\nThis is a test."
        converted_files = {
            "README.md": test_content
        }
        
        organized = organizer.organize_files(converted_files, create_indexes=False)
        
        # Content should be unchanged
        target_file = list(organized.keys())[0]
        assert organized[target_file] == test_content
    
    def test_organize_guides_index_includes_files(self, organizer, temp_docs_root):
        """Test that guides index includes all guide files."""
        converted_files = {
            "BUILD.md": "Build content",
            "BUILD_CN.md": "构建内容",
            "CUSTOM_SOLVER.md": "Solver content"
        }
        
        organized = organizer.organize_files(converted_files, create_indexes=True)
        
        # Find guides index
        guides_index = None
        for path, content in organized.items():
            if "guides/index.rst" in path:
                guides_index = content
                break
        
        assert guides_index is not None
        assert "building" in guides_index
        assert "custom_solver" in guides_index


class TestErrorHandling:
    """Test error handling in file organization."""
    
    def test_invalid_source_file(self, organizer):
        """Test that invalid source files raise appropriate errors."""
        with pytest.raises(OrganizationError):
            organizer.map_source_to_target("invalid_file.txt")
    
    def test_organize_with_invalid_file(self, organizer):
        """Test that organizing with invalid files raises errors."""
        converted_files = {
            "INVALID.md": "Content"
        }
        
        with pytest.raises(OrganizationError):
            organizer.organize_files(converted_files)
