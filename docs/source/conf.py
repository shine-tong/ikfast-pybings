# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
import sphinx_rtd_theme

# Mock imports for C extension modules when building docs
# This allows Sphinx to build documentation without compiling C++ extensions
autodoc_mock_imports = ['ikfast_pybind._ikfast_pybind']


project = 'IKFast_Pybind'
copyright = '2026, shine-tong'
author = 'shine-tong'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Auto-generate API docs from docstrings
    'sphinx.ext.intersphinx',  # Link to other Sphinx documentation
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.napoleon',     # Support Google/NumPy docstring styles
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Code highlighting configuration -----------------------------------------
pygments_style = 'sphinx'
highlight_language = 'python'

# -- Bilingual support configuration -----------------------------------------
locale_dirs = ['locale/']
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -- Theme options -----------------------------------------------------------
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = ['_static']