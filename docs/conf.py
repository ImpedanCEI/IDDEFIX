"""Sphinx configuration for the user guide, API reference, and saved notebooks."""

import subprocess
import sys
from pathlib import Path

# Import the package from this checkout
DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR.parent))
from iddefix import __version__  # noqa: E402

# Copy published notebooks using the same Python environment
subprocess.run([sys.executable, str(DOCS_DIR / "copy_notebooks.py")], check=True)

project = "iddefix"
copyright = "2025, CERN, BE-ABP-CEI"
author = "Sebastien Joly, Malthe Raschke, Elena de la Fuente"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_design",
    "sphinx_last_updated_by_git",
]

# Include constructor parameters and use NumPy-style docstrings
autodoc_preserve_defaults = True
autoclass_content = "both"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto")
nb_execution_mode = "off"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"
pygments_style = "sphinx"

# The logo is copied by Sphinx; the source directory is not a static asset folder
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_logo = "logo.png"
html_theme_options = {"logo_only": True}
htmlhelp_basename = "iddefix"
