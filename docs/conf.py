# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
import os

# sys.path.insert(0, os.path.abspath('../iddefix/'))
sys.path.append(os.path.abspath(".."))

# Copy notebooks
import subprocess
import os

# Run the copy_notebooks.py script before building the docs
copy_script = os.path.join(os.path.dirname(__file__), "copy_notebooks.py")
subprocess.run(["python", copy_script], check=True)

# -- Project information -----------------------------------------------------

project = "iddefix"
copyright = "2025, CERN, BE-ABP-CEI"
author = "Sebastien Joly, Malthe Raschke, Elena de la Fuente"

# The full version, including alpha/beta/rc tags
release = "1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # for eqs
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    #'sphinxemoji',
    "myst_nb",  # for markdown and ipynb
    "sphinx_design",
    "sphinx_last_updated_by_git",
]

autodoc_preserve_defaults = True  # preserves default args as in source code

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")
# myst_enable_extensions = ["deflist", "dollarmath"]
nb_execution_mode = "off"

# The suffix of source filenames.

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_static_path = ["."]
html_logo = "logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = "ReadtheDocsTemplatedoc"
