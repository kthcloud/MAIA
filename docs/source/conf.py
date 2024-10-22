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
import os
import subprocess
import sys

import MAIA

sys.path.insert(0, os.path.abspath("../../MAIA_scripts"))
sys.path.insert(0, os.path.abspath("../../MAIA"))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print(sys.path)

# -- Project information -----------------------------------------------------

project = "MAIA"
copyright = "2023, Simone Bendazzoli"
author = "Simone Bendazzoli"

# The full version, including alpha/beta/rc tags
release = MAIA.__version__

exclude_patterns = ["configs"]


def generate_apidocs(*args):
    """Generate API docs automatically by trawling the available modules"""
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "MAIA"))
    scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "MAIA_scripts"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "apidocs"))
    apidoc_command_path = "sphinx-apidoc"
    if hasattr(sys, "real_prefix"):  # called from a virtualenv
        apidoc_command_path = os.path.join(sys.prefix, "bin", "sphinx-apidoc")
        apidoc_command_path = os.path.abspath(apidoc_command_path)
    print(f"output_path {output_path}")
    print(f"scripts_path {scripts_path}")
    print(f"module_path {module_path}")
    print(f"command_path {apidoc_command_path}")
    subprocess.check_call(
        [apidoc_command_path, "-e"]
        + ["-o", output_path]
        + [module_path]
        + [os.path.join(module_path, p) for p in exclude_patterns]
    )
    #subprocess.check_call([apidoc_command_path, "-e"] + ["-o", output_path] + [scripts_path])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
nbsphinx_execute = 'never'
extensions = [
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxcontrib.contentui",
    "sphinx.ext.autosectionlabel",
    "sphinx-jsonschema",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
]


# source_suffix = '.rst'
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autoclass_content = "both"
add_module_names = True
source_encoding = "utf-8"
autosectionlabel_prefix_document = True
napoleon_use_param = True
napoleon_include_init_with_doc = True
set_type_checking_flag = True
napoleon_use_rtype = False

# always_document_param_types = False
# set_type_checking_flag = False

pygments_style = "sphinx"


def setup(app):
    ...
    # Hook to allow for automatic generation of API docs
    # before doc deployment begins.
    app.connect("builder-inited", generate_apidocs)






