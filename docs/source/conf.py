# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "final_challenge2025"
copyright = "2025, rss-2025-team4"
author = "rss-2025-team4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinxcontrib.restbuilder",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}

autodoc_class_signature = "separated"


intersphinx_mapping = {
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

autodoc_type_aliases = {
    "Line": "Line",
    "Point": "Point",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
