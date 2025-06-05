# Configuration file for the Sphinx TEST_PROJECT builder.
#
# For the full list of built-in configuration values, see the TEST_PROJECT:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "oai-dbt-generator"
copyright = "2025, British Airways"  # pylint: disable=[redefined-builtin]
author = "akatsipanos"
release = "0.01"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_rtd_theme"
html_logo = "assets/ba_logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
