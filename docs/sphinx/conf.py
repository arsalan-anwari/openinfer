import os
import sys
import shutil

project = "OpenInfer"
copyright = "Lucidy 2026 -- www.lucidy.site"
author = "OpenInfer Contributors"
version = "0.1"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib_rust",
    "sphinxcontrib.mermaid",
    "revitron_sphinx_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "out", "api/rustdocgen"]

import revitron_sphinx_theme

html_theme = "revitron_sphinx_theme"
html_theme_path = [revitron_sphinx_theme.get_html_theme_path()]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/mobile_nav.js", "js/copyright_override.js"]
html_logo = "_static/images/OpenInferIcon.png"
html_favicon = "_static/images/OpenInferFavIcon.png"
html_compact_lists = True

html_theme_options = {
    "color_scheme": "dark",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 5,
    "logo_mobile": "images/OpenInferIcon.png",
}

# Enable autodoc for the Python utilities if needed.
sys.path.insert(0, os.path.abspath("../../openinfer-oinf"))

# Sphinx Rustdoc generator configuration.
rust_rustdocgen = shutil.which("sphinx-rustdocgen")
rust_generate_mode = "skip"
rust_doc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "api", "rustdocgen"))
rust_crates = {}

# Theme contexts expected by revitron templates.
html_context = {
    "style": "",
}
