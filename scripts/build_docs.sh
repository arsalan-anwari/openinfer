#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SPHINX_DIR="${ROOT_DIR}/docs/sphinx"
SPHINX_OUT="${SPHINX_DIR}/out"
RUSTDOC_OUT="${SPHINX_DIR}/api/rustdoc"
TARGET_DOC="${ROOT_DIR}/target/doc"
SPHINX_THEME_CSS="${SPHINX_OUT}/_static/css/theme.css"
SPHINX_CUSTOM_CSS="${SPHINX_DIR}/_static/css/custom.css"
SPHINX_STATIC_IMAGES="${SPHINX_DIR}/_static/images"
OPENINFER_ICON="${ROOT_DIR}/res/images/OpenInferIcon.png"
OPENINFER_FAVICON="${ROOT_DIR}/res/images/OpenInferFavIcon.png"
RUSTDOC_CSS="${SPHINX_DIR}/_static/css/rustdoc.css"
RUSTDOC_HEADER="${SPHINX_DIR}/_static/rustdoc_header.html"

if [[ ! -d "${VENV_DIR}" ]]; then
  python -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install -r "${SPHINX_DIR}/requirements.txt"

"${VENV_DIR}/bin/python" "${SPHINX_DIR}/generate_ops_docs.py"

mkdir -p "${SPHINX_STATIC_IMAGES}"
cp "${OPENINFER_ICON}" "${SPHINX_STATIC_IMAGES}/OpenInferIcon.png"
cp "${OPENINFER_FAVICON}" "${SPHINX_STATIC_IMAGES}/OpenInferFavIcon.png"

"${VENV_DIR}/bin/python" -m sphinx -b html "${SPHINX_DIR}" "${SPHINX_OUT}"

mkdir -p "${SPHINX_DIR}/api"
rm -rf "${RUSTDOC_OUT}"
mkdir -p "${RUSTDOC_OUT}"

cat > "${RUSTDOC_CSS}" <<EOF
@import url("${SPHINX_THEME_CSS}");
@import url("${SPHINX_CUSTOM_CSS}");

:root {
  color-scheme: dark;
}

:root,
:root:not([data-theme]),
:root[data-theme="light"],
:root[data-theme="dark"] {
  --font-family: "Inter", "Fira Sans", "Helvetica Neue", Arial, sans-serif;
  --font-family-code: "Fira Mono", "Source Code Pro", monospace;

  --hue: 215;
  --sat: 6%;
  --bg: hsl(var(--hue), var(--sat), 14%);
  --bg-nav: var(--bg);
  --bg-nav-current: hsl(var(--hue), var(--sat), 26%);
  --bg-nav-current-list: hsl(var(--hue), var(--sat), 17%);
  --bg-nav-hover: hsl(var(--hue), var(--sat), 20%);
  --bg-code: hsl(var(--hue), var(--sat), 17%);
  --bg-code-highlight: hsl(var(--hue), var(--sat), 21%);
  --bg-panel: var(--bg-code);
  --text: hsl(var(--hue), var(--sat), 94%);
  --link: inherit;
  --nav: hsl(var(--hue), var(--sat), 52%);
  --code: hsl(var(--hue), var(--sat), 86%);
  --headline: var(--text);
  --nav-current: var(--text);
  --border: hsl(var(--hue), var(--sat), 20%);
  --muted: hsl(var(--hue), var(--sat), 38%);
  --info-text: var(--text);
  --info-bg: hsl(var(--hue), var(--sat), 26%);
  --danger-text: #f28f88;
  --danger-bg: #362b2b;
  --warning-text: #ebdc8e;
  --warning-bg: #323129;

  --revitron-bg: var(--bg);
  --revitron-bg-nav: var(--bg-nav);
  --revitron-nav: var(--nav);
  --revitron-border: var(--border);
  --revitron-text: var(--text);
  --revitron-muted: var(--muted);
  --revitron-link: var(--link);
  --revitron-link-strong: var(--link);
  --revitron-code-bg: var(--bg-code);
  --revitron-code: var(--code);
  --revitron-warn: var(--warning-text);

  --main-background-color: var(--revitron-bg);
  --main-color: var(--revitron-text);
  --settings-input-color: var(--accent, #2196f3);
  --settings-input-border-color: var(--revitron-border);
  --settings-button-color: #000;
  --settings-button-border-focus: var(--accent-strong, #ffb900);
  --sidebar-background-color: var(--revitron-bg-nav);
  --sidebar-background-color-hover: var(--bg-nav-hover, #353c47);
  --sidebar-border-color: var(--revitron-border);
  --code-block-background-color: var(--revitron-code-bg);
  --scrollbar-track-background-color: var(--revitron-bg-nav);
  --scrollbar-thumb-background-color: rgba(32, 34, 37, 0.6);
  --scrollbar-color: rgba(32, 34, 37, 0.6) #5a5a5a;
  --headings-border-bottom-color: var(--revitron-border);
  --border-color: var(--revitron-border);
  --button-background-color: var(--bg-elevated, #2f3640);
  --right-side-color: var(--revitron-muted);
  --code-attribute-color: var(--revitron-muted);
  --toggles-color: var(--revitron-muted);
  --toggle-filter: invert(100%);
  --mobile-sidebar-menu-filter: invert(100%);
  --search-input-focused-border-color: var(--accent, #008dfd);
  --copy-path-button-color: var(--revitron-muted);
  --copy-path-img-filter: invert(50%);
  --copy-path-img-hover-filter: invert(65%);
  --code-example-button-color: var(--revitron-muted);
  --code-example-button-hover-color: var(--revitron-text);
  --codeblock-error-hover-color: rgb(255, 0, 0);
  --codeblock-error-color: rgba(255, 0, 0, 0.5);
  --codeblock-ignore-hover-color: rgb(255, 142, 0);
  --codeblock-ignore-color: rgba(255, 142, 0, 0.6);
  --warning-border-color: var(--revitron-warn);
  --type-link-color: var(--type, var(--revitron-link));
  --trait-link-color: var(--trait, #b78cf2);
  --assoc-item-link-color: var(--assoc, #f2c14f);
  --function-link-color: var(--function, #7fd0a3);
  --macro-link-color: var(--macro, #09bd00);
  --keyword-link-color: var(--keyword, var(--assoc, #f2c14f));
  --attribute-link-color: var(--attribute, var(--assoc, #f2c14f));
  --mod-link-color: var(--module, var(--assoc, #f2c14f));
  --link-color: var(--revitron-link);
  --sidebar-link-color: var(--revitron-nav);
  --sidebar-current-link-background-color: var(--bg-nav-active, #3a404a);
  --search-result-link-focus-background-color: var(--bg-elevated, #3a404a);
  --search-result-border-color: var(--revitron-border);
  --search-color: var(--revitron-text);
  --search-error-code-background-color: var(--bg-elevated, #2a2f38);
  --search-results-alias-color: var(--revitron-text);
  --search-results-grey-color: var(--revitron-muted);
  --search-tab-title-count-color: var(--revitron-muted);
  --search-tab-button-not-selected-border-top-color: var(--revitron-border);
  --search-tab-button-not-selected-background: var(--bg-elevated, #2a2f38);
  --search-tab-button-selected-border-top-color: var(--accent, #0089ff);
  --search-tab-button-selected-background: var(--revitron-bg);
  --settings-menu-filter: invert(50%);
  --settings-menu-hover-filter: invert(65%);
  --stab-background-color: var(--bg-elevated, #314559);
  --stab-code-color: var(--revitron-text);
  --code-highlight-kw-color: var(--code-kw, #ab8ac1);
  --code-highlight-kw-2-color: var(--code-kw-2, #769acb);
  --code-highlight-lifetime-color: var(--code-lifetime, #d97f26);
  --code-highlight-prelude-color: var(--code-prelude, #769acb);
  --code-highlight-prelude-val-color: var(--code-prelude-val, #ee6868);
  --code-highlight-number-color: var(--code-number, #83a300);
  --code-highlight-string-color: var(--code-string, #83a300);
  --code-highlight-literal-color: var(--code-literal, #ee6868);
  --code-highlight-attribute-color: var(--code-attribute, #ee6868);
  --code-highlight-self-color: var(--code-self, #ee6868);
  --code-highlight-macro-color: var(--code-macro, #3e999f);
  --code-highlight-question-mark-color: var(--code-question, #ff9011);
  --code-highlight-comment-color: var(--code-comment, #8d8d8b);
  --code-highlight-doc-comment-color: var(--code-doc-comment, #8ca375);
  --src-line-numbers-span-color: var(--revitron-link);
  --src-line-number-highlighted-background-color: #0a042f;
  --target-background-color: #494a3d;
  --target-border-color: #bb7410;
  --kbd-color: var(--revitron-text);
  --kbd-background: var(--bg-elevated, #2a2f38);
  --kbd-box-shadow-color: var(--revitron-border);
  --rust-logo-filter: drop-shadow(1px 0 0px #fff)
    drop-shadow(0 1px 0 #fff)
    drop-shadow(-1px 0 0 #fff)
    drop-shadow(0 -1px 0 #fff);
  --crate-search-div-filter: invert(94%) sepia(0%) saturate(721%)
    hue-rotate(255deg) brightness(90%) contrast(90%);
  --crate-search-div-hover-filter: invert(69%) sepia(60%) saturate(6613%)
    hue-rotate(184deg) brightness(100%) contrast(91%);
  --crate-search-hover-border: var(--accent, #2196f3);
  --src-sidebar-background-selected: var(--bg-nav-active, #3a404a);
  --src-sidebar-background-hover: var(--bg-nav-hover, #353c47);
  --table-alt-row-background-color: var(--bg-elevated, #2a2f38);
  --codeblock-link-background: var(--bg-elevated, #2a2f38);
  --scrape-example-toggle-line-background: var(--revitron-border);
  --scrape-example-toggle-line-hover-background: var(--revitron-muted);
  --scrape-example-code-line-highlight: #5b3b01;
  --scrape-example-code-line-highlight-focus: #7c4b0f;
  --scrape-example-help-border-color: var(--revitron-border);
  --scrape-example-help-color: var(--revitron-text);
  --scrape-example-help-hover-border-color: var(--revitron-text);
  --scrape-example-help-hover-color: var(--revitron-text);
  --scrape-example-code-wrapper-background-start: rgba(31, 34, 40, 1);
  --scrape-example-code-wrapper-background-end: rgba(31, 34, 40, 0);
  --sidebar-resizer-hover: hsl(207, 30%, 54%);
  --sidebar-resizer-active: hsl(207, 90%, 54%);
}

html,
body {
  background-color: var(--main-background-color);
  color: var(--main-color);
}

div.setting-line#theme {
  display: none !important;
}

.setting-line,
.setting-line span,
.setting-line label {
  color: var(--main-color) !important;
}

pre,
pre code,
.docblock pre,
.docblock pre code {
  white-space: pre !important;
}

EOF

cat > "${RUSTDOC_HEADER}" <<EOF
<script>
  try {
    localStorage.setItem("rustdoc-use-system-theme", "false");
    localStorage.setItem("rustdoc-theme", "dark");
    localStorage.setItem("rustdoc-preferred-dark-theme", "dark");
    localStorage.setItem("rustdoc-preferred-light-theme", "dark");
  } catch (e) {}
  document.documentElement.setAttribute("data-theme", "dark");
</script>
EOF

RUSTDOCFLAGS="${RUSTDOCFLAGS:-} --extend-css ${RUSTDOC_CSS} --html-in-header ${RUSTDOC_HEADER}" \
  cargo doc --workspace --no-deps
cp -R "${TARGET_DOC}/." "${RUSTDOC_OUT}/"

rm -rf "${SPHINX_OUT}/api/rustdoc"
mkdir -p "${SPHINX_OUT}/api"
cp -R "${RUSTDOC_OUT}" "${SPHINX_OUT}/api/rustdoc"

echo "Sphinx HTML output: ${SPHINX_OUT}"
echo "Rustdoc output: ${RUSTDOC_OUT}"
echo "Rustdoc copied to: ${SPHINX_OUT}/api/rustdoc"
