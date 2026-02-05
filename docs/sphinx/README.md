# OpenInfer Sphinx Docs

This directory contains the Sphinx portal for OpenInfer documentation.

## One-shot build

From the repo root:

```bash
./scripts/build_docs.sh
```

This installs Sphinx dependencies in `.venv`, builds Rustdoc into
`docs/sphinx/api/rustdoc`, and generates HTML in `docs/sphinx/out`.

## Manual build

```bash
pip install -r requirements.txt
sphinx-build -b html . out
```
