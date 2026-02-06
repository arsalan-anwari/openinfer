#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found in PATH" >&2
  exit 1
fi

echo "==> cargo check (openinfer-dsl)"
cargo check --manifest-path "${repo_root}/openinfer-dsl/Cargo.toml"

echo "==> cargo check (openinfer-simulator)"
cargo check --manifest-path "${repo_root}/openinfer-simulator/Cargo.toml"

if [[ -x "${repo_root}/.venv/bin/python" ]]; then
  echo "==> python -m compileall (openinfer-oinf)"
  "${repo_root}/.venv/bin/python" -m compileall "${repo_root}/openinfer-oinf"
fi

echo "Build complete."
