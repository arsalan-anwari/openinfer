#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found in PATH" >&2
  exit 1
fi

examples_dir="$repo_root/examples/rust"

if [[ ! -d "$examples_dir" ]]; then
  echo "error: examples directory not found at $examples_dir" >&2
  exit 1
fi

mapfile -t examples < <(find "$examples_dir" -maxdepth 1 -type f -name '*.rs' -printf '%f\n' | sed 's/\.rs$//' | sort)

if [[ ${#examples[@]} -eq 0 ]]; then
  echo "error: no Rust examples found in $examples_dir" >&2
  exit 1
fi

cargo_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --features)
      if [[ -z "${2:-}" ]]; then
        echo "error: --features requires a value" >&2
        exit 1
      fi
      cargo_args+=("$1" "$2")
      shift 2
      ;;
    *)
      cargo_args+=("$1")
      shift
      ;;
  esac
done

echo "Running Rust examples (${#examples[@]})..."

for example in "${examples[@]}"; do
  echo "==> cargo run --package openinfer --example $example ${cargo_args[*]-}"
  cargo run --package openinfer --example "$example" "${cargo_args[@]}"
  echo
  echo "ok: $example"
  echo
  sleep 0.1

done

echo "All Rust examples completed successfully."
