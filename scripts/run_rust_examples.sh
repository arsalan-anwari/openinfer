#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found in PATH" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: ./scripts/run_rust_examples.sh [options] [cargo args...]

Options:
  --features <list>     Cargo features to enable (passed through to cargo).
  --target <value>      Example target: cpu|avx|avx2|vulkan|all.
  --help                Show this help.

Examples:
  ./scripts/run_rust_examples.sh --features=avx,avx2,vulkan --target=vulkan
  ./scripts/run_rust_examples.sh --features=avx,avx2,vulkan --target=all
EOF
}

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
selected_target=""
targets=(cpu avx avx2 vulkan)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --features)
      if [[ -z "${2:-}" ]]; then
        echo "error: --features requires a value" >&2
        exit 1
      fi
      cargo_args+=("$1" "$2")
      shift 2
      ;;
    --target)
      if [[ -z "${2:-}" ]]; then
        echo "error: --target requires a value" >&2
        exit 1
      fi
      selected_target="$2"
      shift 2
      ;;
    --target=*)
      selected_target="${1#*=}"
      shift
      ;;
    *)
      cargo_args+=("$1")
      shift
      ;;
  esac
done

echo "Running Rust examples (${#examples[@]})..."

run_one() {
  local example="$1"
  shift
  local run_args=("$@")
  echo "==> cargo run --package openinfer --example $example ${cargo_args[*]-} -- ${run_args[*]-}"
  cargo run --package openinfer --example "$example" "${cargo_args[@]}" -- "${run_args[@]}"
  echo ""
  sleep 0.1
}

if [[ -n "$selected_target" ]]; then
  valid=false
  if [[ "$selected_target" == "all" ]]; then
    valid=true
  else
    for target in "${targets[@]}"; do
      if [[ "$selected_target" == "$target" ]]; then
        valid=true
        break
      fi
    done
  fi
  if [[ "$valid" != "true" ]]; then
    echo "error: unknown target '$selected_target' (expected cpu|avx|avx2|vulkan|all)" >&2
    exit 1
  fi
  if [[ "$selected_target" == "all" ]]; then
    for example in "${examples[@]}"; do
      echo "=== Example: $example ==="
      for target in "${targets[@]}"; do
        run_one "$example" --target "$target"
      done
    done
  else
    for example in "${examples[@]}"; do
      echo "=== Example: $example ==="
      run_one "$example" --target "$selected_target"
    done
  fi
else
  for example in "${examples[@]}"; do
    echo "=== Example: $example ==="
    run_one "$example"
  done
fi

echo "All Rust examples completed successfully."
