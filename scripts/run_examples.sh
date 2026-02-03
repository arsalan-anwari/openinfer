#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_py_dir="$repo_root/examples/openinfer-oinf"
examples_rs_dir="$repo_root/examples/openinfer"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_examples.sh [options]

Runs OpenInfer examples by generating models (Python) then running Rust examples.

Options:
  --list                    List available examples and exit.
  --target <value>          Target: cpu|vulkan|all (default: cpu).
  --features <list>         Cargo features to enable (comma-separated).
  --example-filter <name>   Run only a single example by name.
  --help                    Show this help.

Examples:
  ./scripts/run_examples.sh --list
  ./scripts/run_examples.sh --target=cpu
  ./scripts/run_examples.sh --target=vulkan --features=vulkan
  ./scripts/run_examples.sh --target=all --features=vulkan
  ./scripts/run_examples.sh --example-filter kv_cache_decode
EOF
}

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    echo "error: python3 not found and .venv/bin/python missing" >&2
    exit 1
  fi
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$examples_py_dir" ]]; then
  echo "error: python examples directory not found at $examples_py_dir" >&2
  exit 1
fi

if [[ ! -d "$examples_rs_dir" ]]; then
  echo "error: rust examples directory not found at $examples_rs_dir" >&2
  exit 1
fi

selected_target="cpu"
features=""
example_filter=""
list_only=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --list)
      list_only=true
      shift
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
    --features)
      if [[ -z "${2:-}" ]]; then
        echo "error: --features requires a value" >&2
        exit 1
      fi
      features="$2"
      shift 2
      ;;
    --features=*)
      features="${1#*=}"
      shift
      ;;
    --example-filter)
      if [[ -z "${2:-}" ]]; then
        echo "error: --example-filter requires a value" >&2
        exit 1
      fi
      example_filter="$2"
      shift 2
      ;;
    --example-filter=*)
      example_filter="${1#*=}"
      shift
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

case "$selected_target" in
  cpu|vulkan|all) ;;
  *)
    echo "error: unknown target '$selected_target' (expected cpu|vulkan|all)" >&2
    exit 1
    ;;
esac

mapfile -t py_examples < <(find "$examples_py_dir" -maxdepth 1 -type f -name '*_oinf.py' -printf '%f\n' | sed 's/_oinf\.py$//' | sort)
mapfile -t rs_examples < <(find "$examples_rs_dir" -maxdepth 1 -type f -name '*.rs' -printf '%f\n' | sed 's/\.rs$//' | sort)

declare -A rs_example_map=()
for example in "${rs_examples[@]}"; do
  rs_example_map["$example"]=1
done

available_examples=()
for example in "${py_examples[@]}"; do
  if [[ -n "${rs_example_map[$example]:-}" ]]; then
    available_examples+=("$example")
  fi
done

if [[ ${#available_examples[@]} -eq 0 ]]; then
  echo "error: no matching examples found across python and rust directories" >&2
  exit 1
fi

if [[ "$list_only" == "true" ]]; then
  printf '%s\n' "${available_examples[@]}"
  exit 0
fi

if [[ -n "$example_filter" ]]; then
  found=false
  for example in "${available_examples[@]}"; do
    if [[ "$example" == "$example_filter" ]]; then
      found=true
      break
    fi
  done
  if [[ "$found" != "true" ]]; then
    echo "error: example '$example_filter' not found (use --list to see options)" >&2
    exit 1
  fi
  available_examples=("$example_filter")
fi

append_feature() {
  local current="$1"
  local feature="$2"
  if [[ -z "$current" ]]; then
    echo "$feature"
    return
  fi
  IFS=',' read -r -a items <<< "$current"
  for item in "${items[@]}"; do
    if [[ "$item" == "$feature" ]]; then
      echo "$current"
      return
    fi
  done
  echo "${current},${feature}"
}

run_one() {
  local example="$1"
  local target="$2"
  local run_features="$features"
  if [[ -n "$target" && "$target" != "cpu" ]]; then
    run_features="$(append_feature "$run_features" "vulkan")"
  fi

  local py_script="$examples_py_dir/${example}_oinf.py"
  local -a cmd=(cargo run --package openinfer --example "$example")
  if [[ -n "$run_features" ]]; then
    cmd+=(--features "$run_features")
  fi
  if [[ -n "$target" ]]; then
    cmd+=(-- --target "$target")
  fi

  echo "==> ${python_bin} ${py_script}"
  "$python_bin" "$py_script"
  echo "==> ${cmd[*]}"
  "${cmd[@]}"
  echo ""
}

if [[ "$selected_target" == "all" ]]; then
  for example in "${available_examples[@]}"; do
    echo "=== Example: $example ==="
    run_one "$example" "cpu"
    run_one "$example" "vulkan"
  done
else
  for example in "${available_examples[@]}"; do
    echo "=== Example: $example ==="
    run_one "$example" "$selected_target"
  done
fi

echo "Done."
