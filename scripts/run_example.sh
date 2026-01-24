#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_example.sh <example_name> [options] [-- rust args...]

Options:
  --list                List available Rust examples and exit.
  --target <value>      Example target: cpu|avx|avx2|vulkan|all.
  --trace <mode>        Trace mode: BASE|FULL|VULKAN.
  --help                Show this help.

Examples:
  ./scripts/run_example.sh minimal --target=avx2
  ./scripts/run_example.sh minimal --trace=FULL --target=vulkan
  ./scripts/run_example.sh minimal -- --target=cpu
EOF
}

examples_dir="$repo_root/examples/rust"

if [[ ! -d "$examples_dir" ]]; then
  echo "error: examples directory not found at $examples_dir" >&2
  exit 1
fi

example=""
selected_target=""
trace_mode=""
list_only=false
rust_args=()
targets=(cpu avx avx2 vulkan)

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
    --trace)
      if [[ -z "${2:-}" ]]; then
        echo "error: --trace requires a value" >&2
        exit 1
      fi
      trace_mode="${2^^}"
      shift 2
      ;;
    --trace=*)
      trace_mode="${1#*=}"
      trace_mode="${trace_mode^^}"
      shift
      ;;
    --)
      shift
      if [[ $# -gt 0 ]]; then
        rust_args+=("$@")
      fi
      break
      ;;
    -*)
      if [[ -z "$example" ]]; then
        echo "error: example name is required before additional args" >&2
        exit 1
      fi
      rust_args+=("$1")
      shift
      ;;
    *)
      if [[ -z "$example" ]]; then
        example="$1"
      else
        rust_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ "$list_only" == "true" ]]; then
  mapfile -t examples < <(find "$examples_dir" -maxdepth 1 -type f -name '*.rs' -printf '%f\n' | sed 's/\.rs$//' | sort)
  if [[ ${#examples[@]} -eq 0 ]]; then
    echo "error: no Rust examples found in $examples_dir" >&2
    exit 1
  fi
  printf '%s\n' "${examples[@]}"
  exit 0
fi

if [[ $# -eq 0 && -z "$example" ]]; then
  usage
  exit 1
fi

if [[ -z "$example" ]]; then
  echo "error: example name is required" >&2
  usage
  exit 1
fi

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
fi

if [[ -n "$trace_mode" ]]; then
  case "$trace_mode" in
    BASE|FULL|VULKAN) ;;
    *)
      echo "error: unknown trace mode '$trace_mode' (expected BASE|FULL|VULKAN)" >&2
      exit 1
      ;;
  esac
fi

py_script="$repo_root/examples/python/${example}_oinf.py"
rust_example="$repo_root/examples/rust/${example}.rs"
model_path="$repo_root/res/models/${example}_model.oinf"

if [[ ! -f "$py_script" ]]; then
  echo "error: python example not found at $py_script" >&2
  exit 1
fi

if [[ ! -f "$rust_example" ]]; then
  echo "error: rust example not found at $rust_example" >&2
  exit 1
fi

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

echo "==> ${python_bin} ${py_script}"
"$python_bin" "$py_script"

echo "==> ${python_bin} ${repo_root}/openinfer-oinf/verify_oinf.py ${model_path}"
"$python_bin" "$repo_root/openinfer-oinf/verify_oinf.py" "$model_path"

trace_env=()
case "$trace_mode" in
  BASE)
    trace_env=(OPENINFER_TRACE=1)
    ;;
  FULL)
    trace_env=(RUST_BACKTRACE=1 OPENINFER_TRACE=1)
    ;;
  VULKAN)
    trace_env=(RUST_BACKTRACE=1 OPENINFER_TRACE=1 OPENINFER_VULKAN_TRACE=1)
    ;;
esac

run_one() {
  local target="$1"
  local -a features=()
  local -a run_args=()

  if [[ -n "$target" ]]; then
    run_args+=(--target "$target")
    if [[ "$target" != "cpu" ]]; then
      features+=(--features "$target")
    fi
  fi

  local -a cmd=(cargo run --package openinfer --example "$example")
  if [[ ${#features[@]} -gt 0 ]]; then
    cmd+=("${features[@]}")
  fi
  cmd+=(-- "${run_args[@]}" "${rust_args[@]}")

  if [[ ${#trace_env[@]} -gt 0 ]]; then
    echo "==> ${trace_env[*]} ${cmd[*]}"
    env "${trace_env[@]}" "${cmd[@]}"
  else
    echo "==> ${cmd[*]}"
    "${cmd[@]}"
  fi
  echo ""
}

if [[ "$selected_target" == "all" ]]; then
  for target in "${targets[@]}"; do
    run_one "$target"
  done
else
  run_one "$selected_target"
fi

echo "Example completed successfully."
