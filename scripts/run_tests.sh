#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_ops="$repo_root/tests/openinfer/ops/baseline/gen_ops_baseline.py"
baseline_graph="$repo_root/tests/openinfer/graph/baseline/gen_graph_baseline.py"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_tests.sh [options]

Runs numpy baselines, then openinfer tests.

Options:
  --list                 List available tests and exit.
  --target <value>       Target: cpu|vulkan|all (default: cpu).
  --features <list>      Cargo features to enable (comma-separated).
  --test-filter <name>   Filter tests (supports openinfer::, openinfer-dsl::, openinfer-oinf::).
  --help                 Show this help.

Examples:
  ./scripts/run_tests.sh --target=cpu
  ./scripts/run_tests.sh --target=vulkan --features=vulkan
  ./scripts/run_tests.sh --target=all --features=vulkan
  ./scripts/run_tests.sh --test-filter openinfer::ops_misc
  ./scripts/run_tests.sh --test-filter openinfer-dsl::parse_tests
  ./scripts/run_tests.sh --test-filter openinfer-oinf::test_common.TestCommon.test_align_up
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

selected_target="cpu"
features=""
test_filter=""
test_module=""
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
    --test-filter)
      if [[ -z "${2:-}" ]]; then
        echo "error: --test-filter requires a value" >&2
        exit 1
      fi
      test_filter="$2"
      shift 2
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

if [[ "$selected_target" == "vulkan" || "$selected_target" == "all" ]]; then
  features="$(append_feature "$features" "vulkan")"
fi

if [[ "$list_only" == "true" ]]; then
  list_cmd=(cargo test -p openinfer --test openinfer -- --list)
  if [[ -n "$features" ]]; then
    list_cmd=(cargo test -p openinfer --test openinfer --features "$features" -- --list)
  fi
  echo "==> ${list_cmd[*]}"
  "${list_cmd[@]}" | awk '{gsub(/: test$/, "", $0); if (NF==0) next; print "openinfer::" $0}'
  dsl_list_cmd=(cargo test -p openinfer-dsl -- --list)
  echo "==> ${dsl_list_cmd[*]}"
  "${dsl_list_cmd[@]}" | awk '{gsub(/: test$/, "", $0); if (NF==0) next; print "openinfer-dsl::" $0}'
  oinf_list_cmd=("$python_bin" "$repo_root/tests/openinfer-oinf/run_oinf_tests.py" --list)
  echo "==> ${oinf_list_cmd[*]}"
  "${oinf_list_cmd[@]}" | awk '{if (NF==0) next; print "openinfer-oinf::" $0}'
  exit 0
fi

if [[ "$test_filter" == openinfer::* ]]; then
  test_module="openinfer"
  test_filter="${test_filter#openinfer::}"
elif [[ "$test_filter" == openinfer-dsl::* ]]; then
  test_module="openinfer-dsl"
  test_filter="${test_filter#openinfer-dsl::}"
elif [[ "$test_filter" == openinfer-oinf::* ]]; then
  test_module="openinfer-oinf"
  test_filter="${test_filter#openinfer-oinf::}"
fi

if [[ -z "$test_module" || "$test_module" == "openinfer" ]]; then
  echo "==> ${python_bin} ${baseline_ops}"
  "$python_bin" "$baseline_ops"
  echo "==> ${python_bin} ${baseline_graph}"
  "$python_bin" "$baseline_graph"
fi

test_targets="cpu"
if [[ "$selected_target" == "vulkan" ]]; then
  test_targets="vulkan"
elif [[ "$selected_target" == "all" ]]; then
  test_targets="cpu,vulkan"
fi

if [[ -z "$test_module" || "$test_module" == "openinfer" ]]; then
  test_cmd=(cargo test -p openinfer --test openinfer)
  if [[ -n "$features" ]]; then
    test_cmd+=(--features "$features")
  fi
  if [[ -n "$test_filter" ]]; then
    test_cmd+=(-- "$test_filter")
  fi
  echo "==> TEST_TARGETS=${test_targets} ${test_cmd[*]}"
  TEST_TARGETS="${test_targets}" "${test_cmd[@]}"
fi

if [[ -z "$test_module" || "$test_module" == "openinfer-dsl" ]]; then
  dsl_cmd=(cargo test -p openinfer-dsl)
  if [[ -n "$test_filter" && "$test_module" == "openinfer-dsl" ]]; then
    dsl_cmd+=("$test_filter")
  fi
  echo "==> ${dsl_cmd[*]}"
  "${dsl_cmd[@]}"
fi

if [[ -z "$test_module" || "$test_module" == "openinfer-oinf" ]]; then
  oinf_cmd=("$python_bin" "$repo_root/tests/openinfer-oinf/run_oinf_tests.py")
  if [[ -n "$test_filter" && "$test_module" == "openinfer-oinf" ]]; then
    oinf_cmd+=(--filter "$test_filter")
  fi
  echo "==> ${oinf_cmd[*]}"
  "${oinf_cmd[@]}"
fi

echo "Done."
