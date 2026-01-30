#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_dir="$repo_root/examples/openinfer-oinf"

if [[ ! -d "$examples_dir" ]]; then
  echo "error: examples directory not found at $examples_dir" >&2
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

mapfile -t examples < <(find "$examples_dir" -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort)

if [[ ${#examples[@]} -eq 0 ]]; then
  echo "error: no Python examples found in $examples_dir" >&2
  exit 1
fi

echo "Running OpenInfer OINF examples (${#examples[@]})..."

for example in "${examples[@]}"; do
  script_path="$examples_dir/$example"
  echo "==> ${python_bin} ${script_path}"
  "$python_bin" "$script_path"
  echo ""
done

echo "All OpenInfer OINF examples completed successfully."
