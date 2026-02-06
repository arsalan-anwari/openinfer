#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${repo_root}/scripts/bootstrap_submodules.sh" ]]; then
  "${repo_root}/scripts/bootstrap_submodules.sh"
fi

venv_dir="${repo_root}/.venv"
if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

python_bin="${venv_dir}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "error: python venv not found at ${python_bin}" >&2
  exit 1
fi

if [[ -f "${repo_root}/requirements.txt" ]]; then
  "${python_bin}" -m pip install -r "${repo_root}/requirements.txt"
fi

if [[ -f "${repo_root}/openinfer-oinf/requirements.txt" ]]; then
  "${python_bin}" -m pip install -r "${repo_root}/openinfer-oinf/requirements.txt"
fi

echo "Setup complete."
