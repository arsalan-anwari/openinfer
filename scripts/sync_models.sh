#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${repo_root}/openinfer-oinf/res/models"
dest_dir="${repo_root}/openinfer-simulator/res/models"

if [[ ! -d "${src_dir}" ]]; then
  echo "error: source models directory not found at ${src_dir}" >&2
  exit 1
fi

mkdir -p "${dest_dir}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "${src_dir}/" "${dest_dir}/"
else
  rm -rf "${dest_dir}"
  cp -a "${src_dir}" "${dest_dir}"
fi

echo "Synced models to ${dest_dir}"
