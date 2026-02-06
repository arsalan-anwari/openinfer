#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/update_submodules.sh [options]

Push commits for submodules, without touching the main repo.

Options:
  --modules <list>   Comma-separated list of submodules to update.
  --list             List available submodules and exit.
  --help             Show this help.

Examples:
  ./scripts/update_submodules.sh
  ./scripts/update_submodules.sh --modules openinfer-oinf,openinfer-dsl
  ./scripts/update_submodules.sh --list
EOF
}

modules_csv=""
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
    --modules)
      if [[ -z "${2:-}" ]]; then
        echo "error: --modules requires a value" >&2
        exit 1
      fi
      modules_csv="$2"
      shift 2
      ;;
    --modules=*)
      modules_csv="${1#*=}"
      shift
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

declare -A module_path_by_name
declare -A module_name_by_path
while read -r key path; do
  name="${key#submodule.}"
  name="${name%.path}"
  module_path_by_name["$name"]="$path"
  module_name_by_path["$path"]="$name"
done < <(git -C "$repo_root" config -f .gitmodules --get-regexp '^submodule\..*\.path$')

if [[ "$list_only" == "true" ]]; then
  if [[ "${#module_path_by_name[@]}" -eq 0 ]]; then
    echo "No submodules found."
    exit 0
  fi
  printf "%s\n" "${!module_path_by_name[@]}" | sort | while read -r name; do
    echo "${name} (${module_path_by_name[$name]})"
  done
  exit 0
fi

selected_modules=()
if [[ -n "$modules_csv" ]]; then
  IFS=',' read -r -a raw_modules <<< "$modules_csv"
  for item in "${raw_modules[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${module_path_by_name[$item]:-}" ]]; then
      selected_modules+=("$item")
      continue
    fi
    if [[ -n "${module_name_by_path[$item]:-}" ]]; then
      selected_modules+=("${module_name_by_path[$item]}")
      continue
    fi
    echo "error: unknown submodule '$item'" >&2
    exit 1
  done
fi

selected_modules_csv=""
if [[ "${#selected_modules[@]}" -gt 0 ]]; then
  selected_modules_csv="$(IFS=','; echo "${selected_modules[*]}")"
fi

# Push commits for each submodule, without touching the main repo.
(cd "$repo_root" && SELECTED_MODULES="$selected_modules_csv" git submodule foreach --recursive '
  if [ -n "${SELECTED_MODULES:-}" ]; then
    case ",$SELECTED_MODULES," in
      *,"$name",*|*,"$path",*) ;;
      *) exit 0 ;;
    esac
  fi
  echo "==> $(pwd)"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [ -n "$(git status --porcelain)" ]; then
      echo "  Skipping: working tree not clean"
      exit 0
    fi
    branch="$(git symbolic-ref -q --short HEAD || true)"
    if [ -z "$branch" ]; then
      detached_sha="$(git rev-parse HEAD)"
      default_remote_branch="$(git symbolic-ref -q --short refs/remotes/origin/HEAD | sed "s#^origin/##")"
      if [ -z "$default_remote_branch" ]; then
        echo "  Skipping: detached HEAD and no origin/HEAD set"
        exit 1
      fi
      git switch -q "$default_remote_branch" 2>/dev/null || git switch -q -c "$default_remote_branch"
      if git merge --ff-only "$detached_sha" >/dev/null 2>&1; then
        branch="$default_remote_branch"
      else
        echo "  Skipping: detached HEAD not fast-forwardable to $default_remote_branch"
        exit 1
      fi
    fi
    # Push current branch if it has an upstream, otherwise set upstream to origin/<branch>.
    if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
      git push
    else
      git push -u origin "$branch"
    fi
  fi
')
