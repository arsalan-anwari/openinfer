#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/update_wiki.sh [--output-dir DIR]

Clones the GitHub wiki repo (if needed), syncs docs/wiki/ into it,
and pushes changes.

Options:
  --output-dir DIR   Parent directory for the wiki clone.
                     Defaults to the parent of the repo root.
EOF
}

output_parent_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --output-dir" >&2
        exit 1
      fi
      output_parent_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
repo_name="$(basename "$repo_root")"
docs_dir="$repo_root/docs/wiki"

if [[ ! -d "$docs_dir" ]]; then
  echo "Missing docs/wiki directory at: $docs_dir" >&2
  exit 1
fi

if [[ -z "$output_parent_dir" ]]; then
  output_parent_dir="$(dirname "$repo_root")"
fi

if [[ ! -d "$output_parent_dir" ]]; then
  echo "Output parent directory does not exist: $output_parent_dir" >&2
  exit 1
fi

origin_url="$(git -C "$repo_root" remote get-url origin)"
if [[ -z "$origin_url" ]]; then
  echo "Unable to determine origin remote URL." >&2
  exit 1
fi

wiki_url=""
if [[ "$origin_url" == git@github.com:* ]]; then
  wiki_url="${origin_url%.git}.wiki.git"
elif [[ "$origin_url" == https://github.com/* ]]; then
  wiki_url="${origin_url%.git}.wiki.git"
else
  echo "Unsupported origin URL format: $origin_url" >&2
  echo "Expected git@github.com:<org>/<repo>.git or https://github.com/<org>/<repo>.git" >&2
  exit 1
fi

wiki_dir="$output_parent_dir/${repo_name}-wiki"

if [[ -d "$wiki_dir" ]]; then
  if [[ ! -d "$wiki_dir/.git" ]]; then
    echo "Wiki directory exists but is not a git repo: $wiki_dir" >&2
    exit 1
  fi
else
  git clone "$wiki_url" "$wiki_dir"
fi

rsync -av --delete --exclude ".git" "$docs_dir/" "$wiki_dir/"

git -C "$wiki_dir" add -A
if git -C "$wiki_dir" diff --staged --quiet; then
  echo "No wiki changes to commit."
  exit 0
fi

git -C "$wiki_dir" commit -m "Sync wiki from docs/wiki"
git -C "$wiki_dir" push
