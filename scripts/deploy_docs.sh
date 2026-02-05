#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/deploy_docs.sh [-m MOUNT_DIR]

Options:
  -m MOUNT_DIR  Local mount directory (default: ~/Workspace/open-infer-website)
  -h            Show this help
EOF
}

mount_dir="$HOME/Workspace/open-infer-website"
remote="open-infer:/home/u214998p479997/domains/open-infer.nl/public_html"
source_dir="docs/sphinx/out"

while getopts ":m:h" opt; do
  case "$opt" in
    m) mount_dir="$OPTARG" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      usage >&2
      exit 2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$source_dir" ]]; then
  echo "Source directory not found: $source_dir" >&2
  exit 1
fi

cleanup() {
  if mountpoint -q "$mount_dir"; then
    fusermount -u "$mount_dir"
  fi
}

trap cleanup EXIT

mkdir -p "$mount_dir"
if mountpoint -q "$mount_dir"; then
  echo "Mount point already active, remounting: $mount_dir"
  fusermount -u "$mount_dir"
fi
echo "Mounting $remote -> $mount_dir"
sshfs "$remote" "$mount_dir"

mkdir -p "$mount_dir/docs"
echo "Syncing $source_dir -> $mount_dir/docs"
rsync -rlt --delete --no-perms --no-owner --no-group --omit-dir-times \
  --info=progress2 --human-readable \
  --filter='P .htaccess' \
  --filter='P .well-known' \
  --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "$source_dir"/ "$mount_dir/docs"/

if [[ ! -f "$mount_dir/docs/.htaccess" ]]; then
  echo "Warning: $mount_dir/docs/.htaccess not found after sync."
fi
