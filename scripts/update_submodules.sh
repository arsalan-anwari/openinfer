#!/usr/bin/env bash
set -euo pipefail

# Push commits for each submodule, without touching the main repo.
git submodule foreach --recursive '
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
'
