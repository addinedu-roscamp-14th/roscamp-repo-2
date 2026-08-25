#!/usr/bin/env bash
set -euo pipefail

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
  shift
elif [[ "${1:-}" == "--dry-run" ]]; then
  shift
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 [--dry-run|--execute] DEST SOURCE..." >&2
  exit 2
fi
DEST="$1"
shift
RSYNC_ARGS=(-avz --delete-delay
  --exclude=build/ --exclude=install/ --exclude=log/
  --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude='*.pyc'
  --exclude=.git/ --exclude=legacy/)
if [[ "$MODE" == "dry-run" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
  echo "[DRY-RUN] no files will be changed"
fi
rsync "${RSYNC_ARGS[@]}" "$@" "$DEST/"
