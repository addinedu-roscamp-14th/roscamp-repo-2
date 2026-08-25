#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
ROOT_DIR="$(realpath "$SCRIPT_DIR/..")"
MODE="--dry-run"
if [[ "${1:-}" == "--execute" || "${1:-}" == "--dry-run" ]]; then MODE="$1"; shift; fi
[[ $# -eq 1 ]] || { echo "usage: $0 [--dry-run|--execute] USER@HOST:/path" >&2; exit 2; }
exec "$SCRIPT_DIR/_deploy.sh" "$MODE" "$1" "$ROOT_DIR/raspberry_left_ws" "$ROOT_DIR/docs"
