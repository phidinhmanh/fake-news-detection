#!/usr/bin/env bash
# verity.sh — Shell wrapper for Verity CLI (Linux/macOS)
# Usage: ./verity.sh <command> [options]
# Or:    bash verity.sh <command> [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec uv run python "$SCRIPT_DIR/verity.py" "$@"
