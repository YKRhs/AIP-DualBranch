#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${AIP_PYTHON:-python}"
exec "$PYTHON_BIN" -u app/web_app.py "$@"

