#!/usr/bin/env bash
#
# Start the GenRoad Framework web application.
#
# Usage:
#   ./scripts/start_gui.sh
#   ./scripts/start_gui.sh --share
#   ./scripts/start_gui.sh --host 0.0.0.0 --port 7860

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

exec python -m app.main "$@"
