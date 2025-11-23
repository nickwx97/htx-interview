#!/usr/bin/env bash
set -euo pipefail

# Run the full backend test suite and write logs/exit code to ../logs/tests
# Uses Conda env name from CONDA_ENV_NAME (default: htx). Falls back to system pytest.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/tests"
mkdir -p "$LOG_DIR"

ENV_NAME="${CONDA_ENV_NAME:-htx}"

echo "Running backend tests (logs -> $LOG_DIR/backend-pytest.log)"

if command -v conda >/dev/null 2>&1; then
  echo "Using conda env: $ENV_NAME"
  # Run pytest inside the conda env
  conda run -n "$ENV_NAME" pytest -q 2>&1 | tee "$LOG_DIR/backend-pytest.log"
  rc=${PIPESTATUS[0]:-0}
else
  echo "Conda not found, falling back to system pytest"
  pytest -q 2>&1 | tee "$LOG_DIR/backend-pytest.log"
  rc=${PIPESTATUS[0]:-0}
fi

echo "$rc" > "$LOG_DIR/backend_exit_code"

if [[ "$rc" -ne 0 ]]; then
  echo "Backend tests failed with exit code $rc. See $LOG_DIR/backend-pytest.log"
else
  echo "Backend tests passed."
fi

exit "$rc"
