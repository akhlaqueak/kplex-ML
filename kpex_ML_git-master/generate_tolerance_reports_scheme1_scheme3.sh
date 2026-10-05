#!/usr/bin/env bash
set -e

RUN_TAG="${RUN_TAG:-03sep2026}"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi

echo "Python: $PYTHON_BIN"
echo "Run tag: $RUN_TAG"

"$PYTHON_BIN" restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
  --run-tag "$RUN_TAG"

echo "All tolerance workbooks completed."
