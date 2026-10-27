#!/usr/bin/env bash
# Install the TwinMem4D core (data, memory, evaluation, tables) into the molab notebook kernel.
# The kernel's Python/numpy are molab's (3.13, numpy 2); the DAAAM GPU stack goes into a separate
# Python 3.12 venv, see molab_install_gpu.sh. Idempotent: skips when pyproject.toml is unchanged.
# Usage: PYTHON=/path/to/kernel/python bash scripts/molab_install.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
MARK="$REPO/.molab_installed_kernel"
STAMP="$(sha256sum "$REPO/pyproject.toml" | cut -c1-16)"

if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$STAMP" ] && "$PY" -c "import twinmem4d, h5py, spark_dsg" 2>/dev/null; then
  echo "TwinMem4D core already installed in the kernel ($STAMP)"; exit 0
fi

# Core only: numpy is left as molab ships it (no numpy<2 pin here)
if command -v uv >/dev/null; then
  uv pip install --python "$PY" -e "$REPO[data,dsg,test]"
else
  "$PY" -m pip install -e "$REPO[data,dsg,test]"
fi
echo "$STAMP" > "$MARK"
"$PY" -c "import sys, numpy, twinmem4d; print('kernel python', sys.version.split()[0], 'numpy', numpy.__version__)"
