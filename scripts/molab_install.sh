#!/usr/bin/env bash
# Install TwinMem4D + DAAAM's GPU stack into the given Python (the molab notebook kernel).
# Idempotent: skips when pyproject.toml is unchanged and the key imports work.
# Usage: PYTHON=/path/to/python bash scripts/molab_install.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
MARK="$REPO/.molab_installed"
STAMP="$(sha256sum "$REPO/pyproject.toml" | cut -c1-16)"

if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$STAMP" ] && \
   "$PY" -c "import twinmem4d, torch, boxmot, dam, open_clip, sentence_transformers, cvxpy" 2>/dev/null; then
  echo "TwinMem4D environment already installed ($STAMP)"; exit 0
fi

"$PY" -m pip install --upgrade pip
# Blackwell (sm_120) needs CUDA 12.8 wheels; same pins as DAAAM/requirements.txt
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -e "$REPO[gpu,data,dsg,test]"
echo "$STAMP" > "$MARK"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
