#!/usr/bin/env bash
# Python 3.12 venv with DAAAM's GPU stack (numpy 1.x, torch cu128, boxmot/DAM forks, cvxpy) for every
# `python -m twinmem4d.cli` GPU job. The molab kernel cannot hold this stack: it is Python 3.13 and
# numpy 1.x has no wheels for 3.13+. Writes $GPU_VENV/.twinmem4d_ready only when every step passed.
# Usage: GPU_VENV=$WORK/venv312 CKPT=$WORK/checkpoints bash scripts/molab_install_gpu.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU_VENV:?set GPU_VENV (e.g. \$TWINMEM_WORK/venv312)}"
: "${CKPT:?set CKPT (e.g. \$TWINMEM_WORK/checkpoints)}"
GPY="$GPU_VENV/bin/python"
READY="$GPU_VENV/.twinmem4d_ready"
STAMP="$(sha256sum "$REPO/pyproject.toml" | cut -c1-16)"
step() { echo; echo "=== $* ($(date -u +%H:%M:%S) UTC)"; }

if [ -f "$READY" ] && [ "$(cat "$READY")" = "$STAMP" ]; then
  echo "GPU venv already installed ($STAMP): $GPY"; exit 0
fi
rm -f "$READY"

step "venv: Python 3.12 via uv (downloads a managed CPython if none is installed)"
[ -x "$GPY" ] || uv venv --python 3.12 "$GPU_VENV"
"$GPY" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.version)"

step "torch 2.8.0 + torchvision 0.23.0, CUDA 12.8 wheels (Blackwell sm_120; same pins as DAAAM)"
uv pip install --python "$GPY" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

step "TwinMem4D with the DAAAM GPU stack"
uv pip install --python "$GPY" -e "$REPO[gpu,data,dsg,test]"

step "weights: FastSAM-x"
mkdir -p "$CKPT"
[ -f "$CKPT/FastSAM-x.pt" ] || curl -L --fail -o "$CKPT/FastSAM-x.pt" \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-x.pt

step "ReID: boxmot downloads osnet_x0_25_msmt17.pt on first use (VERIFY)"
"$GPY" - "$CKPT/osnet_x0_25_msmt17.pt" <<'EOF'
import sys
import numpy as np
from twinmem4d.third_party.daaam.tracking import BotSortTracker

out = BotSortTracker(sys.argv[1], device="cuda").update(
    np.array([[10, 10, 60, 60, 0.9, 0]], dtype=np.float32), np.zeros((128, 128, 3), np.uint8))
print("tracker output", out.shape)
EOF

step "tests (incl. the frame-selection MIP, which needs cvxpy)"
(cd "$REPO" && "$GPY" -m pytest -q)

step "torch sees the GPU"
"$GPY" -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

step "HF models into the HF cache (DAM-3B, sentence-t5-xl, Qwen2.5-VL-7B; ~30 GB)"
"$GPY" -m twinmem4d.cli prefetch

echo "$STAMP" > "$READY"
step "GPU venv ready: $GPY"
