# TwinMem4D

Twin-aware object/event memory on 4D scene graphs for history-dependent tabletop instructions.

The question this code answers: when two objects look identical, how well does a DAAAM-style 4D
scene graph keep their identity through manipulation, and how much does a manipulation-aware
object/event layer on top of it help to find "the bowl that was just moved", compared with the
current frame alone and with a VLM that sees a history of frames? Data: the LIBERO-Mem tabletop
tasks T7 (swap 2 bowls) and T8 (rotate 3 bowls).

The DAAAM front-end (FastSAM, BoT-SORT, prompt-frame selection, batched DAM, CLIP and sentence-T5
features) is vendored from [MIT-SPARK/DAAAM](https://github.com/MIT-SPARK/DAAAM) under its BSD 3-Clause
license; see [THIRD_PARTY.md](THIRD_PARTY.md). The Hydra/Khronos backend is not vendored: it is built
and run from an upstream checkout (`twinmem4d.backend_full`, `scripts/install_backend_full.sh`).

## Layout

```
src/twinmem4d/
  third_party/daaam/   vendored DAAAM front-end pieces (segmentation, tracking, frame selection, DAM, embeddings)
  data/                LIBERO-Mem HDF5 + metainfo readers, camera model, converter, ground truth,
                       L1/L2 gate evidence (preview frames, schema report, seg-id export, camera plane check)
  frontend/            synchronous DAAAM front-end runner, oracle front-end, output store
  memory/              simplified backend (lifting), memory graph (Eq. 1), events (Eq. 3),
                       association: raw / reconcile_proxy (DAAAM Sec. 3E proxy) / twin_aware (Eq. 2), spark_dsg I/O
  queries/             Q1-Q4 templates, generation from GT, solver (Eq. 4)
  baselines/           Set-of-Marks rendering, Qwen2.5-VL current-frame / history baselines
  eval/                accuracy, wrong-twin, identity split/swap, episode bootstrap, tables,
                       paper tables from the registry (CSV + LaTeX), Fig. 2 and front-end overlays
  repro/               DAM batching curve (paper Fig. 3), refCOCOg retrieval (paper Table 4)
  backend_full/        run upstream DAAAM (Hydra/Khronos) on converted episodes; H0 gate check, BLOCKERS entry
  probe.py             N0 machine probe (stdlib only, runs before install)
  pipelines.py, cli.py registered batch jobs (every run appends a row to results/registry.csv)
configs/               YAML for conversion, front-end, memory evaluation, VLM, full-backend pipeline
scripts/               molab_install.sh, install_backend_full.sh
tests/                 CPU tests on synthetic scenes
```

## Backend modes

Every result row carries `backend_mode`:

| mode | what produces the object positions and identities |
|---|---|
| `full` | upstream DAAAM: Hydra + Khronos + reconciliation (only if the build passes the H0 gate) |
| `simplified` | `memory/lift.py`: per-mask back-projection with a static, known camera. A baseline, not DAAAM's backend |
| `oracle` | ground-truth instance masks from the simulator segmentation |

## Install

Local, CPU (tests and result processing):

```bash
python -m venv .venv && .venv/bin/pip install "numpy<2" "scipy<1.15" opencv-python-headless h5py pytest pyyaml pillow pandas matplotlib spark-dsg==1.1.3
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest -q
```

molab uses two Python environments, because its kernel is fixed (Python 3.13, numpy 2) and DAAAM's
GPU stack needs numpy 1.x, which has no wheels for Python 3.13:

| environment | install | runs |
|---|---|---|
| notebook kernel (molab's Python) | `PYTHON=$(which python) bash scripts/molab_install.sh` | core: data, memory, evaluation, tables |
| `$TWINMEM_WORK/venv312` (Python 3.12, uv) | `GPU_VENV=... CKPT=... bash scripts/molab_install_gpu.sh` | every GPU job of the CLI below |

The core is tested on Python 3.12 + numpy 1.26 and on Python 3.13 + numpy 2. The molab notebook
lives outside this repository: it clones the repo, installs the core into the kernel, and runs every
long step as one of the commands below, detached, with `$TWINMEM_WORK/venv312/bin/python`.

## Command line (for unattended runs)

```bash
python -m twinmem4d.cli prefetch                                              # HF models into the HF cache
python -m twinmem4d.cli libero-download --files <hdf5> metainfo.json --dest $WORK/data/libero_mem/raw
python -m twinmem4d.cli frontend    --episodes $WORK/data/libero_mem/episodes --results $WORK/results --config configs/frontend_libero.yaml
python -m twinmem4d.cli frontend    --episodes ... --results ... --oracle
python -m twinmem4d.cli memory-eval --episodes ... --results ... --frontend-run $WORK/results/runs/<id> --config configs/memory_eval.yaml
python -m twinmem4d.cli vlm         --episodes ... --results ... --frontend-run ... --config configs/vlm.yaml --queries-config configs/memory_eval.yaml
python -m twinmem4d.cli dam-curve   --episodes ... --results ... --frontend-run ...                       # R2
python -m twinmem4d.cli refcocog    --ref-dir $WORK/data/refcocog --results ...                           # R3
python -m twinmem4d.cli backend     --stage h0|all --episodes ... --results ... --ws $WORK/ros2_ws \
    --hydra-config <tabletop yaml> --docs $WORK/docs --config configs/backend_full/pipeline_config_libero.yaml
```

## Status

Values marked VERIFY in `configs/` and in the data modules (image flip, depth encoding, metainfo
layout, frame rate) are assumptions until the LIBERO-Mem schema check (plan task L1) confirms them.
No result in this repository is final; numbers for the paper come from `results/registry.csv`.

License: to be chosen by the author before the repository is made public. The vendored DAAAM code
stays under its BSD 3-Clause license.
