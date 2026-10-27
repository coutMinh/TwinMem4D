# Third-party code

## DAAAM (vendored subset)

- Source: https://github.com/MIT-SPARK/DAAAM, commit `159f8daf1cf508b316c813890bdc164626842ff4`
- License: BSD 3-Clause, Copyright (c) 2026, Massachusetts Institute of Technology. The full text is in
  [`src/twinmem4d/third_party/daaam/LICENSE`](src/twinmem4d/third_party/daaam/LICENSE).
- Location: `src/twinmem4d/third_party/daaam/`. Each file's header names its upstream source and lists the changes.

| Vendored file | Upstream source | Change |
|---|---|---|
| `segmentation.py` | `utils/segmentation.py`, `segmentation/services.py` | FastSAM path only; plain checkpoint paths |
| `tracking.py` | `tracking/services.py`, `tracking/models.py` | Plain ReID path; simpler mask cache |
| `frame_selection.py` | `assignment/workers/min_frames_max_size.py` | Functions instead of a multiprocessing worker; same optimisation |
| `dam.py` | `query_manager/dam/services.py`, `grounding/utils.py`, `grounding/workers/dam_grounding.py` | No gradio/OpenAI/LangChain imports; same prompt and sampling |
| `embedding.py` | `utils/embedding.py`, `grounding/workers/dam_grounding.py` | OpenCLIP backend only; no thread locks / provenance helpers |
| `geometry.py` | `utils/geometry.py`, `utils/vision.py` | Only the helpers the front-end uses |

The Hydra/Khronos backend is not vendored. `twinmem4d.backend_full` calls an upstream DAAAM checkout
(`scripts/run_pipeline.py`) built with DAAAM's own installer.

## Runtime dependencies (installed, not vendored)

- DAAAM's forks: `boxmot` (https://github.com/nicogorlo/boxmot), `describe-anything-batch`
  (https://github.com/nicogorlo/describe-anything-batch); see their licenses.
- Models downloaded at run time: FastSAM-x (Ultralytics, AGPL-3.0), NVIDIA DAM-3B, OpenAI CLIP ViT-L/14 (via OpenCLIP),
  sentence-transformers/sentence-t5-xl, Qwen2.5-VL-7B-Instruct, a BoT-SORT ReID weight. Check each model card before
  redistributing weights or outputs.
- Data: LIBERO-Mem (https://huggingface.co/datasets/libero-mem/LIBERO-Mem; MIT on the card, CC BY 4.0 in the repo
  README), refCOCOg (umd) annotations and COCO train2014 images.
