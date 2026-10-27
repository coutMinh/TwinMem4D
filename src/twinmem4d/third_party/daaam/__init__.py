"""Subset of DAAAM (https://github.com/MIT-SPARK/DAAAM) vendored for TwinMem4D.

Upstream commit: 159f8daf1cf508b316c813890bdc164626842ff4
License: BSD 3-Clause, Copyright (c) 2026, Massachusetts Institute of Technology
(see LICENSE in this folder).

Only the front-end pieces used by TwinMem4D are vendored. They were taken out of
DAAAM's multiprocessing services so they can be called synchronously from a
notebook. Each file lists its upstream source and what was changed. The
algorithms (FastSAM post-processing, BoT-SORT wrapping, frame selection, DAM
batch prompting, CLIP / sentence-T5 features) are unchanged.

The Hydra/Khronos backend is *not* vendored. The full DAAAM pipeline runs from
an upstream checkout (see ``twinmem4d.backend_full``).
"""

UPSTREAM_URL = "https://github.com/MIT-SPARK/DAAAM"
UPSTREAM_COMMIT = "159f8daf1cf508b316c813890bdc164626842ff4"
