"""R2: DAM time per mask vs. batch size (DAAAM paper Fig. 3), on real front-end masks.

Batched = one ``describe`` call over n masks grouped by image (DAAAM's multi-image batching).
Sequential = one call per mask. The paper's curve is read only qualitatively, so the claim is
"order-of-magnitude speedup reproduced / not reproduced".
"""

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..third_party.daaam.dam import DAMAgent

DEFAULT_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def _group(pairs: Sequence[Tuple[int, np.ndarray, np.ndarray]]):
    """(image_key, rgb, mask) items -> [(rgb, [masks])] keeping one entry per image."""
    by_image: Dict[int, Tuple[np.ndarray, List[np.ndarray]]] = {}
    for key, rgb, mask in pairs:
        by_image.setdefault(key, (rgb, []))[1].append(mask)
    return list(by_image.values())


def _timed(fn) -> float:
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def batch_curve(dam: DAMAgent, items: Sequence[Tuple[int, np.ndarray, np.ndarray]], sizes=DEFAULT_SIZES,
                repeats: int = 3, sequential_max: int = 16, warmup: int = 1) -> List[Dict]:
    """``items`` are (image_key, rgb, mask). Returns one row per (size, mode, repeat)."""
    for _ in range(warmup):
        dam.describe(_group(items[:1]))
    rows = []
    for n in sizes:
        if n > len(items):
            break
        subset = items[:n]
        for r in range(repeats):
            s = _timed(lambda: dam.describe(_group(subset)))
            rows.append({"n_masks": n, "mode": "batched", "repeat": r, "seconds": s, "s_per_mask": s / n,
                         "n_images": len(_group(subset))})
            m = min(n, sequential_max)
            s = _timed(lambda: [dam.describe(_group([it])) for it in subset[:m]])
            rows.append({"n_masks": n, "mode": "sequential", "repeat": r, "seconds": s, "s_per_mask": s / m,
                         "n_images": m})
    return rows
