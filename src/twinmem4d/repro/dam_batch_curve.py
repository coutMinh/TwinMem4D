"""R2: DAM time per mask vs. batch size (DAAAM paper Fig. 3), on real front-end masks.

Batched = one ``describe`` call over n masks grouped by image (DAAAM's multi-image batching).
Sequential = one call per mask. The paper's curve is read only qualitatively, so the claim is
"order-of-magnitude speedup reproduced / not reproduced".
"""

import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..io_utils import PathLike
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


def sample_items(episodes: Sequence[Episode], frontend_run: PathLike, n: int = 128, tracks_per_episode: int = 16,
                 per_track: int = 2) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """``n`` real (image_key, rgb, mask) items spread over episodes, tracks and frames of a front-end run."""
    items = []
    for ep in episodes:
        res = FrontendResult.load(Path(frontend_run) / ep.name)
        for tid in list(res.tracks)[:tracks_per_episode]:
            rows = res.rows_of(tid)
            for r in rows[:: max(len(rows) // per_track, 1)][:per_track]:
                items.append((hash((ep.name, res.frame[r])), ep.rgb(res.frame[r]), res.mask(r)))
        if len(items) >= n:
            break
    return items[:n]


def curve_summary(rows: List[Dict]) -> Dict:
    """Mean/std seconds per mask per (mode, n_masks) and the sequential/batched ratio at the largest batch."""
    import pandas as pd

    agg = pd.DataFrame(rows).groupby(["mode", "n_masks"]).s_per_mask.agg(["mean", "std"]).reset_index()
    batched = agg[agg["mode"] == "batched"].set_index("n_masks")["mean"]
    sequential = agg[agg["mode"] == "sequential"].set_index("n_masks")["mean"]
    return {"s_per_mask": agg.to_dict("records"), "speedup_at_max": float(sequential.iloc[-1] / batched.iloc[-1])}


def curve_figure(rows: List[Dict]):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for mode, g in pd.DataFrame(rows).groupby("mode"):
        m = g.groupby("n_masks").s_per_mask.agg(["mean", "std"])
        ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", label=mode, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("masks per DAM call")
    ax.set_ylabel("seconds per mask")
    ax.legend()
    return fig
