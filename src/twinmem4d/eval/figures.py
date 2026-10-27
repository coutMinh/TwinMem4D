"""Figures and visual checks: Fig. 2 identity timeline, front-end track overlays (R1-a gate D1)."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..baselines.som import draw_marks
from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..memory.association import build_memory
from ..memory.lift import fragments_from_frontend
from .memory_eval import MemoryEvalConfig
from .metrics import match_observations


def identity_timeline(episode: Episode, frontend: FrontendResult, cfg: MemoryEvalConfig,
                      methods: Optional[Sequence[str]] = None, category: str = "bowl"):
    """Fig. 2: image-u of every GT ``category`` observation over time, one panel per method, coloured by memory node.

    A swap shows up as one colour jumping between the two bowls' tracks.
    """
    import matplotlib.pyplot as plt

    methods = list(methods or cfg.methods)
    frags = fragments_from_frontend(frontend, episode, cfg.space, cfg.keep_categories, cfg.min_observations)
    row_to_gt = match_observations(frontend, episode.gt)
    targets = set(episode.gt.instances_of(category))
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, 2.2 * len(methods)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        graph = build_memory(frags, method, cfg.space, episode.fps, **cfg.method_overrides.get(method, {}))
        for node in graph.of_category(category):
            obs = [o for o in node.observations if row_to_gt.get(o.row) in targets]
            ax.scatter([o.t for o in obs], [o.p2d[0] for o in obs], s=4, label=f"node {node.node_id}")
        ax.set_ylabel("u [px]")
        ax.set_title(method, fontsize=9)
        ax.legend(fontsize=6, markerscale=3, ncol=6)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    return fig


def track_overlays(episode: Episode, frontend: FrontendResult, n: int = 4) -> List[Tuple[int, np.ndarray, List[int]]]:
    """(t, marked image, track ids in mark order) on ``n`` frames spread over the episode."""
    out = []
    for t in np.linspace(0, episode.n_frames - 1, n).astype(int):
        rows = frontend.rows_at(int(t))
        out.append((int(t), draw_marks(episode.rgb(int(t)), [frontend.mask(r) for r in rows]),
                    [frontend.track_id[r] for r in rows]))
    return out


def tracks_table(frontend: FrontendResult, max_chars: int = 160) -> pd.DataFrame:
    """One row per track: observation count, category, DAM prompt frame and description."""
    return pd.DataFrame([
        {"track": t.track_id, "n_obs": len(frontend.rows_of(t.track_id)), "category": t.category,
         "prompt_frame": t.prompt_frame, "description": (t.description or "")[:max_chars]}
        for t in frontend.tracks.values()
    ]).sort_values("n_obs", ascending=False).reset_index(drop=True)
