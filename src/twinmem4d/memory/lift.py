"""Simplified backend (plan 3.1): back-project each tracked mask and keep a robust centroid.

X_w = T_wc [D(u, v) K^-1 (u, v, 1)^T], median over the eroded mask. No fragment fusion,
change detection, optimisation or reconciliation: acceptable only for a static, exactly
known camera, and labelled as a baseline, never as DAAAM's backend.
"""

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..third_party.daaam.geometry import compute_mask_centroid
from .graph import Fragment, Observation

logger = logging.getLogger(__name__)
_KERNEL = np.ones((3, 3), np.uint8)


def mask_centroid_3d(mask: np.ndarray, depth: np.ndarray, episode: Episode, min_points: int = 10) -> Optional[np.ndarray]:
    eroded = cv2.erode(mask.astype(np.uint8), _KERNEL).astype(bool)
    if eroded.sum() >= min_points:
        mask = eroded
    pts = episode.camera.backproject(depth, mask)
    return np.median(pts, axis=0) if len(pts) >= min_points else None


def fragments_from_frontend(
    result: FrontendResult,
    episode: Episode,
    space: str = "3d",
    keep_categories: Sequence[str] = ("bowl", "plate"),
    min_observations: int = 5,
) -> List[Fragment]:
    """One Fragment per front-end track of a kept category, with per-observation centroids."""
    if space == "3d" and not episode.has_metric_depth:
        raise ValueError("3D lifting needs metric depth; use space='2d' (plan L2 fallback)")
    depth_cache = {}
    fragments, dropped = [], {"category": 0, "short": 0}
    for tid, info in sorted(result.tracks.items()):
        if info.category not in keep_categories:
            dropped["category"] += 1
            continue
        rows = result.rows_of(tid)
        if len(rows) < min_observations:
            dropped["short"] += 1
            continue
        obs = []
        for row in rows:
            t = result.frame[row]
            mask = result.mask(row)
            c2 = compute_mask_centroid(mask)
            if c2 is None:
                continue
            p2 = np.asarray(c2, dtype=float)
            if space == "3d":
                if t not in depth_cache:
                    depth_cache = {t: episode.depth(t)}  # rows of a track are time-sorted
                p = mask_centroid_3d(mask, depth_cache[t], episode)
                if p is None:
                    continue
            else:
                p = p2
            obs.append(Observation(t=t, p=p, p2d=p2, row=row))
        if len(obs) >= min_observations:
            fragments.append(Fragment(tid, obs, info.category, info.feature, info.description))
        else:
            dropped["short"] += 1
    logger.info(f"{len(fragments)} fragments kept; dropped {dropped}")
    return fragments
