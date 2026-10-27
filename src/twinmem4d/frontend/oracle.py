"""Oracle front-end: tracks are the ground-truth instance masks (upper bound on perception)."""

import numpy as np

from ..data.episode import Episode
from ..third_party.daaam.geometry import bounding_box_from_mask
from .store import FrontendResult


def oracle_frontend(episode: Episode, frame_stride: int = 1, min_area: int = 20) -> FrontendResult:
    """Track id = GT index + 1, description = object name; features are one-hot per category."""
    gt = episode.gt
    if gt is None:
        raise ValueError(f"{episode.root} has no gt/gt.npz")
    result = FrontendResult(shape=gt.shape, source="oracle")
    cats = sorted(set(gt.categories))
    for t in range(0, gt.n_frames, frame_stride):
        depth = episode.depth(t)
        for i in range(len(gt.names)):
            if gt.area[t, i] < min_area:
                continue
            m = gt.mask(t, i)
            med = float(np.median(depth[m])) if depth is not None else 0.0
            result.add(t, i + 1, m, bounding_box_from_mask(m), med, True)
    for i, (name, cat) in enumerate(zip(gt.names, gt.categories)):
        if i + 1 in result.tracks:
            info = result.tracks[i + 1]
            info.description, info.category = name, cat
            # identical twins must look identical to the memory: the feature is category-only
            info.clip_feature = np.eye(len(cats))[cats.index(cat)].astype(np.float32)
    return result
