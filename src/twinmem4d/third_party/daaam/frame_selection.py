# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/assignment/workers/min_frames_max_size.py
#     (select_prompt_images_greedy, select_prompt_images, track_size_reward)
# Modifications (TwinMem4D): the methods are plain functions instead of a multiprocessing
# worker; corrections are passed as {semantic_id: label}; the optimisation is unchanged
# (paper Eq. 1 greedy set cover -> Eq. 2 binary program with K + slack frames, Eq. 3 quality).
"""Prompt-frame selection (DAAAM paper Sec. 3B)."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .tracking import SimplifiedTrack

logger = logging.getLogger(__name__)


@dataclass
class SelectedGroup:
    """Tracks to describe together on one frame."""
    global_frame_id: int
    track_ids: List[int]


@dataclass
class FrameSelectionConfig:
    min_obs_per_track: int = 8
    N_masks_per_batch: int = 64
    min_frame_margin_slack: int = 1
    min_mask_region_area: int = 300
    position_score_weight: float = 0.5
    size_score_weight: float = 0.5
    solver_time_limit_s: float = 10.0


def track_size_reward(track: SimplifiedTrack, frame_height: int, frame_width: int, cfg: FrameSelectionConfig) -> float:
    """Quality q_ij: position entropy (favours centred masks) + tanh size score."""
    x1, y1, x2, y2 = track.bbox
    px = 0.5 * (x1 + x2) / frame_width
    py = 0.5 * (y1 + y2) / frame_height
    entropy = -px * np.log(px + 1e-8) - (1 - px) * np.log(1 - px + 1e-8)
    entropy += -py * np.log(py + 1e-8) - (1 - py) * np.log(1 - py + 1e-8)
    position_score = entropy / np.log(4)
    size_score = np.tanh(track.region_area / (cfg.min_mask_region_area * 10.0))
    return cfg.position_score_weight * position_score + cfg.size_score_weight * size_score


def _qualified_tracks(
    frame_history: List[List[SimplifiedTrack]],
    corrections: Dict[int, str],
    object_labels: Dict[int, int],
    prompted_track_ids: Set[int],
    cfg: FrameSelectionConfig,
):
    tracks_by_frame: Dict[int, List[SimplifiedTrack]] = {}
    track_observations: Dict[int, Set[int]] = {}
    valid_tracks: List[SimplifiedTrack] = []
    for frame_idx, record in enumerate(frame_history):
        if record:
            tracks_by_frame[frame_idx] = list(record)
            valid_tracks.extend(record)
        for track in record:
            track_observations.setdefault(track.id, set()).add(frame_idx)

    def is_qualified(track: SimplifiedTrack) -> bool:
        label = corrections.get(object_labels.get(track.id, -1))
        if label is not None and label.lower() != "unknown":
            return False
        if not track.depth_valid:
            return False
        return len(track_observations[track.id]) >= cfg.min_obs_per_track and track.id not in prompted_track_ids

    qualified = {t.id for t in valid_tracks if is_qualified(t)}
    return qualified, track_observations, tracks_by_frame


def select_prompt_frames_greedy(
    frame_history, corrections, object_labels, prompted_track_ids, start_frame_count, cfg: FrameSelectionConfig
) -> Tuple[List[SelectedGroup], Set[int], Dict[int, Set[int]], Dict[int, List[SimplifiedTrack]]]:
    """Paper Eq. 1: greedy set cover over frames."""
    qualified, track_observations, tracks_by_frame = _qualified_tracks(
        frame_history, corrections, object_labels, prompted_track_ids, cfg
    )
    uncovered = set(qualified)
    groups: List[SelectedGroup] = []
    while uncovered:
        best_idx, best_tracks = None, []
        for frame_idx, frame_tracks in tracks_by_frame.items():
            in_frame = [t.id for t in frame_tracks if t.id in uncovered]
            if len(in_frame) > len(best_tracks):
                best_idx, best_tracks = frame_idx, in_frame
        if best_idx is None:
            break
        for i in range(0, len(best_tracks), cfg.N_masks_per_batch):
            groups.append(SelectedGroup(start_frame_count + int(best_idx), best_tracks[i:i + cfg.N_masks_per_batch]))
        uncovered.difference_update(best_tracks)
    return groups, qualified, track_observations, tracks_by_frame


def select_prompt_frames(
    frame_history: List[List[SimplifiedTrack]],
    frame_dims: Tuple[int, int],
    corrections: Dict[int, str],
    object_labels: Dict[int, int],
    prompted_track_ids: Set[int],
    start_frame_count: int,
    cfg: FrameSelectionConfig,
) -> Optional[List[SelectedGroup]]:
    """Paper Eq. 2: pick K + slack frames maximising sum of q_ij; falls back to the greedy cover."""
    import cvxpy as cp

    greedy, qualified, track_observations, tracks_by_frame = select_prompt_frames_greedy(
        frame_history, corrections, object_labels, prompted_track_ids, start_frame_count, cfg
    )
    if not qualified:
        return None

    frame_height, frame_width = frame_dims
    n_frames = len(frame_history)
    track_ids = list(qualified)
    index = {tid: i for i, tid in enumerate(track_ids)}

    quality = np.zeros((len(track_ids), n_frames))
    for frame_idx, tracks in tracks_by_frame.items():
        for track in tracks:
            if track.id in index:
                quality[index[track.id], frame_idx] = track_size_reward(track, frame_height, frame_width, cfg)

    x = cp.Variable(n_frames, boolean=True)
    y = cp.Variable((len(track_ids), n_frames), boolean=True)
    constraints = [cp.sum(y, axis=1) == 1]
    for tid in track_ids:
        visible = track_observations[tid]
        for f in range(n_frames):
            if f not in visible:
                constraints.append(y[index[tid], f] == 0)
    for f in range(n_frames):
        constraints.append(y[:, f] <= x[f])
    k = min(len(greedy) + cfg.min_frame_margin_slack, n_frames)
    constraints.append(cp.sum(x) == k)
    problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(y, quality))), constraints)

    try:
        try:
            problem.solve(cp.GLPK_MI, glpk={"tm_lim": int(cfg.solver_time_limit_s * 1000)})
        except Exception:
            problem.solve(cp.SCIPY, scipy_options={"time_limit": cfg.solver_time_limit_s})
    except Exception as e:
        logger.warning(f"Frame-selection MIP failed ({e}); using greedy cover")
        return greedy
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None or y.value is None:
        logger.warning(f"Frame-selection MIP status {problem.status}; using greedy cover")
        return greedy

    assign = y.value > 0.5
    groups = []
    for f in np.where(x.value > 0.5)[0]:
        assigned = [track_ids[i] for i in np.where(assign[:, f])[0]]
        if assigned:
            groups.append(SelectedGroup(start_frame_count + int(f), assigned))
    return groups
