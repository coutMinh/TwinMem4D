"""Generate Q1-Q4 with GT targets from an episode's ground truth (plan M1).

Query times are ``offset`` frames after each GT move ends. GT moves come from the GT
trajectories with the same Eq. 3 detector used by the memory (on noise-free positions).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..data.gt import GroundTruth
from ..memory.events import EventConfig, detect_events
from .templates import Query, on_plate, plate_order


@dataclass
class QueryConfig:
    offset: int = 10
    on_plate_radius_px: float = 25.0
    plate_merge_px: float = 10.0     # memory plate nodes closer than this are one plate (must be < plate spacing)
    min_visible_area: int = 30
    category: str = "bowl"
    space: str = "3d"


def gt_trajectory(gt: GroundTruth, i: int, space: str, min_area: int) -> Tuple[List[int], np.ndarray]:
    src = gt.centroid_3d if space == "3d" else gt.centroid_2d
    ts = [t for t in range(gt.n_frames) if gt.area[t, i] >= min_area and np.all(np.isfinite(src[t, i]))]
    return ts, src[ts, i]


def gt_events(gt: GroundTruth, cfg: QueryConfig) -> Dict[int, List[Tuple[int, int]]]:
    """GT instance -> list of (t_start, t_end) moves."""
    ev_cfg = EventConfig.for_space(cfg.space)
    out = {}
    for i in gt.instances_of(cfg.category):
        ts, ps = gt_trajectory(gt, i, cfg.space, cfg.min_visible_area)
        out[i] = [(a, b) for a, b, _, _ in detect_events(ts, ps, ev_cfg)] if len(ts) else []
    return out


def last_position_2d(gt: GroundTruth, i: int, t: int, min_area: int):
    for s in range(t, -1, -1):
        if gt.area[s, i] >= min_area:
            return gt.centroid_2d[s, i]
    return None


def generate_queries(gt: GroundTruth, episode: str, cfg: QueryConfig) -> List[Query]:
    bowls = gt.instances_of(cfg.category)
    plates = gt.instances_of("plate")
    plate_pos = [np.nanmedian(gt.centroid_2d[:, p], axis=0) for p in plates]
    order = plate_order(plate_pos)                  # rank -> index into plates
    rank_of = {idx: r for r, idx in enumerate(order)}
    events = gt_events(gt, cfg)
    ends = sorted((te, b) for b, evs in events.items() for _, te in evs)

    def plate_rank_at(b: int, t: int):
        p = last_position_2d(gt, b, t, cfg.min_visible_area)
        k = None if p is None else on_plate(p, plate_pos, cfg.on_plate_radius_px)
        return None if k is None else rank_of[k]

    first_t = {b: next((t for t in range(gt.n_frames) if gt.area[t, b] >= cfg.min_visible_area), None) for b in bowls}
    origin = {b: plate_rank_at(b, first_t[b]) for b in bowls if first_t[b] is not None}

    queries, seen = [], set()

    def add(qtype: str, t_q: int, target: int, rank=None):
        key = (qtype, t_q, rank)
        if key in seen or gt.area[t_q, target] < cfg.min_visible_area:
            return
        seen.add(key)
        queries.append(Query(
            qid=f"{episode}/{qtype}/{t_q}/{rank}", episode=episode, qtype=qtype, t_q=t_q, category=cfg.category,
            plate_rank=rank, n_plates=len(plates) if rank is not None else None,
            target=target, target_name=gt.names[target],
        ))

    for t_end, _ in ends:
        t_q = min(t_end + cfg.offset, gt.n_frames - 1)
        moved = {b for b, evs in events.items() if any(te <= t_q for _, te in evs)}
        latest = max(((te, b) for b, evs in events.items() for _, te in evs if te <= t_q), default=None)
        if latest is not None:
            add("just_moved", t_q, latest[1])
        ranks_now = {b: plate_rank_at(b, t_q) for b in bowls}
        for r in range(len(plates)):
            on_r = [b for b, rk in ranks_now.items() if rk == r]
            if len(on_r) == 1:
                add("current_on", t_q, on_r[0], r)
        for b, r0 in origin.items():
            if r0 is not None and ranks_now.get(b) != r0:
                add("origin", t_q, b, r0)
        unmoved = [b for b in bowls if b not in moved]
        if len(unmoved) == 1 and moved:
            add("not_yet_moved", t_q, unmoved[0])
    return queries
