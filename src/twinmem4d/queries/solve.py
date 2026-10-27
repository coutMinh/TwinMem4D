"""Answer queries from a memory graph (plan Eq. 4), using only observations up to t_q."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..frontend.store import FrontendResult
from ..memory.events import EventConfig, node_events
from ..memory.graph import MemoryGraph, ObjectNode
from .templates import Query, on_plate, plate_order


@dataclass
class Answer:
    qid: str
    node_id: Optional[int]
    mask_row: Optional[int]       # FrontendResult row holding the returned mask
    mask_age: Optional[int]       # t_q minus the frame of that mask (0 = currently observed)
    ambiguous: bool = False
    reason: str = ""


def _plates(graph: MemoryGraph, t_q: int, merge_px: float):
    """Static plate landmarks: plate nodes seen up to t_q, merged when their image positions coincide.

    Plates are not query targets, so a plate split by arm occlusion is merged for every method alike.
    """
    centres: List[np.ndarray] = []
    weights: List[int] = []
    for n in graph.of_category("plate"):
        hist = n.history(t_q)
        if not hist:
            continue
        p = np.median([o.p2d for o in hist], axis=0)
        d = [np.linalg.norm(p - c) for c in centres]
        if d and min(d) <= merge_px:
            k = int(np.argmin(d))
            centres[k] = (centres[k] * weights[k] + p * len(hist)) / (weights[k] + len(hist))
            weights[k] += len(hist)
        else:
            centres.append(p)
            weights.append(len(hist))
    return centres, plate_order(centres)


def _rank_at(node: ObjectNode, t: int, plate_pos, rank_of, radius: float) -> Optional[int]:
    obs = node.last_observation(t)
    if obs is None:
        return None
    k = on_plate(obs.p2d, plate_pos, radius)
    return None if k is None else rank_of[k]


def solve(query: Query, graph: MemoryGraph, events_cfg: EventConfig, on_plate_radius_px: float = 25.0,
          plate_merge_px: float = 10.0) -> Answer:
    t_q = query.t_q
    candidates = [n for n in graph.of_category(query.category) if n.t_start <= t_q]
    if not candidates:
        return Answer(query.qid, None, None, None, reason="no candidate node")

    chosen: List[ObjectNode] = []
    if query.qtype in ("current_on", "origin"):
        plate_pos, order = _plates(graph, t_q, merge_px=plate_merge_px)
        if len(plate_pos) != query.n_plates:
            return Answer(query.qid, None, None, None, reason=f"{len(plate_pos)} plates in memory, query assumes {query.n_plates}")
        rank_of = {idx: r for r, idx in enumerate(order)}
        if query.qtype == "current_on":
            chosen = [n for n in candidates if _rank_at(n, t_q, plate_pos, rank_of, on_plate_radius_px) == query.plate_rank]
            chosen.sort(key=lambda n: -n.last_observation(t_q).t)       # most recently confirmed first
        else:
            chosen = [n for n in candidates if _rank_at(n, n.t_start, plate_pos, rank_of, on_plate_radius_px) == query.plate_rank]
            chosen.sort(key=lambda n: n.t_start)                        # earliest origin first
    elif query.qtype == "just_moved":
        ends = [(max(e.t_end for e in evs), n) for n in candidates if (evs := node_events(n, events_cfg, t_q))]
        chosen = [n for _, n in sorted(ends, key=lambda x: -x[0])]
    elif query.qtype == "not_yet_moved":
        chosen = [n for n in candidates if not node_events(n, events_cfg, t_q)]
        chosen.sort(key=lambda n: -len(n.history(t_q)))
    else:
        raise ValueError(query.qtype)

    if not chosen:
        return Answer(query.qid, None, None, None, reason="no node satisfies the predicate")
    node = chosen[0]
    obs = next((o for o in reversed(node.history(t_q)) if o.row is not None), None)
    return Answer(
        query.qid, node.node_id, obs.row if obs else None, (t_q - obs.t) if obs else None,
        ambiguous=len(chosen) > 1, reason="" if obs else "node has no mask up to t_q",
    )


def answer_mask(answer: Answer, frontend: FrontendResult) -> Optional[np.ndarray]:
    return frontend.mask(answer.mask_row) if answer.mask_row is not None else None
