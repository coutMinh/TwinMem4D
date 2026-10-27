"""Turn fragments into object nodes.

* ``raw``             one node per fragment (tracker IDs as they are).
* ``reconcile_proxy`` our re-implementation of DAAAM Sec. 3E: merge fragments with similar
                      geometry and descriptive features. The paper gives no thresholds, so the
                      ones here are guesses and the method is a *proxy*, not DAAAM.
* ``twin_aware``      ours (plan Eq. 2): a new fragment may only join a node that is not
                      currently observed and has a compatible category (exclusion); the rest
                      is a Hungarian assignment on time gap, appearance and distance to the
                      node's last position.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import Fragment, MemoryGraph, ObjectNode

INF = 1e9


@dataclass
class ProxyConfig:
    max_distance: float       # geometry: distance between fragment centroids (m or px)
    min_cosine: float = 0.8   # descriptive features: cosine of CLIP+sentence features

    @classmethod
    def for_space(cls, space: str) -> "ProxyConfig":
        return cls(max_distance=0.05) if space == "3d" else cls(max_distance=12.0)


@dataclass
class TwinAwareConfig:
    lambda_t: float = 0.1       # per second of gap
    lambda_a: float = 1.0       # (1 - cosine)
    lambda_s: float = 1.0       # per unit of ``distance_scale``
    distance_scale: float = 0.1  # 10 cm in 3D, 25 px in 2D (see for_space)
    max_cost: float = 3.0       # gate: above this a fragment starts a new node
    max_gap_s: float = 30.0
    batch_window: int = 3       # fragments starting within this many frames are assigned jointly
    fps: float = 20.0
    tail_k: int = 5

    @classmethod
    def for_space(cls, space: str, fps: float) -> "TwinAwareConfig":
        return cls(distance_scale=0.1 if space == "3d" else 25.0, fps=fps)


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or len(a) != len(b):
        return None
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _compatible(a: Optional[str], b: Optional[str]) -> bool:
    return a is None or b is None or a == b


def build_raw(fragments: List[Fragment], space: str) -> MemoryGraph:
    nodes = {i: ObjectNode(i, f.category, [f]) for i, f in enumerate(sorted(fragments, key=lambda f: f.t_start))}
    return MemoryGraph(nodes, "raw", space)


def build_reconcile_proxy(fragments: List[Fragment], space: str, cfg: Optional[ProxyConfig] = None) -> MemoryGraph:
    """Greedy, time-ordered merge of non-overlapping fragments at the same place with similar features.

    Geometry is each fragment's median position, the analogue of the shape Khronos reconstructs
    while the object is static (brief motion at the ends of a fragment barely moves the median).
    """
    cfg = cfg or ProxyConfig.for_space(space)
    nodes: Dict[int, ObjectNode] = {}
    for frag in sorted(fragments, key=lambda f: f.t_start):
        best, best_d = None, INF
        for node in nodes.values():
            if not _compatible(node.category, frag.category) or node.t_end >= frag.t_start:
                continue
            latest = max(node.fragments, key=lambda f: f.t_end)
            d = float(np.linalg.norm(latest.median_position() - frag.median_position()))
            cos = _cos(node.feature, frag.feature)
            if d <= cfg.max_distance and (cos is None or cos >= cfg.min_cosine) and d < best_d:
                best, best_d = node, d
        if best is None:
            nid = len(nodes)
            nodes[nid] = ObjectNode(nid, frag.category, [frag])
        else:
            best.fragments.append(frag)
            best.category = best.category or frag.category
    return MemoryGraph(nodes, "reconcile_proxy", space)


def twin_aware_cost(node: ObjectNode, frag: Fragment, cfg: TwinAwareConfig) -> float:
    """Plan Eq. 2; INF if the exclusion rule or a gate forbids the link.

    The distance term saturates (tanh): it ranks candidates, but an object carried far while
    occluded can still rejoin its node when it is the only eligible one.
    """
    if not _compatible(node.category, frag.category) or node.t_end >= frag.t_start:
        return INF
    gap_s = (frag.t_start - node.t_end) / cfg.fps
    if gap_s > cfg.max_gap_s:
        return INF
    cos = _cos(node.feature, frag.feature)
    appearance = 0.5 if cos is None else 1.0 - cos
    dist = float(np.linalg.norm(frag.head_position(cfg.tail_k) - node.tail_position(cfg.tail_k)))
    return cfg.lambda_t * gap_s + cfg.lambda_a * appearance + cfg.lambda_s * np.tanh(dist / cfg.distance_scale)


def build_twin_aware(fragments: List[Fragment], space: str, cfg: TwinAwareConfig) -> MemoryGraph:
    nodes: Dict[int, ObjectNode] = {}
    ordered = sorted(fragments, key=lambda f: f.t_start)
    i = 0
    while i < len(ordered):
        batch = [ordered[i]]
        while i + len(batch) < len(ordered) and ordered[i + len(batch)].t_start - batch[0].t_start <= cfg.batch_window:
            batch.append(ordered[i + len(batch)])
        i += len(batch)

        candidates = list(nodes.values())
        assigned = {}
        if candidates:
            cost = np.array([[twin_aware_cost(n, f, cfg) for f in batch] for n in candidates])
            rows, cols = linear_sum_assignment(cost)
            assigned = {c: candidates[r] for r, c in zip(rows, cols) if cost[r, c] <= cfg.max_cost}
        for j, frag in enumerate(batch):
            node = assigned.get(j)
            if node is None:
                nid = len(nodes)
                nodes[nid] = ObjectNode(nid, frag.category, [frag])
            else:
                node.fragments.append(frag)
                node.category = node.category or frag.category
    return MemoryGraph(nodes, "twin_aware", space)


def build_memory(fragments: List[Fragment], method: str, space: str, fps: float, **overrides) -> MemoryGraph:
    if method == "raw":
        return build_raw(fragments, space)
    if method == "reconcile_proxy":
        cfg = ProxyConfig.for_space(space)
        return build_reconcile_proxy(fragments, space, _override(cfg, overrides))
    if method == "twin_aware":
        cfg = TwinAwareConfig.for_space(space, fps)
        return build_twin_aware(fragments, space, _override(cfg, overrides))
    raise ValueError(f"unknown association method {method!r}")


def _override(cfg, overrides: Dict):
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"{type(cfg).__name__} has no field {k!r}")
        setattr(cfg, k, v)
    return cfg
