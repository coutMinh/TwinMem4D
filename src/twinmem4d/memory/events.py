"""Move events (plan Eq. 3): displacement > delta between two stationary periods of >= k observations."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .graph import Event, MemoryGraph, ObjectNode


@dataclass
class EventConfig:
    eps: float          # max distance to the running segment median to count as "stationary"
    delta: float        # min displacement between stationary periods to count as a move
    k_min: int = 5      # min observations in a stationary period

    @classmethod
    def for_space(cls, space: str) -> "EventConfig":
        # 3D in metres, 2D in pixels of a 256-px frame; tuned on nothing yet (VERIFY on GT tracks)
        return cls(eps=0.02, delta=0.05) if space == "3d" else cls(eps=4.0, delta=12.0)


def stationary_segments(times: Sequence[int], positions: np.ndarray, eps: float,
                        k_min: int) -> List[Tuple[int, int, np.ndarray]]:
    """Maximal runs that stay within ``eps`` of their running median -> (t_first, t_last, median)."""
    segments, current = [], []
    for t, p in zip(times, positions):
        if current and np.linalg.norm(p - np.median([q for _, q in current], axis=0)) > eps:
            segments.append(current)
            current = []
        current.append((t, p))
    if current:
        segments.append(current)
    return [(seg[0][0], seg[-1][0], np.median([q for _, q in seg], axis=0)) for seg in segments if len(seg) >= k_min]


def detect_events(times: Sequence[int], positions: np.ndarray, cfg: EventConfig) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    segs = stationary_segments(times, np.asarray(positions, dtype=float), cfg.eps, cfg.k_min)
    events = []
    for (_, a_end, a_pos), (b_start, _, b_pos) in zip(segs, segs[1:]):
        if np.linalg.norm(b_pos - a_pos) > cfg.delta:
            events.append((a_end, b_start, a_pos, b_pos))
    return events


def node_events(node: ObjectNode, cfg: EventConfig, until: int = None) -> List[Event]:
    """Events of one node from its history up to ``until`` (causal: no future observations)."""
    hist = node.history(until)
    if len(hist) < 2 * cfg.k_min:
        return []
    found = detect_events([o.t for o in hist], np.array([o.p for o in hist]), cfg)
    return [Event(node.node_id, ts, te, ps, pe) for ts, te, ps, pe in found]


def graph_events(graph: MemoryGraph, cfg: EventConfig, until: int = None) -> List[Event]:
    return sorted((e for n in graph.nodes.values() for e in node_events(n, cfg, until)), key=lambda e: e.t_end)
