"""Object memory (plan Eq. 1): nodes v = (category, feature, history H_v, events E_v)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    t: int                   # frame index
    p: np.ndarray            # position used for events: world metres (3D) or pixels (2D)
    p2d: np.ndarray          # image (u, v); used for on-plate relations
    row: Optional[int]       # row of the mask in the FrontendResult, if any


@dataclass
class Fragment:
    """A temporally consistent piece of one object: a front-end track or a DSG object."""
    fid: int
    observations: List[Observation]
    category: Optional[str] = None
    feature: Optional[np.ndarray] = None
    description: Optional[str] = None
    source: str = "track"

    def __post_init__(self):
        self.observations.sort(key=lambda o: o.t)

    @property
    def t_start(self) -> int:
        return self.observations[0].t

    @property
    def t_end(self) -> int:
        return self.observations[-1].t

    def head_position(self, k: int = 5) -> np.ndarray:
        return np.median([o.p for o in self.observations[:k]], axis=0)

    def tail_position(self, k: int = 5) -> np.ndarray:
        return np.median([o.p for o in self.observations[-k:]], axis=0)

    def median_position(self) -> np.ndarray:
        return np.median([o.p for o in self.observations], axis=0)


@dataclass
class Event:
    """Plan Eq. 3: a move between two stationary periods."""
    node_id: int
    t_start: int
    t_end: int
    p_start: np.ndarray
    p_end: np.ndarray


@dataclass
class ObjectNode:
    node_id: int
    category: Optional[str]
    fragments: List[Fragment] = field(default_factory=list)

    @property
    def observations(self) -> List[Observation]:
        return sorted((o for f in self.fragments for o in f.observations), key=lambda o: o.t)

    @property
    def t_start(self) -> int:
        return min(f.t_start for f in self.fragments)

    @property
    def t_end(self) -> int:
        return max(f.t_end for f in self.fragments)

    @property
    def feature(self) -> Optional[np.ndarray]:
        feats = [f.feature for f in self.fragments if f.feature is not None]
        if not feats:
            return None
        v = np.mean(feats, axis=0)
        return v / (np.linalg.norm(v) + 1e-12)

    def history(self, until: Optional[int] = None) -> List[Observation]:
        obs = self.observations
        return obs if until is None else [o for o in obs if o.t <= until]

    def last_observation(self, until: int) -> Optional[Observation]:
        hist = self.history(until)
        return hist[-1] if hist else None

    def first_observation(self) -> Observation:
        return self.observations[0]

    def tail_position(self, k: int = 5) -> np.ndarray:
        last = max(self.fragments, key=lambda f: f.t_end)
        return last.tail_position(k)


@dataclass
class MemoryGraph:
    nodes: Dict[int, ObjectNode]
    method: str
    space: str               # "3d" or "2d"

    def of_category(self, category: str) -> List[ObjectNode]:
        return [n for n in self.nodes.values() if n.category == category]

    def node_of_fragment(self) -> Dict[int, int]:
        return {f.fid: n.node_id for n in self.nodes.values() for f in n.fragments}
