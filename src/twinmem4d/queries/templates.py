"""History-dependent query types (plan 3.3) and their wording."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

QTYPES = ("current_on", "just_moved", "origin", "not_yet_moved")
QTYPE_LABEL = {"current_on": "Q1", "just_moved": "Q2", "origin": "Q3", "not_yet_moved": "Q4"}


def plate_rank_name(rank: int, n_plates: int) -> str:
    named = {2: ["left", "right"], 3: ["left", "middle", "right"],
             4: ["leftmost", "second from the left", "second from the right", "rightmost"]}
    if n_plates in named:
        return named[n_plates][rank]
    return f"{rank + 1}-th from the left"


@dataclass
class Query:
    qid: str
    episode: str
    qtype: str
    t_q: int
    category: str = "bowl"
    plate_rank: Optional[int] = None
    n_plates: Optional[int] = None
    target: int = -1                 # GT instance index (never visible to methods)
    target_name: str = ""

    @property
    def label(self) -> str:
        return QTYPE_LABEL[self.qtype]

    @property
    def text(self) -> str:
        c = self.category
        if self.qtype == "current_on":
            return f"the {c} on the {plate_rank_name(self.plate_rank, self.n_plates)} plate"
        if self.qtype == "just_moved":
            return f"the {c} that was just moved"
        if self.qtype == "origin":
            return f"the {c} that started on the {plate_rank_name(self.plate_rank, self.n_plates)} plate"
        if self.qtype == "not_yet_moved":
            return f"the {c} that has not been moved yet"
        raise ValueError(self.qtype)


def plate_order(plate_positions_2d: List[np.ndarray]) -> List[int]:
    """Indices of plates sorted left to right by image u."""
    return list(np.argsort([p[0] for p in plate_positions_2d]))


def on_plate(p2d: np.ndarray, plate_positions_2d: List[np.ndarray], radius_px: float) -> Optional[int]:
    """Index of the nearest plate within ``radius_px`` of an object's image centroid, else None."""
    if not plate_positions_2d:
        return None
    d = [np.linalg.norm(np.asarray(p2d) - p) for p in plate_positions_2d]
    i = int(np.argmin(d))
    return i if d[i] <= radius_px else None
