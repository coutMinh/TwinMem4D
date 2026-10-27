"""LIBERO-Mem ``metainfo.json`` helpers (plan task L1).

The dataset card describes, per frame and camera, ``"<object>": [seg_id, [cx, cy, w, h], obj_subgoal]``.
How frames map to (task, demo, step) is not documented: ``summarize`` prints the real
structure, and ``object_table`` only accepts the per-frame format above (VERIFY first).
"""

from typing import Any, Dict, List, Sequence

import numpy as np


def summarize(obj: Any, max_depth: int = 5, max_keys: int = 6, _depth: int = 0) -> Any:
    """Compact structural sketch of a large JSON object."""
    if _depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        keys = list(obj.keys())
        sketch = {k: summarize(obj[k], max_depth, max_keys, _depth + 1) for k in keys[:max_keys]}
        if len(keys) > max_keys:
            sketch["..."] = f"{len(keys)} keys total, e.g. {keys[-1]!r}"
        return sketch
    if isinstance(obj, list):
        head = [summarize(v, max_depth, max_keys, _depth + 1) for v in obj[:2]]
        return head + ([f"... len={len(obj)}"] if len(obj) > 2 else [])
    return obj


def object_table(frames: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-object arrays from a list of per-frame dicts ``{name: [seg_id, [cx, cy, w, h], subgoal]}``.

    Missing objects in a frame get seg_id -1, NaN box and the previous subgoal count.
    """
    names: List[str] = sorted({name for fr in frames for name in fr})
    T = len(frames)
    table = {}
    for name in names:
        seg = np.full(T, -1, dtype=np.int64)
        box = np.full((T, 4), np.nan)
        sub = np.zeros(T, dtype=np.int64)
        last = 0
        for t, fr in enumerate(frames):
            entry = fr.get(name)
            if entry is None:
                sub[t] = last
                continue
            if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
                raise ValueError(f"unexpected metainfo entry for {name!r} at frame {t}: {entry!r} (VERIFY format)")
            seg[t] = int(entry[0])
            box[t] = np.asarray(entry[1], dtype=float)
            last = int(entry[2])
            sub[t] = last
        table[name] = {"seg_id": seg, "box_cxcywh": box, "subgoal": sub}
    return table


def seg_id_map(table: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """name -> per-frame seg id, the input format of ``gt.build_gt``."""
    return {name: cols["seg_id"] for name, cols in table.items()}


def subgoal_increments(table: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, List[int]]:
    """Frames where an object's interaction counter increases (cross-check for GT moves)."""
    return {name: (np.nonzero(np.diff(cols["subgoal"]) > 0)[0] + 1).tolist() for name, cols in table.items()}
