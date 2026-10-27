"""Front-end output: per-frame track observations (packed masks) + per-track DAM descriptions/features."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..io_utils import PathLike, pack_mask, read_json, unpack_mask, write_json


@dataclass
class TrackInfo:
    track_id: int
    description: Optional[str] = None
    category: Optional[str] = None
    prompt_frame: Optional[int] = None
    clip_feature: Optional[np.ndarray] = None
    sentence_feature: Optional[np.ndarray] = None

    @property
    def feature(self) -> Optional[np.ndarray]:
        """CLIP (+) sentence-T5, the fragment feature DAAAM attaches (paper Sec. 3C)."""
        parts = [f for f in (self.clip_feature, self.sentence_feature) if f is not None]
        if not parts:
            return None
        v = np.concatenate(parts)
        return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class FrontendResult:
    """Observations are rows; row i is (frame[i], track_id[i]) with bbox, area, depth and mask."""
    shape: tuple
    frame: List[int] = field(default_factory=list)
    track_id: List[int] = field(default_factory=list)
    bbox: List[np.ndarray] = field(default_factory=list)
    area: List[int] = field(default_factory=list)
    median_depth: List[float] = field(default_factory=list)
    depth_valid: List[bool] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)  # packed rows
    tracks: Dict[int, TrackInfo] = field(default_factory=dict)
    timings: Dict[str, list] = field(default_factory=dict)
    source: str = "daaam_frontend"
    _index: Optional[Dict] = field(default=None, init=False, repr=False)

    def add(self, t: int, tid: int, mask: np.ndarray, bbox, median_depth: float = 0.0, depth_valid: bool = True) -> int:
        self.frame.append(int(t))
        self.track_id.append(int(tid))
        self.bbox.append(np.asarray(bbox, dtype=np.float32))
        self.area.append(int(mask.sum()))
        self.median_depth.append(float(median_depth))
        self.depth_valid.append(bool(depth_valid))
        self.masks.append(pack_mask(mask))
        self.tracks.setdefault(int(tid), TrackInfo(int(tid)))
        self._index = None
        return len(self.frame) - 1

    def _lookup(self) -> Dict:
        if self._index is None:
            by_key, by_frame, by_track = {}, {}, {}
            for i, (f, k) in enumerate(zip(self.frame, self.track_id)):
                by_key[(f, k)] = i
                by_frame.setdefault(f, []).append(i)
                by_track.setdefault(k, []).append(i)
            self._index = {"key": by_key, "frame": by_frame, "track": by_track}
        return self._index

    def row(self, t: int, tid: int) -> Optional[int]:
        return self._lookup()["key"].get((int(t), int(tid)))

    def mask(self, row: int) -> np.ndarray:
        return unpack_mask(self.masks[row], self.shape)

    def rows_at(self, t: int) -> List[int]:
        return self._lookup()["frame"].get(int(t), [])

    def rows_of(self, tid: int) -> List[int]:
        return self._lookup()["track"].get(int(tid), [])

    def time(self, key: str, value) -> None:
        self.timings.setdefault(key, []).append(value)

    # --- persistence -----------------------------------------------------------------------
    def save(self, out_dir: PathLike) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "observations.npz",
            shape=np.array(self.shape), frame=np.array(self.frame, dtype=np.int32),
            track_id=np.array(self.track_id, dtype=np.int32), bbox=np.array(self.bbox, dtype=np.float32).reshape(-1, 4),
            area=np.array(self.area, dtype=np.int32), median_depth=np.array(self.median_depth, dtype=np.float32),
            depth_valid=np.array(self.depth_valid, dtype=bool),
            masks=np.array(self.masks, dtype=np.uint8) if self.masks else np.zeros((0, 0), np.uint8),
        )
        tids = sorted(self.tracks)
        feats = {}
        for key in ("clip_feature", "sentence_feature"):
            vecs = [getattr(self.tracks[t], key) for t in tids]
            dim = next((len(v) for v in vecs if v is not None), 0)
            feats[key] = np.array([v if v is not None else np.full(dim, np.nan) for v in vecs], dtype=np.float32)
        np.savez_compressed(out / "track_features.npz", track_id=np.array(tids, dtype=np.int32), **feats)
        write_json(out / "tracks.json", {
            "source": self.source,
            "tracks": [{"track_id": t, "description": self.tracks[t].description, "category": self.tracks[t].category,
                        "prompt_frame": self.tracks[t].prompt_frame} for t in tids],
        })
        write_json(out / "timings.json", self.timings)
        return out

    @classmethod
    def load(cls, out_dir: PathLike) -> "FrontendResult":
        out = Path(out_dir)
        z = np.load(out / "observations.npz")
        meta = read_json(out / "tracks.json")
        res = cls(shape=tuple(z["shape"].tolist()), source=meta.get("source", "daaam_frontend"))
        res.frame = z["frame"].tolist()
        res.track_id = z["track_id"].tolist()
        res.bbox = list(z["bbox"])
        res.area = z["area"].tolist()
        res.median_depth = z["median_depth"].tolist()
        res.depth_valid = z["depth_valid"].tolist()
        res.masks = list(z["masks"])
        for t in meta["tracks"]:
            res.tracks[t["track_id"]] = TrackInfo(t["track_id"], t["description"], t["category"], t["prompt_frame"])
        f = np.load(out / "track_features.npz")
        for i, tid in enumerate(f["track_id"].tolist()):
            for key in ("clip_feature", "sentence_feature"):
                if f[key].size and not np.isnan(f[key][i]).all():
                    setattr(res.tracks[tid], key, f[key][i])
        if (out / "timings.json").exists():
            res.timings = read_json(out / "timings.json")
        return res
