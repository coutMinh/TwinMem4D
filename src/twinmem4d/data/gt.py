"""Ground-truth instance masks, trajectories and move intervals from the simulator segmentation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..categories import category_of
from ..io_utils import PathLike, pack_mask, unpack_mask
from ..third_party.daaam.geometry import compute_mask_centroid
from .camera import CameraModel


@dataclass
class GroundTruth:
    names: List[str]
    categories: List[str]
    masks: np.ndarray        # (T, N, packed) uint8
    shape: tuple             # (H, W)
    centroid_2d: np.ndarray  # (T, N, 2) pixels (u, v); NaN when not visible
    centroid_3d: np.ndarray  # (T, N, 3) world metres; NaN when not visible or no depth
    area: np.ndarray         # (T, N) pixels

    @property
    def n_frames(self) -> int:
        return self.masks.shape[0]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def mask(self, t: int, i: int) -> np.ndarray:
        return unpack_mask(self.masks[t, i], self.shape)

    def visible(self, t: int, i: int, min_area: int = 1) -> bool:
        return self.area[t, i] >= min_area

    def instances_of(self, category: str) -> List[int]:
        return [i for i, c in enumerate(self.categories) if c == category]

    def save(self, path: PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, names=np.array(self.names), categories=np.array(self.categories), masks=self.masks,
            shape=np.array(self.shape), centroid_2d=self.centroid_2d, centroid_3d=self.centroid_3d, area=self.area,
        )

    @classmethod
    def load(cls, path: PathLike) -> "GroundTruth":
        z = np.load(path)
        return cls(
            names=z["names"].tolist(), categories=z["categories"].tolist(), masks=z["masks"],
            shape=tuple(z["shape"].tolist()), centroid_2d=z["centroid_2d"], centroid_3d=z["centroid_3d"], area=z["area"],
        )


def build_gt(
    seg: np.ndarray,
    seg_ids: Dict[str, np.ndarray],
    depth_m: Optional[np.ndarray] = None,
    camera: Optional[CameraModel] = None,
    keep_categories: Sequence[str] = ("bowl", "plate"),
) -> GroundTruth:
    """``seg`` (T, H, W) instance ids; ``seg_ids`` name -> per-frame id (-1 = absent) or a constant."""
    T, H, W = seg.shape
    names = [n for n in sorted(seg_ids) if category_of(n) in keep_categories]
    if not names:
        raise ValueError(f"no objects of categories {keep_categories} in {sorted(seg_ids)}")
    N = len(names)
    packed_len = len(pack_mask(np.zeros((H, W), bool)))
    masks = np.zeros((T, N, packed_len), dtype=np.uint8)
    c2 = np.full((T, N, 2), np.nan)
    c3 = np.full((T, N, 3), np.nan)
    area = np.zeros((T, N), dtype=np.int64)
    for i, name in enumerate(names):
        ids = np.broadcast_to(np.asarray(seg_ids[name]), (T,))
        for t in range(T):
            if ids[t] < 0:
                continue
            m = seg[t] == ids[t]
            a = int(m.sum())
            if a == 0:
                continue
            area[t, i] = a
            masks[t, i] = pack_mask(m)
            c2[t, i] = compute_mask_centroid(m)
            if depth_m is not None and camera is not None:
                pts = camera.backproject(depth_m[t], m)
                if len(pts):
                    c3[t, i] = np.median(pts, axis=0)
    return GroundTruth(names, [category_of(n) for n in names], masks, (H, W), c2, c3, area)
