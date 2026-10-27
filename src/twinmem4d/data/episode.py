"""Converted episode folder = DAAAM's ``ImageSequenceDataset`` contract + our ``gt/`` and ``meta.json``.

    <episode>/
      rgb/000000.png            RGB, top row first
      depth/000000.png          uint16 millimetres (depth_scale = 1000), 0 = invalid
      pose/poses.txt            one row-major 4x4 camera->world per line (static camera)
      camera_info.json          width, height, intrinsics (3x3), T_wc, znear, zfar
      meta.json                 task, demo, fps, conversion flags, source file
      gt/gt.npz                 GroundTruth (instance masks, centroids)
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..io_utils import PathLike, read_json
from .camera import CameraModel
from .gt import GroundTruth

DEPTH_SCALE = 1000.0


@dataclass
class Episode:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @cached_property
    def meta(self) -> Dict[str, Any]:
        return read_json(self.root / "meta.json")

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @cached_property
    def n_frames(self) -> int:
        return len(list((self.root / "rgb").glob("*.png")))

    @cached_property
    def camera(self) -> CameraModel:
        return CameraModel.load(self.root / "camera_info.json")

    @cached_property
    def gt(self) -> Optional[GroundTruth]:
        path = self.root / "gt" / "gt.npz"
        return GroundTruth.load(path) if path.exists() else None

    @property
    def has_metric_depth(self) -> bool:
        return bool(self.meta.get("metric_depth", False))

    def rgb(self, t: int) -> np.ndarray:
        img = cv2.imread(str(self.root / "rgb" / f"{t:06d}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / "rgb" / f"{t:06d}.png")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def depth(self, t: int) -> Optional[np.ndarray]:
        """Metric depth in metres, or None if the episode has no usable depth."""
        path = self.root / "depth" / f"{t:06d}.png"
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32) / DEPTH_SCALE


def list_episodes(root: PathLike):
    return [Episode(p.parent) for p in sorted(Path(root).glob("*/meta.json"))]
