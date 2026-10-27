# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/tracking/services.py  (BotSortAdapter, TrackingService.update: +1 on track ids)
#   src/daaam/tracking/models.py    (SimplifiedTrack, Track with polygon-encoded masks)
# Modifications (TwinMem4D): ReID weights are a plain path (not relative to DAAAM's ROOT_DIR);
# the weak-ref mask cache of Track is reduced to a plain cache.
"""BoT-SORT tracking and DAAAM's track data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .geometry import compute_polygon_area, mask_to_polygons, polygons_to_mask


class BotSortTracker:
    """BoT-SORT (DAAAM's boxmot fork) returning (M, 8) = [x1, y1, x2, y2, track_id, conf, cls, det_idx]."""

    def __init__(
        self,
        reid_weights: str,
        device: str = "cuda",
        track_buffer: int = 30,
        with_reid: bool = True,
        reid_half: bool = False,
    ):
        from boxmot import BotSort

        self.tracker = BotSort(
            reid_weights=Path(reid_weights),
            device=0 if device == "cuda" else "cpu",
            half=reid_half,
            track_buffer=track_buffer,
            with_reid=with_reid,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return np.empty((0, 8))
        tracks = self.tracker.update(detections, frame)
        # boxmot returns shape (0,) when a frame has detections but no confirmed track
        if tracks.ndim == 1:
            tracks = tracks.reshape(1, -1)
        if len(tracks) == 0 or tracks.shape[1] < 8:
            return np.empty((0, 8))
        tracks[:, 4] += 1  # DAAAM reserves id 0 for "unknown"
        return tracks


@dataclass
class SimplifiedTrack:
    """Track summary handed to the frame-selection step (no mask)."""
    id: int
    bbox: np.ndarray
    depth_valid: bool = True
    region_area: int = 0
    median_depth: float = 0.0


@dataclass
class Track:
    """Per-frame track observation with its mask stored as polygons."""
    id: int
    segmentation_contours: List[np.ndarray]
    bbox: np.ndarray
    image_shape: Tuple[int, int]
    depth_valid: bool = True
    region_area: int = 0
    median_depth: float = 0.0
    _mask_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @classmethod
    def from_mask(cls, id: int, mask: np.ndarray, bbox: np.ndarray, epsilon_factor: float = 0.001, **kwargs) -> "Track":
        contours = mask_to_polygons(mask, epsilon_factor)
        return cls(
            id=id,
            segmentation_contours=contours,
            bbox=np.asarray(bbox),
            image_shape=mask.shape,
            region_area=compute_polygon_area(contours),
            **kwargs,
        )

    @property
    def segmentation(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = polygons_to_mask(self.segmentation_contours, self.image_shape)
        return self._mask_cache

    def clear_cache(self) -> None:
        self._mask_cache = None

    def get_simplified(self) -> SimplifiedTrack:
        return SimplifiedTrack(
            id=self.id,
            bbox=self.bbox,
            depth_valid=self.depth_valid,
            region_area=self.region_area,
            median_depth=self.median_depth,
        )
