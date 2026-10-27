# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/utils/segmentation.py      (UniversalSegmenter: FastSAM path, _filter_small_masks)
#   src/daaam/segmentation/services.py   (SegmentationService.segment error handling)
# Modifications (TwinMem4D): FastSAM only (EfficientViT/SAM/SAM2 branches removed);
# checkpoint/config are plain paths instead of paths relative to DAAAM's ROOT_DIR;
# TensorRT engines are still accepted.
"""FastSAM segmentation in DAAAM's (dets, masks) format."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FastSAMSegmenter:
    """FastSAM wrapper returning ``dets`` (N, 6) = [x1, y1, x2, y2, score, cls] and N bool masks."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        min_mask_region_area: int = 300,
        imgsz: Optional[Tuple[int, int]] = None,
        conf: float = 0.3,
        iou: float = 0.5,
        retina_masks: bool = True,
    ):
        from ultralytics import FastSAM

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"FastSAM checkpoint not found: {checkpoint}")
        self.device = device
        self.min_mask_region_area = min_mask_region_area
        self.is_tensorrt = str(checkpoint).lower().endswith((".engine", ".onnx"))
        self.params: Dict[str, Any] = {"conf": conf, "iou": iou, "retina_masks": retina_masks, "verbose": False}
        if imgsz is not None:
            self.params["imgsz"] = list(imgsz)
        if self.is_tensorrt:
            self.params["device"] = device
        self.model = FastSAM(str(checkpoint))
        if not self.is_tensorrt:
            self.model.to(device)

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        try:
            results = self.model(source=frame, **self.params)
        except Exception as e:  # DAAAM returns an empty result instead of crashing the stream
            logger.error(f"Error during segmentation: {e}")
            return np.empty((0, 6)), []

        dets, masks = [], []
        if results and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            if r.masks is not None:
                masks = [m.cpu().numpy().astype(bool) for m in r.masks.data]
            if masks and len(masks) != len(boxes):
                n = min(len(boxes), len(masks))
                boxes, scores, masks = boxes[:n], scores[:n], masks[:n]
            elif not masks and len(boxes) > 0:
                masks = [np.zeros(frame.shape[:2], dtype=bool) for _ in boxes]
            dets = [[*boxes[i], scores[i], 0.0] for i in range(len(boxes))]
        return self._filter_small_masks(dets, masks)

    def _filter_small_masks(self, dets: List, masks: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        if self.min_mask_region_area <= 0 or not masks:
            return (np.array(dets) if dets else np.empty((0, 6))), masks
        keep = [i for i, m in enumerate(masks) if m.sum() >= self.min_mask_region_area]
        kept_dets = np.array([dets[i] for i in keep]) if keep else np.empty((0, 6))
        return kept_dets, [masks[i] for i in keep]
