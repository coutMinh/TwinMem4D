"""Set-of-Marks rendering: numbered candidate masks on the current frame."""

from typing import List, Sequence

import cv2
import numpy as np

from ..third_party.daaam.geometry import compute_mask_centroid

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60)]


def draw_marks(rgb: np.ndarray, masks: Sequence[np.ndarray], scale: int = 2) -> np.ndarray:
    """Upscale, outline each mask in its colour and print its 1-based number at the mask centroid."""
    img = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for k, mask in enumerate(masks):
        m = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        color = PALETTE[k % len(PALETTE)]
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
        c = compute_mask_centroid(m.astype(bool))
        if c is None:
            continue
        label = str(k + 1)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x, y = c[0] - w // 2, c[1] + h // 2
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + 3), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def history_frames(t_q: int, k: int) -> List[int]:
    """Plan B1: k = 1 is the current frame only; k in {4, 8, 16} adds k past frames uniformly over [0, t_q)."""
    if k <= 1 or t_q <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, t_q - 1, k)})
