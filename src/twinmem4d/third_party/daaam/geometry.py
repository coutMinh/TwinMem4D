# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/utils/geometry.py  (compute_mask_centroid "median" branch, unproject_pixel_to_3d, pose_to_matrix)
#   src/daaam/utils/vision.py    (fast_median_depth, bounding_box_from_mask, mask_to_polygons,
#                                 polygons_to_mask, compute_polygon_area)
# Modifications (TwinMem4D): only the functions used by the front-end are kept;
# compute_mask_centroid keeps the default "median" method only.
"""Mask and pinhole-geometry helpers used by DAAAM's front-end."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Median pixel of a binary mask as (cx, cy); snaps to the nearest mask pixel if outside."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    cy = int(np.median(ys))
    cx = int(np.median(xs))
    if not mask[cy, cx]:
        nearest = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        cy, cx = int(ys[nearest]), int(xs[nearest])
    return cx, cy


def unproject_pixel_to_3d(u: float, v: float, depth: float,
                          fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pinhole back-projection of one pixel into the camera frame."""
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] -> 4x4 homogeneous transform."""
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return T


def fast_median_depth(depth_image: np.ndarray, mask: np.ndarray) -> float:
    """Median of the finite, positive depth values inside ``mask``; -1 if none."""
    depths = depth_image[mask]
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        return -1.0
    return float(np.median(valid))


def bounding_box_from_mask(mask: np.ndarray) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] of a binary mask (cv2.boundingRect convention)."""
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return np.array([x, y, x + w, y + h])


def mask_to_polygons(mask: np.ndarray, epsilon_factor: float = 0.001) -> List[np.ndarray]:
    """External contours of a mask, simplified with Douglas-Peucker scaled by the image diagonal."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if epsilon_factor > 0 and len(contours) > 0:
        h, w = mask.shape
        epsilon = epsilon_factor * np.sqrt(h * h + w * w)
        contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    return list(contours)


def polygons_to_mask(contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if contours:
        cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool)


def compute_polygon_area(contours: List[np.ndarray]) -> int:
    return int(sum(cv2.contourArea(c) for c in contours))
