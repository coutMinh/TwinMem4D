"""LIBERO-Mem demo -> converted episode folder (plan task L3). See ``episode.py`` for the layout."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from ..io_utils import PathLike, write_json
from .camera import CameraModel
from .episode import DEPTH_SCALE, Episode
from .gt import build_gt


@dataclass
class ConvertConfig:
    flip_vertical: bool = True          # robosuite renders bottom row first; L1 decides
    depth_mode: str = "zbuffer"         # "zbuffer" (normalised [0, 1]), "metric" (metres) or "none"
    fps: float = 20.0                   # LIBERO control rate; VERIFY
    keep_categories: Sequence[str] = field(default_factory=lambda: ("bowl", "plate"))


def depth_to_metric(depth: np.ndarray, camera: CameraModel, mode: str) -> Optional[np.ndarray]:
    depth = np.squeeze(depth, axis=-1) if depth.ndim == 4 else depth
    if mode == "zbuffer":
        return camera.zbuffer_to_metric(depth).astype(np.float32)
    if mode == "metric":
        return depth.astype(np.float32)
    if mode == "none":
        return None
    raise ValueError(f"unknown depth_mode {mode!r}")


def to_depth_png(depth_m: np.ndarray) -> np.ndarray:
    mm = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m * DEPTH_SCALE, 0)
    return np.clip(np.round(mm), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def convert_demo(
    arrays: Dict[str, np.ndarray],
    camera: CameraModel,
    out_dir: PathLike,
    cfg: ConvertConfig,
    seg_ids: Optional[Dict[str, np.ndarray]] = None,
    source: Optional[Dict] = None,
) -> Episode:
    """``arrays`` holds ``rgb`` (T, H, W, 3), optional ``depth`` (T, H, W[, 1]) and ``seg`` (T, H, W[, 1])."""
    out = Path(out_dir)
    for sub in ("rgb", "depth", "pose", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    flip = (lambda a: a[:, ::-1]) if cfg.flip_vertical else (lambda a: a)
    rgb = flip(arrays["rgb"])
    T = len(rgb)
    depth_m = None
    if "depth" in arrays and cfg.depth_mode != "none":
        depth_m = flip(depth_to_metric(arrays["depth"], camera, cfg.depth_mode))
    seg = None
    if "seg" in arrays:
        seg = arrays["seg"]
        seg = flip(np.squeeze(seg, axis=-1) if seg.ndim == 4 else seg)

    for t in range(T):
        cv2.imwrite(str(out / "rgb" / f"{t:06d}.png"), cv2.cvtColor(np.ascontiguousarray(rgb[t]), cv2.COLOR_RGB2BGR))
        if depth_m is not None:
            cv2.imwrite(str(out / "depth" / f"{t:06d}.png"), to_depth_png(depth_m[t]))

    pose_row = " ".join(f"{v:.9g}" for v in camera.T_wc.reshape(-1))
    (out / "pose" / "poses.txt").write_text("\n".join([pose_row] * T) + "\n")
    camera.save(out / "camera_info.json")

    n_gt = 0
    if seg is not None and seg_ids:
        gt = build_gt(seg, seg_ids, depth_m, camera if depth_m is not None else None, cfg.keep_categories)
        gt.save(out / "gt" / "gt.npz")
        n_gt = len(gt.names)

    write_json(out / "meta.json", {
        "n_frames": T,
        "fps": cfg.fps,
        "depth_scale": DEPTH_SCALE,
        "metric_depth": depth_m is not None,
        "convert": asdict(cfg),
        "n_gt_objects": n_gt,
        "source": source or {},
    })
    return Episode(out)
