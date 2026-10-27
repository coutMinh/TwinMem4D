"""Evidence for the L1 (schema) and L2 (camera) gates, written under ``docs/`` and ``data/``."""

import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..io_utils import PathLike
from . import metainfo as mi
from .camera import CameraModel, fit_plane_ransac
from .convert import ConvertConfig, depth_to_metric
from .episode import Episode

SCHEMA_DECISIONS = """## Decisions (fill in; they go into configs/convert_libero.yaml)
- split attribute: ...            -> val demos = ...
- depth encoding: zbuffer | metric -> depth_mode = ...
- vertical flip needed: yes | no  -> flip_vertical = ...
- seg ids are: geom | instance    -> how seg_id maps to object names: ...
- metainfo frame lookup: ...      -> frames_of_demo(...) expression
- obj_subgoal counts interactions: yes | no (evidence: plot)
- fps: ...
"""


def _colorize(seg: np.ndarray) -> np.ndarray:
    lut = np.random.default_rng(0).integers(0, 255, (int(seg.max()) + 1, 3), dtype=np.uint8)
    return lut[seg]


def save_preview_frames(arrays: Mapping[str, np.ndarray], out_dir: PathLike, n: int = 3) -> List[Path]:
    """First / middle / last frame as ``RGB | normalised depth | seg overlay`` PNGs (is the image upside down?)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = len(arrays["rgb"])
    saved = []
    for t in np.linspace(0, T - 1, n).astype(int):
        rgb = arrays["rgb"][t]
        d = np.squeeze(arrays["depth"][t]).astype(np.float32)
        d8 = (255 * (d - d.min()) / (np.ptp(d) + 1e-9)).astype(np.uint8)
        seg = _colorize(np.squeeze(arrays["seg"][t]).astype(np.int64))
        row = np.concatenate([rgb, cv2.cvtColor(d8, cv2.COLOR_GRAY2RGB), (0.5 * rgb + 0.5 * seg).astype(np.uint8)], 1)
        path = out / f"frame_{t:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        saved.append(path)
    return saved


def subgoal_check_figure(frames: Sequence[Dict[str, Any]], seg: np.ndarray, category: str = "bowl"):
    """Mask-centroid u and ``obj_subgoal`` per object over time: does the counter step when the object moves?"""
    import matplotlib.pyplot as plt

    table = mi.object_table(frames)
    seg = np.squeeze(seg)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for name, cols in table.items():
        if category not in name:
            continue
        u = [np.nonzero(seg[t] == sid)[1].mean() if sid >= 0 and (seg[t] == sid).any() else np.nan
             for t, sid in enumerate(cols["seg_id"][:len(seg)])]
        axes[0].plot(u, label=name)
        axes[1].plot(cols["subgoal"], label=name)
    axes[0].set_ylabel("mask centroid u [px]")
    axes[1].set_ylabel("obj_subgoal")
    axes[1].set_xlabel("frame")
    axes[0].legend(fontsize=7)
    return fig


def write_schema_report(path: PathLike, files: Sequence[str], structure: Any, value_ranges: Any, demo: Optional[str],
                        frames: Sequence[Path], generator: str) -> Path:
    """``docs/LIBERO_MEM_SCHEMA.md``: printed evidence plus the decisions the user fills in by hand."""
    path = Path(path)
    path.write_text(
        f"# LIBERO-Mem schema (plan task L1)\n\n"
        f"Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC by {generator}. Files: {list(files)}\n\n"
        f"## Evidence\n- Structure: `{structure}`\n- Value ranges (demo {demo}): `{value_ranges}`\n"
        f"- Frames: {[p.name for p in frames]} in {Path(frames[0]).parent if frames else '-'}\n\n"
        + SCHEMA_DECISIONS
    )
    return path


def export_seg_ids(meta: Any, frames_of_demo: Callable[[Any, str, str], Sequence[Dict]], task_keys: Mapping[str, str],
                   demos: Sequence[str], out_root: PathLike) -> Dict[str, int]:
    """``<out_root>/<task>/<demo>.json`` = per-frame seg id per object name + interaction-counter steps (input of L3)."""
    exported = {}
    for task, key in task_keys.items():
        if not key:
            continue
        out = Path(out_root) / task
        out.mkdir(parents=True, exist_ok=True)
        for demo in demos:
            table = mi.object_table(frames_of_demo(meta, key, demo))
            (out / f"{demo}.json").write_text(json.dumps({
                "seg_ids": {n: c["seg_id"].tolist() for n, c in table.items()},
                "subgoal_increments": mi.subgoal_increments(table),
            }))
        exported[task] = len(demos)
    return exported


def camera_from_libero_env(env_args: Mapping[str, Any], camera: str = "agentview", size: int = 256) -> CameraModel:
    """Build the task's LIBERO env once and read the camera from MuJoCo.

    VERIFY: LIBERO stores ``env_name``/``env_kwargs`` in ``data.attrs["env_args"]``; the bddl path may need re-rooting.
    """
    from libero.libero.envs import OffScreenRenderEnv

    kwargs = dict(env_args["env_kwargs"])
    kwargs.update(camera_heights=size, camera_widths=size)
    env = OffScreenRenderEnv(**{k: v for k, v in kwargs.items() if k in ("bddl_file_name", "camera_heights", "camera_widths")})
    try:
        env.reset()
        sim = env.env.sim
        return CameraModel.from_mujoco(sim.model, sim.data, camera, size, size)
    finally:
        env.close()


def camera_gate(depth_frame: np.ndarray, camera: CameraModel, cfg: ConvertConfig, rms_max_m: float = 0.01) -> Dict:
    """L2 gate on one raw depth frame: the table (lowest large surface) must be a plane with inlier RMS < 1 cm."""
    d = depth_to_metric(depth_frame[None], camera, cfg.depth_mode)[0]
    d = d[::-1] if cfg.flip_vertical else d
    pts = camera.backproject(d)
    low = pts[pts[:, 2] <= np.percentile(pts[:, 2], 60)]   # table = lowest large surface
    plane = fit_plane_ransac(low, threshold=0.005)
    return {"plane": plane, "depth_range_m": [float(np.nanmin(d)), float(np.nanmax(d))],
            "passed_plane": plane["inlier_rms_m"] < rms_max_m}


def object_extents(episode: Episode) -> Dict[str, Optional[List[float]]]:
    """Axis-aligned 3D extent (m) of each GT object at its largest visible frame; bowls/plates should be ~0.1-0.2 m."""
    gt = episode.gt
    sizes = {}
    for i, name in enumerate(gt.names):
        t = int(gt.area[:, i].argmax())
        depth = episode.depth(t)
        pts = episode.camera.backproject(depth, gt.mask(t, i)) if depth is not None else np.zeros((0, 3))
        sizes[name] = (pts.max(0) - pts.min(0)).round(3).tolist() if len(pts) else None
    return sizes
