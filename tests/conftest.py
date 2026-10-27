"""Synthetic tabletop episodes: plates in a row, identical bowls, scripted pick-and-place moves."""

from typing import Dict, List, Tuple

import numpy as np
import pytest

from twinmem4d.data.camera import CameraModel
from twinmem4d.data.convert import ConvertConfig, convert_demo

H = W = 64


def look_down_camera() -> CameraModel:
    """Camera 1 m above the table origin looking straight down (OpenCV frame, world z up)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.0, 0.0, 1.0]
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    return CameraModel(K=K, T_wc=T, width=W, height=H, znear=0.01, zfar=10.0)


def disk(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def scripted_scene(T: int, plates: List[Tuple[float, float]], bowl_paths: Dict[str, List[Tuple[float, float]]]):
    """seg (T, H, W) with ids: plates 1..P, bowls 10+; bowls drawn over plates; depth 1 m table, 0.95 m objects."""
    seg = np.zeros((T, H, W), np.int32)
    depth = np.ones((T, H, W), np.float32)
    rgb = np.full((T, H, W, 3), 90, np.uint8)
    ids = {}
    for k, (u, v) in enumerate(plates):
        ids[f"plate_{k + 1}"] = k + 1
        for t in range(T):
            m = disk(u, v, 7)
            seg[t][m] = k + 1
            depth[t][m] = 0.99
            rgb[t][m] = 230
    for k, (name, path) in enumerate(sorted(bowl_paths.items())):
        ids[name] = 10 + k
        for t, (u, v) in enumerate(path):
            m = disk(u, v, 4)
            seg[t][m] = 10 + k
            depth[t][m] = 0.95
            rgb[t][m] = (200, 40, 40)
    return rgb, depth, seg, ids


def move(path_from, path_to, t_start, t_end, T):
    """Static at ``path_from`` until t_start, linear to ``path_to`` by t_end, static after."""
    out = []
    for t in range(T):
        a = np.clip((t - t_start) / max(t_end - t_start, 1), 0, 1)
        out.append(tuple(np.array(path_from) * (1 - a) + np.array(path_to) * a))
    return out


@pytest.fixture
def swap_episode(tmp_path):
    """T7-like: two identical bowls on the left/middle plates, right plate empty; bowl_1 -> right, then bowl_2 -> left."""
    T = 120
    plates = [(12, 32), (32, 32), (52, 32)]
    paths = {
        "akita_black_bowl_1": move((12, 32), (52, 32), 20, 40, T),
        "akita_black_bowl_2": move((32, 32), (12, 32), 70, 90, T),
    }
    rgb, depth, seg, ids = scripted_scene(T, plates, paths)
    cam = look_down_camera()
    # store depth as a z-buffer to exercise the conversion
    near, far = cam.znear, cam.zfar
    zbuf = (1 - near / depth) / (1 - near / far)
    ep = convert_demo({"rgb": rgb, "depth": zbuf, "seg": seg}, cam, tmp_path / "ep_swap",
                      ConvertConfig(flip_vertical=False, depth_mode="zbuffer", fps=20.0), seg_ids=ids)
    return ep
