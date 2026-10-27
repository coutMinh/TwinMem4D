"""Static pinhole camera for LIBERO/robosuite renders (plan task L2).

Conventions: intrinsics K and ``T_wc`` (camera-to-world) use the OpenCV optical frame
(x right, y down, z forward) for images stored top-row-first. MuJoCo cameras look along
-z with y up, so their rotation is corrected by diag(1, -1, -1), as robosuite's
``get_camera_extrinsic_matrix`` does.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..io_utils import PathLike, read_json, write_json

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraModel:
    K: np.ndarray            # (3, 3)
    T_wc: np.ndarray         # (4, 4) camera -> world
    width: int
    height: int
    znear: Optional[float] = None   # metres (MuJoCo znear * extent)
    zfar: Optional[float] = None

    @classmethod
    def from_fovy(cls, fovy_deg: float, width: int, height: int, cam_pos: np.ndarray,
                  cam_mat_gl: np.ndarray, znear: Optional[float] = None, zfar: Optional[float] = None) -> "CameraModel":
        """``cam_mat_gl`` is MuJoCo's camera orientation (data.cam_xmat reshaped to 3x3)."""
        f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
        K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = np.asarray(cam_mat_gl).reshape(3, 3) @ GL_TO_CV
        T[:3, 3] = np.asarray(cam_pos).reshape(3)
        return cls(K=K, T_wc=T, width=width, height=height, znear=znear, zfar=zfar)

    @classmethod
    def from_mujoco(cls, model, data, camera_name: str, width: int, height: int) -> "CameraModel":
        """Read a camera from MuJoCo ``model``/``data`` (mujoco bindings or robosuite's MjSim wrappers)."""
        cam_id = model.camera_name2id(camera_name) if hasattr(model, "camera_name2id") else _mj_name2id(model, camera_name)
        extent = float(model.stat.extent)
        return cls.from_fovy(
            fovy_deg=float(model.cam_fovy[cam_id]),
            width=width, height=height,
            cam_pos=np.array(data.cam_xpos[cam_id]),
            cam_mat_gl=np.array(data.cam_xmat[cam_id]).reshape(3, 3),
            znear=float(model.vis.map.znear) * extent,
            zfar=float(model.vis.map.zfar) * extent,
        )

    @property
    def intrinsics(self) -> Dict[str, float]:
        return {"fx": float(self.K[0, 0]), "fy": float(self.K[1, 1]), "cx": float(self.K[0, 2]), "cy": float(self.K[1, 2])}

    def zbuffer_to_metric(self, z: np.ndarray) -> np.ndarray:
        """OpenGL depth buffer in [0, 1] -> metric z-depth (robosuite ``get_real_depth_map``)."""
        if self.znear is None or self.zfar is None:
            raise ValueError("znear/zfar are required to linearise a z-buffer")
        near, far = self.znear, self.zfar
        return near / (1.0 - z.astype(np.float64) * (1.0 - near / far))

    def backproject(self, depth: np.ndarray, mask: Optional[np.ndarray] = None, world: bool = True) -> np.ndarray:
        """(N, 3) points for pixels with finite positive depth (inside ``mask`` if given)."""
        valid = np.isfinite(depth) & (depth > 0)
        if mask is not None:
            valid &= mask
        v, u = np.nonzero(valid)
        z = depth[v, u]
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
        if world:
            pts = pts @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]
        return pts

    def project(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """World points -> (pixels (N, 2) as (u, v), depth z (N,))."""
        T_cw = np.linalg.inv(self.T_wc)
        pc = points_world @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = pc[:, 2]
        uv = (pc[:, :2] / z[:, None]) * np.array([self.K[0, 0], self.K[1, 1]]) + np.array([self.K[0, 2], self.K[1, 2]])
        return uv, z

    # --- DAAAM ImageSequenceDataset contract -------------------------------------------------
    def to_camera_info(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "intrinsics": self.K.tolist(),
            "distortion": [0.0] * 5,
            "T_wc": self.T_wc.tolist(),
            "znear": self.znear,
            "zfar": self.zfar,
        }

    @classmethod
    def from_camera_info(cls, info: Dict) -> "CameraModel":
        return cls(
            K=np.array(info["intrinsics"], dtype=float),
            T_wc=np.array(info.get("T_wc", np.eye(4)), dtype=float),
            width=int(info["width"]),
            height=int(info["height"]),
            znear=info.get("znear"),
            zfar=info.get("zfar"),
        )

    def save(self, path: PathLike) -> None:
        write_json(path, self.to_camera_info())

    @classmethod
    def load(cls, path: PathLike) -> "CameraModel":
        return cls.from_camera_info(read_json(path))


def _mj_name2id(model, name: str) -> int:
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise KeyError(f"camera '{name}' not found")
    return cam_id


def fit_plane_ransac(points: np.ndarray, iters: int = 500, threshold: float = 0.005,
                     seed: int = 0) -> Dict[str, float]:
    """RANSAC plane fit; L2 gate uses the inlier RMS residual (target <~ 1 cm on the table)."""
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        dist = np.abs((points - p[0]) @ n)
        inliers = dist < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None:
        raise ValueError("degenerate point set")
    inl = points[best_inliers]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal = vt[-1]
    residual = (inl - centroid) @ normal
    return {
        "normal": normal.tolist(),
        "offset": float(-normal @ centroid),
        "inlier_ratio": float(best_inliers.mean()),
        "inlier_rms_m": float(np.sqrt(np.mean(residual ** 2))),
    }
