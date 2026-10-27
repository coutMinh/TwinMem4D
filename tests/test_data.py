import numpy as np

from twinmem4d.data.camera import CameraModel, fit_plane_ransac
from twinmem4d.data.episode import Episode
from twinmem4d.io_utils import pack_mask, unpack_mask

from conftest import look_down_camera


def test_pack_roundtrip():
    m = np.random.default_rng(0).random((37, 53)) > 0.5
    assert np.array_equal(unpack_mask(pack_mask(m), m.shape), m)


def test_zbuffer_linearisation_inverts_projection():
    cam = look_down_camera()
    d = np.array([0.5, 1.0, 3.0])
    z = (1 - cam.znear / d) / (1 - cam.znear / cam.zfar)
    assert np.allclose(cam.zbuffer_to_metric(z), d)


def test_backproject_project_roundtrip_and_plane():
    cam = look_down_camera()
    depth = np.ones((cam.height, cam.width))
    pts = cam.backproject(depth)
    assert np.allclose(pts[:, 2], 0.0, atol=1e-9)          # table plane z = 0 in world
    uv, z = cam.project(pts[:5])
    assert np.allclose(z, 1.0)
    plane = fit_plane_ransac(pts, iters=50)
    assert plane["inlier_rms_m"] < 1e-6 and plane["inlier_ratio"] > 0.99


def test_from_fovy_matches_robosuite_convention():
    # MuJoCo camera looking down: its -z (view direction) is world -z, its y (up) is world +y
    cam = CameraModel.from_fovy(45.0, 256, 256, cam_pos=[0, 0, 1], cam_mat_gl=np.eye(3))
    assert np.isclose(cam.K[0, 0], 128 / np.tan(np.deg2rad(22.5)))
    forward_world = cam.T_wc[:3, :3] @ np.array([0, 0, 1.0])   # optical +z
    assert np.allclose(forward_world, [0, 0, -1])


def test_converted_episode_contract(swap_episode: Episode):
    ep = swap_episode
    assert ep.n_frames == 120 and ep.has_metric_depth
    assert np.allclose(ep.depth(0)[0, 0], 1.0, atol=1e-3)     # z-buffer -> metres, stored in mm
    rows = (ep.root / "pose" / "poses.txt").read_text().split()
    assert len(rows) == 16 * 120
    T = np.array(rows[:16], float).reshape(4, 4)
    assert np.allclose(T, ep.camera.T_wc)
    gt = ep.gt
    assert gt.categories.count("bowl") == 2 and gt.categories.count("plate") == 3
    p = gt.index("plate_1")
    u = gt.centroid_2d[:, p, 0]                                   # static plate: constant up to bowl occlusion
    assert np.nanmax(np.abs(u - np.nanmedian(u))) <= 3
    b = gt.index("akita_black_bowl_1")
    assert np.allclose(gt.centroid_3d[0, b, 2], 0.05, atol=0.01)  # bowl top 5 cm above the table
