"""Helpers the molab notebook calls directly: L1/L2 evidence, figures, R2 sampling, backend glue, paper tables."""

import json

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

from twinmem4d.backend_full.run import pipeline_command, stage_checkpoints, write_blocker  # noqa: E402
from twinmem4d.cli import main  # noqa: E402
from twinmem4d.data import evidence  # noqa: E402
from twinmem4d.data.convert import ConvertConfig, convert_tasks  # noqa: E402
from twinmem4d.data.episode import Episode  # noqa: E402
from twinmem4d.eval.figures import identity_timeline, track_overlays, tracks_table  # noqa: E402
from twinmem4d.eval.memory_eval import MemoryEvalConfig  # noqa: E402
from twinmem4d.eval.paper import latest_run, to_latex, write_paper_tables  # noqa: E402
from twinmem4d.frontend.oracle import oracle_frontend  # noqa: E402
from twinmem4d.probe import probe_machine  # noqa: E402
from twinmem4d.repro.dam_batch_curve import curve_summary, sample_items  # noqa: E402

from conftest import look_down_camera  # noqa: E402


def test_probe_writes_decision(tmp_path):
    probe = probe_machine(tmp_path)
    assert set(probe["decision"]) == {"full_backend_build", "hdf5_full_download"}
    assert json.loads((tmp_path / "docs" / "molab_probe.json").read_text())["decision"] == probe["decision"]


def test_probe_virtual_filesystem_is_unknown(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(8 * 2**60, 10 * 2**20, 8 * 2**60))   # molab: 8 EiB
    assert probe_machine(tmp_path)["decision"]["hdf5_full_download"] is None


def test_l1_evidence(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"rgb": rng.integers(0, 255, (10, 16, 16, 3), dtype=np.uint8),
              "depth": rng.random((10, 16, 16, 1), dtype=np.float32),
              "seg": rng.integers(0, 4, (10, 16, 16, 1))}
    frames = evidence.save_preview_frames(arrays, tmp_path / "l1")
    assert [p.name for p in frames] == ["frame_0000.png", "frame_0004.png", "frame_0009.png"]

    per_frame = [{"akita_black_bowl_1": [1, [0, 0, 1, 1], int(t > 5)], "plate_1": [2, [0, 0, 1, 1], 0]} for t in range(10)]
    assert evidence.subgoal_check_figure(per_frame, arrays["seg"]) is not None
    meta = {"task7": {"demo_100": per_frame, "demo_101": per_frame}}
    exported = evidence.export_seg_ids(meta, lambda m, task, demo: m[task][demo], {"T7": "task7", "T8": None},
                                       ["demo_100", "demo_101"], tmp_path / "seg_ids")
    assert exported == {"T7": 2}
    saved = json.loads((tmp_path / "seg_ids" / "T7" / "demo_100.json").read_text())
    assert saved["seg_ids"]["plate_1"] == [2] * 10 and saved["subgoal_increments"]["akita_black_bowl_1"] == [6]

    report = evidence.write_schema_report(tmp_path / "SCHEMA.md", ["t7.hdf5"], {"n_demos": 2}, {}, "demo_100",
                                          frames, generator="test")
    assert "## Decisions" in report.read_text()


def test_l2_camera_gate_on_flat_table():
    cam = look_down_camera()
    gate = evidence.camera_gate(np.ones((cam.height, cam.width), np.float32), cam,
                                ConvertConfig(flip_vertical=False, depth_mode="metric"))
    assert gate["passed_plane"] and gate["depth_range_m"] == [1.0, 1.0]


def test_l2_camera_gate_report(tmp_path):
    import h5py

    cam = look_down_camera()
    with h5py.File(tmp_path / "t7.hdf5", "w") as f:
        for i in (0, 1):
            f[f"data/demo_{i}/obs/agentview_depth"] = np.ones((2, cam.height, cam.width, 1), np.float32)
    report = evidence.camera_gate_report(tmp_path / "t7.hdf5", ConvertConfig(flip_vertical=False, depth_mode="metric"),
                                         tmp_path / "docs" / "l2.json", demo_index=1, camera=cam)
    saved = json.loads((tmp_path / "docs" / "l2.json").read_text())
    assert report["demo"] == "demo_1" and saved["passed_plane"]
    assert np.allclose(saved["camera"]["intrinsics"], cam.K)


def test_convert_tasks_is_resumable(tmp_path):
    import h5py

    T, rng = 5, np.random.default_rng(0)
    with h5py.File(tmp_path / "t7.hdf5", "w") as f:
        obs = f.create_group("data/demo_100/obs")
        obs["agentview_rgb"] = rng.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8)
        obs["agentview_depth"] = np.ones((T, 64, 64, 1), np.float32)
        obs["agentview_seg"] = np.full((T, 64, 64, 1), 3, np.int32)
    (tmp_path / "seg_ids" / "T7").mkdir(parents=True)
    (tmp_path / "seg_ids" / "T7" / "demo_100.json").write_text(json.dumps({"seg_ids": {"plate_1": [3] * T}}))
    cfg = ConvertConfig(flip_vertical=False, depth_mode="metric")
    args = ({"T7": tmp_path / "t7.hdf5", "T8": None}, tmp_path / "seg_ids", look_down_camera(), tmp_path / "eps", cfg)
    assert convert_tasks(*args) == ["T7_demo_100"]
    assert convert_tasks(*args) == []                                     # second call skips the done episode
    assert evidence.object_extents(Episode(tmp_path / "eps" / "T7_demo_100"))["plate_1"] is not None


def test_object_extents(swap_episode):
    sizes = evidence.object_extents(swap_episode)
    # plate radius 7 px at 1 m with f = 60 px -> ~0.23 m across; bowl radius 4 px at 0.95 m -> ~0.13 m
    assert 0.18 < sizes["plate_1"][0] < 0.28 and 0.08 < sizes["akita_black_bowl_1"][0] < 0.18


def test_figures(swap_episode):
    fe = oracle_frontend(swap_episode)
    cfg = MemoryEvalConfig(queries=MemoryEvalConfig().queries)
    assert len(identity_timeline(swap_episode, fe, cfg).axes) == len(cfg.methods)
    overlays = track_overlays(swap_episode, fe, n=2)
    assert [t for t, _, _ in overlays] == [0, swap_episode.n_frames - 1]
    assert overlays[0][1].shape[:2] == (2 * swap_episode.camera.height, 2 * swap_episode.camera.width)
    table = tracks_table(fe)
    assert len(table) == len(fe.tracks) and table.n_obs.is_monotonic_decreasing


def test_dam_curve_helpers(swap_episode, tmp_path):
    oracle_frontend(swap_episode).save(tmp_path / "fe" / swap_episode.name)
    items = sample_items([swap_episode], tmp_path / "fe", n=4)
    assert len(items) == 4 and items[0][2].dtype == bool
    rows = [{"mode": m, "n_masks": n, "s_per_mask": s} for m, n, s in
            (("batched", 1, 1.0), ("batched", 8, 0.1), ("sequential", 1, 1.0), ("sequential", 8, 1.0))]
    assert curve_summary(rows)["speedup_at_max"] == 10.0


def test_backend_glue(tmp_path, swap_episode):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "FastSAM-x.pt").write_bytes(b"x")
    stage_checkpoints(tmp_path / "ckpt", tmp_path / "daaam")
    assert (tmp_path / "daaam" / "checkpoints" / "fastsam" / "FastSAM-x.pt").exists()
    assert "full backend" in write_blocker(tmp_path, "probe: no root").read_text()
    cmd = pipeline_command(tmp_path / "daaam", swap_episode.root, "h.yaml", tmp_path / "out", 20.0, python="python3")
    assert cmd[0] == "python3" and cmd[cmd.index("--fps") + 1] == "20.0"


def test_latex_escapes():
    tex = to_latex(pd.DataFrame({"acc_%": ["1 & 2"]}, index=pd.Index(["twin_aware"], name="method")), "cap")
    assert "twin\\_aware" in tex and "acc\\_\\%" in tex and "1 \\& 2" in tex


def test_paper_tables_from_registry(swap_episode, tmp_path, capsys):
    episodes, results = swap_episode.root.parent, tmp_path / "results"
    assert latest_run(results, "frontend") is None
    main(["frontend", "--episodes", str(episodes), "--results", str(results), "--oracle"])
    fe_run = capsys.readouterr().out.strip().splitlines()[-1]
    cfg = tmp_path / "memory_eval.yaml"
    cfg.write_text(yaml.safe_dump({"queries": {"on_plate_radius_px": 8.0, "plate_merge_px": 5.0}}))
    main(["memory-eval", "--episodes", str(episodes), "--results", str(results), "--frontend-run", fe_run,
          "--config", str(cfg)])

    out = write_paper_tables(results)
    assert out["sources"] == {"memory_simplified": latest_run(results, "memory_eval").run_id}
    assert list(out["table_a"]["item"]) == ["Hydra/Khronos backend (H0)"]          # nothing else was run
    assert set(out["table_b_accuracy"].index) == {"raw", "reconcile_proxy", "twin_aware"}
    for name in ("table_a.tex", "table_b.tex", "table_b_accuracy.csv", "sources.json"):
        assert (out["dir"] / name).exists()
