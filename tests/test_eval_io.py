import numpy as np
import pandas as pd
import pytest

from twinmem4d.eval.metrics import bootstrap_accuracy
from twinmem4d.eval.tables import accuracy_table, format_accuracy
from twinmem4d.frontend.oracle import oracle_frontend
from twinmem4d.frontend.store import FrontendResult
from twinmem4d.memory.association import build_memory
from twinmem4d.memory.lift import fragments_from_frontend
from twinmem4d.registry import Run, read_registry
from twinmem4d.repro.refcocog import retrieval_scores, topk_recall


def test_frontend_store_roundtrip(swap_episode, tmp_path):
    fe = oracle_frontend(swap_episode)
    fe.save(tmp_path / "fe")
    back = FrontendResult.load(tmp_path / "fe")
    assert back.frame == fe.frame and back.track_id == fe.track_id
    assert np.array_equal(back.mask(10), fe.mask(10))
    assert back.tracks[1].description == fe.tracks[1].description
    assert np.allclose(back.tracks[1].clip_feature, fe.tracks[1].clip_feature)


def test_bootstrap_resamples_episodes():
    eps = ["a"] * 10 + ["b"] * 10
    ci = bootstrap_accuracy(eps, [1] * 10 + [0] * 10, n_boot=200)
    assert ci["mean"] == 0.5 and ci["lo"] <= 0.5 <= ci["hi"] and ci["n_episodes"] == 2


def test_accuracy_table_shape():
    df = pd.DataFrame({"method": ["m"] * 4, "qtype": ["Q1", "Q1", "Q2", "Q2"], "episode": ["a", "b", "a", "b"],
                       "correct": [1, 0, 1, 1], "wrong_twin": [0, 1, 0, 0], "answered": [1] * 4, "latency_s": [0.0] * 4})
    t = accuracy_table(df, n_boot=50)
    assert set(t.qtype) == {"Q1", "Q2", "all"}
    assert format_accuracy(t).shape == (1, 3)


def test_registry_row(tmp_path):
    with Run(tmp_path, stage="unit") as run:
        run.metrics = {"x": 1}
    reg = read_registry(tmp_path)
    assert reg.iloc[0]["stage"] == "unit" and (tmp_path / reg.iloc[0]["metrics_path"]).exists()


def test_topk_recall():
    sim = np.array([[0.9, 0.1, 0.0], [0.2, 0.3, 0.8]])
    r = topk_recall(sim, np.array([0, 1]), ks=(1, 2))
    assert r == {"top1": 50.0, "top2": 100.0}
    s = np.eye(3)
    out = retrieval_scores(s, s, s, s, np.arange(3))
    assert all(v["top1"] == 100.0 for v in out.values())


def test_dsg_export_roundtrip(swap_episode, tmp_path):
    pytest.importorskip("spark_dsg")
    from twinmem4d.memory.dsg_io import export_memory, load_memory_summary

    fe = oracle_frontend(swap_episode)
    g = build_memory(fragments_from_frontend(fe, swap_episode, "3d"), "twin_aware", "3d", fps=20.0)
    path = export_memory(g, tmp_path / "memory_dsg.json", fps=20.0)
    nodes = load_memory_summary(path)
    assert len(nodes) == len(g.nodes)
    bowls = [n for n in nodes if n["category"] == "bowl"]
    assert len(bowls) == 2 and all(len(n["events"]) == 1 for n in bowls)
