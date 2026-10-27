import numpy as np

from twinmem4d.eval.memory_eval import MemoryEvalConfig, evaluate_episode
from twinmem4d.frontend.oracle import oracle_frontend
from twinmem4d.frontend.store import FrontendResult
from twinmem4d.memory.association import build_memory
from twinmem4d.memory.events import EventConfig, detect_events
from twinmem4d.memory.graph import Fragment
from twinmem4d.memory.lift import fragments_from_frontend
from twinmem4d.queries.generate import QueryConfig, generate_queries

# 64-px synthetic scene: plates are 20 px apart, so the 256-px defaults (25 px) are too wide
SMALL_SCENE = QueryConfig(space="3d", on_plate_radius_px=8.0, plate_merge_px=5.0)


def test_detect_single_move():
    ts = list(range(60))
    ps = np.array([[0.0, 0, 0]] * 20 + [[0.01 * k, 0, 0] for k in range(1, 21)] + [[0.2, 0, 0]] * 20)
    ev = detect_events(ts, ps, EventConfig(eps=0.02, delta=0.05, k_min=5))
    assert len(ev) == 1
    t_s, t_e, p_s, p_e = ev[0]
    assert 18 <= t_s <= 21 and 38 <= t_e <= 41 and np.isclose(p_e[0], 0.2)


def test_jitter_is_not_a_move():
    rng = np.random.default_rng(0)
    ps = rng.normal(0, 0.003, (100, 3))
    assert detect_events(list(range(100)), ps, EventConfig(eps=0.02, delta=0.05)) == []


def split_tracks(result: FrontendResult, name_to_tid, breaks):
    """Simulate tracker breaks: a GT track is cut at each frame in ``breaks`` and gets a fresh id."""
    next_id = max(result.tracks) + 100
    for name, frames in breaks.items():
        tid = name_to_tid[name]
        for cut in frames:
            new = next_id
            next_id += 1
            for i, (f, k) in enumerate(zip(result.frame, result.track_id)):
                if k == tid and f >= cut:
                    result.track_id[i] = new
            old = result.tracks[tid]
            result.tracks[new] = type(old)(new, old.description, old.category, None, old.clip_feature, None)
            tid = new
    result._index = None
    return result


def test_queries_from_gt(swap_episode):
    qs = generate_queries(swap_episode.gt, swap_episode.name, SMALL_SCENE)
    kinds = {q.label for q in qs}
    assert {"Q1", "Q2", "Q3", "Q4"} <= kinds
    first = [q for q in qs if q.label == "Q2"][0]
    assert first.target_name == "akita_black_bowl_1"
    q4 = [q for q in qs if q.label == "Q4"][0]
    assert q4.target_name == "akita_black_bowl_2"


def test_oracle_memory_answers_everything(swap_episode):
    fe = oracle_frontend(swap_episode)
    rows, ident, _ = evaluate_episode(swap_episode, fe, MemoryEvalConfig(space="3d", queries=SMALL_SCENE, methods=("raw", "twin_aware")))
    for method in ("raw", "twin_aware"):
        acc = np.mean([r["correct"] for r in rows if r["method"] == method])
        assert acc == 1.0, method
    assert all(i["splits"] == 0 and i["swaps"] == 0 for i in ident)


def test_twin_aware_repairs_split_tracks_where_raw_fails(swap_episode):
    fe = oracle_frontend(swap_episode)
    gt = swap_episode.gt
    tid = {n: gt.index(n) + 1 for n in gt.names}
    # bowl_1 disappears while carried (frames 25-35) and comes back as a new track
    keep = [i for i, (f, k) in enumerate(zip(fe.frame, fe.track_id))
            if not (k == tid["akita_black_bowl_1"] and 25 <= f <= 35)]
    for attr in ("frame", "track_id", "bbox", "area", "median_depth", "depth_valid", "masks"):
        setattr(fe, attr, [getattr(fe, attr)[i] for i in keep])
    fe = split_tracks(fe, tid, {"akita_black_bowl_1": [36]})

    cfg = MemoryEvalConfig(space="3d", queries=SMALL_SCENE, methods=("raw", "reconcile_proxy", "twin_aware"))
    rows, ident, _ = evaluate_episode(swap_episode, fe, cfg)
    by = {i["method"]: i for i in ident}
    assert by["raw"]["splits"] == 1
    assert by["reconcile_proxy"]["splits"] == 1      # moved fragment is at another place: geometry says "new object"
    assert by["twin_aware"]["splits"] == 0 and by["twin_aware"]["swaps"] == 0
    acc = {m: np.mean([r["correct"] for r in rows if r["method"] == m and r["qtype"] in ("Q3", "Q4")])
           for m in by}
    assert acc["twin_aware"] == 1.0 and acc["raw"] < 1.0


def drop_frames(fe: FrontendResult, track: int, lo: int, hi: int) -> FrontendResult:
    keep = [i for i, (f, k) in enumerate(zip(fe.frame, fe.track_id)) if not (k == track and lo <= f <= hi)]
    for attr in ("frame", "track_id", "bbox", "area", "median_depth", "depth_valid", "masks"):
        setattr(fe, attr, [getattr(fe, attr)[i] for i in keep])
    fe._index = None
    return fe


def test_proxy_swaps_twins_at_a_reused_place_exclusion_does_not(swap_episode):
    """Both bowls are lost while carried; bowl_2 reappears exactly where bowl_1 used to stand."""
    gt = swap_episode.gt
    tid = {n: gt.index(n) + 1 for n in gt.names}
    fe = oracle_frontend(swap_episode)
    fe = drop_frames(fe, tid["akita_black_bowl_1"], 25, 35)
    fe = drop_frames(fe, tid["akita_black_bowl_2"], 75, 85)
    fe = split_tracks(fe, tid, {"akita_black_bowl_1": [36], "akita_black_bowl_2": [86]})

    cfg = MemoryEvalConfig(space="3d", queries=SMALL_SCENE, methods=("reconcile_proxy", "twin_aware"))
    _, ident, _ = evaluate_episode(swap_episode, fe, cfg)
    by = {i["method"]: i for i in ident}
    assert by["reconcile_proxy"]["swaps"] >= 1        # same place + identical features -> merged with the wrong twin
    assert by["twin_aware"]["swaps"] == 0 and by["twin_aware"]["splits"] == 0


def test_exclusion_rule_blocks_observed_nodes():
    def frag(fid, t0, t1, x):
        from twinmem4d.memory.graph import Observation
        obs = [Observation(t, np.array([x, 0, 0.0]), np.array([x * 100, 0.0]), None) for t in range(t0, t1)]
        return Fragment(fid, obs, "bowl", np.ones(4) / 2)

    # node A observed until 50; B continuously observed; new fragment C starts at 30 near B -> cannot join B
    frags = [frag(0, 0, 50, 0.0), frag(1, 0, 100, 0.3), frag(2, 30, 100, 0.31)]
    g = build_memory(frags, "twin_aware", "3d", fps=20.0)
    node_of = g.node_of_fragment()
    assert node_of[2] != node_of[1]
    assert node_of[2] != node_of[0]    # A was still observed when C started


def test_lift_2d_mode(swap_episode):
    fe = oracle_frontend(swap_episode)
    frags = fragments_from_frontend(fe, swap_episode, space="2d")
    assert len(frags) == 5 and frags[0].observations[0].p.shape == (2,)
