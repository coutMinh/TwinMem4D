import numpy as np
import pytest

from twinmem4d.third_party.daaam.frame_selection import (
    FrameSelectionConfig, select_prompt_frames, select_prompt_frames_greedy, track_size_reward,
)
from twinmem4d.third_party.daaam.tracking import SimplifiedTrack


def _t(tid, x, area=2000):
    return SimplifiedTrack(id=tid, bbox=np.array([x, 100, x + 40, 140]), region_area=area)


def test_reward_prefers_centred_masks():
    cfg = FrameSelectionConfig()
    assert track_size_reward(_t(1, 300), 480, 640, cfg) > track_size_reward(_t(1, 0), 480, 640, cfg)


def test_greedy_covers_all_qualified_tracks():
    cfg = FrameSelectionConfig(min_obs_per_track=2)
    history = [[_t(1, 10)], [_t(1, 10), _t(2, 300)], [_t(2, 300), _t(3, 500)], [_t(3, 500)]]
    groups, qualified, _, _ = select_prompt_frames_greedy(history, {}, {}, set(), 0, cfg)
    assert qualified == {1, 2, 3}
    assert {t for g in groups for t in g.track_ids} == {1, 2, 3}
    assert len(groups) == 2


def test_described_and_prompted_tracks_are_skipped():
    cfg = FrameSelectionConfig(min_obs_per_track=1)
    history = [[_t(1, 10), _t(2, 300)]]
    _, qualified, _, _ = select_prompt_frames_greedy(history, {1: "a bowl"}, {1: 1, 2: 2}, {2}, 0, cfg)
    assert qualified == set()


def test_mip_assigns_each_track_once():
    pytest.importorskip("cvxpy")
    cfg = FrameSelectionConfig(min_obs_per_track=2)
    history = [[_t(1, 0), _t(2, 600)], [_t(1, 300), _t(2, 300)], [_t(1, 300)]]
    groups = select_prompt_frames(history, (480, 640), {}, {}, set(), 0, cfg)
    ids = [t for g in groups for t in g.track_ids]
    assert sorted(ids) == [1, 2]
