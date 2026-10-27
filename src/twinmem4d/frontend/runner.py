"""DAAAM front-end (paper stages A-C) run synchronously on one converted episode (plan R1).

Mirrors ``PipelineOrchestrator.process_frame`` of DAAAM: FastSAM -> BoT-SORT -> per-track depth
validation -> sliding window -> prompt-frame selection every ``query_interval_frames`` ->
batched DAM descriptions -> CLIP crop + sentence-T5 features. The multiprocessing queues of
the upstream pipeline are replaced by direct calls, so timings are per stage and per batch.
Stage D (Hydra/Khronos) is not run here.
"""

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ..categories import DEFAULT_VOCAB, category_of
from ..data.episode import Episode
from ..third_party.daaam.frame_selection import FrameSelectionConfig, SelectedGroup, select_prompt_frames
from ..third_party.daaam.geometry import bounding_box_from_mask, fast_median_depth
from ..third_party.daaam.tracking import SimplifiedTrack, Track
from .store import FrontendResult

logger = logging.getLogger(__name__)


@dataclass
class FrontendConfig:
    # segmentation (DAAAM: FastSAM-x, min area 300 px at 640x480; lowered for 256 px frames)
    fastsam_checkpoint: str = "checkpoints/FastSAM-x.pt"
    imgsz: Optional[Tuple[int, int]] = (256, 256)
    min_mask_region_area: int = 50
    max_mask_region_area: int = 256 * 256 // 4
    fastsam_conf: float = 0.3
    fastsam_iou: float = 0.5
    polygon_epsilon_factor: float = 0.001
    # tracking
    reid_weights: str = "checkpoints/osnet_x0_25_msmt17.pt"
    with_reid: bool = True
    reid_half: bool = False
    track_buffer: int = 30
    # depth validation (DAAAM: 0.25-5 m for rooms)
    depth_lb: float = 0.1
    depth_ub: float = 3.0
    # semantic lifting
    query_interval_frames: int = 90
    first_query_frame: int = 60          # DAAAM queries once early at frame 60
    multi_image_min_n_masks: int = 32
    frame_selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)
    dam_model_path: str = "nvidia/DAM-3B"
    dam_conv_mode: str = "v1"
    dam_prompt_mode: str = "focal_prompt"
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    sentence_model: str = "sentence-transformers/sentence-t5-xl"
    min_crop_px: int = 10
    # run control
    device: str = "cuda"
    frame_stride: int = 1
    max_frames: Optional[int] = None
    category_vocab: Tuple[str, ...] = DEFAULT_VOCAB


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync()
        self.seconds = time.perf_counter() - self.t0


class DAAAMFrontend:
    """Loads the models once; ``run`` processes one episode."""

    def __init__(self, cfg: FrontendConfig, load_semantics: bool = True):
        from ..third_party.daaam.segmentation import FastSAMSegmenter

        self.cfg = cfg
        self.segmenter = FastSAMSegmenter(
            cfg.fastsam_checkpoint, device=cfg.device, min_mask_region_area=cfg.min_mask_region_area,
            imgsz=cfg.imgsz, conf=cfg.fastsam_conf, iou=cfg.fastsam_iou,
        )
        self.dam = self.clip = self.sentence = None
        if load_semantics:
            from ..third_party.daaam.dam import DAMAgent
            from ..third_party.daaam.embedding import CLIPHandler, SentenceEmbeddingHandler

            self.dam = DAMAgent(cfg.dam_model_path, cfg.dam_conv_mode, cfg.dam_prompt_mode)
            self.clip = CLIPHandler(cfg.clip_model, cfg.clip_pretrained, cfg.device)
            self.sentence = SentenceEmbeddingHandler(cfg.sentence_model, cfg.device)

    def _new_tracker(self):
        from ..third_party.daaam.tracking import BotSortTracker

        c = self.cfg
        return BotSortTracker(c.reid_weights, c.device, c.track_buffer, c.with_reid, c.reid_half)

    # --- per-frame --------------------------------------------------------------------------
    def _validate_depth(self, tid: int, mask: np.ndarray, depth: Optional[np.ndarray],
                        history: Dict[int, List[float]]) -> Tuple[float, bool]:
        """DAAAM ``_validate_track_depth``: >= 25 % valid pixels, median of last 10 within bounds, or large."""
        if depth is None:
            return 0.0, True
        n = mask.sum()
        if np.sum(depth[mask] > 0) < 0.25 * n:
            return 0.0, False
        med = fast_median_depth(depth, mask)
        if med < 0:
            return 0.0, False
        hist = history.setdefault(tid, [])
        hist.append(med)
        del hist[:-10]
        track_med = float(np.median(hist))
        depth_ok = self.cfg.depth_lb <= track_med <= self.cfg.depth_ub
        size_ok = n >= self.cfg.min_mask_region_area * 30
        return med, bool(depth_ok or size_ok)

    def run(self, episode: Episode) -> FrontendResult:
        c = self.cfg
        tracker = self._new_tracker()
        frames = list(range(0, episode.n_frames, c.frame_stride))
        if c.max_frames:
            frames = frames[:c.max_frames]
        result = FrontendResult(shape=(episode.camera.height, episode.camera.width))
        result.timings["config"] = [asdict(c)]

        window: Deque[Tuple[int, List[SimplifiedTrack]]] = deque(maxlen=c.query_interval_frames + 10)
        depth_history: Dict[int, List[float]] = {}
        prompted: Set[int] = set()
        pending: List[SelectedGroup] = []

        for count, t in enumerate(frames, start=1):
            rgb = episode.rgb(t)
            depth = episode.depth(t)
            with Timer() as ts:
                dets, masks = self.segmenter(rgb)
            if len(dets):
                keep = [i for i, m in enumerate(masks) if m.sum() <= c.max_mask_region_area]
                dets, masks = dets[keep], [masks[i] for i in keep]
            with Timer() as tt:
                tracks = tracker.update(dets, rgb)
            result.time("segment_s", ts.seconds)
            result.time("track_s", tt.seconds)

            simplified = []
            for row in tracks:
                tid, midx = int(row[4]), int(row[7])
                if midx >= len(masks):
                    continue
                mask = masks[midx].astype(bool)
                med, valid = self._validate_depth(tid, mask, depth, depth_history)
                bbox = bounding_box_from_mask(mask)
                result.add(t, tid, mask, bbox, med, valid)
                tr = Track.from_mask(tid, mask, bbox, c.polygon_epsilon_factor, depth_valid=valid, median_depth=med)
                simplified.append(tr.get_simplified())
            window.append((t, simplified))

            last = count == len(frames)
            if count % c.query_interval_frames == 0 or count == c.first_query_frame or last:
                pending.extend(self._select(window, result, prompted))
                n_pending = sum(len(g.track_ids) for g in pending)
                if self.dam is not None and pending and (n_pending >= c.multi_image_min_n_masks or last):
                    self._describe(episode, pending, result)
                    pending = []
        return result

    # --- semantic lifting -------------------------------------------------------------------
    def _select(self, window, result: FrontendResult, prompted: Set[int]) -> List[SelectedGroup]:
        frame_ids = [t for t, _ in window]
        history = [tracks for _, tracks in window]
        corrections = {tid: info.description for tid, info in result.tracks.items() if info.description}
        with Timer() as tm:
            groups = select_prompt_frames(
                history, result.shape, corrections, {tid: tid for tid in result.tracks},
                prompted, start_frame_count=0, cfg=self.cfg.frame_selection,
            ) or []
        result.time("frame_selection_s", tm.seconds)
        for g in groups:
            g.global_frame_id = frame_ids[g.global_frame_id]  # window index -> episode frame
            prompted.update(g.track_ids)
        return groups

    def _describe(self, episode: Episode, groups: List[SelectedGroup], result: FrontendResult) -> None:
        pairs, keys, crops, crop_keys = [], [], [], []
        for g in groups:
            rgb = episode.rgb(g.global_frame_id)
            masks = []
            for tid in g.track_ids:
                row = result.row(g.global_frame_id, tid)
                if row is None:
                    continue
                masks.append(result.mask(row))
                keys.append((g.global_frame_id, tid))
                x1, y1, x2, y2 = [int(v) for v in result.bbox[row]]
                crop = rgb[max(y1, 0):y2, max(x1, 0):x2]
                if min(crop.shape[:2]) >= self.cfg.min_crop_px:
                    crops.append(crop)
                    crop_keys.append(tid)
            if masks:
                pairs.append((rgb, masks))
        if not pairs:
            return

        with Timer() as td:
            descriptions = [d for per_image in self.dam.describe(pairs) for d in per_image]
        with Timer() as tf:
            clip = self.clip.image_features(crops) if crops else np.zeros((0, 0))
            sent = self.sentence.embed_descriptions([d.strip() or "unknown" for d in descriptions])
        result.time("dam_batch", {"seconds": td.seconds, "n_masks": len(keys), "n_images": len(pairs)})
        result.time("features_s", tf.seconds)

        for (frame, tid), desc, s in zip(keys, descriptions, sent):
            info = result.tracks[tid]
            info.description = desc.strip() or "unknown"
            info.category = category_of(info.description, self.cfg.category_vocab)
            info.prompt_frame = frame
            info.sentence_feature = s
        for tid, f in zip(crop_keys, clip):
            result.tracks[tid].clip_feature = f
