"""spark_dsg I/O: our memory as a 4D-SG object layer, and DAAAM's DSG objects as fragments.

Export: one ``KhronosObjectAttributes`` node per memory node. In spark-dsg 1.1.3 the Khronos
temporal fields (first/last observed, trajectory) are read-only from Python, so times,
history and events are stored in node metadata. Import: objects of a DAAAM run
(``dsg.json`` + node metadata ``description`` + graph metadata ``features``) become
fragments; their masks come from matching the projected 3D box to front-end masks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..categories import category_of
from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..io_utils import PathLike, mask_iou
from .events import EventConfig, node_events
from .graph import Fragment, MemoryGraph, Observation

logger = logging.getLogger(__name__)


def export_memory(graph: MemoryGraph, path: PathLike, fps: float, events_cfg: Optional[EventConfig] = None,
                  history_stride: int = 5) -> Path:
    import spark_dsg as dsg

    events_cfg = events_cfg or EventConfig.for_space(graph.space)
    g = dsg.DynamicSceneGraph()
    for node in graph.nodes.values():
        obs = node.observations
        attrs = dsg.KhronosObjectAttributes()
        attrs.name = node.category or "unknown"
        p3 = obs[-1].p if graph.space == "3d" else np.r_[obs[-1].p, 0.0]
        attrs.position = np.asarray(p3, dtype=float)
        attrs.metadata.add({
            "category": node.category,
            "description": next((f.description for f in node.fragments if f.description), None),
            "fragment_ids": [f.fid for f in node.fragments],
            "first_observed_s": obs[0].t / fps,
            "last_observed_s": obs[-1].t / fps,
            "history": [{"t": o.t, "p": np.asarray(o.p).tolist()} for o in obs[::history_stride]],
            "events": [{"t_start": e.t_start, "t_end": e.t_end, "p_start": e.p_start.tolist(), "p_end": e.p_end.tolist()}
                       for e in node_events(node, events_cfg)],
        })
        g.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", node.node_id).value, attrs)
    g.metadata.add({"twinmem4d": {"method": graph.method, "space": graph.space, "fps": fps}})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(str(path))
    return path


def load_memory_summary(path: PathLike) -> List[Dict]:
    """Node metadata of an exported memory graph (round-trip check)."""
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph.load(str(path))
    out = []
    for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        meta = dict(n.attributes.metadata.get())
        meta["node_id"] = int(n.id.category_id)
        meta["position"] = np.asarray(n.attributes.position).tolist()
        out.append(meta)
    return sorted(out, key=lambda m: m["node_id"])


# --- DAAAM full backend -> fragments ------------------------------------------------------------

def _box_mask(attrs, episode: Episode) -> Optional[np.ndarray]:
    """Convex hull of the projected 3D bounding box (None if behind the camera / invalid)."""
    import cv2

    bb = attrs.bounding_box
    if not bb.is_valid():
        return None
    corners = np.asarray(bb.corners())
    uv, z = episode.camera.project(corners)
    if np.any(z <= 0):
        return None
    mask = np.zeros((episode.camera.height, episode.camera.width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
    return mask.astype(bool)


def fragments_from_daaam_dsg(
    dsg_path: PathLike,
    episode: Episode,
    frontend: FrontendResult,
    min_iou: float = 0.3,
    keep_categories=("bowl", "plate"),
) -> List[Fragment]:
    """Objects of a DAAAM DSG -> fragments observed on [first_observed, last_observed].

    Positions are the object position (plus the trajectory for dynamic objects); each
    observation's mask is the front-end mask at that frame that best overlaps the projected box.
    """
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph.load(str(dsg_path))
    features = dict(g.metadata.get()).get("features", {})
    fps = episode.fps
    fragments = []
    for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        a = n.attributes
        meta = dict(a.metadata.get())
        description = meta.get("description") or a.name
        category = category_of(description or "")
        if category not in keep_categories:
            continue
        first = list(getattr(a, "first_observed_ns", []) or [])
        last = list(getattr(a, "last_observed_ns", []) or [])
        if not first or not last:
            logger.warning(f"object {n.id} has no observation times; skipped")
            continue
        t0, t1 = int(round(min(first) * 1e-9 * fps)), int(round(max(last) * 1e-9 * fps))
        traj_t = [int(round(ts * 1e-9 * fps)) for ts in getattr(a, "trajectory_timestamps", [])]
        traj_p = [np.asarray(p) for p in getattr(a, "trajectory_positions", [])]
        box = _box_mask(a, episode)
        obs = []
        for t in range(t0, min(t1, episode.n_frames - 1) + 1):
            p = np.asarray(a.position)
            if traj_t:
                p = traj_p[int(np.argmin([abs(tt - t) for tt in traj_t]))]
            uv, _ = episode.camera.project(p[None])
            row = _best_row(frontend, t, box, min_iou) if box is not None else None
            if row is None and t not in (t0, t1):
                continue  # keep the endpoints even without a mask so the time span is preserved
            obs.append(Observation(t=t, p=p, p2d=uv[0], row=row))
        if not obs:
            continue
        feat = features.get(str(int(a.semantic_label)), {}) or features.get(int(a.semantic_label), {})
        vec = [np.asarray(v) for v in (feat.get("clip_feature"), feat.get("sentence_embedding_feature")) if v]
        feature = np.concatenate(vec) if vec else None
        if feature is not None:
            feature = feature / (np.linalg.norm(feature) + 1e-12)
        fragments.append(Fragment(int(n.id.category_id), obs, category, feature, description, source="daaam_dsg"))
    return fragments


def _best_row(frontend: FrontendResult, t: int, box: np.ndarray, min_iou: float) -> Optional[int]:
    best, best_iou = None, min_iou
    for row in frontend.rows_at(t):
        iou = mask_iou(frontend.mask(row), box)
        if iou > best_iou:
            best, best_iou = row, iou
    return best
