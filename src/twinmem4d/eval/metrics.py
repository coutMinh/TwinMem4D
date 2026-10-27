"""Metrics of plan E1: query accuracy, wrong-twin rate, identity split/swap, episode bootstrap."""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.gt import GroundTruth
from ..frontend.store import FrontendResult
from ..io_utils import mask_iou
from ..memory.graph import MemoryGraph
from ..queries.templates import Query


def score_prediction(pred: Optional[np.ndarray], query: Query, gt: GroundTruth, iou_thr: float = 0.5) -> Dict:
    """Correct if IoU with the GT target at t_q > thr; wrong-twin if it matches another same-category instance."""
    if pred is None:
        return {"correct": 0, "wrong_twin": 0, "iou": 0.0, "answered": 0}
    iou = mask_iou(pred, gt.mask(query.t_q, query.target))
    twin = any(
        mask_iou(pred, gt.mask(query.t_q, j)) > iou_thr
        for j in gt.instances_of(query.category) if j != query.target and gt.area[query.t_q, j] > 0
    )
    return {"correct": int(iou > iou_thr), "wrong_twin": int(twin and iou <= iou_thr), "iou": iou, "answered": 1}


def match_observations(frontend: FrontendResult, gt: GroundTruth, iou_thr: float = 0.5) -> Dict[int, Optional[int]]:
    """Front-end row -> GT instance with IoU > thr at that frame (None if no match)."""
    out = {}
    for row, t in enumerate(frontend.frame):
        if t >= gt.n_frames:
            out[row] = None
            continue
        m = frontend.mask(row)
        best, best_iou = None, iou_thr
        for i in range(len(gt.names)):
            if gt.area[t, i] == 0:
                continue
            iou = mask_iou(m, gt.mask(t, i))
            if iou > best_iou:
                best, best_iou = i, iou
        out[row] = best
    return out


def identity_metrics(graph: MemoryGraph, row_to_gt: Dict[int, Optional[int]], gt: GroundTruth,
                     category: str = "bowl", min_obs: int = 5) -> Dict:
    """Splits: extra nodes per GT instance. Swaps: extra GT instances per node (identity mixing)."""
    instances = set(gt.instances_of(category))
    node_to_gt: Dict[int, Counter] = defaultdict(Counter)
    for node in graph.nodes.values():
        for o in node.observations:
            g = row_to_gt.get(o.row) if o.row is not None else None
            if g in instances:
                node_to_gt[node.node_id][g] += 1
    gt_to_nodes: Dict[int, set] = defaultdict(set)
    swaps = 0
    for nid, counts in node_to_gt.items():
        strong = [g for g, c in counts.items() if c >= min_obs]
        swaps += max(len(strong) - 1, 0)
        for g in strong:
            gt_to_nodes[g].add(nid)
    splits = sum(max(len(gt_to_nodes[g]) - 1, 0) for g in instances)
    consistent = sum(1 for g in instances if len(gt_to_nodes[g]) == 1
                     and all(len([x for x, c in node_to_gt[n].items() if c >= min_obs]) == 1 for n in gt_to_nodes[g]))
    return {
        "n_gt": len(instances),
        "n_nodes": sum(1 for n in graph.nodes.values() if n.category == category),
        "splits": splits,
        "swaps": swaps,
        "identity_consistent": consistent,
        "missed": sum(1 for g in instances if not gt_to_nodes[g]),
    }


def bootstrap_accuracy(episode_ids: Sequence[str], correct: Sequence[int], n_boot: int = 1000,
                       seed: int = 0, alpha: float = 0.05) -> Dict:
    """Mean accuracy with a percentile CI from resampling *episodes* (queries within one are correlated)."""
    per_ep: Dict[str, List[int]] = defaultdict(list)
    for e, c in zip(episode_ids, correct):
        per_ep[e].append(int(c))
    eps = list(per_ep)
    if not eps:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_queries": 0, "n_episodes": 0}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(len(eps), len(eps), replace=True)
        vals = [c for k in sample for c in per_ep[eps[k]]]
        stats.append(np.mean(vals))
    pooled = [c for e in eps for c in per_ep[e]]
    return {
        "mean": float(np.mean(pooled)),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
        "n_queries": len(pooled),
        "n_episodes": len(eps),
    }
