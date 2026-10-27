"""Run the memory variants on one episode and score them (plan M1 + E1)."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..memory.association import build_memory
from ..memory.events import EventConfig
from ..memory.graph import Fragment, MemoryGraph
from ..memory.lift import fragments_from_frontend
from ..queries.generate import QueryConfig, generate_queries
from ..queries.solve import answer_mask, solve
from ..queries.templates import Query
from .metrics import identity_metrics, match_observations, score_prediction


@dataclass
class MemoryEvalConfig:
    space: str = "3d"
    methods: Tuple[str, ...] = ("raw", "reconcile_proxy", "twin_aware")
    keep_categories: Tuple[str, ...] = ("bowl", "plate")
    min_observations: int = 5
    queries: QueryConfig = field(default_factory=QueryConfig)
    method_overrides: Dict[str, Dict] = field(default_factory=dict)


def evaluate_graph(graph: MemoryGraph, frontend: FrontendResult, episode: Episode, queries: List[Query],
                   row_to_gt: Dict, cfg: MemoryEvalConfig, row_label: str) -> Tuple[List[Dict], Dict]:
    ev_cfg = EventConfig.for_space(graph.space)
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        ans = solve(q, graph, ev_cfg, cfg.queries.on_plate_radius_px, cfg.queries.plate_merge_px)
        latency = time.perf_counter() - t0
        s = score_prediction(answer_mask(ans, frontend), q, episode.gt)
        rows.append({
            "episode": episode.name, "method": row_label, "qid": q.qid, "qtype": q.label, "t_q": q.t_q,
            "target": q.target_name, "node": ans.node_id, "mask_age": ans.mask_age, "ambiguous": ans.ambiguous,
            "reason": ans.reason, "latency_s": latency, **s,
        })
    ident = identity_metrics(graph, row_to_gt, episode.gt, cfg.queries.category)
    ident.update({"episode": episode.name, "method": row_label, "memory_nodes": len(graph.nodes)})
    return rows, ident


def evaluate_episode(episode: Episode, frontend: FrontendResult, cfg: MemoryEvalConfig,
                     label_prefix: str = "", extra_fragments: Optional[Dict[str, List[Fragment]]] = None):
    """Returns (query rows, identity rows, queries). ``extra_fragments`` adds e.g. DAAAM-DSG fragments."""
    queries = generate_queries(episode.gt, episode.name, cfg.queries)
    row_to_gt = match_observations(frontend, episode.gt)
    fragment_sets = {"": fragments_from_frontend(frontend, episode, cfg.space, cfg.keep_categories, cfg.min_observations)}
    fragment_sets.update(extra_fragments or {})

    query_rows, ident_rows = [], []
    for set_name, fragments in fragment_sets.items():
        for method in cfg.methods:
            graph = build_memory(fragments, method, cfg.space, episode.fps, **cfg.method_overrides.get(method, {}))
            label = "+".join(x for x in (label_prefix, set_name, method) if x)
            q, i = evaluate_graph(graph, frontend, episode, queries, row_to_gt, cfg, label)
            query_rows += q
            ident_rows.append(i)
    return query_rows, ident_rows, queries
