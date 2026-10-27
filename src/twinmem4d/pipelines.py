"""Registered batch jobs shared by the molab notebooks and the CLI (each appends a registry row)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data.episode import Episode
from .eval.memory_eval import MemoryEvalConfig, evaluate_episode
from .eval.metrics import score_prediction
from .eval.tables import accuracy_table, identity_table
from .frontend.store import FrontendResult
from .io_utils import PathLike, hash_files, write_json
from .queries.generate import generate_queries
from .registry import Run

logger = logging.getLogger(__name__)


def _manifest(episodes: Sequence[Episode]) -> str:
    return hash_files([ep.root / "meta.json" for ep in episodes])


def summarize_timings(results: Dict[str, FrontendResult]) -> Dict:
    """R4: per-stage latency, mean +- std over all episodes (paper: 1.2 +- 0.74 s selection, 9.2 +- 1.4 s per DAM batch)."""
    pools: Dict[str, List[float]] = {}
    for res in results.values():
        for key in ("segment_s", "track_s", "frame_selection_s", "features_s"):
            pools.setdefault(key, []).extend(res.timings.get(key, []))
        for b in res.timings.get("dam_batch", []):
            pools.setdefault("dam_batch_s", []).append(b["seconds"])
            pools.setdefault("dam_s_per_mask", []).append(b["seconds"] / max(b["n_masks"], 1))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in pools.items() if v}


def run_frontend(episodes: Sequence[Episode], cfg, results_dir: PathLike, config_path: str = "",
                 oracle: bool = False, notes: str = "") -> Path:
    from .frontend.oracle import oracle_frontend

    with Run(Path(results_dir), stage="frontend_oracle" if oracle else "frontend", backend_mode="oracle" if oracle else "frontend",
             config_path=config_path, data_manifest_hash=_manifest(episodes), notes=notes) as run:
        if oracle:
            def process(ep):
                return oracle_frontend(ep)
        else:
            from .frontend.runner import DAAAMFrontend
            frontend = DAAAMFrontend(cfg)
            process = frontend.run
        results = {}
        for ep in episodes:
            logger.info(f"front-end on {ep.name}")
            res = process(ep)
            res.save(run.run_dir / ep.name)
            results[ep.name] = res
        run.metrics = {
            "episodes": [ep.name for ep in episodes],
            "timings": summarize_timings(results),
            "tracks_per_episode": {k: len(r.tracks) for k, r in results.items()},
            "described_tracks": {k: sum(t.description is not None for t in r.tracks.values()) for k, r in results.items()},
        }
    return run.run_dir


def run_memory_eval(episodes: Sequence[Episode], frontend_run: PathLike, cfg: MemoryEvalConfig, results_dir: PathLike,
                    config_path: str = "", label: str = "", backend_mode: str = "simplified",
                    dsg_paths: Optional[Dict[str, PathLike]] = None, notes: str = "") -> Path:
    """Memory variants on stored front-end outputs; with ``dsg_paths`` also on DAAAM-DSG fragments."""
    from .memory.dsg_io import fragments_from_daaam_dsg

    with Run(Path(results_dir), stage="memory_eval", backend_mode=backend_mode, config_path=config_path,
             data_manifest_hash=_manifest(episodes), notes=notes or f"frontend_run={Path(frontend_run).name}") as run:
        q_rows, i_rows = [], []
        for ep in episodes:
            fe = FrontendResult.load(Path(frontend_run) / ep.name)
            extra = {}
            if dsg_paths and ep.name in dsg_paths:
                extra["daaam_full"] = fragments_from_daaam_dsg(dsg_paths[ep.name], ep, fe)
            q, i, _ = evaluate_episode(ep, fe, cfg, label_prefix=label, extra_fragments=extra)
            q_rows += q
            i_rows += i
        q_df, i_df = pd.DataFrame(q_rows), pd.DataFrame(i_rows)
        q_df.to_csv(run.run_dir / "query_rows.csv", index=False)
        i_df.to_csv(run.run_dir / "identity_rows.csv", index=False)
        acc = accuracy_table(q_df) if len(q_df) else pd.DataFrame()
        acc.to_csv(run.run_dir / "accuracy.csv", index=False)
        ident = identity_table(i_df) if len(i_df) else pd.DataFrame()
        ident.to_csv(run.run_dir / "identity.csv", index=False)
        run.metrics = {"accuracy": acc.to_dict("records"), "identity": ident.to_dict("records"),
                       "n_queries": len(q_df) // max(q_df["method"].nunique(), 1) if len(q_df) else 0}
    return run.run_dir


def run_vlm_baselines(episodes: Sequence[Episode], frontend_run: PathLike, vlm_cfg, query_cfg, ks: Sequence[int],
                      results_dir: PathLike, config_path: str = "", seed: int = 0, notes: str = "") -> Path:
    from .baselines.vlm import QwenSoM, random_baseline

    with Run(Path(results_dir), stage="vlm_baselines", backend_mode="none", config_path=config_path,
             data_manifest_hash=_manifest(episodes), seed=seed, notes=notes) as run:
        model = QwenSoM(vlm_cfg)
        rng = np.random.default_rng(seed)
        rows = []
        for ep in episodes:
            fe = FrontendResult.load(Path(frontend_run) / ep.name)
            for q in generate_queries(ep.gt, ep.name, query_cfg):
                base = {"episode": ep.name, "qid": q.qid, "qtype": q.label, "t_q": q.t_q, "target": q.target_name}
                r = random_baseline(fe, q, rng)
                rows.append({**base, "method": "random", "latency_s": 0.0,
                             **score_prediction(fe.mask(r) if r is not None else None, q, ep.gt)})
                for k in ks:
                    out = model.answer(ep, fe, q, k)
                    pred = fe.mask(out["mask_row"]) if out["mask_row"] is not None else None
                    rows.append({**base, "method": f"vlm_k{k}", "latency_s": out["latency_s"], "tokens": out["tokens"],
                                 "reply": out["reply"], "n_candidates": out["n_candidates"],
                                 **score_prediction(pred, q, ep.gt)})
        df = pd.DataFrame(rows)
        df.to_csv(run.run_dir / "query_rows.csv", index=False)
        acc = accuracy_table(df) if len(df) else pd.DataFrame()
        acc.to_csv(run.run_dir / "accuracy.csv", index=False)
        cost = df.groupby("method")[["tokens", "latency_s"]].mean().reset_index() if "tokens" in df else pd.DataFrame()
        cost.to_csv(run.run_dir / "cost.csv", index=False)
        run.metrics = {"accuracy": acc.to_dict("records"), "cost": cost.to_dict("records"), "ks": list(ks)}
        write_json(run.run_dir / "vlm_config.json", vlm_cfg)
    return run.run_dir


def run_dam_curve(episodes: Sequence[Episode], frontend_run: PathLike, results_dir: PathLike, repeats: int = 3,
                  notes: str = "") -> Path:
    """R2 on 128 real masks of a front-end run; rows in ``dam_batch_curve.csv``."""
    from .repro.dam_batch_curve import batch_curve, curve_summary, sample_items
    from .third_party.daaam.dam import DAMAgent

    with Run(Path(results_dir), stage="repro_dam_curve", backend_mode="none", data_manifest_hash=_manifest(episodes),
             notes=notes or f"masks from {Path(frontend_run).name}") as run:
        rows = batch_curve(DAMAgent(), sample_items(episodes, frontend_run), repeats=repeats)
        pd.DataFrame(rows).to_csv(run.run_dir / "dam_batch_curve.csv", index=False)
        run.metrics = curve_summary(rows)
    return run.run_dir


def run_refcocog(ref_dir: PathLike, results_dir: PathLike, split: str = "val", notes: str = "") -> Path:
    """R3: refCOCOg (umd) retrieval; fetches the annotations and the needed COCO images first (resumable)."""
    from .repro import refcocog as rc
    from .third_party.daaam.dam import DAMAgent

    files = rc.fetch_annotations(ref_dir)
    subjects = rc.load_refcocog_umd(files["refs"], files["instances"], split)
    images = rc.CocoImages(Path(ref_dir) / "train2014")
    images.fetch([s.image_id for s in subjects])
    with Run(Path(results_dir), stage="repro_refcocog", backend_mode="none", notes=notes or f"refCOCOg umd {split}") as run:
        run.metrics = rc.run_retrieval(DAMAgent(), subjects, images, cache_path=Path(ref_dir) / "dam_descriptions.json")
    return run.run_dir
