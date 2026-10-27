"""Paper tables from ``results/registry.csv`` only (Table A: reproduction, Table B: queries + identity + cost).

Every number comes from the latest registered run of its stage; ``sources.json`` records which run
fed which table.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..io_utils import PathLike, read_json, write_json
from ..registry import read_registry
from .tables import format_accuracy

PAPER_LATENCY = {"frame_selection_s": "1.2 ± 0.74 s", "dam_batch_s": "9.2 ± 1.4 s"}   # DAAAM paper, Sec. 4


def latest_run(results_dir: PathLike, stage: str, backend_mode: Optional[str] = None) -> Optional[pd.Series]:
    """Newest registry row of ``stage`` (optionally of one ``backend_mode``), or None.

    Re-reads the CSV on every call: background jobs append rows while a notebook is open.
    """
    if not (Path(results_dir) / "registry.csv").exists():
        return None
    reg = read_registry(results_dir)
    rows = reg[(reg.stage == stage) & ((reg.backend_mode == backend_mode) if backend_mode else True)]
    return rows.iloc[-1] if len(rows) else None


def run_dir(results_dir: PathLike, row: Optional[pd.Series]) -> Optional[Path]:
    return None if row is None else Path(results_dir) / "runs" / row["run_id"]


def _metrics(results_dir: PathLike, row: pd.Series) -> Dict:
    return read_json(Path(results_dir) / row["metrics_path"])


def reproduction_table(results_dir: PathLike, sources: Dict[str, str]) -> pd.DataFrame:
    """Table A: R3 refCOCOg Top-k vs reported, R2 DAM speed-up, R4 latency, H0 backend status."""
    rows = []
    r3 = latest_run(results_dir, "repro_refcocog")
    if r3 is not None:
        sources["R3"] = r3.run_id
        m = _metrics(results_dir, r3)
        for method, s in m["scores"].items():
            rows.append({"item": f"refCOCOg {method}", "ours": "/".join(f"{s[f'top{k}']:.2f}" for k in (1, 5, 10)),
                         "paper": "/".join(f"{v:.2f}" for v in m["reported"][method])})
    r2 = latest_run(results_dir, "repro_dam_curve")
    if r2 is not None:
        sources["R2"] = r2.run_id
        rows.append({"item": "DAM batched vs sequential speed-up (largest batch)",
                     "ours": f"{_metrics(results_dir, r2)['speedup_at_max']:.1f}x", "paper": "Fig. 3 (qualitative)"})
    fe = latest_run(results_dir, "frontend")
    if fe is not None:
        sources["R4"] = fe.run_id
        timings = _metrics(results_dir, fe)["timings"]
        for key, paper in PAPER_LATENCY.items():
            if key in timings:
                t = timings[key]
                rows.append({"item": key, "ours": f"{t['mean']:.2f} ± {t['std']:.2f} s (n={t['n']})", "paper": paper})
    h0 = latest_run(results_dir, "backend_full_h0")
    passed = h0 is not None and _metrics(results_dir, h0).get("passed")
    rows.append({"item": "Hydra/Khronos backend (H0)", "ours": "passed" if passed else "not reproduced", "paper": "—"})
    return pd.DataFrame(rows, columns=["item", "ours", "paper"])


def query_tables(results_dir: PathLike, sources: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Table B: accuracy per query type × method over the memory runs and the VLM run, identity, VLM cost."""
    acc, ident, cost = [], [], []
    for mode in ("simplified", "oracle", "full"):
        row = latest_run(results_dir, "memory_eval", mode)
        if row is not None:
            sources[f"memory_{mode}"] = row.run_id
            acc.append(pd.read_csv(run_dir(results_dir, row) / "accuracy.csv"))
            ident.append(pd.read_csv(run_dir(results_dir, row) / "identity.csv"))
    vlm = latest_run(results_dir, "vlm_baselines")
    if vlm is not None:
        sources["vlm"] = vlm.run_id
        acc.append(pd.read_csv(run_dir(results_dir, vlm) / "accuracy.csv"))
        cost.append(pd.read_csv(run_dir(results_dir, vlm) / "cost.csv"))
    long = pd.concat(acc, ignore_index=True).drop_duplicates(["method", "qtype"], keep="last") if acc else pd.DataFrame()
    return {
        "accuracy_long": long,
        "accuracy": format_accuracy(long) if len(long) else pd.DataFrame(),
        "identity": pd.concat(ident, ignore_index=True) if ident else pd.DataFrame(),
        "cost": pd.concat(cost, ignore_index=True) if cost else pd.DataFrame(),
    }


def _esc(v) -> str:
    return str(v).replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace("&", "\\&")


def to_latex(df: pd.DataFrame, caption: str) -> str:
    """Booktabs tabular with the index as first column (no pandas Styler / jinja2 dependency)."""
    cols = list(df.columns)
    lines = [f"% {caption}", "\\begin{tabular}{" + "l" * (len(cols) + 1) + "}", "\\toprule",
             " & ".join(_esc(c) for c in [df.index.name or ""] + cols) + " \\\\", "\\midrule"]
    lines += [" & ".join(_esc(v) for v in [i, *r]) + " \\\\" for i, r in zip(df.index, df.values)]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def write_paper_tables(results_dir: PathLike) -> Dict:
    """Write ``results/paper/*.csv``, ``*.tex`` and ``sources.json``; returns the tables and their sources."""
    out = Path(results_dir) / "paper"
    out.mkdir(parents=True, exist_ok=True)
    sources: Dict[str, str] = {}
    table_a = reproduction_table(results_dir, sources)
    table_b = query_tables(results_dir, sources)
    table_a.to_csv(out / "table_a_reproduction.csv", index=False)
    table_b["accuracy_long"].to_csv(out / "table_b_accuracy_long.csv", index=False)
    table_b["accuracy"].to_csv(out / "table_b_accuracy.csv")
    table_b["identity"].to_csv(out / "table_b_identity.csv", index=False)
    if len(table_b["cost"]):
        table_b["cost"].to_csv(out / "table_b_cost.csv", index=False)
    (out / "table_a.tex").write_text(to_latex(table_a.set_index("item"), "Table A, reproduction"), encoding="utf-8")
    if len(table_b["accuracy"]):
        (out / "table_b.tex").write_text(to_latex(table_b["accuracy"], "Table B, query accuracy [95% CI]"), encoding="utf-8")
    write_json(out / "sources.json", sources)
    return {"table_a": table_a, **{f"table_b_{k}": v for k, v in table_b.items()}, "sources": sources, "dir": out}
