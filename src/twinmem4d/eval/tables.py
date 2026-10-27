"""Paper tables from per-query rows (never typed by hand)."""

from typing import List

import pandas as pd

from .metrics import bootstrap_accuracy


def accuracy_table(query_rows: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """method x query type -> accuracy [95 % episode-bootstrap CI], wrong-twin rate, n."""
    records = []
    groups = list(query_rows.groupby(["method", "qtype"])) + [
        ((m, "all"), g) for m, g in query_rows.groupby("method")
    ]
    for (method, qtype), g in groups:
        ci = bootstrap_accuracy(g["episode"].tolist(), g["correct"].tolist(), n_boot=n_boot, seed=seed)
        records.append({
            "method": method, "qtype": qtype, "acc": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
            "wrong_twin": g["wrong_twin"].mean(), "answered": g["answered"].mean(),
            "n_queries": ci["n_queries"], "n_episodes": ci["n_episodes"],
            "latency_ms": 1000 * g["latency_s"].mean() if "latency_s" in g else float("nan"),
        })
    return pd.DataFrame(records).sort_values(["method", "qtype"]).reset_index(drop=True)


def format_accuracy(table: pd.DataFrame) -> pd.DataFrame:
    """Wide view: one row per method, one 'acc [lo, hi]' column per query type."""
    cell = table.apply(lambda r: f"{100 * r.acc:.1f} [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}]", axis=1)
    return table.assign(cell=cell).pivot(index="method", columns="qtype", values="cell")


def identity_table(ident_rows: pd.DataFrame) -> pd.DataFrame:
    cols: List[str] = ["n_gt", "n_nodes", "splits", "swaps", "identity_consistent", "missed"]
    return ident_rows.groupby("method")[cols].sum().reset_index()
