"""Experiment registry: one CSV row per run; paper tables are generated from it."""

import csv
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .io_utils import PathLike, write_json

REGISTRY_FIELDS = [
    "run_id", "stage", "backend_mode", "git_commit", "config_path", "data_manifest_hash",
    "seed", "hardware", "wall_clock_s", "metrics_path", "notes", "created_utc",
]


def git_commit(repo_dir: Optional[PathLike] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir or Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir or Path(__file__).parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def hardware_string() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}"
    except ImportError:
        pass
    return f"cpu | {platform.processor() or platform.machine()}"


@dataclass
class Run:
    """Context manager that times a run and appends its row to ``results/registry.csv``."""
    results_dir: Path
    stage: str
    backend_mode: str = ""
    config_path: str = ""
    data_manifest_hash: str = ""
    seed: int = 0
    notes: str = ""
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / "runs" / self.run_id

    def __enter__(self) -> "Run":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return  # failed runs are not registered; their folder stays for debugging
        metrics_path = write_json(self.run_dir / "metrics.json", self.metrics)
        row = {
            "run_id": self.run_id,
            "stage": self.stage,
            "backend_mode": self.backend_mode,
            "git_commit": git_commit(),
            "config_path": self.config_path,
            "data_manifest_hash": self.data_manifest_hash,
            "seed": self.seed,
            "hardware": hardware_string(),
            "wall_clock_s": round(time.time() - self._t0, 2),
            "metrics_path": str(metrics_path.relative_to(self.results_dir)),
            "notes": self.notes,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        registry = Path(self.results_dir) / "registry.csv"
        new = not registry.exists()
        with open(registry, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow(row)


def read_registry(results_dir: PathLike):
    import pandas as pd

    return pd.read_csv(Path(results_dir) / "registry.csv")
