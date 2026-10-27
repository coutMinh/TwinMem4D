"""Full DAAAM backend (Hydra + Khronos) from an upstream checkout (plan B-build / H0 / R5).

Nothing of the backend is vendored: the upstream ``scripts/run_pipeline.py`` is called on a
converted episode (ImageSequenceDataset contract). The H0 gate needs processed frames > 0,
a saved DSG that re-loads, object positions inside the camera view, and observation times.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data.episode import Episode
from ..io_utils import PathLike


def pipeline_command(daaam_dir: PathLike, episode_dir: PathLike, hydra_config: PathLike, output_dir: PathLike,
                     fps: float, pipeline_config: Optional[PathLike] = None, extra: Optional[List[str]] = None) -> List[str]:
    cmd = [
        sys.executable, str(Path(daaam_dir) / "scripts" / "run_pipeline.py"), str(episode_dir),
        "--dataset-type", "ImageSequenceDataset",
        "--hydra-config-path", str(hydra_config),
        "--fps", str(fps),
        "--depth-scale", "1000",
        "--output-dir", str(output_dir),
        "--zmq-url", "none",
        "--no-throttle",
    ]
    if pipeline_config:
        cmd += ["--config", str(pipeline_config)]
    return cmd + list(extra or [])


def run_pipeline(cmd: List[str], log_path: PathLike, env: Optional[Dict[str, str]] = None,
                 background: bool = False):
    """Run (or start, if ``background``) the upstream pipeline with stdout/stderr to ``log_path``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    if background:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    finally:
        log.close()


def retune_hydra_config(src: PathLike, dst: PathLike, overrides: Dict[str, object]) -> Dict[str, Dict]:
    """Copy a Hydra YAML and set dotted keys (e.g. ``map.voxel_size``); returns {key: {old, new}}.

    Only keys that already exist may be changed, so a typo cannot silently add a new parameter.
    The upstream file is never modified.
    """
    import yaml

    data = yaml.safe_load(Path(src).read_text())
    changes = {}
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"{dotted} not in {src}; inspect the file before retuning")
        changes[dotted] = {"old": node[leaf], "new": value}
        node[leaf] = value
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(yaml.safe_dump(data, sort_keys=False))
    return changes


def find_outputs(output_dir: PathLike) -> Dict[str, Optional[Path]]:
    """Newest DAAAM output folder: the corrected ``dsg.json`` and ``corrections.yaml``."""
    candidates = sorted(Path(output_dir).rglob("dsg.json"), key=lambda p: p.stat().st_mtime)
    corrected = [p for p in candidates if (p.parent / "corrections.yaml").exists()]
    dsg = (corrected or candidates or [None])[-1]
    return {"dsg": dsg, "corrections": (dsg.parent / "corrections.yaml") if dsg and (dsg.parent / "corrections.yaml").exists() else None}


def h0_check(dsg_path: PathLike, episode: Episode, log_path: Optional[PathLike] = None) -> Dict:
    """Evidence for the H0 gate; ``passed`` requires every check to hold."""
    import spark_dsg as dsg

    report: Dict = {"dsg_path": str(dsg_path)}
    g = dsg.DynamicSceneGraph.load(str(dsg_path))
    objects = list(g.get_layer(dsg.DsgLayers.OBJECTS).nodes)
    report["n_objects"] = len(objects)
    in_view, timed, rows = 0, 0, []
    for n in objects:
        a = n.attributes
        p = np.asarray(a.position)
        uv, z = episode.camera.project(p[None])
        visible = bool(z[0] > 0 and 0 <= uv[0, 0] < episode.camera.width and 0 <= uv[0, 1] < episode.camera.height)
        first, last = list(a.first_observed_ns), list(a.last_observed_ns)
        in_view += visible
        timed += bool(first and last)
        rows.append({"id": str(n.id), "description": dict(a.metadata.get()).get("description", a.name),
                     "position": p.tolist(), "uv": uv[0].tolist(), "in_view": visible,
                     "first_s": min(first) * 1e-9 if first else None, "last_s": max(last) * 1e-9 if last else None})
    report.update({"objects_in_view": in_view, "objects_with_times": timed, "objects": rows})
    if log_path and Path(log_path).exists():
        report["log_tail"] = Path(log_path).read_text(errors="replace").splitlines()[-30:]
    report["passed"] = bool(objects) and in_view > 0 and timed > 0
    return report
