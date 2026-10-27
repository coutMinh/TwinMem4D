"""Full DAAAM backend (Hydra + Khronos) from an upstream checkout (plan B-build / H0 / R5).

Nothing of the backend is vendored: the upstream ``scripts/run_pipeline.py`` is called on a
converted episode (ImageSequenceDataset contract). The H0 gate needs processed frames > 0,
a saved DSG that re-loads, object positions inside the camera view, and observation times.
"""

import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.episode import Episode
from ..io_utils import PathLike
from ..registry import Run

ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def pipeline_command(daaam_dir: PathLike, episode_dir: PathLike, hydra_config: PathLike, output_dir: PathLike,
                     fps: float, pipeline_config: Optional[PathLike] = None, extra: Optional[List[str]] = None,
                     python: Optional[str] = None) -> List[str]:
    cmd = [
        python or sys.executable, str(Path(daaam_dir) / "scripts" / "run_pipeline.py"), str(episode_dir),
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


def write_blocker(docs_dir: PathLike, text: str) -> Path:
    """Append a dated entry to ``docs/BLOCKERS.md`` (AGENTS rule: record blockers, then use the fallback)."""
    path = Path(docs_dir) / "BLOCKERS.md"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC — full backend\n{text}\n"
                "Fallback: reconciliation-proxy only (plan §3.2b); paper states the 4D-SG backend was not reproduced.\n")
    return path


def stage_checkpoints(checkpoints_dir: PathLike, daaam_dir: PathLike) -> None:
    """DAAAM resolves model paths under its own ``checkpoints/``; copy ours there without overwriting."""
    import shutil

    for name, sub in (("FastSAM-x.pt", "fastsam"), ("osnet_x0_25_msmt17.pt", "reid_weights")):
        src, dst = Path(checkpoints_dir) / name, Path(daaam_dir) / "checkpoints" / sub / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def run_episode(ws: PathLike, episode: Episode, hydra_config: PathLike, out_dir: PathLike,
                pipeline_config: Optional[PathLike] = None, python: str = "python3") -> Dict[str, Optional[Path]]:
    """Upstream ``run_pipeline.py`` on one converted episode inside the ROS workspace; log next to ``out_dir``."""
    daaam_dir = Path(ws) / "src" / "daaam"
    cmd = pipeline_command(daaam_dir, episode.root, hydra_config, out_dir, episode.fps, pipeline_config, python=python)
    log = Path(f"{out_dir}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    script = (f"source {ROS_SETUP} && source {Path(ws) / 'install' / 'setup.bash'} && cd {daaam_dir} && "
              f"{' '.join(shlex.quote(str(c)) for c in cmd)}")
    with open(log, "w") as f:
        subprocess.run(["bash", "-c", script], stdout=f, stderr=subprocess.STDOUT, check=False)
    return {**find_outputs(out_dir), "log": log}


def run_h0(ws: PathLike, episode: Episode, hydra_config: PathLike, results_dir: PathLike, docs_dir: PathLike,
           pipeline_config: Optional[PathLike] = None, python: str = "python3") -> Dict:
    """H0 gate on one episode (registered as ``backend_full_h0``); also written to ``docs/h0_gate.json``."""
    with Run(Path(results_dir), stage="backend_full_h0", backend_mode="full", config_path=str(pipeline_config or "")) as run:
        out = run_episode(ws, episode, hydra_config, run.run_dir / episode.name, pipeline_config, python)
        if out["dsg"]:
            report = h0_check(out["dsg"], episode, out["log"])
        else:
            report = {"passed": False, "reason": "no dsg.json written",
                      "log_tail": out["log"].read_text(errors="replace").splitlines()[-30:]}
        run.metrics = report
    (Path(docs_dir) / "h0_gate.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def run_all(ws: PathLike, episodes: Sequence[Episode], hydra_config: PathLike, results_dir: PathLike,
            pipeline_config: Optional[PathLike] = None, python: str = "python3") -> Path:
    """R5: every episode (registered as ``backend_full``); ``dsg_paths.json`` maps episode -> DSG for memory_eval."""
    with Run(Path(results_dir), stage="backend_full", backend_mode="full", config_path=str(pipeline_config or "")) as run:
        dsg_paths = {}
        for ep in episodes:
            dsg = run_episode(ws, ep, hydra_config, run.run_dir / ep.name, pipeline_config, python)["dsg"]
            if dsg:
                dsg_paths[ep.name] = str(dsg)
        (run.run_dir / "dsg_paths.json").write_text(json.dumps(dsg_paths, indent=2))
        run.metrics = {"episodes": [ep.name for ep in episodes], "episodes_with_dsg": len(dsg_paths)}
    return run.run_dir
