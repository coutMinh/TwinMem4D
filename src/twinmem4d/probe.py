"""N0 machine probe (plan §5): may the full DAAAM backend be built here, may the HDF5 files be downloaded whole?

Standard library only, so it runs right after ``git clone`` and before anything is installed.
"""

import json
import os
import platform
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Dict, Union

PathLike = Union[str, os.PathLike]
HOSTS = {
    "github_https": ("github.com", 443),
    "github_ssh": ("github.com", 22),
    "huggingface": ("huggingface.co", 443),
    "pypi": ("pypi.org", 443),
    "ros_packages": ("packages.ros.org", 80),
    "coco_images": ("images.cocodataset.org", 80),
}
MIN_FREE_GIB_FULL_HDF5 = 80   # T7 23.4 GB + T8 28.2 GB + converted episodes and weights
VIRTUAL_FS_GIB = 2**20        # molab's sandbox filesystem reports 8 EiB free: no real quota information


def _run(cmd: str) -> Dict:
    try:
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=20)
        return {"rc": r.returncode, "out": (r.stdout + r.stderr).strip()[-600:]}
    except subprocess.TimeoutExpired:
        return {"rc": None, "out": "timeout"}


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def _ram_gib():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        return None


def probe_machine(work_dir: PathLike) -> Dict:
    """Collect the evidence, decide, and write ``<work_dir>/docs/molab_probe.json``."""
    work = Path(work_dir)
    (work / "docs").mkdir(parents=True, exist_ok=True)
    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}
    probe = {
        "os": os_release.get("PRETTY_NAME"),
        "os_version_id": os_release.get("VERSION_ID"),
        "python": platform.python_version(),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "sudo_nopasswd": _run("sudo -n true")["rc"] == 0,
        "tools": {t: shutil.which(t) for t in ("apt-get", "gcc", "g++", "cmake", "git", "curl", "nvcc")},
        "nvidia_smi": _run("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader")["out"],
        "disk_free_gib": {str(p): round(shutil.disk_usage(p).free / 2**30, 1) for p in {work, Path(work.anchor or "/")}},
        "ram_total_gib": _ram_gib(),
        "cpu_count": os.cpu_count(),
        "outbound": {name: _reachable(*hp) for name, hp in HOSTS.items()},
    }
    is_root = probe["uid"] == 0 or probe["sudo_nopasswd"]
    free = min(probe["disk_free_gib"].values())
    probe["decision"] = {
        "full_backend_build": bool(is_root and probe["os_version_id"] == "24.04" and probe["tools"]["apt-get"]),
        # None = unknown: a virtual filesystem reports no usable free-space figure
        "hdf5_full_download": None if free >= VIRTUAL_FS_GIB else free >= MIN_FREE_GIB_FULL_HDF5,
    }
    (work / "docs" / "molab_probe.json").write_text(json.dumps(probe, indent=2))
    return probe


def pid_alive(pid: int) -> bool:
    """Linux only (molab): the process exists and is not a zombie. Elsewhere report False."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def heartbeat(docs_dir: PathLike, minutes: int = 60) -> Dict:
    """Detached job appending one line per minute; report its lines and whether it is still running.

    Re-check after closing and reopening the tab: a growing count means detached jobs survive. Files
    outlive a molab restart but processes do not, so a stale pid file is replaced by a new job.
    """
    docs = Path(docs_dir)
    log, pid_file = docs / "heartbeat.log", docs / "heartbeat.pid"
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    started = False
    if pid is None or not pid_alive(pid):
        p = subprocess.Popen(["bash", "-c", f"for i in $(seq 1 {minutes}); do date -u >> {log}; sleep 60; done"],
                             start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid_file.write_text(str(p.pid))
        started = pid is not None   # a previous job existed but died (restart)
    lines = log.read_text().splitlines() if log.exists() else []
    return {"heartbeat_lines": len(lines), "first": lines[:1], "last": lines[-1:],
            "restarted_after_dead_job": started}
