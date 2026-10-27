"""Read LIBERO-Mem HDF5 files (plan task L1).

Assumed layout (from the dataset card, VERIFY on real files with ``describe``):
``data/demo_<i>/obs/{agentview,eye_in_hand}_{rgb,depth,seg}`` plus ``actions``,
``joint_states``, ``gripper_states``. ``data`` attributes may hold LIBERO's
``env_args`` / ``problem_info`` JSON strings.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..io_utils import PathLike

HF_REPO = "libero-mem/LIBERO-Mem"


def open_hdf5(path_or_url: PathLike):
    """Open a local file, or an http(s) URL with range reads (needs ``fsspec`` + ``aiohttp``)."""
    import h5py

    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        import fsspec

        return h5py.File(fsspec.open(s, "rb", block_size=1 << 22).open(), "r")
    return h5py.File(s, "r")


def hf_url(filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{filename}"


def list_demos(f) -> List[str]:
    """Demo keys sorted by their integer suffix."""
    keys = list(f["data"].keys())
    return sorted(keys, key=lambda k: int(re.sub(r"\D", "", k) or -1))


def describe(group, max_depth: int = 4, _depth: int = 0) -> Dict[str, Any]:
    """Nested summary of datasets: shape, dtype, chunks, compression, attrs."""
    import h5py

    out: Dict[str, Any] = {}
    if group.attrs:
        out["@attrs"] = {k: _short(v) for k, v in group.attrs.items()}
    if _depth >= max_depth:
        return out
    for name, item in group.items():
        if isinstance(item, h5py.Dataset):
            out[name] = {
                "shape": list(item.shape), "dtype": str(item.dtype),
                "chunks": list(item.chunks) if item.chunks else None, "compression": item.compression,
            }
        else:
            out[name] = describe(item, max_depth, _depth + 1)
    return out


def data_attrs(f) -> Dict[str, Any]:
    """``data`` group attributes, JSON-decoded where possible (LIBERO stores env_args this way)."""
    attrs = {}
    for k, v in f["data"].attrs.items():
        v = v.decode() if isinstance(v, bytes) else v
        try:
            attrs[k] = json.loads(v) if isinstance(v, str) else v
        except json.JSONDecodeError:
            attrs[k] = v
    return attrs


def value_stats(arr: np.ndarray, max_unique: int = 64) -> Dict[str, Any]:
    flat = np.asarray(arr).ravel()
    stats = {"dtype": str(flat.dtype), "min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean())}
    if np.issubdtype(flat.dtype, np.integer):
        uniq = np.unique(flat)
        stats["n_unique"] = int(len(uniq))
        stats["unique_head"] = uniq[:max_unique].tolist()
    return stats


def load_demo(f, demo: str, camera: str = "agentview", modalities: Iterable[str] = ("rgb", "depth", "seg"),
              extra: Iterable[str] = (), frames: Optional[slice] = None) -> Dict[str, np.ndarray]:
    """Arrays for one demo; ``frames`` slices time before reading (cheap for remote files)."""
    obs = f["data"][demo]["obs"]
    sl = frames if frames is not None else slice(None)
    out = {m: np.asarray(obs[f"{camera}_{m}"][sl]) for m in modalities}
    for key in extra:
        node = f["data"][demo][key] if key in f["data"][demo] else obs[key]
        out[key] = np.asarray(node[sl])
    return out


def _short(v: Any, n: int = 200) -> Any:
    v = v.decode() if isinstance(v, bytes) else v
    if isinstance(v, str) and len(v) > n:
        return v[:n] + "..."
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 16 else f"array{list(v.shape)}"
    return v


def download(filenames: Iterable[str], local_dir: PathLike, revision: str = "main") -> List[str]:
    """Fetch dataset files into ``local_dir`` (resumable; hf_transfer when installed)."""
    import os

    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return [hf_hub_download(HF_REPO, name, repo_type="dataset", revision=revision, local_dir=str(local_dir))
            for name in filenames]
