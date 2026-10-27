"""Small I/O helpers shared by all stages."""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import yaml

PathLike = Union[str, Path]


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """Bool (H, W) -> packed uint8 row; the shape is stored once per file."""
    return np.packbits(mask.astype(bool).ravel())


def unpack_mask(packed: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: PathLike, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2), encoding="utf-8")
    return path


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path: PathLike) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: PathLike, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_jsonable(obj), sort_keys=False), encoding="utf-8")
    return path


def hash_files(paths: Iterable[PathLike], chunk: int = 1 << 20) -> str:
    """SHA-256 over the contents of ``paths`` in sorted order (data manifest hash)."""
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(str(p.name).encode())
        with open(p, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
    return h.hexdigest()[:16]
