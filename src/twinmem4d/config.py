"""Load YAML files into (nested) dataclasses."""

import dataclasses
import typing
from typing import Any, Dict, Type, TypeVar

from .io_utils import PathLike, read_yaml

T = TypeVar("T")


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build ``cls`` from a dict; nested dataclass fields are built recursively; unknown keys fail."""
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = from_dict(hint, value)
        elif typing.get_origin(hint) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(cls: Type[T], path: PathLike, section: str = "") -> T:
    data = read_yaml(path) or {}
    if section:
        data = data.get(section, {})
    return from_dict(cls, data)
