"""Atomic filesystem helpers and strict payload validation shared by every module."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import yaml


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_distinct_paths(first: str | os.PathLike[str], second: str | os.PathLike[str]) -> None:
    """Reject in-place processing so an input is never clobbered by its output."""

    if Path(first).resolve() == Path(second).resolve():
        raise ValueError(f"input and output paths must differ: {Path(first)}")


def _replace_atomically(path: Path, write: Callable[[BinaryIO], object]) -> Path:
    """Write through a temporary file in the destination directory, then rename."""

    destination = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Atomically replace ``path`` with ``data``."""

    return _replace_atomically(Path(path), lambda handle: handle.write(data))


def atomic_write_text(
    path: str | os.PathLike[str], content: str, *, encoding: str = "utf-8"
) -> Path:
    """Atomically replace ``path`` with ``content`` encoded verbatim (no newline translation)."""

    return atomic_write_bytes(path, content.encode(encoding))


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically write a UTF-8, human-readable JSON object with a trailing newline."""

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return atomic_write_text(path, text)


def atomic_write_yaml(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically write a UTF-8 YAML mapping preserving key order."""

    text = yaml.safe_dump(
        dict(payload), allow_unicode=True, default_flow_style=False, sort_keys=False
    )
    return atomic_write_text(path, text)


def atomic_copy(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> Path:
    """Atomically copy ``source`` over ``destination``; the two paths must differ."""

    require_distinct_paths(source, destination)
    source_path = Path(source)

    def write(handle: BinaryIO) -> None:
        with source_path.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)

    return _replace_atomically(Path(destination), write)


def require_exact_fields(
    value: object, expected: set[str], *, location: str
) -> dict[str, Any]:
    """Return ``value`` when it is a mapping with exactly ``expected`` keys, else raise."""

    if type(value) is not dict:
        raise ValueError(f"{location} must be a JSON object")
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        unknown = sorted(fields - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{location} has invalid fields ({'; '.join(details)})")
    return value
