"""Strict, checkpointable manifest translation for the memoryless first pass."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


MANIFEST_VERSION = 1


@dataclass
class TranslationUnit:
    id: int
    source: str
    translation: str | None


@dataclass
class TranslationManifest:
    version: int
    source_artifact_path: str
    translated_artifact_path: str
    units: list[TranslationUnit]


@dataclass(frozen=True)
class TranslationRequest:
    id: int
    source: str


@dataclass(frozen=True)
class TranslationResult:
    id: int
    translation: str


TranslationBatch = tuple[TranslationRequest, ...]
TranslateBatch = Callable[[TranslationBatch], Sequence[TranslationResult]]


def _require_exact_fields(
    value: object, expected: set[str], *, location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{location} must be a JSON object")
    fields = set(value)
    if fields != expected:
        missing = sorted(expected - fields)
        unknown = sorted(fields - expected)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{location} has invalid fields ({'; '.join(details)})")
    return value


def _validate_manifest(manifest: TranslationManifest) -> None:
    if type(manifest) is not TranslationManifest:
        raise TypeError("manifest must be a TranslationManifest")
    if type(manifest.version) is not int or manifest.version != MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {MANIFEST_VERSION}")
    if type(manifest.source_artifact_path) is not str:
        raise ValueError("source_artifact_path must be a string")
    if type(manifest.translated_artifact_path) is not str:
        raise ValueError("translated_artifact_path must be a string")
    if type(manifest.units) is not list:
        raise ValueError("units must be a list")

    seen_ids: set[int] = set()
    for index, unit in enumerate(manifest.units):
        location = f"units[{index}]"
        if type(unit) is not TranslationUnit:
            raise ValueError(f"{location} must be a TranslationUnit")
        if type(unit.id) is not int:
            raise ValueError(f"{location}.id must be an integer")
        if unit.id in seen_ids:
            raise ValueError(f"duplicate unit id: {unit.id}")
        seen_ids.add(unit.id)
        if type(unit.source) is not str:
            raise ValueError(f"{location}.source must be a string")
        if unit.translation is not None and type(unit.translation) is not str:
            raise ValueError(f"{location}.translation must be a string or null")


def _manifest_from_json(value: object) -> TranslationManifest:
    payload = _require_exact_fields(
        value,
        {"version", "source_artifact_path", "translated_artifact_path", "units"},
        location="manifest",
    )
    raw_units = payload["units"]
    if type(raw_units) is not list:
        raise ValueError("units must be a JSON array")

    units: list[TranslationUnit] = []
    for index, raw_unit in enumerate(raw_units):
        unit = _require_exact_fields(
            raw_unit,
            {"id", "source", "translation"},
            location=f"units[{index}]",
        )
        units.append(
            TranslationUnit(
                id=unit["id"],  # type: ignore[arg-type]
                source=unit["source"],  # type: ignore[arg-type]
                translation=unit["translation"],  # type: ignore[arg-type]
            )
        )

    manifest = TranslationManifest(
        version=payload["version"],  # type: ignore[arg-type]
        source_artifact_path=payload["source_artifact_path"],  # type: ignore[arg-type]
        translated_artifact_path=payload["translated_artifact_path"],  # type: ignore[arg-type]
        units=units,
    )
    _validate_manifest(manifest)
    return manifest


def load_manifest(path: str | os.PathLike[str]) -> TranslationManifest:
    """Load and strictly validate a version-1 translation manifest."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _manifest_from_json(payload)


def save_manifest(
    manifest: TranslationManifest, path: str | os.PathLike[str]
) -> None:
    """Validate and atomically replace a manifest in its destination directory."""

    _validate_manifest(manifest)
    destination = Path(path)
    payload = {
        "version": manifest.version,
        "source_artifact_path": manifest.source_artifact_path,
        "translated_artifact_path": manifest.translated_artifact_path,
        "units": [
            {
                "id": unit.id,
                "source": unit.source,
                "translation": unit.translation,
            }
            for unit in manifest.units
        ],
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_batch_result(
    batch: TranslationBatch, results: Sequence[TranslationResult]
) -> dict[int, str]:
    if type(results) not in (list, tuple):
        raise ValueError("translate_batch must return a list or tuple")
    if len(results) != len(batch):
        raise ValueError("translate_batch returned the wrong number of results")

    expected_ids = {request.id for request in batch}
    translations: dict[int, str] = {}
    for result in results:
        if type(result) is not TranslationResult:
            raise ValueError("translate_batch results must be TranslationResult values")
        if type(result.id) is not int:
            raise ValueError("translation result id must be an integer")
        if result.id in translations:
            raise ValueError(f"translate_batch returned duplicate id: {result.id}")
        if type(result.translation) is not str or not result.translation.strip():
            raise ValueError(
                f"translation for unit {result.id} must be a non-empty string"
            )
        translations[result.id] = result.translation

    if set(translations) != expected_ids:
        raise ValueError("translate_batch result ids do not match the requested batch")
    return translations


def translate_manifest(
    manifest_path: str | os.PathLike[str],
    translate_batch: TranslateBatch,
    batch_size: int,
    max_workers: int,
) -> TranslationManifest:
    """Translate pending units concurrently and checkpoint each completed batch."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")

    manifest = load_manifest(manifest_path)
    units_by_id = {unit.id: unit for unit in manifest.units}
    pending = [unit for unit in manifest.units if unit.translation is None]
    batches = [
        tuple(
            TranslationRequest(id=unit.id, source=unit.source)
            for unit in pending[start : start + batch_size]
        )
        for start in range(0, len(pending), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[Sequence[TranslationResult]], TranslationBatch] = {
            executor.submit(translate_batch, batch): batch for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            translations = _validate_batch_result(batch, future.result())
            for request in batch:
                units_by_id[request.id].translation = translations[request.id]
            save_manifest(manifest, manifest_path)

    return manifest
