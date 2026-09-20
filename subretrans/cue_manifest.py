"""Frozen, content-addressed cue identity for an ASS episode.

The subtitle pipeline has two useful but different numbering schemes: SRT
cue indexes are normally one based, while bilingual ``SubtitlePair`` ids are
zero based.  A cue manifest makes the latter the canonical evidence identity
and records the mapping back to the source ASS events.  The manifest is
created after the bilingual ASS merge and must not be regenerated from a
later, edited artifact when resuming a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .ass_parser import build_pairs_from_ass_lines, parse_ass_file
from .fsutil import atomic_write_json, require_exact_fields, sha256_file, sha256_json


CUE_MANIFEST_VERSION = 1
ID_SCHEME = "pair-zero-based"


@dataclass(frozen=True)
class CueManifestEntry:
    """Stable identity and source text for one bilingual cue pair."""

    pair_id: int
    eng_line_id: int
    chinese_line_id: int | None
    start: str
    end: str
    english: str
    chinese: str

    @property
    def source_locator(self) -> dict[str, int | None]:
        """Return the ASS event ids used to write this pair back."""

        return {
            "eng_line_id": self.eng_line_id,
            "chinese_line_id": self.chinese_line_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_locator": self.source_locator,
            "start": self.start,
            "end": self.end,
            "english": self.english,
            "chinese": self.chinese,
        }


@dataclass(frozen=True)
class CueManifest:
    """Immutable manifest frozen from one merged ASS artifact."""

    version: int
    episode_id: str
    id_scheme: str
    source_artifact_path: str
    source_artifact_hash: str
    cues: tuple[CueManifestEntry, ...]
    canonical_hash: str

    @property
    def artifact_hash(self) -> str:
        """Alias used by evidence-binding callers."""

        return self.source_artifact_hash

    @property
    def source_hash(self) -> str:
        """Short alias for callers that refer to the source hash directly."""

        return self.source_artifact_hash

    @property
    def manifest_hash(self) -> str:
        """The canonical hash of the manifest payload."""

        return self.canonical_hash

    @property
    def entries(self) -> tuple[CueManifestEntry, ...]:
        """Compatibility alias for callers that call entries rather than cues."""

        return self.cues

    @property
    def cue_ids(self) -> frozenset[int]:
        return frozenset(cue.pair_id for cue in self.cues)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "episode_id": self.episode_id,
            "id_scheme": self.id_scheme,
            "source_artifact_path": self.source_artifact_path,
            "source_artifact_hash": self.source_artifact_hash,
            "cues": [cue.to_dict() for cue in self.cues],
        }
        if include_hash:
            payload["canonical_hash"] = self.canonical_hash
        return payload


def _require_int(value: object, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be an integer")
    return value


def _validate_entry(entry: CueManifestEntry, index: int) -> None:
    prefix = f"cues[{index}]"
    _require_int(entry.pair_id, f"{prefix}.pair_id")
    _require_int(entry.eng_line_id, f"{prefix}.source_locator.eng_line_id")
    if entry.chinese_line_id is not None:
        _require_int(entry.chinese_line_id, f"{prefix}.source_locator.chinese_line_id")
    for field in ("start", "end", "english", "chinese"):
        if type(getattr(entry, field)) is not str:
            raise ValueError(f"{prefix}.{field} must be a string")


def validate_cue_manifest(manifest: CueManifest) -> CueManifest:
    """Validate schema, canonical hash, and the frozen zero-based id sequence."""

    if type(manifest) is not CueManifest:
        raise TypeError("manifest must be a CueManifest")
    if type(manifest.version) is not int or manifest.version != CUE_MANIFEST_VERSION:
        raise ValueError(f"cue manifest version must be {CUE_MANIFEST_VERSION}")
    if type(manifest.id_scheme) is not str or manifest.id_scheme != ID_SCHEME:
        raise ValueError(f"cue manifest id_scheme must be {ID_SCHEME}")
    for field in (
        "episode_id",
        "source_artifact_path",
        "source_artifact_hash",
        "canonical_hash",
    ):
        if type(getattr(manifest, field)) is not str or not getattr(manifest, field):
            raise ValueError(f"cue manifest {field} must be a non-empty string")
    if type(manifest.cues) is not tuple:
        raise ValueError("cue manifest cues must be a tuple")

    seen_pair_ids: set[int] = set()
    seen_event_ids: set[int] = set()
    for index, cue in enumerate(manifest.cues):
        if type(cue) is not CueManifestEntry:
            raise ValueError(f"cues[{index}] must be a CueManifestEntry")
        _validate_entry(cue, index)
        if cue.pair_id in seen_pair_ids:
            raise ValueError(f"duplicate cue pair_id: {cue.pair_id}")
        seen_pair_ids.add(cue.pair_id)
        for event_id in (cue.eng_line_id, cue.chinese_line_id):
            if event_id is None:
                continue
            if event_id in seen_event_ids:
                raise ValueError(f"duplicate ASS event id: {event_id}")
            seen_event_ids.add(event_id)

    if sorted(seen_pair_ids) != list(range(len(manifest.cues))):
        raise ValueError("cue manifest pair ids must be contiguous zero-based ids")
    expected_hash = sha256_json(manifest.to_dict(include_hash=False))
    if manifest.canonical_hash != expected_hash:
        raise ValueError("cue manifest canonical_hash does not match its payload")
    return manifest


def _entry_from_pair(pair: Any) -> CueManifestEntry:
    meta = pair.meta or {}
    try:
        eng_line_id = meta["eng_line_id"]
        chinese_line_id = meta.get("chinese_line_id", -1)
        start = meta["start"]
        end = meta["end"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"pair {pair.id} has no complete ASS source metadata") from exc
    if chinese_line_id == -1:
        chinese_line_id = None
    return CueManifestEntry(
        pair_id=pair.id,
        eng_line_id=eng_line_id,
        chinese_line_id=chinese_line_id,
        start=start,
        end=end,
        english=pair.eng,
        chinese=pair.chinese,
    )


def build_cue_manifest(
    ass_path: str | Path,
    *,
    episode_id: str | None = None,
) -> CueManifest:
    """Build and freeze a manifest from the merged bilingual ASS artifact."""

    source_path = Path(ass_path)
    _, ass_lines = parse_ass_file(source_path)
    pairs = build_pairs_from_ass_lines(ass_lines)
    if not pairs:
        raise ValueError(f"no bilingual cue pairs found in {source_path}")
    entries = tuple(_entry_from_pair(pair) for pair in pairs)
    manifest = CueManifest(
        version=CUE_MANIFEST_VERSION,
        episode_id=episode_id or source_path.stem,
        id_scheme=ID_SCHEME,
        source_artifact_path=str(source_path),
        source_artifact_hash=sha256_file(source_path),
        cues=entries,
        canonical_hash="",
    )
    return replace(manifest, canonical_hash=sha256_json(manifest.to_dict(include_hash=False)))


def save_cue_manifest(manifest: CueManifest, path: str | Path) -> Path:
    """Validate and atomically save a frozen manifest."""

    validate_cue_manifest(manifest)
    return atomic_write_json(path, manifest.to_dict())


def _entry_from_dict(value: object, index: int) -> CueManifestEntry:
    payload = require_exact_fields(
        value,
        {"pair_id", "source_locator", "start", "end", "english", "chinese"},
        location=f"cues[{index}]",
    )
    locator = require_exact_fields(
        payload["source_locator"],
        {"eng_line_id", "chinese_line_id"},
        location=f"cues[{index}].source_locator",
    )
    return CueManifestEntry(
        pair_id=payload["pair_id"],
        eng_line_id=locator["eng_line_id"],
        chinese_line_id=locator["chinese_line_id"],
        start=payload["start"],
        end=payload["end"],
        english=payload["english"],
        chinese=payload["chinese"],
    )


def load_cue_manifest(
    path: str | Path,
    *,
    expected_episode_id: str | None = None,
    expected_manifest_hash: str | None = None,
) -> CueManifest:
    """Load a strictly validated manifest and optionally bind it to a run."""

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    mapping = require_exact_fields(
        payload,
        {
            "version",
            "episode_id",
            "id_scheme",
            "source_artifact_path",
            "source_artifact_hash",
            "cues",
            "canonical_hash",
        },
        location="cue manifest",
    )
    if type(mapping["cues"]) is not list:
        raise ValueError("cue manifest cues must be a JSON array")
    manifest = CueManifest(
        version=mapping["version"],
        episode_id=mapping["episode_id"],
        id_scheme=mapping["id_scheme"],
        source_artifact_path=mapping["source_artifact_path"],
        source_artifact_hash=mapping["source_artifact_hash"],
        cues=tuple(_entry_from_dict(value, index) for index, value in enumerate(mapping["cues"])),
        canonical_hash=mapping["canonical_hash"],
    )
    validate_cue_manifest(manifest)
    if expected_episode_id is not None and manifest.episode_id != expected_episode_id:
        raise ValueError("cue manifest episode_id does not match")
    if expected_manifest_hash is not None and manifest.manifest_hash != expected_manifest_hash:
        raise ValueError("cue manifest hash does not match")
    return manifest


def validate_cue_manifest_against_ass(
    manifest: CueManifest,
    ass_path: str | Path,
    *,
    require_source_hash: bool = True,
) -> None:
    """Validate the manifest against the original frozen ASS artifact."""

    validate_cue_manifest(manifest)
    path = Path(ass_path)
    if require_source_hash and sha256_file(path) != manifest.source_artifact_hash:
        raise ValueError("cue manifest source_artifact_hash does not match ASS artifact")
    _, ass_lines = parse_ass_file(path)
    pairs = build_pairs_from_ass_lines(ass_lines)
    _validate_pair_identity(manifest, pairs)


def validate_cue_identity(manifest: CueManifest, ass_path: str | Path) -> None:
    """Validate stable cue identity against a later artifact whose Chinese text changed."""

    validate_cue_manifest(manifest)
    _, ass_lines = parse_ass_file(Path(ass_path))
    _validate_pair_identity(manifest, build_pairs_from_ass_lines(ass_lines))


def _validate_pair_identity(manifest: CueManifest, pairs: list[Any]) -> None:
    if len(pairs) != len(manifest.cues):
        raise ValueError("cue manifest pair count does not match ASS artifact")
    for entry, pair in zip(manifest.cues, pairs, strict=True):
        meta = pair.meta or {}
        if pair.id != entry.pair_id:
            raise ValueError(f"cue manifest pair id mismatch at {entry.pair_id}")
        if (
            meta.get("eng_line_id") != entry.eng_line_id
            or (None if meta.get("chinese_line_id", -1) == -1 else meta.get("chinese_line_id"))
            != entry.chinese_line_id
            or meta.get("start") != entry.start
            or meta.get("end") != entry.end
            or pair.eng != entry.english
        ):
            raise ValueError(f"cue manifest source identity mismatch at pair {entry.pair_id}")


__all__ = [
    "CUE_MANIFEST_VERSION",
    "ID_SCHEME",
    "CueManifest",
    "CueManifestEntry",
    "build_cue_manifest",
    "load_cue_manifest",
    "save_cue_manifest",
    "validate_cue_identity",
    "validate_cue_manifest",
    "validate_cue_manifest_against_ass",
]
