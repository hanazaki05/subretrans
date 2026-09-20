"""Intermediate representations exchanged with the refine model.

Three formats are supported: JSON arrays, an XML-like ``<pair>`` block format
and a pseudo-TOML ``[pair]`` format. Strict parsers raise
:class:`SerializationError`; recovery helpers salvage what they can from
malformed model output.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from .pairs import SubtitlePair


logger = logging.getLogger(__name__)

REPRESENTATIONS = ("json", "xml-pair", "pseudo-toml")

_FIELD_ASSIGNMENT_SEPARATORS = "=>:|"
_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_XML_BLOCK_RE = re.compile(r"(?is)<pair>\s*(.*?)\s*</pair>")
_UNESCAPED_BACKSLASH_RE = re.compile(r'\\(?!["\\/bfnrtu])')


class SerializationError(Exception):
    """Raised when text cannot be parsed as the requested representation."""


def _normalize_format(format_type: str) -> str:
    normalized = (format_type or "").lower()
    if normalized not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported format: {format_type}. Supported formats: {', '.join(REPRESENTATIONS)}"
        )
    return normalized


def _pair_from_fields(raw_id: object, eng: object, chinese: object) -> SubtitlePair:
    try:
        pair_id = int(str(raw_id).strip())
    except ValueError as exc:
        raise SerializationError(f"Invalid ID value: {raw_id!r}") from exc
    if not isinstance(eng, str) or not isinstance(chinese, str):
        raise SerializationError(f"eng and chinese must be strings for ID {pair_id}")
    return SubtitlePair(id=pair_id, eng=eng, chinese=chinese)


# JSON -----------------------------------------------------------------------


def serialize_json(pairs: Sequence[SubtitlePair]) -> str:
    return json.dumps([pair.to_dict() for pair in pairs], ensure_ascii=False, indent=2)


def deserialize_json(text: str) -> list[SubtitlePair]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Failed to parse JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise SerializationError("JSON must be an array")
    pairs: list[SubtitlePair] = []
    for item in payload:
        if not isinstance(item, dict):
            raise SerializationError(f"Invalid JSON item: {item!r}")
        if not all(key in item for key in ("id", "eng", "chinese")):
            raise SerializationError(f"Missing required fields in: {item!r}")
        pairs.append(_pair_from_fields(item["id"], item["eng"], item["chinese"]))
    return pairs


# XML-pair -------------------------------------------------------------------


def serialize_xml_pair(pairs: Sequence[SubtitlePair]) -> str:
    blocks = [
        f"<pair>\nID={pair.id}\neng={pair.eng}\nchinese={pair.chinese}\n</pair>"
        for pair in pairs
    ]
    return "\n\n".join(blocks)


def _parse_field_assignment(line: str, expected_field: str) -> tuple[str | None, str | None]:
    """Return ``(value, separator)``; separator is ``None`` for a strict ``field=`` match."""

    if "=" in line:
        key, value = line.split("=", 1)
        if key == expected_field:
            return value, None
    match = re.match(
        rf"^({re.escape(expected_field)})\s*([{re.escape(_FIELD_ASSIGNMENT_SEPARATORS)}])\s*(.*)$",
        line.strip(),
    )
    if match:
        _, separator, value = match.groups()
        return value.strip(), separator
    return None, None


def deserialize_xml_pair(text: str) -> list[SubtitlePair]:
    pairs: list[SubtitlePair] = []
    lines = text.strip().split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line != "<pair>":
            raise SerializationError(f"Expected '<pair>' at line {index + 1}, got: {line}")
        index += 1
        fields: dict[str, str] = {}
        for field in ("ID", "eng", "chinese"):
            if index >= len(lines):
                raise SerializationError(f"Unexpected end of input while reading {field}")
            line = lines[index].strip()
            value, separator = _parse_field_assignment(line, field)
            if value is None:
                raise SerializationError(f"Expected '{field}=...' at line {index + 1}, got: {line}")
            if separator is not None and separator != "=":
                logger.warning(
                    "Non-standard separator %r for field %r at line %d; auto-corrected",
                    separator,
                    field,
                    index + 1,
                )
            fields[field] = value
            index += 1
        if index >= len(lines):
            raise SerializationError("Expected '</pair>' tag")
        line = lines[index].strip()
        if line != "</pair>":
            raise SerializationError(f"Expected '</pair>' at line {index + 1}, got: {line}")
        pairs.append(_pair_from_fields(fields["ID"], fields["eng"], fields["chinese"]))
        index += 1
    return pairs


def _extract_xml_pair_field(block: str, field: str) -> str:
    pattern = (
        rf"(?is)\b{re.escape(field)}\s*[{re.escape(_FIELD_ASSIGNMENT_SEPARATORS)}]\s*"
        rf"(.*?)(?=\b(?:ID|eng|chinese)\s*[{re.escape(_FIELD_ASSIGNMENT_SEPARATORS)}]|$)"
    )
    match = re.search(pattern, block)
    if not match:
        raise SerializationError(f"Missing field '{field}'")
    return match.group(1).strip()


def deserialize_xml_pair_best_effort(text: str) -> tuple[list[SubtitlePair], list[str]]:
    """Parse every well-formed ``<pair>`` block and report the ones skipped."""

    matches = list(_XML_BLOCK_RE.finditer(text or ""))
    if not matches:
        return [], ["No <pair>...</pair> blocks found"]
    pairs: list[SubtitlePair] = []
    errors: list[str] = []
    for position, match in enumerate(matches, start=1):
        block = match.group(1)
        try:
            raw_id = _extract_xml_pair_field(block, "ID")
            id_match = re.match(r"\s*(\d+)", raw_id)
            if not id_match:
                raise SerializationError(f"Invalid ID value: {raw_id}")
            pairs.append(
                SubtitlePair(
                    id=int(id_match.group(1)),
                    eng=_extract_xml_pair_field(block, "eng"),
                    chinese=_extract_xml_pair_field(block, "chinese"),
                )
            )
        except SerializationError as exc:
            errors.append(f"pair#{position}: {exc}")
    return pairs, errors


# Pseudo-TOML ----------------------------------------------------------------


def serialize_pseudo_toml(pairs: Sequence[SubtitlePair]) -> str:
    blocks = [
        f"[pair]\nid = {pair.id}\neng = {pair.eng}\nchinese = {pair.chinese}" for pair in pairs
    ]
    return "\n\n".join(blocks)


def deserialize_pseudo_toml(text: str) -> list[SubtitlePair]:
    pairs: list[SubtitlePair] = []
    lines = text.strip().split("\n")
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line != "[pair]":
            raise SerializationError(f"Expected '[pair]' at line {index + 1}, got: {line}")
        index += 1
        fields: dict[str, str] = {}
        for field in ("id", "eng", "chinese"):
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index >= len(lines):
                raise SerializationError(f"Unexpected end of input while reading {field}")
            line = lines[index].strip()
            if "=" not in line:
                raise SerializationError(f"Expected '{field} = ...' at line {index + 1}, got: {line}")
            key, value = (part.strip() for part in line.split("=", 1))
            if key != field:
                raise SerializationError(f"Expected field '{field}' at line {index + 1}, got: {key}")
            fields[field] = value
            index += 1
        pairs.append(_pair_from_fields(fields["id"], fields["eng"], fields["chinese"]))
    return pairs


# Format-agnostic interface --------------------------------------------------


def serialize(pairs: Sequence[SubtitlePair], format_type: str) -> str:
    """Serialize ``pairs`` in the requested representation."""

    normalized = _normalize_format(format_type)
    if normalized == "json":
        return serialize_json(pairs)
    if normalized == "xml-pair":
        return serialize_xml_pair(pairs)
    return serialize_pseudo_toml(pairs)


def deserialize(text: str, format_type: str) -> list[SubtitlePair]:
    """Strictly parse ``text`` in the requested representation."""

    normalized = _normalize_format(format_type)
    if normalized == "json":
        return deserialize_json(text)
    if normalized == "xml-pair":
        return deserialize_xml_pair(text)
    return deserialize_pseudo_toml(text)


def extract_from_format_marker(text: str, format_type: str) -> str | None:
    """Return the portion of ``text`` starting at the first format marker, if any.

    Used after the strict parser fails because a model added commentary before
    the payload. The text is expected to be already cleaned of code fences.
    """

    normalized = _normalize_format(format_type)
    if normalized == "xml-pair":
        index = text.find("<pair>")
        return text[index:].strip() if index != -1 else None
    if normalized == "pseudo-toml":
        index = text.find("[pair]")
        return text[index:].strip() if index != -1 else None
    arrays = _JSON_ARRAY_RE.findall(text)
    if arrays:
        return max(arrays, key=len)
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return None


def deserialize_best_effort(text: str, format_type: str) -> tuple[list[SubtitlePair], list[str]]:
    """Last-resort recovery that keeps well-formed items and reports skipped ones."""

    normalized = _normalize_format(format_type)
    if normalized == "xml-pair":
        return deserialize_xml_pair_best_effort(text)
    return [], [f"Best-effort deserialization not implemented for format: {format_type}"]


def convert_json_examples_to_format(json_text: str, target_format: str) -> str:
    """Convert a JSON example array from a prompt template to ``target_format``.

    Prompt templates may contain raw ASS tags such as ``{\\i1}`` inside JSON
    strings; unescaped backslashes are repaired before the second attempt.
    """

    normalized = _normalize_format(target_format)
    if normalized == "json":
        return json_text
    try:
        pairs = deserialize_json(json_text)
    except SerializationError as original:
        fixed = _UNESCAPED_BACKSLASH_RE.sub(r"\\\\", json_text)
        try:
            pairs = deserialize_json(fixed)
        except SerializationError:
            raise SerializationError(f"Failed to parse JSON: {original}") from original
    return serialize(pairs, normalized)
