import hashlib
import json
from pathlib import Path

import pytest
import yaml

from subretrans.fsutil import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    require_distinct_paths,
    require_exact_fields,
    sha256_file,
)


def no_temporaries(directory: Path) -> bool:
    return not list(directory.glob(".*.tmp"))


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello" * 1000)

    assert sha256_file(path) == hashlib.sha256(b"hello" * 1000).hexdigest()


def test_atomic_writers_replace_content_without_leftovers(tmp_path: Path) -> None:
    text_path = tmp_path / "out.txt"
    text_path.write_text("old", encoding="utf-8")

    assert atomic_write_text(text_path, "新\r\ncontent", encoding="utf-8-sig") == text_path
    assert text_path.read_bytes() == "新\r\ncontent".encode("utf-8-sig")

    bytes_path = tmp_path / "out.bin"
    atomic_write_bytes(bytes_path, b"\x00\x01")
    assert bytes_path.read_bytes() == b"\x00\x01"

    json_path = tmp_path / "out.json"
    atomic_write_json(json_path, {"b": "值", "a": 1})
    assert json_path.read_text(encoding="utf-8") == '{\n  "b": "值",\n  "a": 1\n}\n'
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"b": "值", "a": 1}

    yaml_path = tmp_path / "out.yaml"
    atomic_write_yaml(yaml_path, {"zeta": ["值"], "alpha": "x"})
    assert yaml_path.read_text(encoding="utf-8") == "zeta:\n- 值\nalpha: x\n"
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == {
        "zeta": ["值"],
        "alpha": "x",
    }
    assert no_temporaries(tmp_path)


def test_atomic_copy_replaces_destination_and_rejects_same_path(tmp_path: Path) -> None:
    source = tmp_path / "source.ass"
    source.write_bytes(b"\xef\xbb\xbfcontent")
    destination = tmp_path / "nested" / "destination.ass"
    destination.parent.mkdir()
    destination.write_bytes(b"stale")

    assert atomic_copy(source, destination) == destination
    assert destination.read_bytes() == b"\xef\xbb\xbfcontent"
    assert no_temporaries(destination.parent)
    with pytest.raises(ValueError, match="must differ"):
        atomic_copy(source, tmp_path / "." / "source.ass")


def test_failed_write_keeps_original_and_removes_temporary(tmp_path: Path) -> None:
    destination = tmp_path / "keep.txt"
    destination.write_text("original", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        atomic_copy(tmp_path / "missing.txt", destination)

    assert destination.read_text(encoding="utf-8") == "original"
    assert no_temporaries(tmp_path)


def test_require_distinct_paths(tmp_path: Path) -> None:
    require_distinct_paths(tmp_path / "a", tmp_path / "b")
    with pytest.raises(ValueError, match="must differ"):
        require_distinct_paths(tmp_path / "a", tmp_path / "sub" / ".." / "a")


@pytest.mark.parametrize(
    "value, match",
    [
        ([], "must be a JSON object"),
        ({"id": 1}, "missing fields: translation"),
        ({"id": 1, "translation": "x", "extra": 2}, "unknown fields: extra"),
        ({"translation": "x", "extra": 2}, "missing fields: id; unknown fields: extra"),
    ],
)
def test_require_exact_fields_rejects_shape_violations(value, match) -> None:
    with pytest.raises(ValueError, match=match):
        require_exact_fields(value, {"id", "translation"}, location="item")


def test_require_exact_fields_returns_matching_mapping() -> None:
    value = {"id": 1, "translation": "x"}

    assert require_exact_fields(value, {"id", "translation"}, location="item") is value
