import json
from dataclasses import replace
from pathlib import Path

import pytest

from subretrans.cue_manifest import (
    build_cue_manifest,
    load_cue_manifest,
    save_cue_manifest,
    validate_cue_identity,
    validate_cue_manifest,
    validate_cue_manifest_against_ass,
)
from subretrans.fsutil import sha256_file, sha256_json


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "real"


@pytest.mark.parametrize("name, episode_id", [("S07E08", "S07E08"), ("S07E11", "S07E11")])
def test_real_fixture_freezes_zero_based_cue_identity(name: str, episode_id: str) -> None:
    source = FIXTURE_DIR / f"{name}-minimal-merged.ass"

    manifest = build_cue_manifest(source, episode_id=episode_id)

    assert manifest.episode_id == episode_id
    assert manifest.id_scheme == "pair-zero-based"
    assert [cue.pair_id for cue in manifest.cues] == [0, 1, 2]
    assert manifest.source_artifact_hash == sha256_file(source)
    assert manifest.canonical_hash == sha256_json(manifest.to_dict(include_hash=False))
    assert manifest.cues[0].source_locator == {"eng_line_id": 0, "chinese_line_id": 1}
    assert manifest.cues[1].source_locator == {"eng_line_id": 2, "chinese_line_id": 3}
    validate_cue_manifest_against_ass(manifest, source)


def test_manifest_roundtrip_and_later_chinese_edit_keep_source_identity(tmp_path: Path) -> None:
    source = FIXTURE_DIR / "S07E08-minimal-merged.ass"
    manifest = build_cue_manifest(source, episode_id="S07E08")
    output = tmp_path / "cue-manifest.json"

    save_cue_manifest(manifest, output)
    loaded = load_cue_manifest(output, expected_episode_id="S07E08", expected_manifest_hash=manifest.manifest_hash)
    assert loaded == manifest

    changed = tmp_path / "postprocessed.ass"
    changed.write_text(
        source.read_text(encoding="utf-8").replace("女士，您没事吧？", "女士，您还好吗？"),
        encoding="utf-8",
    )
    validate_cue_identity(manifest, changed)


def test_manifest_rejects_hash_tampering() -> None:
    source = FIXTURE_DIR / "S07E11-minimal-merged.ass"
    manifest = build_cue_manifest(source, episode_id="S07E11")
    tampered = replace(manifest, canonical_hash="0" * 64)

    with pytest.raises(ValueError, match="canonical_hash"):
        validate_cue_manifest(tampered)


def test_manifest_json_has_strict_top_level_shape(tmp_path: Path) -> None:
    source = FIXTURE_DIR / "S07E11-minimal-merged.ass"
    manifest = build_cue_manifest(source, episode_id="S07E11")
    output = tmp_path / "cue-manifest.json"
    save_cue_manifest(manifest, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid fields"):
        load_cue_manifest(output)
