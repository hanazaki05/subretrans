import json
from pathlib import Path

import pytest

from subretrans.fsutil import atomic_write_json, sha256_file
from subretrans.run_manifest import (
    commit_manifest,
    create_manifest,
    load_manifest,
    mutable_manifest,
    reconcile_state,
    register_artifact,
)
from subretrans.state import validate_pipeline_state


def new_run(tmp_path: Path):
    source = tmp_path / "source.ass"
    source.write_text("source", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    return create_manifest(
        tmp_path / "run.json",
        run_id="episode-01",
        translation_mode="serial_memory",
        source_path=source,
        config_path=config,
        prompt_paths={name: config for name in ("shared", "refine", "qa", "repair")},
        release_path=tmp_path / "release.ass",
        budget_limits={
            "repair_attempts": 2,
            "tool_steps": 8,
            "full_sweeps": 1,
            "glossary_repairs": 1,
            "research_requests": 2,
        },
    )


def test_v3_manifest_roundtrip_has_exact_authoritative_shape(tmp_path: Path) -> None:
    state = new_run(tmp_path)
    manifest = load_manifest(state["manifest_path"])

    assert manifest["version"] == 3
    assert manifest["revision"] == 0
    assert manifest["next_stage"] == "preprocess"
    assert manifest["heads"]["current"] is None
    assert manifest["budgets"]["repair_attempts"] == {"used": 0, "max": 2}
    assert Path(manifest["review"]["path"]).is_relative_to(tmp_path)


def test_register_artifact_rejects_paths_outside_run(tmp_path: Path) -> None:
    state = new_run(tmp_path)
    manifest = load_manifest(state["manifest_path"])
    outside = tmp_path.parent / "outside.ass"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="inside run directory"):
        register_artifact(
            manifest,
            state["manifest_path"],
            name="outside",
            path=outside,
            kind="ass",
        )


def test_exactly_one_manifest_ahead_commit_is_reconciled(tmp_path: Path) -> None:
    old_state = new_run(tmp_path)
    checked, manifest = mutable_manifest(old_state, expected_stage="preprocess")
    committed = commit_manifest(
        checked,
        stage="preprocess",
        next_stage="freeze_manifest",
        route_reason="serial_input_ready",
        manifest=manifest,
    )

    reconciled, advanced = reconcile_state(old_state)
    assert advanced is True
    assert reconciled == committed


def test_more_than_one_manifest_ahead_is_rejected(tmp_path: Path) -> None:
    original = new_run(tmp_path)
    checked, manifest = mutable_manifest(original, expected_stage="preprocess")
    first = commit_manifest(
        checked,
        stage="preprocess",
        next_stage="freeze_manifest",
        route_reason="first",
        manifest=manifest,
    )
    checked, manifest = mutable_manifest(first, expected_stage="freeze_manifest")
    commit_manifest(
        checked,
        stage="freeze_manifest",
        next_stage="refine_serial",
        route_reason="second",
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="more than one commit"):
        reconcile_state(original)


def test_manifest_rejects_tampered_operation_id(tmp_path: Path) -> None:
    state = new_run(tmp_path)
    checked, manifest = mutable_manifest(state, expected_stage="preprocess")
    commit_manifest(
        checked,
        stage="preprocess",
        next_stage="freeze_manifest",
        route_reason="complete",
        manifest=manifest,
    )
    path = Path(state["manifest_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed_operation_id"] = "0" * 64
    atomic_write_json(path, payload)

    with pytest.raises(ValueError, match="operation_id is not stable"):
        load_manifest(path, verify_artifacts=False)


def test_manifest_detects_artifact_hash_changes(tmp_path: Path) -> None:
    state = new_run(tmp_path)
    artifact = tmp_path / "artifact.ass"
    artifact.write_text("one", encoding="utf-8")
    checked, manifest = mutable_manifest(state, expected_stage="preprocess")
    register_artifact(
        manifest,
        state["manifest_path"],
        name="artifact",
        path=artifact,
        kind="ass",
        head="current",
    )
    committed = commit_manifest(
        checked,
        stage="preprocess",
        next_stage="freeze_manifest",
        route_reason="complete",
        manifest=manifest,
    )
    assert committed["manifest_hash"] == sha256_file(state["manifest_path"])

    artifact.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash does not match"):
        load_manifest(state["manifest_path"])


def test_state_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    state = new_run(tmp_path)
    missing = dict(state)
    missing.pop("route_reason")
    with pytest.raises(ValueError, match="missing fields"):
        validate_pipeline_state(missing)
    with pytest.raises(ValueError, match="unknown fields"):
        validate_pipeline_state({**state, "artifact_path": "legacy.ass"})
