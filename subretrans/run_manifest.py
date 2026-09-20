"""Strict, atomically committed run manifest and checkpoint reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .fsutil import atomic_write_json, require_exact_fields, sha256_file
from .state import NextStage, PipelineState, Stage, TranslationMode, validate_pipeline_state


RUN_MANIFEST_VERSION = 3
MANIFEST_FIELDS = {
    "version",
    "run_id",
    "revision",
    "previous_manifest_hash",
    "completed_operation_id",
    "completed_stage",
    "next_stage",
    "route_reason",
    "translation_mode",
    "inputs",
    "configuration",
    "artifacts",
    "heads",
    "budgets",
    "review",
    "release",
}
ARTIFACT_FIELDS = {"kind", "path", "sha256", "schema_version", "depends_on"}
HEAD_FIELDS = {
    "preprocessed",
    "translation_manifest",
    "translated",
    "current",
    "cue_manifest",
    "refined",
    "memory",
    "effective_glossary",
    "suggestion_pool",
    "decision_log",
    "repair_state",
    "repair_history",
    "repair_staged",
    "repair_exchanges",
    "coverage",
    "review",
    "approved",
}
PROMPT_NAMES = {"shared", "refine", "qa", "repair"}
BUDGET_NAMES = {
    "repair_attempts",
    "tool_steps",
    "full_sweeps",
    "glossary_repairs",
    "research_requests",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: object, location: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{location} must be a lowercase SHA-256 digest")
    return value


def stable_operation_id(run_id: str, stage: str, previous_manifest_hash: str) -> str:
    """Return a retry-stable operation id for one manifest transition."""

    value = f"{run_id}\0{stage}\0{previous_manifest_hash}".encode()
    return hashlib.sha256(value).hexdigest()


def _validate_artifact_ref(value: object, *, location: str) -> dict[str, Any]:
    ref = require_exact_fields(value, ARTIFACT_FIELDS, location=location)
    if type(ref["kind"]) is not str or not ref["kind"]:
        raise ValueError(f"{location}.kind must be a non-empty string")
    raw_path = ref["path"]
    if type(raw_path) is not str or not raw_path:
        raise ValueError(f"{location}.path must be a non-empty relative path")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{location}.path must stay inside the run directory")
    _require_sha256(ref["sha256"], f"{location}.sha256")
    if type(ref["schema_version"]) is not int or ref["schema_version"] <= 0:
        raise ValueError(f"{location}.schema_version must be a positive integer")
    if type(ref["depends_on"]) is not list or not all(
        type(name) is str and name for name in ref["depends_on"]
    ):
        raise ValueError(f"{location}.depends_on must be a string list")
    return ref


def validate_manifest(value: object, *, manifest_path: Path) -> dict[str, Any]:
    """Validate the full v3 schema and its internal references."""

    payload = require_exact_fields(value, MANIFEST_FIELDS, location="run manifest")
    if payload["version"] != RUN_MANIFEST_VERSION:
        raise ValueError(f"run manifest version must be {RUN_MANIFEST_VERSION}")
    if type(payload["run_id"]) is not str or not payload["run_id"]:
        raise ValueError("run manifest run_id must be a non-empty string")
    if type(payload["revision"]) is not int or payload["revision"] < 0:
        raise ValueError("run manifest revision must be a non-negative integer")
    _require_sha256(
        payload["previous_manifest_hash"],
        "run manifest previous_manifest_hash",
        allow_empty=payload["revision"] == 0,
    )
    completed_stage = payload["completed_stage"]
    completed_operation_id = payload["completed_operation_id"]
    if payload["revision"] == 0:
        if completed_operation_id is not None or completed_stage is not None:
            raise ValueError("initial run manifest cannot have a completed operation")
    else:
        if type(completed_stage) is not str:
            raise ValueError("run manifest completed_stage must be a string")
        _require_sha256(completed_operation_id, "run manifest completed_operation_id")
        expected = stable_operation_id(
            payload["run_id"], completed_stage, payload["previous_manifest_hash"]
        )
        if completed_operation_id != expected:
            raise ValueError("run manifest completed_operation_id is not stable")
    if payload["translation_mode"] not in {"parallel_initial", "serial_memory"}:
        raise ValueError("run manifest translation_mode is invalid")

    inputs = require_exact_fields(payload["inputs"], {"source"}, location="run manifest.inputs")
    source = require_exact_fields(
        inputs["source"], {"path", "sha256", "kind"}, location="run manifest.inputs.source"
    )
    if any(type(source[field]) is not str or not source[field] for field in ("path", "kind")):
        raise ValueError("run manifest source path and kind must be non-empty strings")
    _require_sha256(source["sha256"], "run manifest.inputs.source.sha256")
    configuration = require_exact_fields(
        payload["configuration"],
        {"path", "sha256", "prompts"},
        location="run manifest.configuration",
    )
    if type(configuration["path"]) is not str or not configuration["path"]:
        raise ValueError("run manifest configuration.path must be a non-empty string")
    _require_sha256(configuration["sha256"], "run manifest.configuration.sha256")
    prompts = require_exact_fields(
        configuration["prompts"], PROMPT_NAMES, location="run manifest.configuration.prompts"
    )
    for name, prompt_value in prompts.items():
        prompt = require_exact_fields(
            prompt_value,
            {"path", "sha256"},
            location=f"run manifest.configuration.prompts.{name}",
        )
        if type(prompt["path"]) is not str or not prompt["path"]:
            raise ValueError(f"run manifest prompt {name} path must be a non-empty string")
        _require_sha256(prompt["sha256"], f"run manifest prompt {name} sha256")

    artifacts = payload["artifacts"]
    if type(artifacts) is not dict:
        raise ValueError("run manifest artifacts must be an object")
    for name, ref in artifacts.items():
        if type(name) is not str or not name:
            raise ValueError("run manifest artifact names must be non-empty strings")
        _validate_artifact_ref(ref, location=f"run manifest.artifacts.{name}")
    for name, ref in artifacts.items():
        unknown = set(ref["depends_on"]) - set(artifacts)
        if unknown:
            raise ValueError(f"artifact {name} has unknown dependencies: {sorted(unknown)}")

    heads = require_exact_fields(payload["heads"], HEAD_FIELDS, location="run manifest.heads")
    for name, artifact_name in heads.items():
        if artifact_name is not None and artifact_name not in artifacts:
            raise ValueError(f"run manifest head {name} references an unknown artifact")

    budgets = require_exact_fields(
        payload["budgets"], BUDGET_NAMES, location="run manifest.budgets"
    )
    for name, counter in budgets.items():
        counter = require_exact_fields(
            counter, {"used", "max"}, location=f"run manifest.budgets.{name}"
        )
        if any(type(counter[field]) is not int or counter[field] < 0 for field in ("used", "max")):
            raise ValueError(f"run manifest budget {name} must contain non-negative integers")
        if counter["used"] > counter["max"]:
            raise ValueError(f"run manifest budget {name} exceeds its maximum")

    review = require_exact_fields(
        payload["review"], {"path", "status", "sha256"}, location="run manifest.review"
    )
    release = require_exact_fields(
        payload["release"], {"path", "status", "sha256"}, location="run manifest.release"
    )
    run_dir = manifest_path.parent.resolve()
    review_path = Path(review["path"])
    if not review_path.is_absolute() or not review_path.resolve().is_relative_to(run_dir):
        raise ValueError("run manifest review.path must be inside the run directory")
    if review["status"] not in {"pending", "awaiting", "approved", "rejected"}:
        raise ValueError("run manifest review.status is invalid")
    _require_sha256(review["sha256"], "run manifest.review.sha256", allow_empty=True)
    if type(release["path"]) is not str or not release["path"]:
        raise ValueError("run manifest release.path must be a non-empty string")
    if release["status"] not in {"pending", "released"}:
        raise ValueError("run manifest release.status is invalid")
    _require_sha256(release["sha256"], "run manifest.release.sha256", allow_empty=True)
    validate_pipeline_state(
        {
            "manifest_path": str(manifest_path),
            "manifest_hash": "0" * 64,
            "next_stage": payload["next_stage"],
            "route_reason": payload["route_reason"],
        },
        location="run manifest routing",
    )
    return payload


def load_manifest(path: str | Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = validate_manifest(payload, manifest_path=manifest_path)
    if verify_artifacts:
        verify_manifest_artifacts(manifest_path, manifest)
    return manifest


def verify_manifest_artifacts(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    allow_modified_review: bool = False,
) -> None:
    run_dir = Path(manifest_path).resolve().parent
    mutable_review = manifest["heads"]["review"] if allow_modified_review else None
    for name, ref in manifest["artifacts"].items():
        path = (run_dir / ref["path"]).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValueError(f"manifest artifact is missing or outside run directory: {name}")
        if name != mutable_review and sha256_file(path) != ref["sha256"]:
            raise ValueError(f"manifest artifact hash does not match: {name}")


def create_manifest(
    path: str | Path,
    *,
    run_id: str,
    translation_mode: TranslationMode,
    source_path: Path,
    config_path: Path,
    prompt_paths: Mapping[str, Path],
    release_path: Path,
    budget_limits: Mapping[str, int],
) -> PipelineState:
    manifest_path = Path(path).resolve()
    review_path = manifest_path.parent / "review" / f"{release_path.stem}.review{release_path.suffix}"
    limits = {name: int(budget_limits.get(name, 0)) for name in BUDGET_NAMES}
    payload: dict[str, Any] = {
        "version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "revision": 0,
        "previous_manifest_hash": "",
        "completed_operation_id": None,
        "completed_stage": None,
        "next_stage": "preprocess",
        "route_reason": "new_run",
        "translation_mode": translation_mode,
        "inputs": {
            "source": {
                "path": str(source_path.resolve()),
                "sha256": sha256_file(source_path),
                "kind": source_path.suffix.lower() or "input",
            }
        },
        "configuration": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "prompts": {
                name: {
                    "path": str(prompt_paths[name].resolve()),
                    "sha256": sha256_file(prompt_paths[name]),
                }
                for name in PROMPT_NAMES
            },
        },
        "artifacts": {},
        "heads": {name: None for name in HEAD_FIELDS},
        "budgets": {name: {"used": 0, "max": limits[name]} for name in BUDGET_NAMES},
        "review": {"path": str(review_path), "status": "pending", "sha256": ""},
        "release": {"path": str(release_path.resolve()), "status": "pending", "sha256": ""},
    }
    validate_manifest(payload, manifest_path=manifest_path)
    atomic_write_json(manifest_path, payload)
    return state_from_manifest(manifest_path, payload)


def state_from_manifest(path: str | Path, manifest: Mapping[str, Any] | None = None) -> PipelineState:
    manifest_path = Path(path).resolve()
    current = load_manifest(manifest_path, verify_artifacts=False) if manifest is None else manifest
    return validate_pipeline_state(
        {
            "manifest_path": str(manifest_path),
            "manifest_hash": sha256_file(manifest_path),
            "next_stage": current["next_stage"],
            "route_reason": current["route_reason"],
        }
    )


def reconcile_state(state: PipelineState) -> tuple[PipelineState, bool]:
    """Accept only an exact match or one verified manifest-ahead commit."""

    checked = validate_pipeline_state(state)
    manifest_path = Path(checked["manifest_path"]).resolve()
    manifest = load_manifest(manifest_path, verify_artifacts=False)
    current_hash = sha256_file(manifest_path)
    if current_hash == checked["manifest_hash"]:
        return checked, False
    previous_hash = manifest["previous_manifest_hash"]
    if previous_hash != checked["manifest_hash"]:
        raise ValueError("LangGraph state and run manifest differ by more than one commit")
    expected = stable_operation_id(manifest["run_id"], manifest["completed_stage"], previous_hash)
    if manifest["completed_operation_id"] != expected:
        raise ValueError("run manifest one-step-ahead operation id does not match")
    return state_from_manifest(manifest_path, manifest), True


def artifact_path(manifest_path: str | Path, manifest: Mapping[str, Any], head: str) -> Path:
    artifact_name = manifest["heads"][head]
    if artifact_name is None:
        raise ValueError(f"run manifest has no {head} artifact")
    ref = manifest["artifacts"][artifact_name]
    path = (Path(manifest_path).resolve().parent / ref["path"]).resolve()
    if sha256_file(path) != ref["sha256"]:
        raise ValueError(f"run manifest {head} artifact hash does not match")
    return path


def register_artifact(
    manifest: dict[str, Any],
    manifest_path: str | Path,
    *,
    name: str,
    path: Path,
    kind: str,
    head: str | None = None,
    schema_version: int = 1,
    depends_on: Sequence[str] = (),
) -> None:
    run_dir = Path(manifest_path).resolve().parent
    resolved = path.resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError(f"artifact must stay inside run directory: {path}")
    manifest["artifacts"][name] = {
        "kind": kind,
        "path": str(resolved.relative_to(run_dir)),
        "sha256": sha256_file(resolved),
        "schema_version": schema_version,
        "depends_on": list(depends_on),
    }
    if head is not None:
        if head not in HEAD_FIELDS:
            raise ValueError(f"unsupported manifest head: {head}")
        manifest["heads"][head] = name


def mutable_manifest(
    state: PipelineState,
    *,
    expected_stage: Stage,
    allow_modified_review: bool = False,
) -> tuple[PipelineState, dict[str, Any]]:
    checked, advanced = reconcile_state(state)
    manifest = load_manifest(checked["manifest_path"], verify_artifacts=False)
    verify_manifest_artifacts(
        checked["manifest_path"], manifest, allow_modified_review=allow_modified_review
    )
    if not advanced and manifest["next_stage"] != expected_stage:
        raise ValueError(f"run manifest expected {manifest['next_stage']}, not {expected_stage}")
    return checked, copy.deepcopy(manifest)


def commit_manifest(
    state: PipelineState,
    *,
    stage: Stage,
    next_stage: NextStage,
    route_reason: str,
    manifest: Mapping[str, Any],
    allow_modified_review: bool = False,
) -> PipelineState:
    checked, advanced = reconcile_state(state)
    if advanced:
        if checked["next_stage"] == next_stage:
            return checked
        raise ValueError("cannot commit from a one-step-ahead manifest")
    manifest_path = Path(checked["manifest_path"])
    current = load_manifest(manifest_path, verify_artifacts=False)
    verify_manifest_artifacts(
        manifest_path, current, allow_modified_review=allow_modified_review
    )
    if current["next_stage"] != stage:
        raise ValueError(f"manifest expected stage {current['next_stage']}, cannot commit {stage}")
    updated = copy.deepcopy(dict(manifest))
    previous_hash = checked["manifest_hash"]
    updated.update(
        {
            "version": RUN_MANIFEST_VERSION,
            "run_id": current["run_id"],
            "revision": current["revision"] + 1,
            "previous_manifest_hash": previous_hash,
            "completed_operation_id": stable_operation_id(current["run_id"], stage, previous_hash),
            "completed_stage": stage,
            "next_stage": next_stage,
            "route_reason": route_reason,
        }
    )
    validate_manifest(updated, manifest_path=manifest_path)
    atomic_write_json(manifest_path, updated)
    return state_from_manifest(manifest_path, updated)
