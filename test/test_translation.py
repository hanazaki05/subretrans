import json
import threading
import time

import pytest

from subretrans.translation import (
    TranslationManifest,
    TranslationRequest,
    TranslationResult,
    TranslationUnit,
    load_manifest,
    save_manifest,
    translate_manifest,
)


def write_payload(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_payload() -> dict:
    return {
        "version": 1,
        "source_artifact_path": "episode.source.json",
        "translated_artifact_path": "episode.zh.json",
        "units": [
            {"id": 1, "source": "one", "translation": None},
            {"id": 2, "source": "two", "translation": "已有"},
        ],
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payload: payload.update(extra=True), "unknown fields"),
        (lambda payload: payload.pop("source_artifact_path"), "missing fields"),
        (lambda payload: payload.update(version=2), "version must be 1"),
        (lambda payload: payload.update(units="bad"), "units must be a JSON array"),
        (
            lambda payload: payload["units"][0].update(extra=True),
            "unknown fields",
        ),
        (
            lambda payload: payload["units"].append(
                {"id": 1, "source": "again", "translation": None}
            ),
            "duplicate unit id",
        ),
        (lambda payload: payload["units"][0].update(id=True), "id must be an integer"),
        (
            lambda payload: payload["units"][0].update(translation=3),
            "translation must be a string or null",
        ),
    ],
)
def test_load_manifest_rejects_schema_violations(tmp_path, mutate, match) -> None:
    payload = valid_payload()
    mutate(payload)
    path = tmp_path / "manifest.json"
    write_payload(path, payload)

    with pytest.raises(ValueError, match=match):
        load_manifest(path)


def test_save_and_load_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = TranslationManifest(
        version=1,
        source_artifact_path="source.json",
        translated_artifact_path="translated.json",
        units=[TranslationUnit(id=7, source="Hello", translation=None)],
    )

    save_manifest(manifest, path)

    assert load_manifest(path) == manifest
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_translate_manifest_runs_parallel_and_preserves_unit_order(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    payload = valid_payload()
    payload["units"] = [
        {"id": 1, "source": "slow", "translation": None},
        {"id": 2, "source": "done", "translation": "完成"},
        {"id": 3, "source": "fast", "translation": None},
    ]
    write_payload(path, payload)
    barrier = threading.Barrier(2)
    received: list[tuple[TranslationRequest, ...]] = []

    def translate(batch):
        received.append(batch)
        barrier.wait()
        if batch[0].id == 1:
            time.sleep(0.03)
        return tuple(
            TranslationResult(item.id, f"zh:{item.source}") for item in reversed(batch)
        )

    result = translate_manifest(path, translate, batch_size=1, max_workers=2)

    assert all(type(batch) is tuple for batch in received)
    assert all(type(item) is TranslationRequest for batch in received for item in batch)
    assert [unit.id for unit in result.units] == [1, 2, 3]
    assert [unit.translation for unit in result.units] == ["zh:slow", "完成", "zh:fast"]
    assert load_manifest(path) == result


def test_translate_manifest_checkpoints_each_batch_and_resumes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manifest.json"
    payload = valid_payload()
    payload["units"] = [
        {"id": 1, "source": "one", "translation": None},
        {"id": 2, "source": "two", "translation": "existing"},
        {"id": 3, "source": "three", "translation": None},
        {"id": 4, "source": "four", "translation": None},
    ]
    write_payload(path, payload)
    calls: list[tuple[int, ...]] = []
    checkpoints: list[list[str | None]] = []

    def translate(batch):
        calls.append(tuple(item.id for item in batch))
        return [TranslationResult(item.id, item.source.upper()) for item in batch]

    from subretrans import translation

    real_save = translation.save_manifest

    def recording_save(manifest, manifest_path):
        checkpoints.append([unit.translation for unit in manifest.units])
        real_save(manifest, manifest_path)

    monkeypatch.setattr(translation, "save_manifest", recording_save)
    translate_manifest(path, translate, batch_size=2, max_workers=1)
    translate_manifest(path, translate, batch_size=2, max_workers=1)

    assert calls == [(1, 3), (4,)]
    assert len(checkpoints) == 2
    assert checkpoints[-1] == ["ONE", "existing", "THREE", "FOUR"]


@pytest.mark.parametrize(
    "bad_results, match",
    [
        ([TranslationResult(1, "ok")], "wrong number"),
        (
            [TranslationResult(1, "ok"), TranslationResult(99, "bad")],
            "do not match",
        ),
        (
            [TranslationResult(1, "ok"), TranslationResult(2, "   ")],
            "non-empty string",
        ),
    ],
)
def test_translate_manifest_fails_without_checkpointing_invalid_batch(
    tmp_path, bad_results, match
) -> None:
    path = tmp_path / "manifest.json"
    payload = valid_payload()
    payload["units"][1]["translation"] = None
    write_payload(path, payload)

    with pytest.raises(ValueError, match=match):
        translate_manifest(path, lambda batch: bad_results, batch_size=2, max_workers=1)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_translate_manifest_propagates_batch_failure(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_payload(path, valid_payload())

    def fail(batch):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        translate_manifest(path, fail, batch_size=1, max_workers=1)

    assert load_manifest(path).units[0].translation is None
