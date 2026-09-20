import hashlib
import json
from pathlib import Path

import pytest

from subretrans.model_agent import AgentQAResult, AgentRepair
from subretrans.subtitle_processing import read_srt
from subretrans.stage_handlers import WorkflowSettings, build_stage_handlers
from subretrans.state import PipelineState
from subretrans.translation import TranslationResult, load_manifest


VALID_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,你好
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_for(path: Path, mode: str) -> PipelineState:
    return {
        "artifact_path": str(path),
        "artifact_hash": sha256(path),
        "translation_manifest_path": None,
        "translation_mode": mode,  # type: ignore[typeddict-item]
        "stage": "preprocess",
        "refine_chunk_cursor": 0,
        "memory_checkpoint_path": None,
        "memory_hash": "",
        "model_versions": {
            "primer": "test-primer",
            "refine": "test-refine",
            "extraction": "test-extraction",
            "agent": "test-agent",
        },
        "prompt_version": "test-prompt",
        "qa_conclusion": "pending",
        "qa_passed": False,
        "qa_repair_applied": False,
        "agent_repair_attempts": 0,
    }


def apply_update(state: PipelineState, update) -> PipelineState:
    state.update(update)
    return state


def make_refine(*, progress_mutation=None):
    def refine(
        input_path: Path,
        output_path: Path,
        checkpoint_path: Path,
        progress_path: Path,
    ) -> None:
        output_path.write_bytes(input_path.read_bytes())
        checkpoint_path.write_text("story: test\n", encoding="utf-8")
        payload = {
            "version": 1,
            "next_pair": 7,
            "artifact_path": str(output_path),
            "artifact_hash": sha256(output_path),
            "memory_checkpoint_path": str(checkpoint_path),
            "memory_hash": sha256(checkpoint_path),
        }
        if progress_mutation is not None:
            progress_mutation(payload)
        progress_path.write_text(json.dumps(payload), encoding="utf-8")

    return refine


def pass_agent_qa(pairs, structural_qa):
    return AgentQAResult(True, (), ())


def test_parallel_handlers_run_core_chain_without_memory_in_manifest(tmp_path) -> None:
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"fake media")
    run_dir = tmp_path / "run"
    release_path = tmp_path / "release.ass"
    preprocess_calls = []
    seen_batches = []

    def preprocess_subtitle(input_path: Path, output_path: Path) -> Path:
        preprocess_calls.append((input_path, output_path))
        output_path.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
            encoding="utf-8",
        )
        return output_path

    def translate(batch):
        seen_batches.append(batch)
        return [
            TranslationResult(request.id, f"中:{request.source}")
            for request in batch
        ]

    handlers = build_stage_handlers(
        WorkflowSettings(
            run_dir,
            release_path,
            batch_size=1,
            max_workers=1,
            agent_max_repair_attempts=2,
            episode_replacements=(),
        ),
        preprocess_subtitle=preprocess_subtitle,
        translate_batch=translate,
        refine=make_refine(),
        agent_qa=pass_agent_qa,
    )
    state = state_for(source, "parallel_initial")

    apply_update(state, handlers["preprocess"](state))
    manifest_path = Path(state["translation_manifest_path"])  # type: ignore[arg-type]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest_payload) == {
        "version",
        "source_artifact_path",
        "translated_artifact_path",
        "units",
    }
    assert all("memory" not in key for key in manifest_payload)
    assert manifest_payload["source_artifact_path"] == str(
        run_dir / "preprocessed.en.srt"
    )
    assert preprocess_calls == [(source, run_dir / "preprocessed.en.srt")]
    assert state["artifact_path"] == str(source)

    translation_state = {
        "artifact_path": state["artifact_path"],
        "artifact_hash": state["artifact_hash"],
        "translation_manifest_path": state["translation_manifest_path"],
        "stage": "translate_parallel",
        "model_version": state["model_versions"]["primer"],
        "prompt_version": state["prompt_version"],
    }
    assert handlers["translate_parallel"](translation_state) == {}
    assert [request.id for batch in seen_batches for request in batch] == [1, 2]
    translated = read_srt(run_dir / "translated.zh.srt")
    assert [(cue.start, cue.end, cue.text) for cue in translated] == [
        ("00:00:01,000", "00:00:02,000", "中:Hello"),
        ("00:00:03,000", "00:00:04,000", "中:World"),
    ]
    assert all(unit.translation for unit in load_manifest(manifest_path).units)

    apply_update(state, handlers["merge_ass"](state))
    assert Path(state["artifact_path"]).name == "merged.ass"
    assert state["artifact_hash"] == sha256(Path(state["artifact_path"]))
    apply_update(state, handlers["refine_serial"](state))
    assert state["refine_chunk_cursor"] == 7
    assert state["memory_hash"] == sha256(run_dir / "memory.yaml")
    apply_update(state, handlers["postprocess"](state))
    apply_update(state, handlers["qa"](state))
    assert state["qa_conclusion"] == "passed"
    assert handlers["human_review"](state) == {}
    apply_update(state, handlers["release"](state))
    assert state["artifact_path"] == str(release_path)
    assert state["artifact_hash"] == sha256(release_path)
    assert release_path.read_bytes() == (run_dir / "postprocessed.ass").read_bytes()


def test_serial_handlers_skip_manifest_and_run_core_chain(tmp_path) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    run_dir = tmp_path / "serial-run"
    release_path = tmp_path / "serial-release.ass"
    handlers = build_stage_handlers(
        WorkflowSettings(
            run_dir,
            release_path,
            batch_size=2,
            max_workers=2,
            agent_max_repair_attempts=2,
            episode_replacements=(),
        ),
        preprocess_subtitle=lambda input_path, output_path: pytest.fail(
            "serial mode preprocessed"
        ),
        translate_batch=lambda batch: pytest.fail("serial mode translated"),
        refine=make_refine(),
        agent_qa=pass_agent_qa,
    )
    state = state_for(source, "serial_memory")

    assert handlers["preprocess"](state) == {"translation_manifest_path": None}
    assert not (run_dir / "translation.json").exists()
    apply_update(state, handlers["refine_serial"](state))
    apply_update(state, handlers["postprocess"](state))
    apply_update(state, handlers["qa"](state))
    assert state["qa_conclusion"] == "passed"
    apply_update(state, handlers["release"](state))
    assert release_path.exists()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payload: payload.update(version=2), "version must be 1"),
        (
            lambda payload: payload.update(artifact_hash="0" * 64),
            "artifact_hash does not match",
        ),
        (
            lambda payload: payload.update(memory_hash="0" * 64),
            "memory_hash does not match",
        ),
        (lambda payload: payload.update(next_pair=True), "non-negative integer"),
        (lambda payload: payload.update(extra=True), "unknown fields: extra"),
    ],
)
def test_refine_rejects_invalid_progress(tmp_path, mutate, match) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    handlers = build_stage_handlers(
        WorkflowSettings(
            tmp_path / "run",
            tmp_path / "release.ass",
            batch_size=1,
            max_workers=1,
            agent_max_repair_attempts=2,
            episode_replacements=(),
        ),
        preprocess_subtitle=lambda input_path, output_path: output_path,
        translate_batch=lambda batch: (),
        refine=make_refine(progress_mutation=mutate),
        agent_qa=pass_agent_qa,
    )

    with pytest.raises(ValueError, match=match):
        handlers["refine_serial"](state_for(source, "serial_memory"))


def test_qa_failure_reports_counts(tmp_path) -> None:
    source = tmp_path / "bad.ass"
    source.write_text(VALID_ASS.replace(",,你好", ",,"), encoding="utf-8")
    handlers = build_stage_handlers(
        WorkflowSettings(
            tmp_path / "run",
            tmp_path / "release.ass",
            batch_size=1,
            max_workers=1,
            agent_max_repair_attempts=2,
            episode_replacements=(),
        ),
        preprocess_subtitle=lambda input_path, output_path: output_path,
        translate_batch=lambda batch: (),
        refine=make_refine(),
        agent_qa=lambda pairs, structural_qa: AgentQAResult(
            False, ("Missing translation",), ()
        ),
    )

    conclusion = handlers["qa"](state_for(source, "serial_memory"))["qa_conclusion"]

    assert conclusion.startswith("structural=failed: ")
    assert "empty_chinese_events=1" in conclusion


def test_agent_qa_applies_bounded_targeted_repair_to_new_artifact(tmp_path) -> None:
    source = tmp_path / "wrong.ass"
    source.write_text(VALID_ASS.replace("你好", "错误"), encoding="utf-8")
    run_dir = tmp_path / "run"
    handlers = build_stage_handlers(
        WorkflowSettings(
            run_dir,
            tmp_path / "release.ass",
            batch_size=1,
            max_workers=1,
            agent_max_repair_attempts=1,
            episode_replacements=(),
        ),
        preprocess_subtitle=lambda input_path, output_path: output_path,
        translate_batch=lambda batch: (),
        refine=make_refine(),
        agent_qa=lambda pairs, structural_qa: AgentQAResult(
            False,
            ("The Chinese translation is incorrect.",),
            (AgentRepair(0, "你好"),),
        ),
    )
    state = state_for(source, "serial_memory")

    update = handlers["qa"](state)

    repaired = Path(update["artifact_path"])
    assert repaired == run_dir / "qa-repair-001.ass"
    assert "你好" in repaired.read_text(encoding="utf-8-sig")
    assert "Hello" in repaired.read_text(encoding="utf-8-sig")
    assert "错误" in source.read_text(encoding="utf-8")
    assert update["agent_repair_attempts"] == 1
    assert update["qa_repair_applied"] is True

    exhausted_state = state_for(source, "serial_memory")
    exhausted_state["agent_repair_attempts"] = 1
    exhausted = handlers["qa"](exhausted_state)
    assert exhausted["qa_repair_applied"] is False
    assert "artifact_path" not in exhausted


def test_serial_preprocess_rejects_wrong_artifact_type_and_malformed_ass(tmp_path) -> None:
    malformed_ass = tmp_path / "bad.ass"
    malformed_ass.write_text("not ass\n", encoding="utf-8")
    srt = tmp_path / "input.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")
    handlers = build_stage_handlers(
        WorkflowSettings(
            tmp_path / "run",
            tmp_path / "release.ass",
            batch_size=1,
            max_workers=1,
            agent_max_repair_attempts=2,
            episode_replacements=(),
        ),
        preprocess_subtitle=lambda input_path, output_path: pytest.fail(
            "serial mode preprocessed"
        ),
        translate_batch=lambda batch: (),
        refine=make_refine(),
        agent_qa=pass_agent_qa,
    )

    with pytest.raises(ValueError, match="expected a .ass"):
        handlers["preprocess"](state_for(srt, "serial_memory"))
    with pytest.raises(ValueError, match="not parseable"):
        handlers["preprocess"](state_for(malformed_ass, "serial_memory"))
