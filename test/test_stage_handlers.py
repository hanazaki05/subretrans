import json
from pathlib import Path

from subretrans.fsutil import atomic_copy, atomic_write_json, sha256_file
from subretrans.model_agent import (
    AgentQAEvidence,
    AgentQAMemory,
    AgentQAResult,
    AgentQASuggestion,
    AgentQATranslation,
)
from subretrans.run_manifest import artifact_path, create_manifest, load_manifest
from subretrans.stage_handlers import (
    WorkflowSettings,
    _default_freeze_effective_glossary,
    build_stage_handlers,
)
from subretrans.cue_manifest import build_cue_manifest, save_cue_manifest
from subretrans.subtitle_processing import SrtCue, write_srt
from subretrans.translation import TranslationResult


VALID_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,错误
"""


def setup_run(tmp_path: Path):
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    state = create_manifest(
        tmp_path / "run.json",
        run_id="episode",
        translation_mode="serial_memory",
        source_path=source,
        config_path=config,
        prompt_paths={name: config for name in ("shared", "refine", "qa", "repair")},
        release_path=tmp_path / "release.ass",
        budget_limits={
            "repair_attempts": 1,
            "tool_steps": 4,
            "full_sweeps": 1,
            "glossary_repairs": 0,
            "research_requests": 0,
        },
    )

    def refine(input_path, output_path, memory_path, progress_path):
        atomic_copy(input_path, output_path)
        memory_path.write_text(
            "user_glossary: []\nglossary: []\nstory_description: Context.\n",
            encoding="utf-8",
        )
        atomic_write_json(
            progress_path,
            {
                "version": 1,
                "next_pair": 1,
                "artifact_path": str(output_path),
                "artifact_hash": sha256_file(output_path),
                "memory_checkpoint_path": str(memory_path),
                "memory_hash": sha256_file(memory_path),
            },
        )

    def freeze_cues(input_path, output_path):
        atomic_write_json(output_path, {"version": 1, "source": sha256_file(input_path)})

    def freeze_glossary(**kwargs):
        atomic_write_json(
            kwargs["output_path"],
            {
                "version": 1,
                "authoritative": [{"eng": "Commander", "zh": "指挥官"}],
                "learned": [
                    {
                        "eng": "Turner",
                        "zh": "特纳",
                        "confidence": 0.9,
                        "evidence_ids": [0],
                    }
                ],
            },
        )
        atomic_write_json(kwargs["decisions_path"], {"version": 1, "decisions": []})

    return state, refine, freeze_cues, freeze_glossary


def make_handlers(
    tmp_path: Path,
    *,
    qa_result: AgentQAResult,
    repair_calls: list[Path],
    qa_memories: list[AgentQAMemory] | None = None,
    qa_histories: list[tuple] | None = None,
    dismiss_suggestions: bool = False,
    max_tool_steps: int = 4,
):
    state, refine, freeze_cues, freeze_glossary = setup_run(tmp_path)

    def repair_runner(**kwargs):
        repair_calls.append(Path(kwargs["current_artifact_path"]))
        output = tmp_path / "repair" / "candidate.ass"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_copy(kwargs["current_artifact_path"], output)
        pool = json.loads(Path(kwargs["suggestion_pool_path"]).read_text(encoding="utf-8"))
        decisions = []
        if dismiss_suggestions:
            decisions = [
                {
                    "issue_id": entry["issue_id"],
                    "issue_key": entry["key"],
                    "status": "dismissed",
                    "reason": "verified false positive",
                }
                for entry in pool["suggestions"]
            ]
        atomic_write_json(
            kwargs["decision_log_path"], {"version": 1, "decisions": decisions}
        )
        atomic_write_json(
            kwargs["coverage_path"], {"version": 1, "full_sweeps_completed": 1}
        )
        repair_state = tmp_path / "repair" / "repair-state.json"
        history = tmp_path / "repair" / "history.json"
        staged = tmp_path / "repair" / "staged.json"
        exchanges = tmp_path / "repair" / "exchanges.json"
        for path, payload in (
            (repair_state, {"version": 1, "generation": 1}),
            (history, {"version": 1, "history": []}),
            (staged, {"version": 1, "groups": []}),
            (exchanges, {"version": 1, "exchanges": []}),
        ):
            atomic_write_json(path, payload)
        return {
            "artifact_path": output,
            "repair_state_path": repair_state,
            "history_path": history,
            "staged_path": staged,
            "exchanges_path": exchanges,
            "tool_steps_used": 1,
            "full_sweeps_used": 1,
            "escalated": False,
        }

    handlers = build_stage_handlers(
        WorkflowSettings(
            run_dir=tmp_path,
            primer_batch_size=1,
            primer_max_workers=1,
            episode_replacements=(),
            qa_batch_size=10,
            max_repair_attempts=1,
            max_tool_steps=max_tool_steps,
            max_full_sweeps=1,
        ),
        preprocess_subtitle=lambda input_path, output_path: output_path,
        translate_batch=lambda batch: (),
        refine=refine,
        agent_qa=lambda pairs, structural, history, memory: (
            qa_memories.append(memory) if qa_memories is not None else None
        )
        or (qa_histories.append(history) if qa_histories is not None else None)
        or qa_result,
        freeze_cues=freeze_cues,
        freeze_glossary=freeze_glossary,
        repair_runner=repair_runner,
    )
    return state, handlers


def run_to_qa(state, handlers):
    for stage in (
        "preprocess",
        "freeze_manifest",
        "refine_serial",
        "postprocess",
        "glossary",
        "qa",
    ):
        state = handlers[stage](state)
    return state


def test_default_glossary_freeze_escalates_unavailable_rank_research(tmp_path: Path) -> None:
    artifact = tmp_path / "rank.ass"
    artifact.write_text(
        VALID_ASS.replace("Hello", "Commander Harmon Rabb reporting"), encoding="utf-8"
    )
    cue_path = tmp_path / "cue-manifest.json"
    save_cue_manifest(build_cue_manifest(artifact, episode_id="JAG.S07E07"), cue_path)
    memory = tmp_path / "memory.yaml"
    memory.write_text(
        """user_glossary:
  - eng: Commander
    zh: 中校
glossary:
  - eng: Commander Harmon Rabb
    zh: 哈蒙·拉布中校
    type: title
    confidence: 0.9
    evidence_ids: [0]
story_description: Episode context.
""",
        encoding="utf-8",
    )
    output = tmp_path / "effective.json"
    decisions = tmp_path / "decisions.json"
    research = tmp_path / "research.json"
    repairs = tmp_path / "repairs.json"

    result = _default_freeze_effective_glossary(
        memory_path=memory,
        cue_manifest_path=cue_path,
        artifact_path=artifact,
        output_path=output,
        decisions_path=decisions,
        research_path=research,
        repair_path=repairs,
        episode_replacements=(),
        research_runner=None,
        glossary_repair_runner=None,
        max_research_requests=4,
        max_glossary_repairs=1,
    )

    effective = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads(research.read_text(encoding="utf-8"))
    assert effective["learned"] == []
    assert report["human_review_required"] is True
    assert report["decisions"][0]["passes"][0]["error"] == "research_unavailable"
    assert result["research_requests_used"] == 0


def test_invalid_rank_evidence_cannot_bypass_research_through_glossary_repair(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rank.ass"
    artifact.write_text(
        VALID_ASS.replace("Hello", "Chaplain Turner reports"), encoding="utf-8"
    )
    cue_path = tmp_path / "cue-manifest.json"
    save_cue_manifest(build_cue_manifest(artifact, episode_id="JAG.S07E11"), cue_path)
    memory = tmp_path / "memory.yaml"
    memory.write_text(
        """user_glossary:
  - eng: Commander
    zh: 中校
glossary:
  - eng: Commander Turner
    zh: 特纳指挥官
    type: title
    confidence: 0.9
    evidence_ids: [0]
story_description: Episode context.
""",
        encoding="utf-8",
    )
    repair_called = False

    def repair_runner(**kwargs):
        nonlocal repair_called
        repair_called = True
        return {"attempts_used": 1, "actions": []}

    output = tmp_path / "effective.json"
    _default_freeze_effective_glossary(
        memory_path=memory,
        cue_manifest_path=cue_path,
        artifact_path=artifact,
        output_path=output,
        decisions_path=tmp_path / "decisions.json",
        research_path=tmp_path / "research.json",
        repair_path=tmp_path / "repairs.json",
        episode_replacements=(),
        research_runner=None,
        glossary_repair_runner=repair_runner,
        max_research_requests=4,
        max_glossary_repairs=1,
    )

    effective = json.loads(output.read_text(encoding="utf-8"))
    assert effective["learned"] == []
    assert repair_called is False
    assert any(
        entry["reason"] == "research-insufficient"
        for entry in effective["unresolved"]
    )


def test_glossary_repair_correction_is_revalidated_before_freeze(tmp_path: Path) -> None:
    artifact = tmp_path / "term.ass"
    artifact.write_text(
        VALID_ASS.replace("Hello", "Naval Aviator Harmon Rabb reports"), encoding="utf-8"
    )
    cue_path = tmp_path / "cue-manifest.json"
    save_cue_manifest(build_cue_manifest(artifact, episode_id="episode"), cue_path)
    memory = tmp_path / "memory.yaml"
    memory.write_text(
        """user_glossary: []
glossary:
  - eng: Naval Aviator
    zh: 海军飞行员
    type: title
    confidence: 0.9
    evidence_ids: [99]
story_description: Context.
""",
        encoding="utf-8",
    )
    output = tmp_path / "effective.json"

    result = _default_freeze_effective_glossary(
        memory_path=memory,
        cue_manifest_path=cue_path,
        artifact_path=artifact,
        output_path=output,
        decisions_path=tmp_path / "decisions.json",
        research_path=tmp_path / "research.json",
        repair_path=tmp_path / "repairs.json",
        episode_replacements=(),
        research_runner=None,
        glossary_repair_runner=lambda **kwargs: {
            "attempts_used": 1,
            "actions": [
                {
                    "action": "correct",
                    "eng": "Naval Aviator",
                    "reason": "repair evidence binding",
                    "candidate": {
                        "eng": "Naval Aviator",
                        "zh": "海军飞行员",
                        "type": "title",
                        "confidence": 0.9,
                        "evidence_ids": [0],
                    },
                }
            ],
        },
        max_research_requests=0,
        max_glossary_repairs=2,
    )

    effective = json.loads(output.read_text(encoding="utf-8"))
    assert effective["learned"][0]["eng"] == "Naval Aviator"
    assert effective["learned"][0]["evidence_ids"] == [0]
    assert result["glossary_repairs_used"] == 1


def test_qa_persists_suggestions_without_modifying_artifact(tmp_path: Path) -> None:
    suggestion = AgentQASuggestion(
        affected_ids=(0,),
        kind="accuracy",
        diagnosis="Meaning is reversed.",
        evidence=(AgentQAEvidence((0,), "English and Chinese disagree."),),
        suggested_translations=(AgentQATranslation(0, "你好"),),
    )
    state, handlers = make_handlers(
        tmp_path, qa_result=AgentQAResult(False, (suggestion,)), repair_calls=[]
    )
    state = run_to_qa(state, handlers)
    manifest = load_manifest(state["manifest_path"])

    current = artifact_path(state["manifest_path"], manifest, "current")
    pool = artifact_path(state["manifest_path"], manifest, "suggestion_pool")
    assert "错误" in current.read_text(encoding="utf-8-sig")
    assert json.loads(pool.read_text(encoding="utf-8"))["suggestions"][0]["affected_ids"] == [0]
    assert state["next_stage"] == "repair"


def test_parallel_translation_keeps_committed_seed_immutable(tmp_path: Path) -> None:
    source = tmp_path / "input.mkv"
    source.write_text("container", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")
    state = create_manifest(
        tmp_path / "run.json",
        run_id="parallel",
        translation_mode="parallel_initial",
        source_path=source,
        config_path=config,
        prompt_paths={name: config for name in ("shared", "refine", "qa", "repair")},
        release_path=tmp_path / "release.ass",
        budget_limits={},
    )

    def preprocess(input_path, output_path):
        write_srt((SrtCue(1, "00:00:01,000", "00:00:02,000", "Hello"),), output_path)
        return output_path

    handlers = build_stage_handlers(
        WorkflowSettings(tmp_path, 1, 1, ()),
        preprocess_subtitle=preprocess,
        translate_batch=lambda batch: tuple(
            TranslationResult(request.id, "你好") for request in batch
        ),
        refine=lambda *args: None,
        agent_qa=lambda *args: AgentQAResult(True, ()),
        repair_runner=lambda **kwargs: {},
    )
    state = handlers["preprocess"](state)
    before = load_manifest(state["manifest_path"])
    seed_ref = before["artifacts"]["translation_seed"]

    state = handlers["translate_parallel"](state)
    after = load_manifest(state["manifest_path"])

    assert after["artifacts"]["translation_seed"] == seed_ref
    assert after["heads"]["translation_manifest"] == "translation_manifest"


def test_repair_runs_even_when_qa_pool_is_empty(tmp_path: Path) -> None:
    calls: list[Path] = []
    state, handlers = make_handlers(
        tmp_path, qa_result=AgentQAResult(True, ()), repair_calls=calls
    )
    state = run_to_qa(state, handlers)
    state = handlers["repair"](state)

    assert len(calls) == 1
    assert state["next_stage"] == "qa_verify"
    manifest = load_manifest(state["manifest_path"])
    assert manifest["budgets"]["repair_attempts"]["used"] == 1
    assert manifest["heads"]["current"] == "postprocessed"
    for head in (
        "repair_state",
        "decision_log",
        "repair_history",
        "repair_staged",
        "repair_exchanges",
        "coverage",
    ):
        assert manifest["heads"][head] is not None


def test_qa_verify_routes_to_review_when_tool_budget_is_exhausted(tmp_path: Path) -> None:
    suggestion = AgentQASuggestion(
        affected_ids=(0,),
        kind="accuracy",
        diagnosis="Meaning is reversed.",
        evidence=(AgentQAEvidence((0,), "English and Chinese disagree."),),
        suggested_translations=(AgentQATranslation(0, "你好"),),
    )
    state, handlers = make_handlers(
        tmp_path,
        qa_result=AgentQAResult(False, (suggestion,)),
        repair_calls=[],
        max_tool_steps=1,
    )

    state = run_to_qa(state, handlers)
    state = handlers["repair"](state)
    state = handlers["qa_verify"](state)

    assert state["next_stage"] == "review_export"
    assert state["route_reason"] == "repair_budget_exhausted"


def test_qa_receives_only_the_frozen_effective_glossary(tmp_path: Path) -> None:
    memories: list[AgentQAMemory] = []
    state, handlers = make_handlers(
        tmp_path,
        qa_result=AgentQAResult(True, ()),
        repair_calls=[],
        qa_memories=memories,
    )

    run_to_qa(state, handlers)

    assert memories[0].user_glossary[0].eng == "Commander"
    assert memories[0].user_glossary[0].zh == "指挥官"
    assert memories[0].glossary[0].eng == "Turner"
    assert memories[0].glossary[0].zh == "特纳"


def test_qa_verify_uses_prior_decisions_and_suppresses_dismissed_repeat(
    tmp_path: Path,
) -> None:
    suggestion = AgentQASuggestion(
        affected_ids=(0,),
        kind="accuracy",
        diagnosis="Meaning is reversed.",
        evidence=(AgentQAEvidence((0,), "English and Chinese disagree."),),
        suggested_translations=(AgentQATranslation(0, "你好"),),
    )
    histories: list[tuple] = []
    state, handlers = make_handlers(
        tmp_path,
        qa_result=AgentQAResult(False, (suggestion,)),
        repair_calls=[],
        qa_histories=histories,
        dismiss_suggestions=True,
    )

    state = run_to_qa(state, handlers)
    state = handlers["repair"](state)
    state = handlers["qa_verify"](state)
    manifest = load_manifest(state["manifest_path"])
    pool = artifact_path(state["manifest_path"], manifest, "suggestion_pool")

    assert histories[0] == ()
    assert histories[1][0].status == "dismissed"
    assert json.loads(pool.read_text(encoding="utf-8"))["suggestions"] == []
    assert state["next_stage"] == "review_export"


def test_review_is_run_local_and_approval_freezes_edited_copy(tmp_path: Path) -> None:
    calls: list[Path] = []
    state, handlers = make_handlers(
        tmp_path, qa_result=AgentQAResult(True, ()), repair_calls=calls
    )
    state = run_to_qa(state, handlers)
    state = handlers["repair"](state)
    state = handlers["qa_verify"](state)
    state = handlers["review_export"](state)
    manifest = load_manifest(state["manifest_path"])
    review = Path(manifest["review"]["path"])
    assert review.is_relative_to(tmp_path)
    review.write_text(review.read_text(encoding="utf-8-sig").replace("错误", "人工修改"), encoding="utf-8-sig")

    state = handlers["human_review"](state, "approve")
    manifest = load_manifest(state["manifest_path"])
    approved = artifact_path(state["manifest_path"], manifest, "approved")
    assert "人工修改" in approved.read_text(encoding="utf-8-sig")
    assert approved != review
