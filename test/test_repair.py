import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import subretrans.repair as repair_module
from subretrans.config import RepairSettings
from subretrans.model_agent import AgentQAEvidence, AgentQASuggestion
from subretrans.pairs import SubtitlePair
from subretrans.providers import ToolAction, ToolExchange, ToolLoopResult
from subretrans.repair import RepairSession, host_qa_suggestions, run_repair_agent
from subretrans.reference_reader import ReferenceReader
from subretrans.stats import UsageStats


def _settings() -> RepairSettings:
    return RepairSettings(24, 1, 2, 1, 3, 2)


def _pairs() -> tuple[SubtitlePair, ...]:
    return (
        SubtitlePair(0, "First", "第一", {"start": "0:00:01.00", "end": "0:00:02.00"}),
        SubtitlePair(1, "Second", "第二", {"start": "0:00:02.00", "end": "0:00:03.00"}),
        SubtitlePair(2, "Third", "第三", {"start": "0:00:03.00", "end": "0:00:04.00"}),
    )


def _hosted():
    return host_qa_suggestions(
        (
            AgentQASuggestion(
                (1,),
                "meaning",
                "The second cue is inaccurate.",
                (AgentQAEvidence((1,), "Second does not mean 第二。"),),
            ),
        ),
        source_window=(0, 2),
        source_pass=0,
        artifact_hash="artifact",
        effective_glossary_hash="glossary",
        manifest_hash="manifest",
    )


def _session(tmp_path: Path, *, suggestions=None) -> RepairSession:
    return RepairSession(
        state_dir=tmp_path / "repair",
        settings=_settings(),
        refined_pairs=_pairs(),
        current_pairs=_pairs(),
        suggestions=_hosted() if suggestions is None else suggestions,
        operations=("normalize_punctuation", "episode_replacements"),
        episode_replacements=(("罗伯茨", "罗伯特"),),
        manifest_hash="manifest",
        effective_glossary={"authoritative": [], "learned": []},
    )


def test_host_generates_stable_issue_id_and_key() -> None:
    assert _hosted() == _hosted()
    assert _hosted()[0].issue_id.startswith("issue-")
    assert len(_hosted()[0].key) == 64


def test_finish_requires_ledger_backed_full_sweep_even_without_qa(tmp_path: Path) -> None:
    session = _session(tmp_path, suggestions=())
    with pytest.raises(ValueError, match="full-episode sweep"):
        session.execute("finish", {})
    with pytest.raises(ValueError, match="provided first"):
        session.execute("inspect_context", {"start_id": 0, "end_id": 2, "completed": True})

    session.execute("inspect_context", {"start_id": 0, "end_id": 2, "completed": False})
    session.execute("inspect_context", {"start_id": 0, "end_id": 2, "completed": True})
    assert session.execute("finish", {})["status"] == "finish"
    assert (tmp_path / "repair" / "coverage.json").exists()
    assert (tmp_path / "repair" / "repair-state.json").exists()


def test_group_is_atomic_and_episode_replacement_conflict_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    before_hash = session.current_hash
    issue_id = _hosted()[0].issue_id

    result = session.execute(
        "stage_group_repair",
        {
            "group_id": "group-1",
            "base_artifact_hash": before_hash,
            "affected_ids": [1],
            "translations": [{"id": 1, "translation": "罗伯茨"}],
            "issue_ids": [issue_id],
            "reason": "correct the name",
        },
    )

    assert result["status"] == "rejected"
    assert "episode replacements" in result["reason"]
    assert session.current_hash == before_hash
    assert session.current[1].chinese == "第二"
    assert session.issue_states[issue_id] == "open"


def test_group_applies_complete_consecutive_cues_and_invalidates_coverage(tmp_path: Path) -> None:
    session = _session(tmp_path)
    issue_id = _hosted()[0].issue_id
    session.execute("inspect_context", {"start_id": 0, "end_id": 2, "completed": False})
    session.execute("inspect_context", {"start_id": 0, "end_id": 2, "completed": True})
    result = session.execute(
        "stage_group_repair",
        {
            "group_id": "group-2",
            "base_artifact_hash": session.current_hash,
            "affected_ids": [1],
            "translations": [{"id": 1, "translation": "第二句……"}],
            "issue_ids": [issue_id],
            "reason": "restore the complete meaning",
        },
    )

    assert result["status"] == "applied"
    assert session.current[1].chinese == "第二句..."
    assert session.issue_states[issue_id] == "resolved"
    assert session.covered_ids == set()
    with pytest.raises(ValueError, match="complete coverage"):
        session.execute("finish", {})


def test_resume_preserves_cumulative_budgets_and_hashes(tmp_path: Path) -> None:
    session = _session(tmp_path, suggestions=())
    session.tool_steps_used = 7
    session.repair_attempts_used = 1
    session._commit()

    resumed = RepairSession.resume(
        state_dir=tmp_path / "repair",
        settings=_settings(),
        operations=("normalize_punctuation", "episode_replacements"),
        episode_replacements=(("罗伯茨", "罗伯特"),),
        manifest_hash="manifest",
    )

    assert resumed.tool_steps_used == 7
    assert resumed.repair_attempts_used == 1
    assert resumed.current_hash == session.current_hash


def test_group_rejects_authoritative_glossary_violation(tmp_path: Path) -> None:
    pairs = (
        SubtitlePair(
            0,
            "Commander Turner reports.",
            "特纳中校报告。",
            {"start": "0:00:01.00", "end": "0:00:02.00"},
        ),
    )
    suggestion = host_qa_suggestions(
        (
            AgentQASuggestion(
                (0,),
                "terminology",
                "Check the rank.",
                (AgentQAEvidence((0,), "Commander is present."),),
            ),
        ),
        source_window=(0, 1),
        source_pass=1,
        artifact_hash="artifact",
        effective_glossary_hash="glossary",
        manifest_hash="manifest",
    )
    session = RepairSession(
        state_dir=tmp_path / "repair",
        settings=_settings(),
        refined_pairs=pairs,
        current_pairs=pairs,
        suggestions=suggestion,
        operations=("episode_replacements",),
        episode_replacements=(),
        manifest_hash="manifest",
        effective_glossary={
            "authoritative": [{"eng": "Commander", "zh": "中校"}],
            "learned": [],
        },
    )

    result = session.execute(
        "stage_group_repair",
        {
            "group_id": "wrong-rank",
            "base_artifact_hash": session.current_hash,
            "affected_ids": [0],
            "translations": [{"id": 0, "translation": "特纳指挥官报告。"}],
            "issue_ids": [suggestion[0].issue_id],
            "reason": "model preferred a literal title",
        },
    )

    assert result["status"] == "rejected"
    assert "authoritative:Commander->中校" in result["reason"]
    assert session.current[0].chinese == "特纳中校报告。"


def test_group_glossary_gate_prefers_longest_overlapping_authority(tmp_path: Path) -> None:
    pairs = (
        SubtitlePair(
            0,
            "Lieutenant Commander Rabb reports.",
            "拉布少校报告。",
            {"start": "0:00:01.00", "end": "0:00:02.00"},
        ),
    )
    session = RepairSession(
        state_dir=tmp_path / "repair",
        settings=_settings(),
        refined_pairs=pairs,
        current_pairs=pairs,
        suggestions=(),
        operations=(),
        episode_replacements=(),
        manifest_hash="manifest",
        effective_glossary={
            "authoritative": [
                {"eng": "Lieutenant Commander", "zh": "少校"},
                {"eng": "Commander", "zh": "中校"},
            ],
            "learned": [],
        },
    )

    assert session._glossary_violations((0,), session.current) == ()


def test_group_rejects_nonconsecutive_or_partial_translations(tmp_path: Path) -> None:
    session = _session(tmp_path)
    issue_id = _hosted()[0].issue_id
    with pytest.raises(ValueError, match="consecutive"):
        session.execute(
            "stage_group_repair",
            {
                "group_id": "bad",
                "base_artifact_hash": session.current_hash,
                "affected_ids": [0, 2],
                "translations": [
                    {"id": 0, "translation": "一"},
                    {"id": 2, "translation": "三"},
                ],
                "issue_ids": [issue_id],
                "reason": "bad group",
            },
        )


def test_reference_and_webfetch_tools_are_advertised_and_dispatched(tmp_path: Path) -> None:
    root = tmp_path / "references"
    root.mkdir()
    (root / "one.srt").write_text("1\nCommander Turner\n", encoding="utf-8")
    (root / "two.srt").write_text("1\n特纳中校\n", encoding="utf-8")
    fetched: list[str] = []

    def webfetch(url: str):
        fetched.append(url)
        return {"final_url": url, "body_sha256": "a" * 64}

    session = RepairSession(
        state_dir=tmp_path / "repair-tools",
        settings=_settings(),
        refined_pairs=_pairs(),
        current_pairs=_pairs(),
        suggestions=(),
        operations=("episode_replacements",),
        episode_replacements=(),
        manifest_hash="manifest",
        effective_glossary={"authoritative": [], "learned": []},
        reference_reader=ReferenceReader(root),
        webfetch_executor=webfetch,
    )

    names = {tool.name for tool in session.tool_definitions()}
    assert {
        "list_resources",
        "search_subtitles",
        "read_subtitle_context",
        "compare",
        "webfetch",
    } <= names
    assert len(session.execute("list_resources", {"limit": 10})) == 2
    matches = session.execute(
        "search_subtitles",
        {"query": "Turner", "paths": [], "max_results": 5},
    )
    assert matches[0]["relative_path"] == "one.srt"
    context = session.execute(
        "read_subtitle_context", {"path": "one.srt", "line": 2, "radius": 1}
    )
    assert "Commander Turner" in context["lines"]
    comparison = session.execute(
        "compare",
        {"left": "one.srt", "right": "two.srt", "max_diff_lines": 20},
    )
    assert comparison["diff"]
    assert session.execute("webfetch", {"url": "https://example.com"})["final_url"] == "https://example.com"
    assert fetched == ["https://example.com"]


def test_begin_round_preserves_budget_and_does_not_reopen_same_dismissed_key(tmp_path: Path) -> None:
    session = _session(tmp_path)
    first = _hosted()[0]
    session.execute(
        "dismiss_issue", {"issue_id": first.issue_id, "reason": "confirmed false positive"}
    )
    session.tool_steps_used = 5
    session.repair_attempts_used = 1
    session.force_escalation("round complete")
    repeated = host_qa_suggestions(
        (
            AgentQASuggestion(
                (1,),
                "meaning",
                "The second cue is inaccurate.",
                (AgentQAEvidence((1,), "Second does not mean 第二。"),),
            ),
        ),
        source_window=(0, 2),
        source_pass=2,
        artifact_hash="artifact",
        effective_glossary_hash="glossary",
        manifest_hash="manifest",
    )

    result = session.begin_round(current_pairs=_pairs(), suggestions=repeated)

    assert result["linked_prior_decisions"] == 1
    assert session.issue_states[repeated[0].issue_id] == "dismissed"
    assert session.tool_steps_used == 5
    assert session.repair_attempts_used == 1
    assert session.terminal_status is None


def test_incomplete_generation_does_not_replace_last_complete_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    session = _session(tmp_path, suggestions=())
    original_pointer = (session.state_dir / "repair-state.json").read_bytes()
    original_write = repair_module.atomic_write_json

    def interrupted(path, payload):
        path = Path(path)
        if "generations" in path.parts and path.name == "history.json":
            raise OSError("simulated interruption")
        return original_write(path, payload)

    monkeypatch.setattr(repair_module, "atomic_write_json", interrupted)
    session.tool_steps_used = 9
    with pytest.raises(OSError, match="simulated interruption"):
        session._commit()
    assert (session.state_dir / "repair-state.json").read_bytes() == original_pointer

    resumed = RepairSession.resume(
        state_dir=session.state_dir,
        settings=_settings(),
        operations=("normalize_punctuation", "episode_replacements"),
        episode_replacements=(("罗伯茨", "罗伯特"),),
        manifest_hash="manifest",
    )
    assert resumed.tool_steps_used == 0


def test_tool_exchanges_are_persisted_and_hashed(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path, suggestions=())
    exchange = ToolExchange(
        1,
        (("system", "repair"),),
        '{"action":"final","result":{"status":"done"}}',
        ToolAction("final", result={"status": "done"}),
        None,
        None,
    )
    loop_result = ToolLoopResult(
        {"status": "done"}, (exchange,), UsageStats(), 1
    )

    def fake_invoke(model, messages, tools, execute, *, max_tool_steps, on_exchange):
        on_exchange(exchange)
        return loop_result

    monkeypatch.setattr(repair_module, "build_chat_model", lambda config: object())
    monkeypatch.setattr(repair_module, "invoke_with_tools", fake_invoke)
    model_settings = Mock()
    model_settings.config = Mock()
    run_repair_agent(
        model_settings=model_settings,
        system_prompt="repair",
        session=session,
        effective_glossary={"authoritative": [], "learned": []},
        episode_memory={},
    )

    exchanges_path = session.latest_artifact_path("exchanges.json")
    payload = json.loads(exchanges_path.read_text(encoding="utf-8"))
    assert payload["exchanges"][0]["response"] == exchange.response
    pointer = json.loads(
        (session.state_dir / "repair-state.json").read_text(encoding="utf-8")
    )
    assert "exchanges.json" in pointer["artifacts"]
