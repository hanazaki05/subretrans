import json
from pathlib import Path

import pytest

from subretrans.fsutil import sha256_file
from subretrans.unresolved_report import (
    build_report_source,
    parse_agent_report,
    render_markdown,
    report_path_for_review,
)


ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,你好
Dialogue: -1,0:00:03.00,0:00:04.00,English3,,0,0,0,,Commander Test
Dialogue:  1,0:00:03.00,0:00:04.00,Chinese3,,0,0,0,,测试指挥官
"""


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _source(tmp_path: Path):
    current = tmp_path / "current.ass"
    review = tmp_path / "review" / "episode.review.ass"
    current.write_text(ASS, encoding="utf-8")
    review.parent.mkdir()
    review.write_text(ASS, encoding="utf-8")
    suggestion = {
        "issue_id": "issue-1",
        "affected_ids": [0],
        "kind": "meaning",
        "diagnosis": "可能误译",
        "evidence": [],
        "suggested_translations": [{"id": 0, "translation": "您好"}],
    }
    generation = tmp_path / "repair" / "generations" / "0001"
    suggestions = _write_json(generation / "suggestions.json", {"suggestions": [suggestion]})
    state = _write_json(
        tmp_path / "repair" / "repair-state-001.json",
        {
            "generation_dir": "generations/0001",
            "artifacts": {"suggestions.json": sha256_file(suggestions)},
        },
    )
    decisions = _write_json(
        tmp_path / "repair" / "decisions-001.json",
        {"decisions": [{"issue_id": "issue-1", "status": "escalated", "reason": "uncertain"}]},
    )
    coverage = _write_json(
        tmp_path / "repair" / "coverage-001.json",
        {
            "full_sweeps_completed": 1,
            "coverage": [{"completed": True, "covered_ids": [0]}],
        },
    )
    glossary = _write_json(
        tmp_path / "glossary-decisions.json",
        {
            "unresolved": [
                {
                    "eng": "Commander Test",
                    "zh": "",
                    "reason": "insufficient evidence",
                    "source": "learned",
                    "evidence_ids": [1],
                }
            ]
        },
    )
    pool = _write_json(tmp_path / "suggestion-pool.json", {"suggestions": [suggestion]})
    return build_report_source(
        decision_log_path=decisions,
        repair_state_path=state,
        coverage_path=coverage,
        glossary_decisions_path=glossary,
        suggestion_pool_path=pool,
        current_artifact_path=current,
        review_artifact_path=review,
    )


def test_builds_trusted_unresolved_source_and_coverage_gap(tmp_path: Path) -> None:
    source = _source(tmp_path)

    assert [item["issue_id"] for item in source["unresolved_issues"]] == ["issue-1"]
    assert source["unresolved_issues"][0]["cues"][0]["english"] == "Hello"
    assert [item["eng"] for item in source["unresolved_glossary"]] == ["Commander Test"]
    assert source["coverage"] == {
        "covered": 1,
        "total": 2,
        "missing_ids": [1],
        "full_sweeps_completed": 1,
    }


def test_agent_report_requires_exact_issue_and_term_sets() -> None:
    valid = {
        "coverage_note": "第二条未覆盖。",
        "glossary_items": [
            {
                "eng": "Commander Test",
                "priority": "medium",
                "summary": "军衔待确认",
                "review_action": "核对剧集语境",
            }
        ],
        "groups": [
            {
                "issue_ids": ["issue-1", "issue-2"],
                "priority": "high",
                "title": "同一译义问题",
                "summary": "译义待确认",
                "review_action": "听辨后决定",
            }
        ],
    }
    issue_cues = {"issue-1": {0}, "issue-2": {0, 1}}
    assert parse_agent_report(valid, issue_cues, {"Commander Test"})["groups"][0][
        "issue_ids"
    ] == ["issue-1", "issue-2"]
    invalid = {**valid, "groups": []}
    with pytest.raises(ValueError, match="issue ids differ"):
        parse_agent_report(invalid, issue_cues, {"Commander Test"})


def test_agent_report_rejects_duplicate_or_disconnected_grouping() -> None:
    base = {
        "coverage_note": "仍需人工复核。",
        "glossary_items": [],
        "groups": [
            {
                "issue_ids": ["issue-1", "issue-2"],
                "priority": "high",
                "title": "问题",
                "summary": "摘要",
                "review_action": "复核",
            }
        ],
    }
    with pytest.raises(ValueError, match="without overlapping cues"):
        parse_agent_report(base, {"issue-1": {0}, "issue-2": {2}}, set())
    duplicate = {
        **base,
        "groups": [
            base["groups"][0],
            {
                **base["groups"][0],
                "issue_ids": ["issue-2"],
            },
        ],
    }
    with pytest.raises(ValueError, match="duplicate issue ids"):
        parse_agent_report(duplicate, {"issue-1": {0}, "issue-2": {0}}, set())


def test_renders_report_beside_review_with_unconfirmed_warning(tmp_path: Path) -> None:
    source = _source(tmp_path)
    guidance = {
        "coverage_note": "cue 1 未完成有效复核。",
        "glossary_items": [
            {
                "eng": "Commander Test",
                "priority": "medium",
                "summary": "军衔待确认",
                "review_action": "核对剧集语境",
            }
        ],
        "groups": [
            {
                "issue_ids": ["issue-1"],
                "priority": "high",
                "title": "译义问题",
                "summary": "译义待确认",
                "review_action": "听辨后决定",
            }
        ],
    }

    rendered = render_markdown(source, guidance)

    assert "不代表已经确认的翻译错误" in rendered
    assert "原始未决 issue：1" in rendered
    assert "合并后审核项：1" in rendered
    assert "未覆盖 cue：1（1）" in rendered
    assert "Commander Test" in rendered
    assert "`0` EN: Hello" in rendered
    review = tmp_path / "episode.review.ass"
    assert report_path_for_review(review).name == "episode.review.unresolved.md"
