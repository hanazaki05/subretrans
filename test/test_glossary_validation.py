from types import SimpleNamespace

from subretrans.glossary_validation import (
    build_effective_glossary,
    normalize_term_key,
    order_terms_longest_first,
    term_occurs,
    validate_candidate_evidence,
    validate_learned_candidates,
)


def _cue(pair_id: int, english: str) -> SimpleNamespace:
    return SimpleNamespace(pair_id=pair_id, english=english)


def test_normalization_and_complete_token_matching() -> None:
    assert normalize_term_key("\ufeff Ｕ．Ｓ．Ｓ． Gaines\u200b ") == "u.s.s. gaines"
    assert term_occurs("Rabb", "Rabb's report is complete.")
    assert not term_occurs("Web", "Webb is here.")
    assert term_occurs("U.S.S. Gaines", "They boarded the U.S.S. Gaines.")
    assert order_terms_longest_first(("Gaines", "U.S.S. Gaines", "USS")) == (
        "U.S.S. Gaines",
        "Gaines",
        "USS",
    )


def test_effective_glossary_locks_authority_and_records_conflicts() -> None:
    effective = build_effective_glossary(
        [{"eng": "Harm", "zh": "哈姆"}],
        [
            {"eng": "harm", "zh": "哈蒙", "type": "person"},
            {"eng": "Rabb", "zh": "拉布", "type": "person"},
            {"eng": "Other", "zh": "罗伯茨", "type": "other"},
            {"eng": "Rabb", "zh": "拉布2", "type": "person"},
        ],
        [("罗伯茨", "罗伯特")],
        episode_id="S07E08",
        manifest_hash="manifest-hash",
        artifact_hash="artifact-hash",
    )

    assert [(term.eng, term.source) for term in effective.authoritative] == [
        ("Harm", "user_glossary"),
        ("罗伯茨", "episode_replacement"),
    ]
    assert [term.eng for term in effective.learned] == ["Rabb"]
    assert effective.learned[0].provenance == {
        "source": "learned",
        "episode_id": "S07E08",
        "manifest_hash": "manifest-hash",
        "artifact_hash": "artifact-hash",
    }
    reasons = {decision.reason for decision in effective.decisions}
    assert "duplicate-authoritative-key" in reasons
    assert "episode-replacement-left-value-conflict" in reasons
    assert "duplicate-learned-key" in reasons


def test_rank_phrase_and_embedded_episode_replacement_conflicts_are_rejected() -> None:
    effective = build_effective_glossary(
        [{"eng": "Commander", "zh": "中校"}],
        [
            {"eng": "Commander Turner", "zh": "特纳指挥官", "evidence_ids": [365]},
            {"eng": "Midshipman Roberts", "zh": "海军学员罗伯茨", "evidence_ids": [803]},
            {"eng": "Commander Rabb", "zh": "拉布中校", "evidence_ids": [1]},
        ],
        [("罗伯茨", "罗伯特")],
        episode_id="S07E11",
        manifest_hash="manifest",
        artifact_hash="artifact",
    )

    assert [term.eng for term in effective.learned] == ["Commander Rabb"]
    decisions = {decision.eng: decision.reason for decision in effective.decisions}
    assert decisions["Commander Turner"] == "authoritative-phrase-conflict"
    assert decisions["Midshipman Roberts"] == "episode-replacement-left-value-conflict"


def test_evidence_is_strict_bound_to_cue_text_and_errors_are_isolated() -> None:
    cues = (
        _cue(0, "They boarded the U.S.S. Gaines."),
        _cue(1, "Rabb's report is complete."),
    )
    candidate = {
        "eng": "U.S.S. Gaines",
        "zh": "盖恩斯号",
        "evidence_ids": [0, "1", 99, 0],
    }

    valid_ids, issues = validate_candidate_evidence(candidate, cues)
    assert valid_ids == (0,)
    assert {issue.code for issue in issues} == {"invalid-evidence-id", "unknown-cue", "duplicate-evidence-id"}

    authority = build_effective_glossary([], [], episode_id="S07E08", manifest_hash="m", artifact_hash="a")
    result = validate_learned_candidates(
        [
            candidate,
            {"eng": "Rabb", "zh": "拉布", "evidence_ids": [1]},
            {"eng": "NoSuchCue", "zh": "不存在", "evidence_ids": [99]},
        ],
        cues,
        authority,
        episode_id="S07E08",
        manifest_hash="m",
        artifact_hash="a",
    )

    assert [term.eng for term in result.accepted] == ["U.S.S. Gaines", "Rabb"]
    assert result.accepted[0].evidence_ids == (0,)
    assert result.accepted[0].episode_id == "S07E08"
    assert result.accepted[0].manifest_hash == "m"
    assert result.accepted[0].artifact_hash == "a"
    assert [entry.eng for entry in result.unresolved] == ["NoSuchCue"]
    assert any(issue.code == "unknown-cue" and issue.eng == "NoSuchCue" for issue in result.evidence_issues)
