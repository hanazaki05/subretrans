from subretrans.memory import TerminologyEntry, _parse_terminology_entries
from subretrans.prompts import build_memory_update_system_prompt


RAW = [
    {"eng": "Bryer", "zh": "布赖尔", "type": "person", "confidence": 0.8, "evidence_ids": [20, 20, "21", "x"]},
    {"eng": "Chris", "zh": "克里斯", "type": "person", "confidence": 0.5, "evidence_ids": [2]},
    {"eng": "Bad", "zh": "坏", "type": "unknown-type", "confidence": 0.9},
    {"eng": "", "zh": "空", "type": "person", "confidence": 0.9},
    {"eng": "NoConf", "zh": "无", "type": "person", "confidence": "high"},
]


def test_system_prompt_shows_configured_threshold() -> None:
    prompt = build_memory_update_system_prompt(0.6)

    assert "Only keep entries with confidence >= 0.6" in prompt
    assert '{"glossary": [{"eng": "..."' in prompt


def test_parsing_filters_by_threshold_type_and_shape() -> None:
    high = _parse_terminology_entries(RAW, min_confidence=0.6)
    low = _parse_terminology_entries(RAW, min_confidence=0.4)

    assert high == [TerminologyEntry("Bryer", "布赖尔", "person", 0.8, (20, 21))]
    assert [entry.eng for entry in low] == ["Bryer", "Chris"]
    assert high[0].to_dict()["evidence_ids"] == [20, 21]
    assert _parse_terminology_entries("not a list", min_confidence=0.6) == []
