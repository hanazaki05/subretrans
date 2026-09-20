import json
from types import SimpleNamespace

import pytest
import yaml

from subretrans.memory import (
    GlobalMemory,
    compress_memory,
    load_memory_checkpoint,
    normalize_term_key,
    prune_learned_glossary_against_user_glossary,
    save_memory_checkpoint,
    set_user_glossary,
    update_global_memory,
    validate_memory_structure,
)
from subretrans.pairs import SubtitlePair
from subretrans.prompts import inject_memory_into_template
from subretrans.stats import UsageStats


SETTINGS = SimpleNamespace(max_entries=100, terminology_min_confidence=0.6)


def test_memory_roundtrip_drops_legacy_style_notes() -> None:
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}],
        story_description="Harm asks Webb about the investigation.",
    )

    payload = memory.to_dict()
    assert set(payload) == {"user_glossary", "glossary", "story_description"}
    assert GlobalMemory.from_dict(payload) == memory
    assert validate_memory_structure(payload)
    assert validate_memory_structure({**payload, "style_notes": ""})
    assert GlobalMemory.from_dict({**payload, "style_notes": "legacy"}) == memory
    assert not validate_memory_structure({**payload, "summary": "legacy"})
    assert not validate_memory_structure({**payload, "glossary": [{"eng": "x"}]})
    with pytest.raises(ValueError, match="invalid memory structure"):
        GlobalMemory.from_dict({"glossary": []})


def test_checkpoint_roundtrip_is_atomic_and_complete(tmp_path) -> None:
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}],
        story_description="Harm asks Webb about the investigation.",
    )
    checkpoint = tmp_path / "episode.memory.yaml"

    save_memory_checkpoint(memory, checkpoint)

    assert load_memory_checkpoint(checkpoint) == memory
    assert load_memory_checkpoint(tmp_path / "missing.yaml") is None
    assert not list(tmp_path.glob(".*.tmp"))
    assert list(yaml.safe_load(checkpoint.read_text(encoding="utf-8"))) == [
        "user_glossary",
        "glossary",
        "story_description",
    ]


def test_normalized_keys_drive_pruning_and_user_glossary_lock() -> None:
    memory = GlobalMemory(
        glossary=[
            {"eng": "﻿ commander ", "zh": "指挥官"},
            {"eng": "SecNav", "zh": "海军部长"},
        ]
    )

    removed = set_user_glossary(memory, [{"eng": "Commander", "zh": "中校"}])

    assert normalize_term_key("﻿ Ｃommander ") == "commander"
    assert removed == 1
    assert [entry["eng"] for entry in memory.glossary] == ["SecNav"]
    assert prune_learned_glossary_against_user_glossary(memory) == (0, [])


def test_template_injection_adds_story_block_after_glossary() -> None:
    memory = GlobalMemory(
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}],
        story_description="Webb is currently questioning a witness.",
    )
    template = """Rules.

### 1. User Terminology (Authoritative Glossary)
- Harm: 哈姆

### 2. Input/Output Format & Constraint
Return JSON.
"""

    rendered = inject_memory_into_template(template, memory)

    assert "### 1. User Terminology (Authoritative Glossary)\n- Harm: 哈姆\n\n" in rendered
    assert "**Learned Terminology (Supplement):**\n- Webb (person): 韦布" in rendered
    assert "### 2. Incremental Story Description\nWebb is currently questioning a witness." in rendered
    assert "### 3. Input/Output Format & Constraint" in rendered
    with pytest.raises(ValueError, match="User Terminology"):
        inject_memory_into_template("### 1. Other\n", memory)


def test_incremental_update_merges_glossary_and_replaces_story(monkeypatch) -> None:
    calls = []

    def fake_invoke(model, messages, **kwargs):
        calls.append((model, list(messages)))
        return (
            json.dumps(
                {
                    "glossary": [
                        {"eng": "Webb", "zh": "韦布", "type": "person", "confidence": 0.95, "evidence_ids": [8]},
                        {"eng": "harm", "zh": "哈蒙", "type": "person", "confidence": 0.9, "evidence_ids": [8]},
                        {"eng": " mac ", "zh": "麦克", "type": "person", "confidence": 0.9, "evidence_ids": [8]},
                        {"eng": "Low", "zh": "低", "type": "person", "confidence": 0.2, "evidence_ids": [8]},
                    ],
                    "story_description": "Harm briefs Mac; Webb is questioning a witness.",
                }
            ),
            UsageStats(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    monkeypatch.setattr("subretrans.memory.invoke_text", fake_invoke)
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Mac", "zh": "麦可", "type": "person"}],
        story_description="Harm briefs Mac.",
    )
    pairs = [SubtitlePair(id=8, eng="Webb questions her.", chinese="韦布盘问她")]

    updated, usage = update_global_memory(memory, pairs, model="extraction", settings=SETTINGS)

    assert updated is memory
    assert usage.total_tokens == 5
    assert calls[0][0] == "extraction"
    system_prompt, user_prompt = calls[0][1][0][1], calls[0][1][1][1]
    assert "confidence >= 0.6" in system_prompt
    assert "Harm briefs Mac." in user_prompt
    assert "Webb questions her." in user_prompt
    assert [entry["eng"] for entry in updated.glossary] == ["Mac", "Webb"]
    assert updated.user_glossary == [{"eng": "Harm", "zh": "哈姆"}]
    assert updated.story_description == "Harm briefs Mac; Webb is questioning a witness."
    assert update_global_memory(memory, [], model="extraction", settings=SETTINGS) == (memory, UsageStats())


def test_incremental_update_rejects_malformed_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "subretrans.memory.invoke_text",
        lambda model, messages, **kwargs: ("```json\n{\"glossary\": []}\n```", UsageStats()),
    )
    with pytest.raises(ValueError, match="missing fields: story_description"):
        update_global_memory(
            GlobalMemory(), [SubtitlePair(1, "a", "b")], model=None, settings=SETTINGS
        )


def test_compression_keeps_user_glossary_and_sends_only_learned_state(monkeypatch) -> None:
    calls = []

    def fake_invoke(model, messages, **kwargs):
        calls.append(list(messages))
        return (
            json.dumps(
                {
                    "glossary": [
                        {"eng": "Webb", "zh": "韦布", "type": "person"},
                        {"eng": "Harm", "zh": "哈蒙", "type": "person"},
                    ],
                    "story_description": "Short story.",
                }
            ),
            UsageStats(total_tokens=1),
        )

    monkeypatch.setattr("subretrans.memory.invoke_text", fake_invoke)
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}, {"eng": "Old", "zh": "旧"}],
        story_description="A very long story.",
    )

    compressed, usage = compress_memory(memory, model=None, target_tokens=50)

    assert usage.total_tokens == 1
    assert "user_glossary" not in calls[0][1][1]
    assert compressed.user_glossary == [{"eng": "Harm", "zh": "哈姆"}]
    assert [entry["eng"] for entry in compressed.glossary] == ["Webb"]
    assert compressed.story_description == "Short story."
    assert memory.glossary[1] == {"eng": "Old", "zh": "旧"}
