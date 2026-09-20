import json
from types import SimpleNamespace

from subretrans.memory import (
    GlobalMemory,
    compress_memory_simple,
    update_global_memory,
    validate_memory_structure,
)
from subretrans.cli import load_memory_checkpoint, save_memory_checkpoint
from subretrans.pairs import SubtitlePair
from subretrans.prompts import build_system_prompt_legacy, inject_memory_into_template


def test_global_memory_roundtrip_and_compression_preserve_story_and_user_glossary() -> None:
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}],
        style_notes="Keep dialogue concise.",
        story_description="Harm asks Webb about the investigation.",
    )

    payload = json.loads(memory.to_json())
    restored = GlobalMemory.from_dict(payload)

    assert restored == memory
    assert "summary" not in payload
    assert not hasattr(restored, "summary")
    assert validate_memory_structure(payload)
    assert not validate_memory_structure({**payload, "summary": "legacy"})

    compressed = compress_memory_simple(memory, max_entries=1)
    assert compressed.user_glossary == memory.user_glossary
    assert compressed.story_description == memory.story_description


def test_checkpoint_roundtrip_persists_complete_episode_memory(tmp_path) -> None:
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Webb", "zh": "韦布", "type": "person"}],
        style_notes="Keep dialogue concise.",
        story_description="Harm asks Webb about the investigation.",
    )
    checkpoint = tmp_path / "episode.ass.memory.yaml"

    save_memory_checkpoint(memory, str(checkpoint))

    assert load_memory_checkpoint(str(checkpoint)) == memory


def test_legacy_and_template_prompts_inject_independent_story_block() -> None:
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

    legacy = build_system_prompt_legacy(memory)
    rendered = inject_memory_into_template(template, memory)

    assert "**Incremental Story Description:**" in legacy
    assert memory.story_description in legacy
    assert "### 2. Incremental Story Description" in rendered
    assert memory.story_description in rendered
    assert rendered.index("Incremental Story Description") < rendered.index(
        "Input/Output Format & Constraint"
    )


def test_incremental_update_merges_glossary_and_replaces_story(monkeypatch) -> None:
    calls = []

    def fake_call(messages, config, model_settings):
        calls.append((messages, config, model_settings))
        return (
            json.dumps(
                {
                    "glossary": [
                        {
                            "eng": "Webb",
                            "zh": "韦布",
                            "type": "person",
                            "confidence": 0.95,
                            "evidence_ids": [8],
                        }
                    ],
                    "story_description": (
                        "Harm briefs Mac; Webb is questioning a witness."
                    ),
                }
            ),
            None,
        )

    monkeypatch.setattr("subretrans.llm.call_role_api_sdk", fake_call)
    config = SimpleNamespace(
        terminology_min_confidence=0.6,
        extraction=SimpleNamespace(model="memory-model"),
        glossary_policy="lock",
        glossary_max_entries=100,
        verbose=False,
        very_verbose=False,
    )
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[{"eng": "Mac", "zh": "麦可", "type": "person"}],
        story_description="Harm briefs Mac.",
    )
    pairs = [SubtitlePair(id=8, eng="Webb questions her.", chinese="韦布盘问她")]

    updated = update_global_memory(memory, pairs, config)

    assert len(calls) == 1
    user_prompt = calls[0][0][1]["content"]
    assert "Harm briefs Mac." in user_prompt
    assert "Webb questions her." in user_prompt
    assert [entry["eng"] for entry in updated.glossary] == ["Mac", "Webb"]
    assert updated.user_glossary == [{"eng": "Harm", "zh": "哈姆"}]
    assert updated.story_description == (
        "Harm briefs Mac; Webb is questioning a witness."
    )
