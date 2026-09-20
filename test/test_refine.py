import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from langchain_core.messages import AIMessage, AIMessageChunk

import subretrans.refine as refine_module
from subretrans.config import AppConfig, load_config
from subretrans.fsutil import sha256_file
from subretrans.pairs import SubtitlePair
from subretrans.pricing import ModelPricing
from subretrans.refine import (
    RefineOptions,
    load_refine_progress,
    parse_refined_pairs,
    refine_serial,
    save_refine_progress,
)


ASS_HEADER = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_with(rows: list[tuple[str, str]]) -> str:
    lines = [ASS_HEADER]
    for index, (eng, zh) in enumerate(rows, start=1):
        start, end = f"0:00:0{index}.00", f"0:00:0{index}.50"
        lines.append(f"Dialogue: -1,{start},{end},English3,,0,0,0,,{eng}\n")
        lines.append(f"Dialogue:  1,{start},{end},Chinese3,,0,0,0,,{zh}\n")
    return "".join(lines)


SHARED_PROMPT = """# Shared rules

### User Terminology (Authoritative Glossary)
- Harm: 哈姆
- Mac: 麦可
"""

REFINE_PROMPT = """You refine subtitles based on the provided JSON input.

### Input/Output Format & Constraint
- **Input:** A JSON array of subtitle pairs (`id`, `eng`, `chinese`).
- **Output:** A JSON array with the SAME structure containing corrections.
- **STRICT ADHERENCE REQUIRED:** You MUST **ONLY** return the JSON array.

### Few-Shot Examples
Input:
[
  {"id": 1, "eng": "hello", "chinese": "你好。"}
]
Output:
[
  {"id": 1, "eng": "Hello.", "chinese": "你好"}
]
"""


def write_config(tmp_path: Path, **refine_overrides: object) -> Path:
    key = tmp_path / "key"
    key.write_text("test-key\n", encoding="utf-8")
    (tmp_path / "shared.md").write_text(SHARED_PROMPT, encoding="utf-8")
    (tmp_path / "refine.md").write_text(REFINE_PROMPT, encoding="utf-8")
    (tmp_path / "qa.md").write_text("QA task.\n", encoding="utf-8")
    role = {
        "protocol": "openai-chat-compatible",
        "key_file": str(key),
        "base_url": "https://example.test/v1",
        "timeout": 30,
        "max_retries": 0,
        "max_output_tokens": 1000,
        "reasoning_effort": None,
        "temperature": None,
    }
    refine = {
        "batch_size": 1,
        "chunk_token_soft_limit": 80000,
        "memory_token_limit": 4000,
        "intermediate_representation": "json",
    }
    refine.update(refine_overrides)
    payload = {
        "api": {
            name: {**role, "model": f"{name}-model"}
            for name in ("primer", "refine", "extraction", "agent")
        },
        "pipeline": {
            "state_dir": str(tmp_path / "state"),
            "checkpoint_db": str(tmp_path / "state/checkpoints.sqlite3"),
            "agent_max_repair_attempts": 1,
        },
        "prompts": {
            "shared_path": str(tmp_path / "shared.md"),
            "refine_path": str(tmp_path / "refine.md"),
            "qa_path": str(tmp_path / "qa.md"),
        },
        "primer": {
            "batch_size": 2,
            "max_workers": 1,
            "source_language": "English",
            "target_language": "Simplified Chinese",
            "user_instruction": None,
        },
        "refine": refine,
        "qa": {"batch_size": 10, "max_workers": 1, "window_offsets": [0]},
        "postprocess": {"operations": [], "episode_replacements": []},
        "subtitle_edit": {
            "repository_url": "https://example.test/subtitleedit.git",
            "revision": "7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9",
            "source_dir": str(tmp_path / "se-src"),
            "build_dir": str(tmp_path / "se-build"),
            "dotnet_executable": "dotnet",
            "settings_file": str(tmp_path / "se-settings.json"),
            "multiple_replace_file": str(tmp_path / "se-replace.template"),
            "first_pass_operations": ["--remove-text-for-hi"],
            "second_pass_operations": ["--fix-common-errors-rules:FixUnneededSpaces"],
        },
        "glossary": {"max_entries": 100, "terminology_min_confidence": 0.6},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


class FakeModel:
    """Minimal stand-in for a LangChain chat model driven by a responder callable."""

    def __init__(self, responder: Callable[[list[tuple[str, str]]], str]) -> None:
        self.responder = responder
        self.calls: list[list[tuple[str, str]]] = []
        self.stream_calls = 0

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(
            content=self.responder(list(messages)),
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    def stream(self, messages):
        self.calls.append(list(messages))
        self.stream_calls += 1
        text = self.responder(list(messages))
        middle = len(text) // 2
        yield AIMessageChunk(content=text[:middle])
        yield AIMessageChunk(
            content=text[middle:],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def refine_responder(messages: list[tuple[str, str]]) -> str:
    system, human = messages[0][1], messages[1][1]
    if system.startswith("You compress episode memory"):
        return json.dumps(
            {
                "glossary": [{"eng": "Webb", "zh": "韦布", "type": "person"}],
                "story_description": "compressed",
            }
        )
    pairs = json.loads(human)
    return json.dumps(
        [
            {"id": pair["id"], "eng": pair["eng"].capitalize(), "chinese": pair["chinese"] + "！"}
            for pair in pairs
        ]
    )


def extraction_responder(messages: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "glossary": [
                {"eng": "Webb", "zh": "韦布", "type": "person", "confidence": 0.9, "evidence_ids": [0]},
                {"eng": "Harm", "zh": "哈蒙", "type": "person", "confidence": 0.9, "evidence_ids": [0]},
            ],
            "story_description": "Webb meets Harm.",
        }
    )


@pytest.fixture(autouse=True)
def offline_pricing(monkeypatch):
    def unavailable(model_name: str):
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr("subretrans.pricing.load_model_pricing", unavailable)


@pytest.fixture
def models(monkeypatch) -> dict[str, FakeModel]:
    built: dict[str, FakeModel] = {}

    def build(config):
        responder = refine_responder if config.model == "refine-model" else extraction_responder
        built[config.model] = FakeModel(responder)
        return built[config.model]

    monkeypatch.setattr(refine_module, "build_chat_model", build)
    return built


def pairs(*ids: int) -> list[SubtitlePair]:
    return [SubtitlePair(id=pair_id, eng=f"e{pair_id}", chinese=f"c{pair_id}") for pair_id in ids]


# parse_refined_pairs ---------------------------------------------------------


def test_parse_recovers_fenced_json_after_thinking() -> None:
    text = '<think>plan</think>\n```json\n[{"id": 3, "eng": "Hi.", "chinese": "嗨"}]\n```'

    parsed = parse_refined_pairs(text, "json", pairs(3))

    assert [(p.id, p.eng, p.chinese) for p in parsed.pairs] == [(3, "Hi.", "嗨")]
    assert parsed.warnings == ()
    assert parsed.missing_ids == ()


def test_parse_extracts_json_after_leading_commentary() -> None:
    text = 'Here are the corrections:\n[{"id": 3, "eng": "Hi.", "chinese": "嗨"}]'

    parsed = parse_refined_pairs(text, "json", pairs(3))

    assert [p.id for p in parsed.pairs] == [3]
    assert any("first format marker" in warning for warning in parsed.warnings)


def test_parse_accepts_malformed_xml_separators() -> None:
    text = "<pair>\nID=7\neng>Hello\nchinese: 你好\n</pair>"

    parsed = parse_refined_pairs(text, "xml-pair", pairs(7))

    assert [(p.id, p.eng, p.chinese) for p in parsed.pairs] == [(7, "Hello", "你好")]


def test_parse_salvages_valid_xml_blocks_and_reports_missing() -> None:
    text = "<pair>\nID=7\neng=Hello\nchinese=你好\n</pair>\n<pair>\nID=8\neng=Bye\n</pair>"

    parsed = parse_refined_pairs(text, "xml-pair", pairs(7, 8))

    assert [p.id for p in parsed.pairs] == [7]
    assert parsed.missing_ids == (8,)
    assert any("best-effort" in warning for warning in parsed.warnings)


def test_parse_keeps_last_duplicate() -> None:
    text = json.dumps(
        [
            {"id": 1, "eng": "first", "chinese": "一"},
            {"id": 2, "eng": "two", "chinese": "二"},
            {"id": 1, "eng": "second", "chinese": "壹"},
        ]
    )

    parsed = parse_refined_pairs(text, "json", pairs(1, 2))

    assert [(p.id, p.eng) for p in parsed.pairs] == [(1, "second"), (2, "two")]
    assert any("duplicate pair ids [1]" in warning for warning in parsed.warnings)


@pytest.mark.parametrize("first_local_id", [0, 1])
def test_parse_remaps_local_ids(first_local_id: int) -> None:
    text = json.dumps(
        [
            {"id": first_local_id, "eng": "a", "chinese": "甲"},
            {"id": first_local_id + 1, "eng": "b", "chinese": "乙"},
        ]
    )

    parsed = parse_refined_pairs(text, "json", pairs(40, 41))

    assert [(p.id, p.eng) for p in parsed.pairs] == [(40, "a"), (41, "b")]
    assert any("local ids" in warning for warning in parsed.warnings)


def test_parse_refuses_foreign_ids_and_drops_partial_strays() -> None:
    foreign = json.dumps([{"id": 900, "eng": "x", "chinese": "x"}])
    with pytest.raises(ValueError, match="do not match the chunk"):
        parse_refined_pairs(foreign, "json", pairs(40, 41))

    partial = json.dumps(
        [{"id": 40, "eng": "a", "chinese": "甲"}, {"id": 900, "eng": "x", "chinese": "x"}]
    )
    parsed = parse_refined_pairs(partial, "json", pairs(40, 41))
    assert [p.id for p in parsed.pairs] == [40]
    assert parsed.missing_ids == (41,)


def test_parse_raises_when_nothing_recoverable() -> None:
    with pytest.raises(ValueError, match="failed to parse json response"):
        parse_refined_pairs("no payload here", "json", pairs(1))


# progress manifest ------------------------------------------------------------


def write_progress_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "refined.ass"
    output.write_text(ass_with([("Hello", "你好")]), encoding="utf-8")
    memory = tmp_path / "memory.yaml"
    memory.write_text("user_glossary: []\nglossary: []\nstory_description: ''\n", encoding="utf-8")
    progress = tmp_path / "progress.json"
    return progress, output, memory


def test_progress_roundtrip(tmp_path: Path) -> None:
    progress, output, memory = write_progress_inputs(tmp_path)

    saved = save_refine_progress(progress, next_pair=1, output_path=output, checkpoint_path=memory)
    loaded = load_refine_progress(progress, output, memory)

    assert loaded == saved
    assert loaded.next_pair == 1
    assert loaded.artifact_hash == sha256_file(output)
    assert loaded.memory_hash == sha256_file(memory)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda payload: payload.update(version=2), "version must be 1"),
        (lambda payload: payload.update(artifact_hash="0" * 64), "artifact_hash does not match"),
        (lambda payload: payload.update(memory_hash="0" * 64), "memory_hash does not match"),
        (lambda payload: payload.update(next_pair=True), "non-negative integer"),
        (lambda payload: payload.update(extra=True), "unknown fields: extra"),
        (lambda payload: payload.update(artifact_path="/elsewhere"), "artifact_path does not match"),
    ],
)
def test_progress_rejects_invalid_manifest(tmp_path: Path, mutate, match: str) -> None:
    progress, output, memory = write_progress_inputs(tmp_path)
    save_refine_progress(progress, next_pair=1, output_path=output, checkpoint_path=memory)
    payload = json.loads(progress.read_text(encoding="utf-8"))
    mutate(payload)
    progress.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_refine_progress(progress, output, memory)


# refine_serial ----------------------------------------------------------------


def test_refine_serial_commits_artifact_memory_and_progress(tmp_path: Path, models, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好"), ("bye", "再见")]), encoding="utf-8")
    output = tmp_path / "refined.ass"
    memory = tmp_path / "memory.yaml"
    progress = tmp_path / "progress.json"
    atomic_calls: list[Path] = []
    real_atomic = refine_module.save_memory_checkpoint.__globals__["atomic_write_yaml"]

    def spy(path, payload):
        atomic_calls.append(Path(path))
        return real_atomic(path, payload)

    monkeypatch.setattr("subretrans.memory.atomic_write_yaml", spy)

    result = refine_serial(
        source, output, config, checkpoint_path=memory, progress_path=progress
    )

    assert (result.total_pairs, result.committed_pairs, result.chunks) == (2, 2, 2)
    assert result.usage.total_tokens == 30
    assert result.extraction_usage.total_tokens == 30
    assert result.cost is None
    rendered = output.read_text(encoding="utf-8-sig")
    assert "Hello" in rendered and "你好！" in rendered and "再见！" in rendered
    saved = yaml.safe_load(memory.read_text(encoding="utf-8"))
    assert saved["user_glossary"] == [{"eng": "Harm", "zh": "哈姆"}, {"eng": "Mac", "zh": "麦可"}]
    assert [entry["eng"] for entry in saved["glossary"]] == ["Webb"]
    assert saved["story_description"] == "Webb meets Harm."
    assert "style_notes" not in saved
    assert atomic_calls == [memory, memory]
    loaded = load_refine_progress(progress, output, memory)
    assert loaded.next_pair == 2
    assert models["refine-model"].calls[0][0][1].count("- Harm: 哈姆") == 1
    assert "### 2. Incremental Story Description" in models["refine-model"].calls[1][0][1]
    assert "Webb meets Harm." in models["refine-model"].calls[1][0][1]


def test_refine_serial_resume_is_noop_when_everything_committed(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")
    output = tmp_path / "refined.ass"

    result = refine_serial(source, output, config, options=RefineOptions(resume_index=1))

    assert result.committed_pairs == 1
    assert result.chunks == 0
    assert not output.exists()
    assert models == {}


def test_refine_serial_resume_preserves_committed_pairs(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好"), ("bye", "再见")]), encoding="utf-8")
    output = tmp_path / "refined.ass"
    output.write_text(ass_with([("Hello!", "已精修"), ("bye", "再见")]), encoding="utf-8")

    result = refine_serial(source, output, config, options=RefineOptions(resume_index=1))

    assert result.committed_pairs == 2
    rendered = output.read_text(encoding="utf-8-sig")
    assert "已精修" in rendered and "Hello!" in rendered and "再见！" in rendered
    assert len(models["refine-model"].calls) == 1
    assert json.loads(models["refine-model"].calls[0][1][1])[0]["id"] == 1


def test_refine_serial_resume_rejects_misaligned_output(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好"), ("bye", "再见")]), encoding="utf-8")
    output = tmp_path / "refined.ass"
    output.write_text(ass_with([("only", "一条")]), encoding="utf-8")

    with pytest.raises(ValueError, match="existing output has 1 pairs"):
        refine_serial(source, output, config, options=RefineOptions(resume_index=1))
    with pytest.raises(ValueError, match="resume requires the existing refined output"):
        refine_serial(source, tmp_path / "missing.ass", config, options=RefineOptions(resume_index=1))


def test_refine_serial_reapplies_user_glossary_after_compression(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path, memory_token_limit=1))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")
    memory = tmp_path / "memory.yaml"

    refine_serial(source, tmp_path / "refined.ass", config, checkpoint_path=memory)

    compression_calls = [
        call for call in models["refine-model"].calls if call[0][1].startswith("You compress")
    ]
    assert len(compression_calls) == 1
    assert "user_glossary" not in compression_calls[0][1][1]
    saved = yaml.safe_load(memory.read_text(encoding="utf-8"))
    assert saved["user_glossary"] == [{"eng": "Harm", "zh": "哈姆"}, {"eng": "Mac", "zh": "麦可"}]
    assert saved["story_description"] == "compressed"


def test_refine_serial_reports_cost_when_pricing_available(tmp_path: Path, models, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")
    pricing = ModelPricing("refine-model", "vendor", "v1", "2026-09-21", Decimal("1"), Decimal("2"))
    monkeypatch.setattr("subretrans.pricing.load_model_pricing", lambda model_name: pricing)

    result = refine_serial(source, tmp_path / "refined.ass", config)

    assert result.cost is not None
    assert result.cost.cost == Decimal("0.00002")


def test_refine_serial_streams_when_requested(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")
    seen: list[str] = []

    result = refine_serial(
        source,
        tmp_path / "refined.ass",
        config,
        options=RefineOptions(stream=True, on_stream_chunk=seen.append),
    )

    assert models["refine-model"].stream_calls == 1
    assert json.loads("".join(seen))[0]["eng"] == "Hello"
    assert result.usage.total_tokens == 15


def test_refine_serial_rejects_progress_without_checkpoint(tmp_path: Path, models) -> None:
    config = load_config(write_config(tmp_path))
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a memory checkpoint"):
        refine_serial(source, tmp_path / "out.ass", config, progress_path=tmp_path / "p.json")


def test_refine_serial_requires_glossary_section(tmp_path: Path, models) -> None:
    config_path = write_config(tmp_path)
    (tmp_path / "shared.md").write_text("no glossary here\n", encoding="utf-8")
    config: AppConfig = load_config(config_path)
    source = tmp_path / "input.ass"
    source.write_text(ass_with([("hello", "你好")]), encoding="utf-8")

    with pytest.raises(ValueError, match="User Terminology"):
        refine_serial(source, tmp_path / "out.ass", config)
