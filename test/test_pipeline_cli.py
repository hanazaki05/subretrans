import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from subretrans.cli import file_sha256
from subretrans.model_agent import AgentQAResult
from subretrans.pipeline_cli import _refine_callable, review_pipeline, run_pipeline


VALID_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,Hello
Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,你好
"""


def test_serial_pipeline_cli_persists_review_and_releases(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "input.ass"
    source.write_text(VALID_ASS, encoding="utf-8")
    release = tmp_path / "release.ass"
    key = tmp_path / "key"
    key.write_text("test-key\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""api:
  primer: &model
    protocol: openai-responses
    model: test-model
    key_file: {key}
    base_url: https://example.test/v1
    timeout: 30
    max_retries: 0
    max_output_tokens: 1000
    reasoning_effort: null
    temperature: null
  refine: *model
  extraction: *model
  agent: *model
pipeline:
  state_dir: {tmp_path / 'state'}
  checkpoint_db: {tmp_path / 'state/checkpoints.sqlite3'}
  batch_size: 2
  max_workers: 2
  agent_max_repair_attempts: 1
  source_language: English
  target_language: Simplified Chinese
  user_instruction: null
  episode_replacements: []
""",
        encoding="utf-8",
    )

    def fake_refine_factory(config_path):
        def refine(input_path, output_path, checkpoint_path, progress_path):
            output_path.write_bytes(input_path.read_bytes())
            checkpoint_path.write_text("story_description: test\n", encoding="utf-8")
            progress_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "next_pair": 1,
                        "artifact_path": str(output_path),
                        "artifact_hash": file_sha256(str(output_path)),
                        "memory_checkpoint_path": str(checkpoint_path),
                        "memory_hash": file_sha256(str(checkpoint_path)),
                    }
                ),
                encoding="utf-8",
            )

        return refine

    monkeypatch.setattr(
        "subretrans.pipeline_cli._refine_callable", fake_refine_factory
    )
    monkeypatch.setattr(
        "subretrans.pipeline_cli.build_agent_qa",
        lambda settings: lambda pairs, structural_qa: AgentQAResult(True, (), ()),
    )
    run_args = argparse.Namespace(
        input=str(source),
        output=str(release),
        mode="serial_memory",
        thread_id="episode-01",
        config=str(config),
    )
    review_args = argparse.Namespace(
        thread_id="episode-01",
        decision="approve",
        config=str(config),
    )

    assert run_pipeline(run_args) == 0
    assert not release.exists()
    assert review_pipeline(review_args) == 0
    assert release.exists()
    assert "你好" in release.read_text(encoding="utf-8")


def test_refine_resume_continues_from_committed_output(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "original.ass"
    original.write_text(VALID_ASS.replace("你好", ""), encoding="utf-8")
    output = tmp_path / "refined.ass"
    output.write_text(
        VALID_ASS
        + "Dialogue: -1,0:00:03.00,0:00:04.00,English3,,0,0,0,,Bye\n"
        + "Dialogue:  1,0:00:03.00,0:00:04.00,Chinese3,,0,0,0,,再见\n",
        encoding="utf-8",
    )
    memory = tmp_path / "memory.yaml"
    memory.write_text("story_description: test\n", encoding="utf-8")
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "version": 1,
                "next_pair": 1,
                "artifact_path": str(output),
                "artifact_hash": file_sha256(str(output)),
                "memory_checkpoint_path": str(memory),
                "memory_hash": file_sha256(str(memory)),
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "subretrans.pipeline_cli.load_config_sdk",
        lambda **kwargs: SimpleNamespace(
            refine=SimpleNamespace(protocol=SimpleNamespace(value="openai-responses")),
            use_stream=False,
        ),
    )

    def fake_process(input_path, output_path, config_value, **kwargs):
        seen.update(input_path=input_path, output_path=output_path, kwargs=kwargs)
        return True

    monkeypatch.setattr("subretrans.pipeline_cli.process_subtitles", fake_process)

    _refine_callable(config)(original, output, memory, progress)

    assert seen["input_path"] == str(output)
    assert seen["output_path"] == str(output)
    assert seen["kwargs"]["resume_index"] == 1  # type: ignore[index]
