#!/usr/bin/env python3
"""
Test script for per-block update feature (formerly incremental_output).

Goals:
- Verify YAML config loading supports `runtime.per_block_update` and the deprecated
  `runtime.incremental_output` fallback.
- Verify `process_subtitles()` writes output after each chunk when per-block update
  is enabled, and only once at the end when disabled.

This test is network-free: it stubs the LLM refinement and memory update functions.
"""

import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairs import SubtitlePair
from stats import UsageStats
from memory import GlobalMemory
from experiment.config_sdk import ConfigSDK, load_config_from_yaml
import experiment.main_sdk as main_sdk


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _make_minimal_ass(path: str) -> None:
    # Two timestamp groups => 2 SubtitlePair objects.
    # Ensure styles include "English" / "Chinese" so pairing works.
    content = """[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: English,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Chinese,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,English,,0,0,0,,Hello there
Dialogue: 0,0:00:01.00,0:00:02.00,Chinese,,0,0,0,,你好
Dialogue: 0,0:00:03.00,0:00:04.00,English,,0,0,0,,Second line
Dialogue: 0,0:00:03.00,0:00:04.00,Chinese,,0,0,0,,第二句
"""
    _write_text(path, content)


def test_yaml_loading_per_block_update():
    print("=" * 70)
    print("Test 1: YAML loading supports per_block_update + deprecated incremental_output")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        key_path = os.path.join(td, "key")
        _write_text(key_path, "sk-test\n")

        # Case A: per_block_update present
        yaml_a = os.path.join(td, "a.yaml")
        _write_text(
            yaml_a,
            """api:
  key_file: key
runtime:
  per_block_update: true
""",
        )
        cfg_a = load_config_from_yaml(yaml_a)
        assert cfg_a.per_block_update is True, "per_block_update should load from YAML"

        # Case B: fallback to deprecated incremental_output
        yaml_b = os.path.join(td, "b.yaml")
        _write_text(
            yaml_b,
            """api:
  key_file: key
runtime:
  incremental_output: false
""",
        )
        cfg_b = load_config_from_yaml(yaml_b)
        assert cfg_b.per_block_update is False, "per_block_update should fall back to incremental_output"

    print("\n✅ Test 1 PASSED")
    return True


def test_process_subtitles_writes_per_chunk():
    print("\n" + "=" * 70)
    print("Test 2: process_subtitles per-block write behavior")
    print("=" * 70)

    # Stub LLM refinement and memory update to avoid network calls.
    def stub_refine_chunk_sdk(pairs_chunk, global_memory, config, print_system_prompt=False):
        corrected = []
        for p in pairs_chunk:
            corrected.append(
                SubtitlePair(
                    id=p.id,
                    eng=p.eng,
                    chinese=f"{p.chinese}（已更新）",
                    meta=p.meta,
                )
            )
        return corrected, UsageStats(total_tokens=1, prompt_tokens=1, completion_tokens=0, reasoning_tokens=0), ""

    original_refine = main_sdk.refine_chunk_sdk
    original_update_memory = main_sdk.update_global_memory
    original_init_memory = main_sdk.init_global_memory
    original_estimate_memory = main_sdk.estimate_memory_tokens
    original_estimate_base_prompt = main_sdk.estimate_base_prompt_tokens
    original_write_ass = main_sdk.write_ass_file
    original_print_chunk_stats = main_sdk.print_chunk_statistics

    try:
        main_sdk.refine_chunk_sdk = stub_refine_chunk_sdk
        main_sdk.update_global_memory = lambda memory, corrected_pairs, config: memory
        main_sdk.init_global_memory = lambda: GlobalMemory()
        main_sdk.estimate_memory_tokens = lambda memory, model_name=None: 0
        main_sdk.estimate_base_prompt_tokens = lambda config, global_memory: 0
        main_sdk.print_chunk_statistics = lambda *args, **kwargs: None

        with tempfile.TemporaryDirectory() as td:
            input_path = os.path.join(td, "input.ass")
            output_path = os.path.join(td, "output.ass")
            _make_minimal_ass(input_path)

            written_contents = []

            def capture_write(path: str, content: str) -> None:
                # Keep original behavior of writing so later reads work.
                written_contents.append(content)
                original_write_ass(path, content)

            main_sdk.write_ass_file = capture_write

            # 2 pairs => 2 chunks (pairs_per_chunk=1)
            cfg = ConfigSDK(api_key="sk-test", pairs_per_chunk=1, per_block_update=True)
            ok = main_sdk.process_subtitles(
                input_path=input_path,
                output_path=output_path,
                config=cfg,
                use_streaming=False,
                resume_index=None,
                enable_checkpoint=False,
            )
            assert ok is True, "process_subtitles should succeed with stubbed LLM"

            # When enabled: write after each chunk + final write
            assert len(written_contents) == 3, f"expected 3 writes (2 chunks + final), got {len(written_contents)}"
            assert "你好（已更新）" in written_contents[0], "first per-block write should include first pair update"
            assert "第二句（已更新）" not in written_contents[0], "first per-block write should not include second pair update yet"
            assert "第二句（已更新）" in written_contents[1], "second per-block write should include second pair update"
            assert "第二句（已更新）" in written_contents[2], "final write should include second pair update"

            # Disabled: only final write
            written_contents.clear()
            cfg.per_block_update = False
            ok = main_sdk.process_subtitles(
                input_path=input_path,
                output_path=output_path,
                config=cfg,
                use_streaming=False,
                resume_index=None,
                enable_checkpoint=False,
            )
            assert ok is True
            assert len(written_contents) == 1, f"expected 1 write (final only), got {len(written_contents)}"

    finally:
        main_sdk.refine_chunk_sdk = original_refine
        main_sdk.update_global_memory = original_update_memory
        main_sdk.init_global_memory = original_init_memory
        main_sdk.estimate_memory_tokens = original_estimate_memory
        main_sdk.estimate_base_prompt_tokens = original_estimate_base_prompt
        main_sdk.write_ass_file = original_write_ass
        main_sdk.print_chunk_statistics = original_print_chunk_stats

    print("\n✅ Test 2 PASSED")
    return True


if __name__ == "__main__":
    try:
        all_passed = True
        all_passed &= test_yaml_loading_per_block_update()
        all_passed &= test_process_subtitles_writes_per_chunk()

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL TESTS PASSED")
            print("=" * 70)
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 70)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

