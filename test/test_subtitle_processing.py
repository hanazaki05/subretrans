from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from subretrans.subtitle_processing import (
    AssAudit,
    SrtCue,
    audit_ass,
    merge_srt_to_ass,
    postprocess_ass,
    read_srt,
    write_srt,
)


def _write(path: Path, content: str, *, bom: bool = False) -> Path:
    path.write_text(content, encoding="utf-8-sig" if bom else "utf-8")
    return path


def test_merge_srt_to_complete_ass_with_midpoint_matching(tmp_path: Path) -> None:
    english = _write(
        tmp_path / "english.srt",
        """1
00:00:01,000 --> 00:00:02,000
First line
continued

2
00:00:04,000 --> 00:00:05,000
Second
""",
        bom=True,
    )
    chinese = _write(
        tmp_path / "chinese.srt",
        """1
00:00:01,700 --> 00:00:02,700
第一行

2
00:00:06,000 --> 00:00:07,000
额外
""",
    )
    output = tmp_path / "merged.ass"

    assert merge_srt_to_ass(english, chinese, output) == output
    content = output.read_text(encoding="utf-8")

    assert content.startswith("[Script Info]\n")
    assert "[V4+ Styles]\n" in content
    assert "[Events]\n" in content
    assert (
        r"Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,First line\Ncontinued"
        in content
    )
    assert "Dialogue:  1,0:00:01.00,0:00:02.00,Chinese3,,0,0,0,,第一行" in content
    assert "Dialogue:  1,0:00:04.00,0:00:05.00,Chinese3,,0,0,0,,\n" in content
    assert "Dialogue:  1,0:00:06.00,0:00:07.00,Chinese3,,0,0,0,,额外" in content


def test_merge_rejects_malformed_srt_instead_of_dropping_it(tmp_path: Path) -> None:
    english = _write(tmp_path / "bad.srt", "1\nnot a timestamp\nText\n")
    chinese = _write(
        tmp_path / "chinese.srt", "1\n00:00:01,000 --> 00:00:02,000\n文本\n"
    )

    with pytest.raises(ValueError, match="Malformed SRT block"):
        merge_srt_to_ass(english, chinese, tmp_path / "unused.ass")


def test_srt_read_write_roundtrip_preserves_timeline_and_text(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.srt",
        """7
00:00:01,250 --> 00:00:02,750
First line
second line

9
00:00:04,000 --> 00:00:05,000
""",
        bom=True,
    )
    expected = [
        SrtCue(7, "00:00:01,250", "00:00:02,750", "First line\nsecond line"),
        SrtCue(9, "00:00:04,000", "00:00:05,000", ""),
    ]

    assert read_srt(source) == expected
    output = tmp_path / "rebuilt.srt"
    assert write_srt(expected, output) == output
    assert read_srt(output) == expected


def test_postprocess_ass_separates_generic_and_episode_rules(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "input.ass",
        """[V4+ Styles]
Style: English3,Arial
Style: Chinese3,Arial
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:01.00,0:00:02.00,Chinese3,,0000,0000,0000,,{\\i1}萨拉。罗伯茨，{\\i0}
Dialogue: -1,0:00:01.00,0:00:02.00,English3,,0,0,0,,<I>A.J. -- SECNAV</I>
""",
        bom=True,
    )
    output = tmp_path / "output.ass"

    assert postprocess_ass(
        source,
        output,
        (
            ("萨拉", "莎拉"),
            ("罗伯茨", "罗伯特"),
            ("SECNAV", "SecNav"),
            ("A.J.", "AJ"),
        ),
    ) == output
    content = output.read_text(encoding="utf-8")

    assert "Style: E3,Arial" in content
    assert "Style: C3,Arial" in content
    assert "Dialogue:  1,0:00:01.00,0:00:02.00,C3,,0,0,0,,莎拉 罗伯特" in content
    assert r"Dialogue: -1,0:00:01.00,0:00:02.00,E3,,0,0,0,,{\i1}AJ ... SecNav{\i0}" in content
    assert source.read_text(encoding="utf-8-sig").startswith("[V4+ Styles]")


def test_audit_ass_reports_structural_failures(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.ass",
        """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: -1,0:00:02.00,0:00:03.00,E3,,0,0,0,,English
Dialogue:  1,0:00:02.00,0:00:03.00,C3,,0,0,0,,
Dialogue: -1,0:00:01.00,0:00:02.00,E3,,0,0,0,,Earlier
Dialogue:  1,broken,0:00:02.00,C3,,0,0,0,,无效
""",
    )

    result = audit_ass(path)

    assert result == AssAudit(
        english_events=2,
        chinese_events=2,
        empty_english_events=0,
        empty_chinese_events=1,
        unpaired_timestamps=1,
        non_monotonic_events=1,
        parse_errors=1,
        passed=False,
    )
    with pytest.raises(FrozenInstanceError):
        result.passed = True  # type: ignore[misc]
