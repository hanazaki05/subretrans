"""Generic subtitle parsing, merging, normalization, and structural QA."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .ass_parser import is_chinese_style, is_english_style
from .fsutil import atomic_write_text


SCRIPT_INFO_BLOCK = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: Yes
PlayResX: 1920
PlayResY: 1080
Collisions: Normal
"""

STYLES_BLOCK = """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: English3,Calibri,54,&H00F8FFF4,&H000000FF,&H00703A22,&H19000000,0,0,0,0,100,100,0,0,1,2.3,0.8,2,30,30,24,1
Style: Chinese3,Microsoft YaHei,69,&H00F8FFF4,&H0000FFFF,&H00703A22,&H19000000,0,0,0,0,100,100,0,0,1,2.5,0.8,2,30,30,77,1
Style: Ctop,Microsoft YaHei,54,&H00F8FFF4,&H0000FFFF,&H00703A22,&H19000000,0,0,0,0,100,100,0,0,1,2.5,0.8,8,30,30,28,1
"""

EVENTS_HEADER = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_SRT_TIME_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2}),(?P<milliseconds>\d{3})$"
)
_ASS_TIME_RE = re.compile(
    r"^(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2})\.(?P<centiseconds>\d{2})$"
)


@dataclass(frozen=True)
class SrtCue:
    """One SRT cue with its source timeline and possibly multiline text."""

    index: int
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class _AssCue:
    start: str
    end: str
    text_lines: tuple[str, ...]


@dataclass(frozen=True)
class AssAudit:
    """Counts produced by structural ASS QA; no semantic or visual QA is done."""

    english_events: int
    chinese_events: int
    empty_english_events: int
    empty_chinese_events: int
    unpaired_timestamps: int
    non_monotonic_events: int
    parse_errors: int
    passed: bool


def _srt_timestamp_to_ass(timestamp: str, *, path: Path, block_number: int) -> str:
    match = _SRT_TIME_RE.fullmatch(timestamp.strip())
    if match is None:
        raise ValueError(
            f"Malformed SRT timestamp in {path} block {block_number}: {timestamp!r}"
        )

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    milliseconds = int(match.group("milliseconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(
            f"Malformed SRT timestamp in {path} block {block_number}: {timestamp!r}"
        )

    centiseconds = int(round(milliseconds / 10.0))
    if centiseconds >= 100:
        centiseconds -= 100
        seconds += 1
        if seconds >= 60:
            seconds -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
                hours += 1
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_time_to_seconds(timestamp: str) -> float:
    match = _ASS_TIME_RE.fullmatch(timestamp.strip())
    if match is None:
        raise ValueError(f"Malformed ASS timestamp: {timestamp!r}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    centiseconds = int(match.group("centiseconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Malformed ASS timestamp: {timestamp!r}")
    return hours * 3600.0 + minutes * 60.0 + seconds + centiseconds / 100.0


def _ass_time_to_centiseconds(timestamp: str) -> int:
    return round(_ass_time_to_seconds(timestamp) * 100)


def read_srt(path: Path) -> list[SrtCue]:
    """Read an SRT file strictly, accepting UTF-8 with or without a BOM."""

    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", normalized.strip()) if normalized.strip() else []
    cues: list[SrtCue] = []
    seen_indices: set[int] = set()

    for block_number, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            cue_index = int(lines[0].strip())
            lines = lines[1:]
        else:
            cue_index = block_number
        if cue_index <= 0 or cue_index in seen_indices:
            raise ValueError(
                f"Malformed SRT block {block_number} in {path}: invalid cue index"
            )
        seen_indices.add(cue_index)
        if not lines or "-->" not in lines[0]:
            raise ValueError(f"Malformed SRT block {block_number} in {path}: missing timing")
        if lines[0].count("-->") != 1:
            raise ValueError(f"Malformed SRT block {block_number} in {path}: invalid timing")

        start_raw, end_raw = (part.strip() for part in lines[0].split("-->", 1))
        start_ass = _srt_timestamp_to_ass(start_raw, path=path, block_number=block_number)
        end_ass = _srt_timestamp_to_ass(end_raw, path=path, block_number=block_number)
        if _ass_time_to_seconds(end_ass) < _ass_time_to_seconds(start_ass):
            raise ValueError(f"Malformed SRT block {block_number} in {path}: end before start")
        cues.append(
            SrtCue(
                index=cue_index,
                start=start_raw,
                end=end_raw,
                text="\n".join(line for line in lines[1:] if line.strip()),
            )
        )

    return cues


def write_srt(cues: Sequence[SrtCue], path: Path) -> Path:
    """Write cues as canonical UTF-8 SRT using an atomic replacement."""

    output = Path(path)
    blocks: list[str] = []
    seen_indices: set[int] = set()
    for block_number, cue in enumerate(cues, start=1):
        if (
            not isinstance(cue.index, int)
            or isinstance(cue.index, bool)
            or cue.index <= 0
            or cue.index in seen_indices
        ):
            raise ValueError(f"Malformed SRT cue {block_number}: invalid cue index")
        seen_indices.add(cue.index)
        start_ass = _srt_timestamp_to_ass(cue.start, path=output, block_number=block_number)
        end_ass = _srt_timestamp_to_ass(cue.end, path=output, block_number=block_number)
        if _ass_time_to_seconds(end_ass) < _ass_time_to_seconds(start_ass):
            raise ValueError(f"Malformed SRT cue {block_number}: end before start")
        if "\r" in cue.text or re.search(r"\n[ \t]*\n", cue.text):
            raise ValueError(f"Malformed SRT cue {block_number}: blank line in text")

        block = f"{cue.index}\n{cue.start} --> {cue.end}"
        if cue.text:
            block += f"\n{cue.text}"
        blocks.append(block)

    content = "\n\n".join(blocks)
    if blocks:
        content += "\n"
    atomic_write_text(output, content)
    return output


def _read_ass_cues(path: Path) -> list[_AssCue]:
    return [
        _AssCue(
            start=_srt_timestamp_to_ass(cue.start, path=Path(path), block_number=position),
            end=_srt_timestamp_to_ass(cue.end, path=Path(path), block_number=position),
            text_lines=tuple(cue.text.split("\n")) if cue.text else (),
        )
        for position, cue in enumerate(read_srt(path), start=1)
    ]


def _merge_cues(
    english: list[_AssCue], chinese: list[_AssCue]
) -> list[tuple[_AssCue | None, _AssCue | None]]:
    # Twice each midpoint, in integer centiseconds, keeps the inclusive
    # 0.7-second boundary exact rather than depending on binary float rounding.
    english_midpoints = [
        _ass_time_to_centiseconds(cue.start) + _ass_time_to_centiseconds(cue.end)
        for cue in english
    ]
    chinese_midpoints = [
        _ass_time_to_centiseconds(cue.start) + _ass_time_to_centiseconds(cue.end)
        for cue in chinese
    ]
    used_chinese: set[int] = set()
    merged: list[tuple[_AssCue | None, _AssCue | None]] = []
    chinese_cursor = 0

    for index, english_cue in enumerate(english):
        best_index: int | None = None
        best_difference: float | None = None
        target = english_midpoints[index]
        candidate = chinese_cursor
        while candidate < len(chinese):
            if candidate in used_chinese:
                candidate += 1
                continue
            midpoint = chinese_midpoints[candidate]
            difference = abs(midpoint - target)
            if difference <= 140 and (
                best_difference is None or difference < best_difference
            ):
                best_difference = difference
                best_index = candidate
            if midpoint > target + 140:
                break
            candidate += 1

        if best_index is None:
            merged.append((english_cue, None))
        else:
            merged.append((english_cue, chinese[best_index]))
            used_chinese.add(best_index)
            if best_index > chinese_cursor:
                chinese_cursor = best_index

    for index, chinese_cue in enumerate(chinese):
        if index not in used_chinese:
            merged.append((None, chinese_cue))

    merged.sort(
        key=lambda pair: _ass_time_to_seconds(
            pair[0].start if pair[0] is not None else pair[1].start  # type: ignore[union-attr]
        )
    )
    return merged


def _dialogue(layer: int, start: str, end: str, style: str, text: str) -> str:
    spacing = "  " if layer == 1 else " "
    return (
        f"Dialogue:{spacing}{layer},{start},{end},{style},,0,0,0,,"
        f"{text.replace(chr(10), r'\N')}"
    )


def merge_srt_to_ass(en_path: Path, zh_path: Path, out_path: Path) -> Path:
    """Merge English and Chinese SRT files using the JAG S07 timing policy."""

    english = _read_ass_cues(Path(en_path))
    chinese = _read_ass_cues(Path(zh_path))
    lines = [
        SCRIPT_INFO_BLOCK.rstrip("\n"),
        "",
        STYLES_BLOCK.rstrip("\n"),
        "",
        EVENTS_HEADER.rstrip("\n"),
    ]

    for english_cue, chinese_cue in _merge_cues(english, chinese):
        canonical = english_cue if english_cue is not None else chinese_cue
        assert canonical is not None
        if english_cue is not None and english_cue.text_lines:
            lines.append(
                _dialogue(
                    -1,
                    canonical.start,
                    canonical.end,
                    "English3",
                    "\n".join(english_cue.text_lines),
                )
            )
        if english_cue is not None:
            chinese_text = (
                "\n".join(chinese_cue.text_lines)
                if chinese_cue is not None and chinese_cue.text_lines
                else ""
            )
            lines.append(
                _dialogue(1, canonical.start, canonical.end, "Chinese3", chinese_text)
            )
        elif chinese_cue is not None and chinese_cue.text_lines:
            lines.append(
                _dialogue(
                    1,
                    canonical.start,
                    canonical.end,
                    "Chinese3",
                    "\n".join(chinese_cue.text_lines),
                )
            )

    output = Path(out_path)
    atomic_write_text(output, "\n".join((*lines, "")))
    return output


def _strip_dot(content: str) -> str:
    lines = content.splitlines()
    try:
        events_start = next(
            index for index, line in enumerate(lines) if line.strip() == "[Events]"
        )
    except StopIteration:
        return content

    events_end = len(lines)
    for index in range(events_start + 1, len(lines)):
        if lines[index].startswith("["):
            events_end = index
            break

    format_index: int | None = None
    format_fields: list[str] | None = None
    for index in range(events_start + 1, events_end):
        if lines[index].startswith("Format:"):
            format_index = index
            format_fields = [
                field.strip() for field in lines[index].split(":", 1)[1].split(",")
            ]
            break
    if format_index is None or format_fields is None:
        return content

    lowered_fields = [field.lower() for field in format_fields]
    if "style" not in lowered_fields or "text" not in lowered_fields:
        return content
    style_index = lowered_fields.index("style")
    text_index = lowered_fields.index("text")
    field_count = len(format_fields)

    for index in range(format_index + 1, events_end):
        line = lines[index]
        if not (line.startswith("Dialogue:") or line.startswith("Comment:")):
            continue
        tag, remainder = line.split(":", 1)
        parts = remainder.lstrip().split(",", field_count - 1)
        if len(parts) < field_count or not is_chinese_style(parts[style_index]):
            continue
        updated = parts[text_index].replace("。", " ")
        updated = updated.replace(r"{\i1}", "").replace(r"{\i0}", "")
        if updated.endswith("，"):
            updated = updated.rstrip("，")
        if updated != parts[text_index]:
            parts[text_index] = updated
            lines[index] = f"{tag}: " + ",".join(parts)
    return "\n".join(lines)


def _normalize_punctuation(content: str) -> str:
    replacements = (
        ("……", "..."),
        ("…", "..."),
        ("......", "..."),
        ("--", "..."),
        ("- ", "-"),
        ("————", "-"),
        ("—", "-"),
    )
    for old, new in replacements:
        content = content.replace(old, new)
    return content


def _normalize_style_names(content: str) -> str:
    return content.replace("Chinese3", "C3").replace("English3", "E3")


def _normalize_event_fields(content: str) -> str:
    return content.replace("0000,0000,0000,,", "0,0,0,,").replace(
        "Dialogue: 1,", "Dialogue:  1,"
    )


def _normalize_italics(content: str) -> str:
    content = re.sub(r"<i>", r"{\\i1}", content, flags=re.IGNORECASE)
    content = re.sub(r"</i>", r"{\\i0}", content, flags=re.IGNORECASE)
    return content


POSTPROCESS_OPERATIONS = (
    "clean_chinese_dialogue",
    "normalize_punctuation",
    "normalize_style_names",
    "normalize_event_fields",
    "normalize_italics",
    "episode_replacements",
)

_OPERATION_HANDLERS = {
    "clean_chinese_dialogue": _strip_dot,
    "normalize_punctuation": _normalize_punctuation,
    "normalize_style_names": _normalize_style_names,
    "normalize_event_fields": _normalize_event_fields,
    "normalize_italics": _normalize_italics,
}


def postprocess_ass(
    input_path: Path,
    output_path: Path,
    operations: Sequence[str],
    episode_replacements: Sequence[tuple[str, str]],
) -> Path:
    """Apply the configured deterministic operations in order."""

    content = Path(input_path).read_text(encoding="utf-8-sig")
    for operation in operations:
        if operation == "episode_replacements":
            for old, new in episode_replacements:
                if not old:
                    raise ValueError("episode replacement source must not be empty")
                content = content.replace(old, new)
            continue
        try:
            handler = _OPERATION_HANDLERS[operation]
        except KeyError as exc:
            raise ValueError(f"unsupported postprocess operation: {operation}") from exc
        content = handler(content)
    output = Path(output_path)
    atomic_write_text(output, content)
    return output


def audit_ass(path: Path) -> AssAudit:
    """Perform structural bilingual-event QA without semantic or visual claims."""

    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    parse_errors = 0
    english_events = 0
    chinese_events = 0
    empty_english_events = 0
    empty_chinese_events = 0
    non_monotonic_events = 0
    timestamp_counts: dict[tuple[str, str], list[int]] = {}

    try:
        events_start = next(
            index for index, line in enumerate(lines) if line.strip() == "[Events]"
        )
    except StopIteration:
        return AssAudit(0, 0, 0, 0, 0, 0, 1, False)

    events_end = len(lines)
    for index in range(events_start + 1, len(lines)):
        if lines[index].startswith("["):
            events_end = index
            break

    format_fields: list[str] | None = None
    format_index = events_start
    for index in range(events_start + 1, events_end):
        if lines[index].startswith("Format:"):
            format_fields = [
                field.strip().lower() for field in lines[index].split(":", 1)[1].split(",")
            ]
            format_index = index
            break
    required_fields = {"start", "end", "style", "text"}
    if format_fields is None or not required_fields.issubset(format_fields):
        return AssAudit(0, 0, 0, 0, 0, 0, 1, False)

    start_index = format_fields.index("start")
    end_index = format_fields.index("end")
    style_index = format_fields.index("style")
    text_index = format_fields.index("text")
    previous_start: float | None = None

    for line in lines[format_index + 1 : events_end]:
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(":", 1)[1].lstrip().split(",", len(format_fields) - 1)
        if len(parts) < len(format_fields):
            parse_errors += 1
            continue

        style = parts[style_index]
        language_index: int | None = None
        if is_english_style(style):
            english_events += 1
            language_index = 0
            if not parts[text_index].strip():
                empty_english_events += 1
        elif is_chinese_style(style):
            chinese_events += 1
            language_index = 1
            if not parts[text_index].strip():
                empty_chinese_events += 1

        try:
            start_seconds = _ass_time_to_seconds(parts[start_index])
            end_seconds = _ass_time_to_seconds(parts[end_index])
            if end_seconds < start_seconds:
                raise ValueError("end before start")
        except ValueError:
            parse_errors += 1
            continue

        if previous_start is not None and start_seconds < previous_start:
            non_monotonic_events += 1
        previous_start = start_seconds
        if language_index is not None:
            counts = timestamp_counts.setdefault(
                (parts[start_index].strip(), parts[end_index].strip()), [0, 0]
            )
            counts[language_index] += 1

    unpaired_timestamps = sum(
        english_count != chinese_count
        for english_count, chinese_count in timestamp_counts.values()
    )
    passed = (
        english_events > 0
        and chinese_events > 0
        and empty_english_events == 0
        and empty_chinese_events == 0
        and unpaired_timestamps == 0
        and non_monotonic_events == 0
        and parse_errors == 0
    )
    return AssAudit(
        english_events=english_events,
        chinese_events=chinese_events,
        empty_english_events=empty_english_events,
        empty_chinese_events=empty_chinese_events,
        unpaired_timestamps=unpaired_timestamps,
        non_monotonic_events=non_monotonic_events,
        parse_errors=parse_errors,
        passed=passed,
    )
