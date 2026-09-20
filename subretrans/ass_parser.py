"""ASS subtitle parsing, bilingual pair matching, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from .fsutil import atomic_write_text
from .pairs import SubtitlePair


ENGLISH_STYLE_NAMES = {"e3"}
CHINESE_STYLE_NAMES = {"c3"}


@dataclass
class AssLine:
    """One ``Dialogue:`` event from the ``[Events]`` section.

    ``id`` is the sequential index of the event; ``text`` keeps every ASS
    override tag (``{\\i1}``, ``\\N``) verbatim.
    """

    id: int
    raw: str
    layer: str
    start: str
    end: str
    style: str
    name: str
    margin_l: str
    margin_r: str
    margin_v: str
    effect: str
    text: str


def parse_dialogue_line(line: str, line_id: int) -> AssLine | None:
    """Parse ``Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text``."""

    if not line.startswith("Dialogue:"):
        return None
    parts = line[len("Dialogue:") :].strip().split(",", 9)
    if len(parts) < 10:
        return None
    return AssLine(
        id=line_id,
        raw=line,
        layer=parts[0],
        start=parts[1],
        end=parts[2],
        style=parts[3],
        name=parts[4],
        margin_l=parts[5],
        margin_r=parts[6],
        margin_v=parts[7],
        effect=parts[8],
        text=parts[9],
    )


def is_english_style(style: str) -> bool:
    """Whether an ASS style name denotes the English track."""

    style_lower = style.strip().lower()
    return "english" in style_lower or style_lower in ENGLISH_STYLE_NAMES


def is_chinese_style(style: str) -> bool:
    """Whether an ASS style name denotes the Chinese track."""

    style_lower = style.strip().lower()
    return "chinese" in style_lower or style_lower in CHINESE_STYLE_NAMES


def parse_ass_file(file_path: str | PathLike[str]) -> tuple[str, list[AssLine]]:
    """Split an ASS file into its header text and parsed dialogue events."""

    with open(file_path, encoding="utf-8-sig") as handle:
        lines = handle.readlines()

    header_lines: list[str] = []
    dialogue_lines: list[AssLine] = []
    in_events = False
    line_id = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "[Events]":
            in_events = True
            header_lines.append(line)
            continue
        if in_events and stripped.startswith("Dialogue:"):
            ass_line = parse_dialogue_line(stripped, line_id)
            if ass_line is not None:
                dialogue_lines.append(ass_line)
                line_id += 1
        elif not in_events or stripped.startswith("Format:"):
            header_lines.append(line)

    return "".join(header_lines), dialogue_lines


def build_pairs_from_ass_lines(ass_lines: list[AssLine]) -> list[SubtitlePair]:
    """Pair English and Chinese events that share start and end timestamps.

    Pairs are numbered sequentially in timestamp order; a pair needs an
    English event and may have an empty Chinese side.
    """

    timestamp_groups: dict[tuple[str, str], list[AssLine]] = {}
    for line in ass_lines:
        timestamp_groups.setdefault((line.start, line.end), []).append(line)

    pairs: list[SubtitlePair] = []
    for _, group_lines in sorted(timestamp_groups.items()):
        eng_line: AssLine | None = None
        chinese_line: AssLine | None = None
        for line in group_lines:
            if is_english_style(line.style):
                eng_line = line
            elif is_chinese_style(line.style):
                chinese_line = line
        if eng_line is None:
            continue
        pairs.append(
            SubtitlePair(
                id=len(pairs),
                eng=eng_line.text,
                chinese=chinese_line.text if chinese_line else "",
                meta={
                    "start": eng_line.start,
                    "end": eng_line.end,
                    "style_eng": eng_line.style,
                    "style_chinese": chinese_line.style if chinese_line else "",
                    "layer": eng_line.layer,
                    "name": eng_line.name,
                    "margin_l": eng_line.margin_l,
                    "margin_r": eng_line.margin_r,
                    "margin_v": eng_line.margin_v,
                    "effect": eng_line.effect,
                    "eng_line_id": eng_line.id,
                    "chinese_line_id": chinese_line.id if chinese_line else -1,
                },
            )
        )
    return pairs


def apply_pairs_to_ass_lines(
    ass_lines: list[AssLine], pairs: list[SubtitlePair]
) -> list[AssLine]:
    """Write corrected pair text back onto the events they were built from."""

    line_map = {line.id: line for line in ass_lines}
    for pair in pairs:
        if not pair.meta:
            continue
        eng_line_id = pair.meta.get("eng_line_id")
        if eng_line_id is not None and eng_line_id in line_map:
            line_map[eng_line_id].text = pair.eng
        chinese_line_id = pair.meta.get("chinese_line_id")
        if (
            chinese_line_id is not None
            and chinese_line_id >= 0
            and chinese_line_id in line_map
        ):
            line_map[chinese_line_id].text = pair.chinese
    return sorted(line_map.values(), key=lambda line: line.id)


def render_ass_file(header: str, ass_lines: list[AssLine]) -> str:
    """Render the header followed by every dialogue event."""

    rendered = [header]
    for line in ass_lines:
        rendered.append(
            f"Dialogue: {line.layer},{line.start},{line.end},{line.style},"
            f"{line.name},{line.margin_l},{line.margin_r},{line.margin_v},"
            f"{line.effect},{line.text}\n"
        )
    return "".join(rendered)


def write_ass_file(file_path: str | PathLike[str], content: str) -> None:
    """Atomically write ASS content with a UTF-8 BOM."""

    atomic_write_text(file_path, content, encoding="utf-8-sig")
