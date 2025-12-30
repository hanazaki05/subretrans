"""
ASS subtitle file parser and generator.

Handles parsing of .ass subtitle files, extracting dialogue lines,
building subtitle pairs, and generating output .ass files.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import re
import os
import tempfile

from pairs import SubtitlePair


@dataclass
class AssLine:
    """
    Represents a single Dialogue line from an ASS subtitle file.

    Attributes:
        id: Sequential index in the Events section
        raw: Original raw line text (for debugging)
        layer: Layer field from Dialogue line
        start: Start timestamp (e.g., "0:00:01.00")
        end: End timestamp
        style: Style name (e.g., "English3", "Chinese3")
        name: Name field
        margin_l: Left margin
        margin_r: Right margin
        margin_v: Vertical margin
        effect: Effect field
        text: Text content (may contain ASS tags like {\\i1}, \\N, etc.)
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


def parse_dialogue_line(line: str, line_id: int) -> Optional[AssLine]:
    """
    Parse a single Dialogue line from an ASS file.

    Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text

    Args:
        line: Raw dialogue line from ASS file
        line_id: Sequential ID for this line

    Returns:
        AssLine object if parsing succeeds, None otherwise
    """
    if not line.startswith("Dialogue:"):
        return None

    # Remove "Dialogue: " prefix
    content = line[9:].strip()

    # Split by comma, but only the first 9 commas (Text field may contain commas)
    parts = content.split(",", 9)

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
        text=parts[9]
    )


def parse_ass_file(file_path: str) -> Tuple[str, List[AssLine]]:
    """
    Parse an ASS subtitle file into header and dialogue lines.

    Args:
        file_path: Path to the .ass file

    Returns:
        Tuple of (header_text, list of AssLine objects)
        header_text includes everything before the Dialogue lines
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_lines = []
    dialogue_lines = []
    in_events = False
    line_id = 0

    for line in lines:
        stripped = line.strip()

        # Check if we're in the Events section
        if stripped == "[Events]":
            in_events = True
            header_lines.append(line)
            continue

        # If we're in Events and hit a Dialogue line, parse it
        if in_events and stripped.startswith("Dialogue:"):
            ass_line = parse_dialogue_line(stripped, line_id)
            if ass_line:
                dialogue_lines.append(ass_line)
                line_id += 1
        else:
            # Everything else goes to header
            if not in_events or stripped.startswith("Format:"):
                header_lines.append(line)

    header_text = "".join(header_lines)
    return header_text, dialogue_lines


def build_pairs_from_ass_lines(ass_lines: List[AssLine]) -> List[SubtitlePair]:
    """
    Build SubtitlePair list from AssLine list by matching English and Chinese lines.

    Matches lines based on:
    - Same start/end timestamps
    - One line with "English" in style name, another with "Chinese" in style name

    Args:
        ass_lines: List of parsed AssLine objects

    Returns:
        List of SubtitlePair objects
    """
    # Group lines by timestamp (start, end)
    timestamp_groups: Dict[Tuple[str, str], List[AssLine]] = {}

    for line in ass_lines:
        key = (line.start, line.end)
        if key not in timestamp_groups:
            timestamp_groups[key] = []
        timestamp_groups[key].append(line)

    pairs = []
    pair_id = 0

    # Process each timestamp group
    for timestamp, group_lines in sorted(timestamp_groups.items()):
        eng_line = None
        chinese_line = None

        # Find English and Chinese lines in this group
        for line in group_lines:
            style_lower = line.style.lower()
            if "english" in style_lower:
                eng_line = line
            elif "chinese" in style_lower:
                chinese_line = line

        # Create pair if we have at least English (Chinese can be empty for some cases)
        if eng_line:
            chinese_text = chinese_line.text if chinese_line else ""
            eng_text = eng_line.text

            # Store metadata for later reconstruction
            meta = {
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
                "chinese_line_id": chinese_line.id if chinese_line else -1
            }

            pair = SubtitlePair(
                id=pair_id,
                eng=eng_text,
                chinese=chinese_text,
                meta=meta
            )
            pairs.append(pair)
            pair_id += 1

    return pairs


def apply_pairs_to_ass_lines(ass_lines: List[AssLine], pairs: List[SubtitlePair]) -> List[AssLine]:
    """
    Apply corrected subtitle pairs back to AssLine list.

    Args:
        ass_lines: Original list of AssLine objects
        pairs: List of corrected SubtitlePair objects

    Returns:
        Updated list of AssLine objects with corrected text
    """
    # Create a mapping from line_id to AssLine for fast lookup
    line_map = {line.id: line for line in ass_lines}

    # Apply corrections from pairs
    for pair in pairs:
        if pair.meta:
            # Update English line
            eng_line_id = pair.meta.get("eng_line_id")
            if eng_line_id is not None and eng_line_id in line_map:
                line_map[eng_line_id].text = pair.eng

            # Update Chinese line
            chinese_line_id = pair.meta.get("chinese_line_id")
            if chinese_line_id is not None and chinese_line_id >= 0 and chinese_line_id in line_map:
                line_map[chinese_line_id].text = pair.chinese

    # Return sorted by original ID
    return sorted(line_map.values(), key=lambda x: x.id)


def render_ass_file(header: str, ass_lines: List[AssLine]) -> str:
    """
    Render ASS file content from header and dialogue lines.

    Args:
        header: Header text (everything before Dialogue lines)
        ass_lines: List of AssLine objects

    Returns:
        Complete ASS file content as string
    """
    lines = [header]

    # Add dialogue lines
    for line in ass_lines:
        dialogue = (
            f"Dialogue: {line.layer},{line.start},{line.end},{line.style},"
            f"{line.name},{line.margin_l},{line.margin_r},{line.margin_v},"
            f"{line.effect},{line.text}\n"
        )
        lines.append(dialogue)

    return "".join(lines)


def write_ass_file(file_path: str, content: str) -> None:
    """
    Write ASS file content to disk.

    Args:
        file_path: Output file path
        content: Complete ASS file content
    """
    # Atomic write to avoid corrupting output on crashes/interruption:
    # write to a temp file in the same directory, then replace.
    output_dir = os.path.dirname(os.path.abspath(file_path)) or "."
    base_name = os.path.basename(file_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base_name}.tmp.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Some filesystems may not support fsync; atomic replace still prevents truncation.
                pass
        os.replace(tmp_path, file_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
