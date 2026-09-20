"""Prompt composition, glossary/story injection, and memory-maintenance prompts.

The refine and QA system prompts are composed from Markdown components on
disk (shared rules plus a stage task). The refine prompt additionally has its
"User Terminology" section rewritten with the runtime glossary and receives an
"Incremental Story Description" section built from episode memory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

from .serializers import convert_json_examples_to_format

if TYPE_CHECKING:
    from .config import PromptPaths
    from .memory import GlobalMemory


logger = logging.getLogger(__name__)

GLOSSARY_SECTION_TITLE = "User Terminology (Authoritative Glossary)"
STORY_SECTION_TITLE = "Incremental Story Description"
EMPTY_STORY_PLACEHOLDER = "(No story events revealed yet.)"

_TEMPLATE_CACHE: dict[tuple[str, int, int], str] = {}
_GLOSSARY_LINE_RE = re.compile(r"^\s*-\s+(.+?):\s*(.+?)\s*$")
_SECTION_NUMBER_RE = re.compile(r"^\d+\.\s*")
_JSON_INPUT_SENTENCE_RE = re.compile(r"based on the provided JSON input\.")
_FORMAT_SECTION_RE = re.compile(
    r"(###\s*\d+\.\s*Input/Output Format & Constraint.*?)"
    r"(- \*\*Input:\*\*.*?- \*\*STRICT ADHERENCE REQUIRED:\*\*.*?)(?=###|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_EXAMPLES_SECTION_RE = re.compile(
    r"(###\s*\d+\.\s*Few-Shot Examples[^\n]*\n)(.*?)(?=###|\Z)", re.DOTALL | re.IGNORECASE
)
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\n\]")
_FORMAT_CONSTRAINTS = {
    "xml-pair": (
        "XML-pair input",
        "- **Input:** Subtitle pairs in XML-pair format with `<pair>` tags containing "
        "`ID`, `eng`, and `chinese` fields.\n"
        "- **Output:** The same XML-pair format with corrections applied.\n"
        "- **STRICT ADHERENCE REQUIRED:** You MUST **ONLY** return the XML-pair format. "
        "No explanations, no markdown blocks (unless requested), no extra text.\n",
    ),
    "pseudo-toml": (
        "pseudo-TOML input",
        "- **Input:** Subtitle pairs in pseudo-TOML format with `[pair]` sections containing "
        "`id`, `eng`, and `chinese` fields.\n"
        "- **Output:** The same pseudo-TOML format with corrections applied.\n"
        "- **STRICT ADHERENCE REQUIRED:** You MUST **ONLY** return the pseudo-TOML format. "
        "No explanations, no markdown blocks (unless requested), no extra text.\n",
    ),
}


# Composition ----------------------------------------------------------------


def load_prompt_file(path: str | os.PathLike[str]) -> str:
    """Load one prompt component, cached per file content version."""

    full_path = os.path.abspath(os.fspath(path))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Prompt file not found: {full_path}")
    stat = os.stat(full_path)
    key = (full_path, stat.st_mtime_ns, stat.st_size)
    if key not in _TEMPLATE_CACHE:
        with open(full_path, encoding="utf-8") as handle:
            _TEMPLATE_CACHE[key] = handle.read()
    return _TEMPLATE_CACHE[key]


def compose_prompt(*components: str) -> str:
    """Join non-empty prompt components in their declared precedence order."""

    normalized: list[str] = []
    for index, component in enumerate(components):
        if not isinstance(component, str) or not component.strip():
            raise ValueError(f"prompt component {index} must be a non-empty string")
        normalized.append(component.strip())
    return "\n\n".join(normalized) + "\n"


def load_refine_prompt_template(prompt_paths: PromptPaths) -> str:
    """Compose the shared rules and the refine task."""

    return compose_prompt(
        load_prompt_file(prompt_paths.shared), load_prompt_file(prompt_paths.refine)
    )


def load_qa_prompt_template(prompt_paths: PromptPaths) -> str:
    """Compose the shared rules and the semantic-QA task."""

    return compose_prompt(load_prompt_file(prompt_paths.shared), load_prompt_file(prompt_paths.qa))


# Section handling -----------------------------------------------------------


def _normalize_section_title(title: str) -> str:
    stripped = title.strip()
    if stripped.startswith("###"):
        stripped = stripped[3:].strip()
    return _SECTION_NUMBER_RE.sub("", stripped, count=1).strip()


def _find_section(template: str, title: str) -> tuple[int, int] | None:
    """Return ``(content_start, content_end)`` for the ``###`` section titled ``title``."""

    position = 0
    content_start: int | None = None
    for line in template.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("###"):
            if content_start is not None:
                return content_start, position
            if _normalize_section_title(stripped) == title:
                content_start = position + len(line)
        position += len(line)
    if content_start is not None:
        return content_start, len(template)
    return None


def parse_template_glossary(section_content: str) -> list[dict[str, str]]:
    """Parse ``- Term: 术语`` lines into glossary entries."""

    glossary: list[dict[str, str]] = []
    for line in section_content.splitlines():
        match = _GLOSSARY_LINE_RE.match(line)
        if match:
            eng, zh = match.group(1).strip(), match.group(2).strip()
            if eng and zh:
                glossary.append({"eng": eng, "zh": zh})
    return glossary


def parse_authoritative_glossary(template: str) -> list[dict[str, str]]:
    """Return the template's authoritative glossary; the section is mandatory."""

    bounds = _find_section(template, GLOSSARY_SECTION_TITLE)
    if bounds is None:
        raise ValueError(f"prompt template has no '{GLOSSARY_SECTION_TITLE}' section")
    return parse_template_glossary(template[bounds[0] : bounds[1]])


def _merge_glossaries(
    template_glossary: list[dict[str, str]], runtime_glossary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Template order first, runtime entries override by case-insensitive key."""

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in (*template_glossary, *runtime_glossary):
        key = str(entry.get("eng", "")).casefold()
        if not key:
            continue
        if key not in merged:
            order.append(key)
        merged[key] = entry
    return [merged[key] for key in order]


def render_terminology_section(
    user_glossary: list[dict[str, Any]], learned_glossary: list[dict[str, Any]]
) -> str:
    """Render the glossary section body: authoritative entries then the learned supplement."""

    lines = [
        f"- {entry['eng']}: {entry['zh']}"
        for entry in user_glossary
        if entry.get("eng") and entry.get("zh")
    ]
    if learned_glossary:
        if lines:
            lines.append("")
        lines.append("**Learned Terminology (Supplement):**")
        for entry in learned_glossary:
            eng, zh = entry.get("eng", ""), entry.get("zh", "")
            if eng and zh:
                entry_type = entry.get("type", "")
                suffix = f" ({entry_type})" if entry_type else ""
                lines.append(f"- {eng}{suffix}: {zh}")
    return "\n".join(lines)


def render_memory_sections(memory: GlobalMemory) -> str:
    """Render memory exactly as injected into the refine prompt, for token estimation."""

    story = memory.story_description or EMPTY_STORY_PLACEHOLDER
    return (
        f"### {GLOSSARY_SECTION_TITLE}\n"
        f"{render_terminology_section(memory.user_glossary, memory.glossary)}\n\n"
        f"### {STORY_SECTION_TITLE}\n{story}\n"
    )


def _renumber_sections(template: str) -> str:
    lines: list[str] = []
    number = 0
    for line in template.splitlines():
        if line.strip().startswith("###"):
            number += 1
            lines.append(f"### {number}. {_normalize_section_title(line)}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _inject_story_block(template: str, story: str) -> str:
    bounds = _find_section(template, STORY_SECTION_TITLE)
    if bounds is not None:
        return template[: bounds[0]] + story + "\n\n" + template[bounds[1] :].lstrip()
    glossary_bounds = _find_section(template, GLOSSARY_SECTION_TITLE)
    block = f"### {STORY_SECTION_TITLE}\n{story}\n\n"
    if glossary_bounds is None:
        return template.rstrip() + "\n\n" + block.rstrip()
    end = glossary_bounds[1]
    return template[:end] + block + template[end:]


def inject_memory_into_template(template: str, memory: GlobalMemory) -> str:
    """Rewrite the glossary section with runtime memory and add the story block.

    The authoritative section is mandatory; template entries are kept and
    runtime ``user_glossary`` entries override them by key. Learned entries are
    appended as a supplement and all ``###`` sections are renumbered.
    """

    bounds = _find_section(template, GLOSSARY_SECTION_TITLE)
    if bounds is None:
        raise ValueError(f"prompt template has no '{GLOSSARY_SECTION_TITLE}' section")
    template_glossary = parse_template_glossary(template[bounds[0] : bounds[1]])
    merged = _merge_glossaries(template_glossary, memory.user_glossary)
    content = render_terminology_section(merged, memory.glossary)
    rewritten = template[: bounds[0]] + content + "\n\n" + template[bounds[1] :].lstrip()
    story = memory.story_description or EMPTY_STORY_PLACEHOLDER
    return _renumber_sections(_inject_story_block(rewritten, story))


def convert_examples_to_format(template: str, target_format: str) -> str:
    """Rewrite format wording and few-shot JSON examples for a non-JSON representation."""

    normalized = target_format.lower()
    if normalized == "json":
        return template
    if normalized not in _FORMAT_CONSTRAINTS:
        raise ValueError(f"unsupported intermediate representation: {target_format}")
    input_description, constraint = _FORMAT_CONSTRAINTS[normalized]

    template = _JSON_INPUT_SENTENCE_RE.sub(
        f"based on the provided {input_description}.", template, count=1
    )
    format_match = _FORMAT_SECTION_RE.search(template)
    if format_match:
        template = (
            template[: format_match.start()]
            + format_match.group(1)
            + constraint
            + template[format_match.end() :]
        )
    else:
        logger.warning("Prompt template has no 'Input/Output Format & Constraint' section")

    examples_match = _EXAMPLES_SECTION_RE.search(template)
    if not examples_match:
        logger.warning("Prompt template has no 'Few-Shot Examples' section")
        return template

    def convert(match: re.Match[str]) -> str:
        return convert_json_examples_to_format(match.group(0), normalized)

    converted = _JSON_ARRAY_RE.sub(convert, examples_match.group(2))
    return (
        template[: examples_match.start()]
        + examples_match.group(1)
        + converted
        + template[examples_match.end() :]
    )


def build_refine_system_prompt(
    memory: GlobalMemory, prompt_paths: PromptPaths, representation: str
) -> str:
    """Compose, inject memory into, and format-convert the refine system prompt."""

    template = inject_memory_into_template(load_refine_prompt_template(prompt_paths), memory)
    return convert_examples_to_format(template, representation)


# Memory maintenance prompts -------------------------------------------------


MEMORY_UPDATE_SYSTEM_PROMPT_TEMPLATE = """You update episode memory from paired English and Chinese subtitles.

Update both the learned glossary and the incremental story description. Follow these rules:
- Focus on proper nouns: people, places, organizations, ships, military units, project/operation code names, legal statute names, show or work titles, and stable acronyms (e.g., JAG, NCIS)
- Always extract all person names (character names) you can confidently identify, even if they appear only once in the current chunk.
- Include keywords that need unified translations across the episode
- Ignore generic conversational words and function words even if they are capitalized at the beginning of a sentence. This does not apply to names (e.g., "Chris", "Benny", "Bryer").
- Do not invent translations or entries if the Chinese counterpart cannot be determined confidently
- For every glossary item output: eng (trimmed, original casing), zh (trimmed), type (one of person/place/organization/title/acronym/unit/ship/project/law/other), confidence (0.0-1.0), evidence_ids (list of up to 5 subtitle ids where the term appears)
- Only keep entries with confidence >= {min_conf}
- Treat the previous story description as the established account of this same episode
- Rewrite story_description as a concise cumulative description of only what has been revealed so far
- Preserve relevant prior events while adding newly revealed events, character relationships or identities, and current states
- Do not guess motives, identities, relationships, outcomes, or off-screen events
- Output strictly one JSON object with exactly these keys, without explanations or Markdown:
{{"glossary": [{{"eng": "...", "zh": "...", "type": "...", "confidence": 0.0, "evidence_ids": [1]}}], "story_description": "..."}}
"""

MEMORY_UPDATE_USER_TEMPLATE = """Update the episode memory from the corrected subtitle pairs below.

You will also receive an optional "user glossary" that already defines some eng→zh mappings.
- Do NOT output entries whose eng already appears in the user glossary.
- Do NOT output any entry whose zh conflicts with the user glossary for the same eng.

Previous story description:
{previous_story_description}

Corrected subtitle pairs (JSON):
{pairs_json}

User glossary (JSON array, may be empty):
{user_glossary_json}

Return ONLY the JSON object specified by the system instructions.
"""

MEMORY_COMPRESSION_SYSTEM_PROMPT = """You compress episode memory used for subtitle refinement.

Your task is to:
1. Keep all unique learned terminology mappings while merging duplicates
2. Shorten story_description without inventing events, identities, relationships, or states
3. Return exactly the two fields shown below

Return a compressed version in the same JSON format:
{
  "glossary": [{"eng": "...", "zh": "...", "type": "..."}],
  "story_description": "..."
}

Be aggressive in compression but preserve all unique terminology mappings."""


def build_memory_update_system_prompt(min_confidence: float) -> str:
    """System prompt for one incremental memory update, showing the confidence threshold."""

    return MEMORY_UPDATE_SYSTEM_PROMPT_TEMPLATE.format(min_conf=min_confidence)


def build_memory_update_user_prompt(
    *,
    pairs_json: str,
    user_glossary_json: str,
    previous_story_description: str,
) -> str:
    """User prompt carrying the corrected pairs, the user glossary, and the prior story."""

    return MEMORY_UPDATE_USER_TEMPLATE.format(
        pairs_json=pairs_json,
        user_glossary_json=user_glossary_json,
        previous_story_description=previous_story_description,
    )


def build_memory_compression_prompt(memory: GlobalMemory, target_tokens: int) -> str:
    """User prompt asking the model to compress learned glossary and story only."""

    payload = {"glossary": memory.glossary, "story_description": memory.story_description}
    return (
        f"Current memory is too large. Please compress it to approximately {target_tokens} "
        "tokens or less.\n\nCurrent memory:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Return ONLY the compressed JSON object, no explanations."
    )
