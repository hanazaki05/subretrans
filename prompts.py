"""Prompt templates and helpers for LLM interactions.

Contains system prompts for subtitle refinement, memory compression, and
terminology extraction, plus utilities to inject user customization.
"""

import os
import re
import json
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional

if TYPE_CHECKING:
    from memory import GlobalMemory


_USER_INSTRUCTION: str = ""

# Cache for loaded template
_TEMPLATE_CACHE: Dict[str, str] = {}


def set_user_instruction(text: str) -> None:
    """Set extra user-defined instructions to append after BASE_SYSTEM_PROMPT.

    This is populated early in the run (e.g., from custom_main_prompt.md) and
    is treated as high-priority guidance for the main model.
    """

    global _USER_INSTRUCTION
    _USER_INSTRUCTION = text.strip() if text else ""


# Base system prompt for subtitle refinement (split into core and critical tail)
BASE_SYSTEM_PROMPT_CORE = """You are a professional subtitle editor specializing in bilingual (English-Chinese) subtitle refinement.

Your task is to review and correct subtitle pairs while following these rules:

**English Subtitle Rules:**
1. Perform recaptioning fixes for words and nouns when they should be capitalized
2. Fix spacing and ending punctuation
3. DO NOT modify wording, phrasing, or meaning in any way
4. Preserve ALL ASS formatting tags (e.g., {\\i1}, {\\b1}, \\N) exactly as they appear
5. Ensure proper sentence capitalization 
6. Add missing periods at the end of complete sentences
7. Fix obvious spacing issues (e.g., "Hello,world" → "Hello, world")

**Chinese Subtitle Rules:**
Optimized Chinese Subtitle Rules:
  1.Ensure translation accuracy and natural flow
  2.Maintain consistency with context and character voices
  3.Use conversational, natural Chinese (avoid overly formal language)
  4.Preserve all ASS formatting tags exactly as they appear
Terminology & Name Handling (Strict Order):
  5. PRIORITY 1 (Glossary Override): If the terminology glossary explicitly defines a term or name, strictly follow that definition regardless of the language (English or Chinese). This rule overrides all others.
  6. PRIORITY 2 (Acronyms): Keep initial-based nicknames (e.g., "AJ", "DJ", "CC") in their original English form.
  7. PRIORITY 3 (Standard Translation): For all other personal names (e.g., Chris, Benny), translate them into standard Mandarin Chinese transliterations (标准普通话音译). Ensure the translation chosen is consistent across all subtitles.
Other Rules:
  8. Fix any awkward or unnatural phrasing
  9. **If a sentence ends with a period or comma, remove that period or comma**

Examples:
  Input: [
    {"id": 1, "eng": "Did you talk to chris?", "chinese": "你克里斯说话了吗。"},
    {"id": 2, "eng": "AJ is on the phone.{\\i1} I need to go.", "chinese": "AJ在电话上。{\\i1} 我需要走了。"},
    {"id": 3, "eng": "we need to check the ios version", "chinese": "我们需要检查ios版本"},
    {"id": 4, "eng": "i told benny, let's go.", "chinese": "我告诉了本尼，我们走吧。"}
    ]
  Output: [
    {"id": 1, "eng": "Did you talk to Chris?", "chinese": "你和克里斯说话了吗"},
    {"id": 2, "eng": "AJ is on the phone.{\\i1} I need to go.", "chinese": "AJ在电话上。{\\i1} 我得走了"},
    {"id": 3, "eng": "We need to check the iOS version.", "chinese": "我们需要检查 iOS 版本"},
    {"id": 4, "eng": "I told Benny. Let's go.", "chinese": "我告诉了本尼。我们走吧"}
   ]

**Important Guidelines:**
- Maintain the subtitle's original intent and meaning
- Keep subtitles concise and readable
- Preserve timing and formatting information
- DO NOT add explanations or comments
- Output ONLY valid JSON"""

BASE_SYSTEM_PROMPT_CRITICAL = """STRICT ADHERENCE REQUIRED: You MUST follow the Input/Output format exactly as defined below. 
**Input/Output Format:**
You will receive a JSON array of subtitle pairs. Each pair has:
- "id": unique identifier
- "eng": English subtitle text (with ASS tags)
- "chinese": Chinese subtitle text (with ASS tags)

Return a JSON array with the SAME structure, containing your corrections.

Example:
  Input: [{"id": 0, "eng": "hello world", "chinese": "你好世界"}]
  Output: [{"id": 0, "eng": "Hello world.", "chinese": "你好，世界"}]

**Additional Few-Shot Examples for Clarification:**
  Input: [
    {"id": 1, "eng": "Did you talk to chris?", "chinese": "你克里斯说话了吗。"},
    {"id": 2, "eng": "AJ is on the phone.{\\i1} I need to go.", "chinese": "AJ在电话上。{\\i1} 我需要走了。"},
    {"id": 3, "eng": "we need to check the ios version", "chinese": "我们需要检查ios版本"},
    {"id": 4, "eng": "i told benny, let's go.", "chinese": "我告诉了本尼，我们走吧。"}
    ]
  Output: [
    {"id": 1, "eng": "Did you talk to Chris?", "chinese": "你和克里斯说话了吗"},
    {"id": 2, "eng": "AJ is on the phone.{\\i1} I need to go.", "chinese": "AJ在电话上。{\\i1} 我得走了"},
    {"id": 3, "eng": "We need to check the iOS version.", "chinese": "我们需要检查 iOS 版本"},
    {"id": 4, "eng": "I told Benny. Let's go.", "chinese": "我告诉了本尼。我们走吧"}
   ]

**CRITICAL:** Return ONLY the JSON array. No explanations, no markdown, no extra text."""


def build_memory_section(global_memory: 'GlobalMemory') -> str:
    """
    Build the memory section to append to system prompt.

    Args:
        global_memory: GlobalMemory object containing glossary and style notes

    Returns:
        Formatted memory section as string
    """
    if not global_memory:
        return ""

    sections = []

    # Add user glossary first (highest priority)
    user_glossary = getattr(global_memory, "user_glossary", None)
    if user_glossary:
        sections.append("\n\n**User Terminology (authoritative):**")
        for entry in user_glossary:
            eng = entry.get("eng", "")
            zh = entry.get("zh", "")
            if eng and zh:
                sections.append(f"- {eng}: {zh}")

    # Add learned glossary if present
    if global_memory.glossary:
        sections.append("\n\n**Learned Terminology (supplement):**")
        for entry in global_memory.glossary:
            eng = entry.get("eng", "")
            zh = entry.get("zh", "")
            entry_type = entry.get("type", "")
            if eng and zh:
                type_str = f" ({entry_type})" if entry_type else ""
                sections.append(f"- {eng}{type_str}: {zh}")

    # Add style notes if present
    if global_memory.style_notes:
        sections.append(f"\n\n**Style Guidelines:**\n{global_memory.style_notes}")

    # Add summary if present
    if global_memory.summary:
        sections.append(f"\n\n**Context:**\n{global_memory.summary}")

    return "".join(sections)


def build_system_prompt_legacy(global_memory: 'GlobalMemory') -> str:
    """
    Build complete system prompt including memory section (legacy version).

    This is the old implementation kept for backward compatibility.
    Use build_system_prompt() with config for the new template-based approach.

    Args:
        global_memory: GlobalMemory object

    Returns:
        Complete system prompt with memory
    """
    # Order: base core rules → user instructions → memory section → CRITICAL tail
    prompt = BASE_SYSTEM_PROMPT_CORE

    if _USER_INSTRUCTION:
        prompt += "\n\n" + _USER_INSTRUCTION

    if global_memory:
        memory_section = build_memory_section(global_memory)
        if memory_section:
            prompt += memory_section

    # Ensure the CRITICAL output constraint is always the last part
    prompt += "\n\n" + BASE_SYSTEM_PROMPT_CRITICAL

    return prompt


# ============================================================================
# NEW TEMPLATE-BASED PROMPT CONSTRUCTION (plan3.md)
# ============================================================================


def load_main_prompt_template(config) -> str:
    """
    Load main prompt template from config.user_prompt_path.

    Args:
        config: Config object with user_prompt_path attribute

    Returns:
        Template text as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    global _TEMPLATE_CACHE

    prompt_path = getattr(config, "user_prompt_path", "main_prompt.md")

    # Resolve relative path
    if not os.path.isabs(prompt_path):
        # Try relative to current working directory first
        if os.path.exists(prompt_path):
            full_path = os.path.abspath(prompt_path)
        else:
            # Try relative to this file's directory (project root)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, prompt_path)
    else:
        full_path = prompt_path

    # Check cache
    if full_path in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[full_path]

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Template file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        template = f.read()

    _TEMPLATE_CACHE[full_path] = template
    return template


def _normalize_section_title(title: str) -> str:
    """
    Normalize a section title by removing leading number and dot.

    Examples:
        "### 4. User Terminology (Authoritative Glossary)" -> "User Terminology (Authoritative Glossary)"
        "### User Terminology (Authoritative Glossary)" -> "User Terminology (Authoritative Glossary)"
        "### 10. User Terminology (Authoritative Glossary)" -> "User Terminology (Authoritative Glossary)"
    """
    # Remove "### " prefix
    title = title.strip()
    if title.startswith("###"):
        title = title[3:].strip()

    # Remove leading number and dot (e.g., "4. " or "10. ")
    match = re.match(r"^\d+\.\s*", title)
    if match:
        title = title[match.end():]

    return title.strip()


def _parse_template_glossary(section_content: str) -> List[Dict[str, str]]:
    """
    Parse glossary entries from a template section content.

    Parses lines like:
        - Term: 术语
        - Another Term: 另一个术语

    Args:
        section_content: The content of the terminology section

    Returns:
        List of {"eng": ..., "zh": ...} dictionaries
    """
    glossary = []
    # Match "- eng: zh" pattern
    pattern = re.compile(r"^\s*-\s+(.+?):\s*(.+?)\s*$")

    for line in section_content.splitlines():
        match = pattern.match(line)
        if match:
            eng = match.group(1).strip()
            zh = match.group(2).strip()
            if eng and zh:
                glossary.append({"eng": eng, "zh": zh})

    return glossary


def _find_section_boundaries(template: str, target_title: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Find the start and end positions of a section in the template.

    Args:
        template: Full template text
        target_title: Normalized section title to find (e.g., "User Terminology (Authoritative Glossary)")

    Returns:
        Tuple of (section_start, section_end, full_header_line)
        - section_start: Position after the header line (start of content)
        - section_end: Position of next ### header or end of file
        - full_header_line: The original header line (e.g., "### 4. User Terminology...")
        Returns (None, None, None) if not found
    """
    lines = template.splitlines(keepends=True)
    section_start = None
    section_header = None
    current_pos = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("###"):
            normalized = _normalize_section_title(stripped)
            if normalized == target_title:
                # Found the target section
                section_header = stripped
                section_start = current_pos + len(line)
            elif section_start is not None:
                # Found next section header - this is the end
                return section_start, current_pos, section_header

        current_pos += len(line)

    # If section was found but no next header, end is at file end
    if section_start is not None:
        return section_start, len(template), section_header

    return None, None, None


def _merge_glossaries(template_glossary: List[Dict[str, str]],
                      user_glossary: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Merge template glossary with user glossary.

    User glossary entries take precedence over template entries (by eng key, case-insensitive).

    Args:
        template_glossary: Glossary from template file
        user_glossary: Glossary from GlobalMemory.user_glossary

    Returns:
        Merged glossary list
    """
    # Use casefold for case-insensitive key comparison
    merged = {}

    # Add template entries first
    for entry in template_glossary:
        key = entry["eng"].casefold()
        merged[key] = entry

    # User entries override template entries
    for entry in user_glossary:
        key = entry["eng"].casefold()
        merged[key] = entry

    # Return in order (template order preserved, then any new user entries)
    result = []
    seen_keys = set()

    # First, add entries in template order (with possible user overrides)
    for entry in template_glossary:
        key = entry["eng"].casefold()
        if key not in seen_keys:
            result.append(merged[key])
            seen_keys.add(key)

    # Then add any user entries not in template
    for entry in user_glossary:
        key = entry["eng"].casefold()
        if key not in seen_keys:
            result.append(entry)
            seen_keys.add(key)

    return result


def _build_terminology_section(user_glossary: List[Dict[str, str]],
                               learned_glossary: List[Dict[str, str]]) -> str:
    """
    Build the terminology section content with both user and learned glossaries.

    Args:
        user_glossary: Merged authoritative glossary (template + runtime user_glossary)
        learned_glossary: Learned terminology from GlobalMemory.glossary

    Returns:
        Formatted terminology section content (without header)
    """
    lines = []

    # User glossary (authoritative)
    if user_glossary:
        for entry in user_glossary:
            eng = entry.get("eng", "")
            zh = entry.get("zh", "")
            if eng and zh:
                lines.append(f"- {eng}: {zh}")

    # Add learned glossary as supplement if present
    if learned_glossary:
        if lines:
            lines.append("")  # Blank line before supplement section
        lines.append("**Learned Terminology (Supplement):**")
        for entry in learned_glossary:
            eng = entry.get("eng", "")
            zh = entry.get("zh", "")
            entry_type = entry.get("type", "")
            if eng and zh:
                type_str = f" ({entry_type})" if entry_type else ""
                lines.append(f"- {eng}{type_str}: {zh}")

    return "\n".join(lines)


def _renumber_sections(template: str) -> str:
    """
    Renumber all ### sections in the template sequentially.

    Args:
        template: Template text with potentially non-sequential section numbers

    Returns:
        Template with sections renumbered 1, 2, 3, ...
    """
    lines = template.splitlines()
    result_lines = []
    section_num = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("###"):
            section_num += 1
            # Extract the title part (after any existing number)
            normalized_title = _normalize_section_title(stripped)
            # Rebuild with new number
            result_lines.append(f"### {section_num}. {normalized_title}")
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def inject_memory_into_template(template: str, global_memory: 'GlobalMemory') -> str:
    """
    Inject GlobalMemory terminology into the template's User Terminology section.

    This implements the plan3.md strategy:
    1. Find "### X. User Terminology (Authoritative Glossary)" section
    2. Parse existing glossary entries from template
    3. Merge with GlobalMemory.user_glossary (runtime entries take precedence)
    4. Add GlobalMemory.glossary as "Learned Terminology (Supplement)"
    5. Replace the section content
    6. Renumber all sections

    Args:
        template: Main prompt template text
        global_memory: GlobalMemory object with terminology

    Returns:
        Template with injected terminology
    """
    TARGET_SECTION = "User Terminology (Authoritative Glossary)"

    # Find section boundaries
    section_start, section_end, header_line = _find_section_boundaries(template, TARGET_SECTION)

    if section_start is None:
        # Section not found - return template as-is with warning
        print(f"  Warning: '{TARGET_SECTION}' section not found in template")
        return template

    # Get original section content
    original_content = template[section_start:section_end]

    # Parse template glossary
    template_glossary = _parse_template_glossary(original_content)

    # Get memory glossaries
    user_glossary = getattr(global_memory, "user_glossary", []) or []
    learned_glossary = getattr(global_memory, "glossary", []) or []

    # Merge template + user glossary
    merged_user_glossary = _merge_glossaries(template_glossary, user_glossary)

    # Build new section content
    new_content = _build_terminology_section(merged_user_glossary, learned_glossary)

    # Reconstruct template
    # Keep the header line, replace content up to next section
    new_template = template[:section_start] + "\n" + new_content + "\n\n" + template[section_end:].lstrip()

    # Renumber sections
    new_template = _renumber_sections(new_template)

    return new_template


def convert_examples_to_format(template: str, target_format: str) -> str:
    """
    Convert JSON examples in Few-Shot Examples section to target format.
    Also updates the "Input/Output Format & Constraint" section to match the format.

    Args:
        template: Template text containing JSON examples
        target_format: Target format (json, xml-pair, pseudo-toml)

    Returns:
        Template with examples and format constraints converted to target format
    """
    # Only convert if not JSON
    if target_format.lower() == "json":
        return template

    # Import here to avoid circular dependency
    try:
        import sys
        import os
        # Add experiment directory to path
        experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment")
        if experiment_dir not in sys.path:
            sys.path.insert(0, experiment_dir)

        from serializers import convert_json_examples_to_format
    except ImportError as e:
        print(f"  Warning: Could not import serializers: {e}")
        print("  Returning template with JSON examples unchanged")
        return template

    # Step 1: Update the "Input/Output Format & Constraint" section
    format_constraint_pattern = r'(###\s*\d+\.\s*Input/Output Format & Constraint.*?)(- \*\*Input:\*\*.*?- \*\*STRICT ADHERENCE REQUIRED:\*\*.*?)(?=###|\Z)'
    format_match = re.search(format_constraint_pattern, template, re.DOTALL | re.IGNORECASE)

    if format_match:
        header = format_match.group(1)

        # Generate format-specific constraint text
        if target_format.lower() == "xml-pair":
            new_constraint = """- **Input:** Subtitle pairs in XML-pair format with `<pair>` tags containing `ID`, `eng`, and `chinese` fields.
- **Output:** The same XML-pair format with corrections applied.
- **STRICT ADHERENCE REQUIRED:** You MUST **ONLY** return the XML-pair format. No explanations, no markdown blocks (unless requested), no extra text.
"""
        elif target_format.lower() == "pseudo-toml":
            new_constraint = """- **Input:** Subtitle pairs in pseudo-TOML format with `[pair]` sections containing `id`, `eng`, and `chinese` fields.
- **Output:** The same pseudo-TOML format with corrections applied.
- **STRICT ADHERENCE REQUIRED:** You MUST **ONLY** return the pseudo-TOML format. No explanations, no markdown blocks (unless requested), no extra text.
"""
        else:
            new_constraint = format_match.group(2)  # Keep original if unknown format

        # Replace in template
        new_section = header + new_constraint
        template = template[:format_match.start()] + new_section + template[format_match.end():]

    # Step 2: Find Few-Shot Examples section
    section_pattern = r'(###\s*\d+\.\s*Few-Shot Examples.*?)(Input:.*?Output:.*?)(?=###|\Z)'
    match = re.search(section_pattern, template, re.DOTALL | re.IGNORECASE)

    if not match:
        print("  Warning: Few-Shot Examples section not found in template")
        return template

    header = match.group(1)
    examples_block = match.group(2)

    # Extract JSON arrays from examples
    json_pattern = r'\[.*?\]'
    json_matches = re.findall(json_pattern, examples_block, re.DOTALL)

    if len(json_matches) < 2:
        print("  Warning: Could not find Input/Output JSON examples")
        return template

    input_json = json_matches[0]
    output_json = json_matches[1]

    try:
        # Convert examples to target format
        input_converted = convert_json_examples_to_format(input_json, target_format)
        output_converted = convert_json_examples_to_format(output_json, target_format)

        # Rebuild examples section
        new_examples_block = f"Input:\n{input_converted}\n\nOutput:\n{output_converted}\n"

        # Replace in template
        new_section = header + new_examples_block
        new_template = template[:match.start()] + new_section + template[match.end():]

        return new_template

    except Exception as e:
        print(f"  Warning: Failed to convert examples: {e}")
        print("  Returning template with JSON examples unchanged")
        return template


def build_system_prompt(global_memory: 'GlobalMemory', config=None) -> str:
    """
    Build complete system prompt with memory injection.

    If config is provided and config.user_prompt_path exists, uses the new
    template-based approach from plan3.md. Otherwise falls back to legacy behavior.

    Args:
        global_memory: GlobalMemory object
        config: Optional config object with user_prompt_path

    Returns:
        Complete system prompt with memory
    """
    # Try new template-based approach if config is provided
    if config is not None:
        prompt_path = getattr(config, "user_prompt_path", None)
        if prompt_path:
            try:
                template = load_main_prompt_template(config)
                template = inject_memory_into_template(template, global_memory)

                # Convert examples to target format if specified
                intermediate_format = getattr(config, "intermediate_format", "json")
                if intermediate_format and intermediate_format.lower() != "json":
                    template = convert_examples_to_format(template, intermediate_format)

                return template
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                print("  Falling back to legacy prompt construction")
            except Exception as e:
                print(f"  Warning: Failed to load template: {e}")
                print("  Falling back to legacy prompt construction")

    # Fallback to legacy behavior
    return build_system_prompt_legacy(global_memory)


# System prompt for memory compression
MEMORY_COMPRESSION_SYSTEM_PROMPT = """You are a helpful assistant that compresses terminology and style information for subtitle refinement.

Your task is to:
1. Keep ALL terminology entries (people, places, organizations, technical terms)
2. Merge duplicate entries
3. Remove less important or redundant information
4. Keep style notes concise but informative
5. Maintain the overall structure

Return a compressed version in the same JSON format:
{
  "glossary": [{"eng": "...", "zh": "...", "type": "..."}],
  "style_notes": "...",
  "summary": "..."
}

Be aggressive in compression but preserve all unique terminology mappings."""


def build_memory_compression_prompt(global_memory: 'GlobalMemory', target_tokens: int) -> str:
    """
    Build user prompt for memory compression.

    Args:
        global_memory: Current GlobalMemory object
        target_tokens: Target token count for compressed memory

    Returns:
        User prompt for compression task
    """
    import json

    memory_dict = {
        "glossary": global_memory.glossary,
        "style_notes": global_memory.style_notes,
        "summary": global_memory.summary
    }

    prompt = f"""Current memory is too large. Please compress it to approximately {target_tokens} tokens or less.

Current memory:
{json.dumps(memory_dict, ensure_ascii=False, indent=2)}

Return ONLY the compressed JSON object, no explanations."""

    return prompt


def build_user_prompt_for_chunk(pairs_json: str) -> str:
    """
    Build user prompt containing subtitle pairs to refine.

    Args:
        pairs_json: JSON string of subtitle pairs

    Returns:
        User prompt (just the JSON for simplicity)
    """
    # For subtitle refinement, we just pass the JSON directly
    return pairs_json


def validate_response_format(response: str) -> bool:
    """
    Validate that LLM response is in expected JSON format.

    Args:
        response: Raw response from LLM

    Returns:
        True if format appears valid, False otherwise
    """
    import json

    try:
        data = json.loads(response)
        if not isinstance(data, list):
            return False
        if not data:  # Empty list is valid
            return True
        # Check first item has required fields
        first_item = data[0]
        return "id" in first_item and "eng" in first_item and "chinese" in first_item
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


TERMINOLOGY_EXTRACTION_SYSTEM_PROMPT_TEMPLATE = """You are a bilingual terminology extractor for paired English and Chinese subtitles.

Your task is to identify only the terms that require a consistent translation and produce a clean glossary. Follow these rules:
- Focus on proper nouns: people, places, organizations, ships, military units, project/operation code names, legal statute names, show or work titles, and stable acronyms (e.g., JAG, NCIS)
- Always extract all person names (character names) you can confidently identify, even if they appear only once in the current chunk.
- Include keywords that need unified translations across the episode
- Ignore generic conversational words and function words even if they are capitalized at the beginning of a sentence. This does not apply to names (e.g., "Chris", "Benny", "Bryer").
- Do not invent translations or entries if the Chinese counterpart cannot be determined confidently
- For every glossary item output: eng (trimmed, original casing), zh (trimmed), type (one of person/place/organization/title/acronym/unit/ship/project/law/other), confidence (0.0-1.0), evidence_ids (list of up to 5 subtitle ids where the term appears)
- Only keep entries with confidence >= {min_conf}
- Output strictly a JSON array with objects sorted by first appearance, without explanations or Markdown
"""


TERMINOLOGY_EXTRACTION_USER_TEMPLATE = """Here is the list of corrected subtitle pairs. Each entry contains id, eng, and chinese fields.
Extract the terminology glossary following the system instructions and return ONLY a JSON array.

You will also receive an optional "user glossary" that already defines some eng→zh mappings.
- Do NOT output entries whose eng already appears in the user glossary.
- Do NOT output any entry whose zh conflicts with the user glossary for the same eng.

Subtitle pairs (JSON):
{{PAIRS_JSON}}

User glossary (JSON array, may be empty):
{{USER_GLOSSARY_JSON}}
"""


def build_terminology_system_prompt(min_confidence: float) -> str:
    """Build the system prompt for terminology extraction using the configured confidence.

    The numerical threshold shown to the model is kept in sync with the
    post-filtering threshold used in memory.py via Config.terminology_min_confidence.
    """

    # Keep a short, human-friendly representation (e.g., 0.6 rather than 0.600000)
    if min_confidence is None:
        min_confidence = 0.6
    return TERMINOLOGY_EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(min_conf=min_confidence)


def split_user_prompt_and_glossary(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Split a custom main prompt into instructions and a simple eng→zh glossary.

    Lines like "* Term -> 术语" or "- Term -> 术语" are parsed into glossary entries
    and removed from the instruction text.
    """
    import re

    instructions: List[str] = []
    glossary: List[Dict[str, str]] = []

    pattern = re.compile(r"^\s*[-*]\s+(.+?)\s*->\s*(.+?)\s*$")

    skip_prefixes = {
        "Use the following name translations consistently:",
        "Use the following institutional correspondences:"
    }

    in_comment = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        # Handle multi-line HTML comments <!-- ... -->
        working = line
        while True:
            if in_comment:
                end_idx = working.find("-->")
                if end_idx == -1:
                    # Entire line is inside a comment block; drop it
                    working = ""
                    break
                # Strip comment block and continue scanning remainder
                working = working[end_idx + 3 :]
                in_comment = False
                continue
            else:
                start_idx = working.find("<!--")
                if start_idx == -1:
                    break
                end_idx = working.find("-->", start_idx + 4)
                if end_idx == -1:
                    # Comment starts here and continues on later lines
                    working = working[:start_idx]
                    in_comment = True
                    break
                # Remove inline comment and keep surrounding text
                before = working[:start_idx]
                after = working[end_idx + 3 :]
                working = before + after
                # Loop again in case there are multiple comment blocks

        line = working

        # Drop section headers that only introduce the glossary list
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in skip_prefixes:
            continue

        match = pattern.match(line)
        if match:
            eng = match.group(1).strip()
            zh = match.group(2).strip()
            if eng and zh:
                glossary.append({"eng": eng, "zh": zh})
            # Skip adding this line to instructions
            continue
        instructions.append(line)

    instructions_text = "\n".join(instructions).strip()
    return instructions_text, glossary
