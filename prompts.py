"""Prompt templates and helpers for LLM interactions.

Contains system prompts for subtitle refinement, memory compression, and
terminology extraction, plus utilities to inject user customization.
"""

from typing import TYPE_CHECKING, List, Dict, Tuple

if TYPE_CHECKING:
    from memory import GlobalMemory


_USER_INSTRUCTION: str = ""


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


def build_system_prompt(global_memory: 'GlobalMemory') -> str:
    """
    Build complete system prompt including memory section.

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
