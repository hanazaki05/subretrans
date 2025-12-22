"""
Serializers for subtitle pair intermediate representations.

Supports three formats:
- JSON: Standard JSON array format
- XML-pair: Custom XML-like format
- Pseudo-TOML: TOML-like format

All formats preserve ASS formatting tags and handle special characters.
"""

import json
import re
from typing import List, Dict, Any
from pairs import SubtitlePair


class SerializationError(Exception):
    """Exception raised for serialization/deserialization errors."""
    pass


# ============================================================================
# JSON Format (existing)
# ============================================================================

def serialize_json(pairs: List[SubtitlePair]) -> str:
    """
    Serialize subtitle pairs to JSON format.

    Args:
        pairs: List of SubtitlePair objects

    Returns:
        JSON string representation
    """
    json_list = [pair.to_dict() for pair in pairs]
    return json.dumps(json_list, ensure_ascii=False, indent=2)


def deserialize_json(text: str) -> List[SubtitlePair]:
    """
    Deserialize JSON format to subtitle pairs.

    Args:
        text: JSON string representation

    Returns:
        List of SubtitlePair objects

    Raises:
        SerializationError: If JSON parsing fails
    """
    try:
        json_list = json.loads(text)
        if not isinstance(json_list, list):
            raise SerializationError("JSON must be an array")

        pairs = []
        for item in json_list:
            if not isinstance(item, dict):
                raise SerializationError(f"Invalid JSON item: {item}")
            if not all(k in item for k in ["id", "eng", "chinese"]):
                raise SerializationError(f"Missing required fields in: {item}")

            pairs.append(SubtitlePair(
                id=item["id"],
                eng=item["eng"],
                chinese=item["chinese"]
            ))

        return pairs
    except json.JSONDecodeError as e:
        raise SerializationError(f"Failed to parse JSON: {str(e)}")
    except Exception as e:
        raise SerializationError(f"Deserialization error: {str(e)}")


# ============================================================================
# XML-pair Format
# ============================================================================

def serialize_xml_pair(pairs: List[SubtitlePair]) -> str:
    """
    Serialize subtitle pairs to XML-pair format.

    Format:
        <pair>
        ID=0
        eng=Tonight, on JAG...
        chinese=今晚，在《军法署》...
        </pair>

    Args:
        pairs: List of SubtitlePair objects

    Returns:
        XML-pair string representation
    """
    lines = []
    for pair in pairs:
        lines.append("<pair>")
        lines.append(f"ID={pair.id}")
        lines.append(f"eng={pair.eng}")
        lines.append(f"chinese={pair.chinese}")
        lines.append("</pair>")
        lines.append("")  # Empty line between pairs

    return "\n".join(lines).rstrip()  # Remove trailing empty line


def deserialize_xml_pair(text: str) -> List[SubtitlePair]:
    """
    Deserialize XML-pair format to subtitle pairs.

    Args:
        text: XML-pair string representation

    Returns:
        List of SubtitlePair objects

    Raises:
        SerializationError: If parsing fails
    """
    pairs = []
    lines = text.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Expect <pair> tag
        if line != "<pair>":
            raise SerializationError(f"Expected '<pair>' at line {i+1}, got: {line}")

        i += 1
        pair_data = {}

        # Read ID, eng, chinese
        for field in ["ID", "eng", "chinese"]:
            if i >= len(lines):
                raise SerializationError(f"Unexpected end of input while reading {field}")

            line = lines[i].strip()

            # Parse field=value
            if "=" not in line:
                raise SerializationError(f"Expected '{field}=...' at line {i+1}, got: {line}")

            key, value = line.split("=", 1)

            if key != field:
                raise SerializationError(f"Expected field '{field}' at line {i+1}, got: {key}")

            pair_data[field.lower() if field != "ID" else "id"] = value
            i += 1

        # Expect </pair> tag
        if i >= len(lines):
            raise SerializationError("Expected '</pair>' tag")

        line = lines[i].strip()
        if line != "</pair>":
            raise SerializationError(f"Expected '</pair>' at line {i+1}, got: {line}")

        # Create SubtitlePair
        try:
            pair_id = int(pair_data["id"])
        except ValueError:
            raise SerializationError(f"Invalid ID value: {pair_data['id']}")

        pairs.append(SubtitlePair(
            id=pair_id,
            eng=pair_data["eng"],
            chinese=pair_data["chinese"]
        ))

        i += 1

    return pairs


# ============================================================================
# Pseudo-TOML Format
# ============================================================================

def serialize_pseudo_toml(pairs: List[SubtitlePair]) -> str:
    """
    Serialize subtitle pairs to pseudo-TOML format.

    Format:
        [pair]
        id = 0
        eng = Tonight, on JAG...
        chinese = 今晚，在《军法署》...

        [pair]
        id = 1
        eng = Good evening...
        chinese = 晚上好...

    Args:
        pairs: List of SubtitlePair objects

    Returns:
        Pseudo-TOML string representation
    """
    lines = []
    for pair in pairs:
        lines.append("[pair]")
        lines.append(f"id = {pair.id}")
        lines.append(f"eng = {pair.eng}")
        lines.append(f"chinese = {pair.chinese}")
        lines.append("")  # Empty line between pairs

    return "\n".join(lines).rstrip()  # Remove trailing empty line


def deserialize_pseudo_toml(text: str) -> List[SubtitlePair]:
    """
    Deserialize pseudo-TOML format to subtitle pairs.

    Args:
        text: Pseudo-TOML string representation

    Returns:
        List of SubtitlePair objects

    Raises:
        SerializationError: If parsing fails
    """
    pairs = []
    lines = text.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Expect [pair] section
        if line != "[pair]":
            raise SerializationError(f"Expected '[pair]' at line {i+1}, got: {line}")

        i += 1
        pair_data = {}

        # Read id, eng, chinese
        for field in ["id", "eng", "chinese"]:
            if i >= len(lines):
                raise SerializationError(f"Unexpected end of input while reading {field}")

            line = lines[i].strip()

            # Skip empty lines within a pair (shouldn't happen but be lenient)
            if not line:
                i += 1
                if i >= len(lines):
                    raise SerializationError(f"Unexpected end of input while reading {field}")
                line = lines[i].strip()

            # Parse field = value
            if "=" not in line:
                raise SerializationError(f"Expected '{field} = ...' at line {i+1}, got: {line}")

            parts = line.split("=", 1)
            if len(parts) != 2:
                raise SerializationError(f"Invalid field format at line {i+1}: {line}")

            key = parts[0].strip()
            value = parts[1].strip()

            if key != field:
                raise SerializationError(f"Expected field '{field}' at line {i+1}, got: {key}")

            pair_data[field] = value
            i += 1

        # Create SubtitlePair
        try:
            pair_id = int(pair_data["id"])
        except ValueError:
            raise SerializationError(f"Invalid ID value: {pair_data['id']}")

        pairs.append(SubtitlePair(
            id=pair_id,
            eng=pair_data["eng"],
            chinese=pair_data["chinese"]
        ))

    return pairs


# ============================================================================
# Format-agnostic Interface
# ============================================================================

def serialize(pairs: List[SubtitlePair], format_type: str) -> str:
    """
    Serialize subtitle pairs using the specified format.

    Args:
        pairs: List of SubtitlePair objects
        format_type: One of "json", "xml-pair", "pseudo-toml"

    Returns:
        Serialized string representation

    Raises:
        ValueError: If format_type is unsupported
    """
    format_type = format_type.lower()

    if format_type == "json":
        return serialize_json(pairs)
    elif format_type == "xml-pair":
        return serialize_xml_pair(pairs)
    elif format_type == "pseudo-toml":
        return serialize_pseudo_toml(pairs)
    else:
        raise ValueError(f"Unsupported format: {format_type}. "
                        f"Supported formats: json, xml-pair, pseudo-toml")


def deserialize(text: str, format_type: str) -> List[SubtitlePair]:
    """
    Deserialize text to subtitle pairs using the specified format.

    Args:
        text: Serialized string representation
        format_type: One of "json", "xml-pair", "pseudo-toml"

    Returns:
        List of SubtitlePair objects

    Raises:
        ValueError: If format_type is unsupported
        SerializationError: If deserialization fails
    """
    format_type = format_type.lower()

    if format_type == "json":
        return deserialize_json(text)
    elif format_type == "xml-pair":
        return deserialize_xml_pair(text)
    elif format_type == "pseudo-toml":
        return deserialize_pseudo_toml(text)
    else:
        raise ValueError(f"Unsupported format: {format_type}. "
                        f"Supported formats: json, xml-pair, pseudo-toml")


# ============================================================================
# Example Conversion (for prompts)
# ============================================================================

def convert_json_examples_to_format(json_text: str, target_format: str) -> str:
    """
    Convert JSON examples to target format.

    Used for converting the few-shot examples in main_prompt.md to the
    selected intermediate representation format.

    Args:
        json_text: JSON text containing examples
        target_format: Target format ("json", "xml-pair", "pseudo-toml")

    Returns:
        Converted text in target format

    Raises:
        ValueError: If target_format is unsupported
        SerializationError: If conversion fails
    """
    if target_format.lower() == "json":
        return json_text  # No conversion needed

    # Try to parse JSON, but first attempt to fix common issues with
    # unescaped backslashes in ASS tags (e.g., {\i1} -> {\\i1})
    try:
        # First, try parsing as-is
        pairs = deserialize_json(json_text)
    except (json.JSONDecodeError, SerializationError) as e:
        # If that fails, try fixing unescaped backslashes
        # This handles cases where markdown templates have raw backslashes
        # like {"eng": "text {\i1}italics{\i0}"} which need to be escaped
        # We need to escape backslashes that appear inside JSON string values
        # Strategy: Replace single backslashes with double backslashes, but be careful
        # not to break already-escaped sequences like \\n, \", etc.

        # Use a regex to find backslashes that aren't followed by valid JSON escape chars
        # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
        import re

        # Replace backslashes that are NOT part of valid escape sequences
        # This regex matches a backslash NOT followed by: ", \, /, b, f, n, r, t, u
        fixed_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_text)

        try:
            pairs = deserialize_json(fixed_json)
        except (json.JSONDecodeError, SerializationError):
            # If still failing, raise the original error
            raise SerializationError(f"Failed to parse JSON: {e}")

    # Convert to target format
    return serialize(pairs, target_format)
