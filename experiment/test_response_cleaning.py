#!/usr/bin/env python3
"""
Test script for LLM response post-processing.

Tests the response cleaning pipeline that handles:
1. Thinking blocks (<think>...</think>)
2. Markdown code blocks (```...```)
3. Combined scenarios (both thinking + code blocks)
4. All three intermediate formats (JSON, XML-pair, pseudo-TOML)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, experiment_dir)

import re
from typing import Optional, List
from serializers import deserialize
from pairs import SubtitlePair


# Copy the helper functions from llm_client_sdk.py for testing
def _strip_thinking_blocks(text: str) -> str:
    """
    Remove thinking blocks from LLM response.

    Args:
        text: Raw text that may contain thinking blocks

    Returns:
        Text with thinking blocks removed
    """
    # Remove <think>...</think> blocks (case-insensitive, multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _extract_from_code_blocks(text: str) -> Optional[str]:
    """
    Extract content from markdown code blocks.

    Args:
        text: Raw text that may contain code blocks

    Returns:
        Extracted content or None if no code blocks found
    """
    # Try to find content within code blocks (```...```)
    code_block_pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        # Return the first code block content
        return matches[0].strip()

    # No code blocks found
    return None


def _clean_llm_response(text: str) -> str:
    """
    Clean LLM response by removing extraneous content.

    Processing order (CRITICAL):
    1. Remove thinking blocks (<think>...</think>)
    2. Extract from markdown code blocks (```...```)
    3. Return cleaned text

    Args:
        text: Raw LLM response text

    Returns:
        Cleaned text ready for deserialization
    """
    # Step 1: Remove thinking blocks FIRST
    text = _strip_thinking_blocks(text)

    # Step 2: Extract from code blocks if present
    extracted = _extract_from_code_blocks(text)
    if extracted is not None:
        return extracted

    # No code blocks, return as-is (already stripped of thinking blocks)
    return text.strip()


def test_thinking_block_removal():
    """Test removal of thinking blocks."""
    print("\n" + "=" * 60)
    print("Test 1: Thinking Block Removal")
    print("=" * 60)

    test_cases = [
        # Simple thinking block
        (
            "<think>Analyzing the task...</think>\nActual content here",
            "Actual content here"
        ),
        # Thinking block with formatting
        (
            "<think>\n**Examining the Task**\n\nI'm currently focused on...\n</think>\n\nActual response",
            "Actual response"
        ),
        # Case-insensitive
        (
            "<THINK>Some thoughts</THINK>\nContent",
            "Content"
        ),
        # Multiple thinking blocks
        (
            "<think>First thought</think>\nSome text\n<think>Second thought</think>\nMore text",
            "Some text\n\nMore text"
        ),
        # No thinking blocks
        (
            "Just plain content",
            "Just plain content"
        ),
    ]

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = _strip_thinking_blocks(input_text)
        assert result == expected, f"Case {i} failed:\nExpected: {expected}\nGot: {result}"
        print(f"  ✓ Case {i} passed")

    print("\n✓ All thinking block removal tests passed!")
    return True


def test_code_block_extraction():
    """Test extraction from markdown code blocks."""
    print("\n" + "=" * 60)
    print("Test 2: Code Block Extraction")
    print("=" * 60)

    test_cases = [
        # JSON code block
        (
            '```json\n[{"id": 0}]\n```',
            '[{"id": 0}]'
        ),
        # XML code block
        (
            '```xml\n<pair>ID=0</pair>\n```',
            '<pair>ID=0</pair>'
        ),
        # Code block without language
        (
            '```\ncontent here\n```',
            'content here'
        ),
        # Multiple code blocks (returns first)
        (
            '```json\n{"first": true}\n```\n```json\n{"second": true}\n```',
            '{"first": true}'
        ),
        # No code blocks
        (
            'Plain text without code blocks',
            None
        ),
    ]

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = _extract_from_code_blocks(input_text)
        assert result == expected, f"Case {i} failed:\nExpected: {expected}\nGot: {result}"
        print(f"  ✓ Case {i} passed")

    print("\n✓ All code block extraction tests passed!")
    return True


def test_combined_cleaning():
    """Test combined thinking block + code block scenarios."""
    print("\n" + "=" * 60)
    print("Test 3: Combined Cleaning (Thinking + Code Blocks)")
    print("=" * 60)

    test_cases = [
        # Thinking block THEN code block (most common)
        (
            '<think>Analyzing...</think>\n```json\n[{"id": 0, "eng": "Hello", "chinese": "你好"}]\n```',
            '[{"id": 0, "eng": "Hello", "chinese": "你好"}]'
        ),
        # Thinking block with XML code block
        (
            '<think>Processing pairs...</think>\n```xml\n<pair>\nID=0\neng=Test\nchinese=测试\n</pair>\n```',
            '<pair>\nID=0\neng=Test\nchinese=测试\n</pair>'
        ),
        # Only thinking block (no code block)
        (
            '<think>Just thinking</think>\n{"id": 0}',
            '{"id": 0}'
        ),
        # Only code block (no thinking)
        (
            '```\n[pair]\nid = 0\n```',
            '[pair]\nid = 0'
        ),
        # Neither thinking nor code blocks
        (
            'Plain response text',
            'Plain response text'
        ),
        # Complex case: thinking, text, code block
        (
            '<think>Step 1: Analysis</think>\nHere is the result:\n```json\n{"done": true}\n```',
            '{"done": true}'
        ),
    ]

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = _clean_llm_response(input_text)
        assert result == expected, f"Case {i} failed:\nExpected: {expected}\nGot: {result}"
        print(f"  ✓ Case {i} passed")

    print("\n✓ All combined cleaning tests passed!")
    return True


def test_full_pipeline_json():
    """Test full pipeline with JSON format."""
    print("\n" + "=" * 60)
    print("Test 4: Full Pipeline - JSON Format")
    print("=" * 60)

    # Case 1: Thinking + code block
    response = '''<think>
**Examining the Task**

I'm currently focused on refining the subtitle pairs.
</think>

```json
[
  {
    "id": 120,
    "eng": "She'll forgive you.",
    "chinese": "她会原谅你"
  },
  {
    "id": 121,
    "eng": "You'll pay, but she'll forgive you.",
    "chinese": "你会付出代价，但她会原谅你"
  }
]
```'''

    cleaned = _clean_llm_response(response)
    pairs = deserialize(cleaned, "json")

    assert len(pairs) == 2
    assert pairs[0].id == 120
    assert pairs[0].eng == "She'll forgive you."
    assert pairs[0].chinese == "她会原谅你"
    assert pairs[1].id == 121

    print("  ✓ JSON with thinking + code block")

    # Case 2: Code block only
    response2 = '''```
[
  {"id": 0, "eng": "Hello", "chinese": "你好"}
]
```'''

    cleaned2 = _clean_llm_response(response2)
    pairs2 = deserialize(cleaned2, "json")

    assert len(pairs2) == 1
    assert pairs2[0].eng == "Hello"

    print("  ✓ JSON with code block only")

    print("\n✓ Full JSON pipeline tests passed!")
    return True


def test_full_pipeline_xml():
    """Test full pipeline with XML-pair format."""
    print("\n" + "=" * 60)
    print("Test 5: Full Pipeline - XML-pair Format")
    print("=" * 60)

    # Case: Thinking + code block with XML
    response = '''<think>Converting to XML-pair format...</think>

```xml
<pair>
ID=10
eng=Good morning.
chinese=早上好
</pair>

<pair>
ID=11
eng=How are you?
chinese=你好吗
</pair>
```'''

    cleaned = _clean_llm_response(response)
    pairs = deserialize(cleaned, "xml-pair")

    assert len(pairs) == 2
    assert pairs[0].id == 10
    assert pairs[0].eng == "Good morning."
    assert pairs[0].chinese == "早上好"
    assert pairs[1].id == 11

    print("  ✓ XML-pair with thinking + code block")

    print("\n✓ Full XML-pair pipeline tests passed!")
    return True


def test_full_pipeline_toml():
    """Test full pipeline with pseudo-TOML format."""
    print("\n" + "=" * 60)
    print("Test 6: Full Pipeline - Pseudo-TOML Format")
    print("=" * 60)

    # Case: Thinking + code block with TOML
    response = '''<think>Processing in TOML format...</think>

```toml
[pair]
id = 5
eng = Testing pseudo-TOML
chinese = 测试伪TOML

[pair]
id = 6
eng = Another test
chinese = 另一个测试
```'''

    cleaned = _clean_llm_response(response)
    pairs = deserialize(cleaned, "pseudo-toml")

    assert len(pairs) == 2
    assert pairs[0].id == 5
    assert pairs[0].eng == "Testing pseudo-TOML"
    assert pairs[0].chinese == "测试伪TOML"
    assert pairs[1].id == 6

    print("  ✓ Pseudo-TOML with thinking + code block")

    print("\n✓ Full pseudo-TOML pipeline tests passed!")
    return True


def _extract_from_format_marker(text: str, format_type: str) -> Optional[str]:
    """
    Extract content starting from format-specific marker.
    Used as fallback when normal deserialization fails.
    """
    if format_type.lower() == "xml-pair":
        idx = text.find("<pair>")
        if idx != -1:
            return text[idx:].strip()
    elif format_type.lower() == "json":
        # Find first [ or {
        for char in ['[', '{']:
            idx = text.find(char)
            if idx != -1:
                return text[idx:].strip()
    elif format_type.lower() == "pseudo-toml":
        idx = text.find("[pair]")
        if idx != -1:
            return text[idx:].strip()
    return None


def _detect_duplicate_pairs(pairs: List[SubtitlePair]) -> List[int]:
    """Detect duplicate pair IDs."""
    id_counts = {}
    for pair in pairs:
        id_counts[pair.id] = id_counts.get(pair.id, 0) + 1
    duplicates = [id for id, count in id_counts.items() if count > 1]
    return duplicates


def test_edge_cases():
    """Test edge cases and potential issues."""
    print("\n" + "=" * 60)
    print("Test 7: Edge Cases")
    print("=" * 60)

    # Empty thinking block
    result = _clean_llm_response("<think></think>Content")
    assert result == "Content"
    print("  ✓ Empty thinking block")

    # Nested angle brackets in content
    result = _clean_llm_response("Text with <html> tags")
    assert result == "Text with <html> tags"
    print("  ✓ Nested angle brackets")

    # Thinking-like text but not in tags
    result = _clean_llm_response("I think this is good\nActual content")
    assert "I think" in result
    print("  ✓ Non-tag 'think' text preserved")

    # Code block-like text but incomplete
    result = _clean_llm_response("Some text ```incomplete")
    assert "```incomplete" in result
    print("  ✓ Incomplete code blocks ignored")

    # Multiple newlines and whitespace
    result = _clean_llm_response("<think>X</think>\n\n\n```\ncontent\n```")
    assert result == "content"
    print("  ✓ Whitespace handling")

    print("\n✓ All edge case tests passed!")
    return True


def test_leading_commentary_extraction():
    """Test pattern-based extraction for leading commentary."""
    print("\n" + "=" * 60)
    print("Test 8: Leading Commentary Extraction (Fallback)")
    print("=" * 60)

    # XML with leading commentary
    response_xml = "I have reviewed and corrected the subtitles.\n<pair>\nID=0\neng=Hello\nchinese=你好\n</pair>"
    cleaned = _clean_llm_response(response_xml)
    extracted = _extract_from_format_marker(cleaned, "xml-pair")
    assert extracted is not None
    assert extracted.startswith("<pair>")
    pairs = deserialize(extracted, "xml-pair")
    assert len(pairs) == 1
    assert pairs[0].eng == "Hello"
    print("  ✓ XML with leading commentary")

    # JSON with leading commentary
    response_json = 'Here are the corrections:\n[{"id": 0, "eng": "Hello", "chinese": "你好"}]'
    cleaned_json = _clean_llm_response(response_json)
    extracted_json = _extract_from_format_marker(cleaned_json, "json")
    assert extracted_json is not None
    assert extracted_json.startswith("[")
    pairs_json = deserialize(extracted_json, "json")
    assert len(pairs_json) == 1
    assert pairs_json[0].eng == "Hello"
    print("  ✓ JSON with leading commentary")

    # TOML with leading commentary
    response_toml = "Analysis complete.\n[pair]\nid = 0\neng = Hello\nchinese = 你好"
    cleaned_toml = _clean_llm_response(response_toml)
    extracted_toml = _extract_from_format_marker(cleaned_toml, "pseudo-toml")
    assert extracted_toml is not None
    assert extracted_toml.startswith("[pair]")
    pairs_toml = deserialize(extracted_toml, "pseudo-toml")
    assert len(pairs_toml) == 1
    assert pairs_toml[0].eng == "Hello"
    print("  ✓ TOML with leading commentary")

    print("\n✓ All leading commentary extraction tests passed!")
    return True


def test_duplicate_pair_detection():
    """Test duplicate pair detection and deduplication."""
    print("\n" + "=" * 60)
    print("Test 9: Duplicate Pair Detection")
    print("=" * 60)

    # Create test pairs with duplicates
    pairs = [
        SubtitlePair(id=0, eng="First", chinese="第一"),
        SubtitlePair(id=1, eng="Second", chinese="第二"),
        SubtitlePair(id=0, eng="Duplicate", chinese="重复"),  # Duplicate ID
        SubtitlePair(id=2, eng="Third", chinese="第三"),
        SubtitlePair(id=1, eng="Another duplicate", chinese="另一个重复"),  # Another duplicate
    ]

    # Detect duplicates
    duplicates = _detect_duplicate_pairs(pairs)
    assert set(duplicates) == {0, 1}
    print("  ✓ Duplicate detection works")

    # Deduplicate (keep last occurrence)
    id_to_pair = {}
    for pair in pairs:
        id_to_pair[pair.id] = pair  # Last one wins

    seen_ids = set()
    deduplicated = []
    for pair in pairs:
        if pair.id not in seen_ids:
            deduplicated.append(id_to_pair[pair.id])  # Use last occurrence
            seen_ids.add(pair.id)

    assert len(deduplicated) == 3
    assert deduplicated[0].id == 0
    assert deduplicated[0].eng == "Duplicate"  # Last occurrence kept
    assert deduplicated[1].id == 1
    assert deduplicated[1].eng == "Another duplicate"  # Last occurrence kept
    assert deduplicated[2].id == 2
    print("  ✓ Deduplication keeps last occurrence")

    # No duplicates case
    pairs_no_dup = [
        SubtitlePair(id=0, eng="First", chinese="第一"),
        SubtitlePair(id=1, eng="Second", chinese="第二"),
    ]
    duplicates_none = _detect_duplicate_pairs(pairs_no_dup)
    assert len(duplicates_none) == 0
    print("  ✓ No false positives for unique pairs")

    print("\n✓ All duplicate pair detection tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RESPONSE POST-PROCESSING TESTS")
    print("=" * 60)

    tests = [
        test_thinking_block_removal,
        test_code_block_extraction,
        test_combined_cleaning,
        test_full_pipeline_json,
        test_full_pipeline_xml,
        test_full_pipeline_toml,
        test_edge_cases,
        test_leading_commentary_extraction,
        test_duplicate_pair_detection,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed successfully!")
        print("\nResponse cleaning is working correctly for:")
        print("  • Thinking block removal (<think>...</think>)")
        print("  • Markdown code block extraction (```...```)")
        print("  • Combined scenarios (thinking + code blocks)")
        print("  • All formats (JSON, XML-pair, pseudo-TOML)")
        print("  • Leading commentary extraction (fallback recovery)")
        print("  • Duplicate pair detection and deduplication")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
