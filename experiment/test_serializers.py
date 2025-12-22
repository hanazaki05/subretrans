#!/usr/bin/env python3
"""
Test script for intermediate representation serializers.

Tests all three formats (JSON, XML-pair, pseudo-TOML) for:
- Basic serialization/deserialization
- ASS formatting tag preservation
- Special characters handling
- Round-trip consistency
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairs import SubtitlePair
from serializers import (
    serialize_json, deserialize_json,
    serialize_xml_pair, deserialize_xml_pair,
    serialize_pseudo_toml, deserialize_pseudo_toml,
    serialize, deserialize,
    SerializationError
)


def test_basic_serialization():
    """Test basic serialization for all formats."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Serialization")
    print("=" * 60)

    test_pairs = [
        SubtitlePair(id=0, eng="Hello, world!", chinese="你好，世界"),
        SubtitlePair(id=1, eng="How are you?", chinese="你好吗"),
        SubtitlePair(id=2, eng="Good morning.", chinese="早上好"),
    ]

    formats = ["json", "xml-pair", "pseudo-toml"]

    for fmt in formats:
        print(f"\nTesting {fmt.upper()} format:")
        print("-" * 40)

        try:
            # Serialize
            serialized = serialize(test_pairs, fmt)
            print(f"Serialized:\n{serialized[:200]}...")

            # Deserialize
            deserialized = deserialize(serialized, fmt)

            # Verify
            assert len(deserialized) == len(test_pairs), \
                f"Length mismatch: expected {len(test_pairs)}, got {len(deserialized)}"

            for i, (original, recovered) in enumerate(zip(test_pairs, deserialized)):
                assert original.id == recovered.id, \
                    f"ID mismatch at index {i}: {original.id} != {recovered.id}"
                assert original.eng == recovered.eng, \
                    f"English mismatch at index {i}: {original.eng} != {recovered.eng}"
                assert original.chinese == recovered.chinese, \
                    f"Chinese mismatch at index {i}: {original.chinese} != {recovered.chinese}"

            print(f"✓ {fmt.upper()} passed: {len(test_pairs)} pairs")

        except Exception as e:
            print(f"✗ {fmt.upper()} failed: {str(e)}")
            return False

    print("\n✓ All basic serialization tests passed!")
    return True


def test_ass_tags_preservation():
    """Test preservation of ASS formatting tags."""
    print("\n" + "=" * 60)
    print("Test 2: ASS Tags Preservation")
    print("=" * 60)

    test_pairs = [
        SubtitlePair(
            id=0,
            eng=r"This is {\i1}italic{\i0} text",
            chinese=r"这是{\i1}斜体{\i0}文本"
        ),
        SubtitlePair(
            id=1,
            eng=r"Line break here:\NNext line",
            chinese=r"换行在此：\N下一行"
        ),
        SubtitlePair(
            id=2,
            eng=r"{\b1}Bold{\b0} and {\i1}italic{\i0}",
            chinese=r"{\b1}粗体{\b0}和{\i1}斜体{\i0}"
        ),
    ]

    formats = ["json", "xml-pair", "pseudo-toml"]

    for fmt in formats:
        print(f"\nTesting {fmt.upper()} format with ASS tags:")
        print("-" * 40)

        try:
            # Round-trip test
            serialized = serialize(test_pairs, fmt)
            deserialized = deserialize(serialized, fmt)

            # Verify tags are preserved
            for i, (original, recovered) in enumerate(zip(test_pairs, deserialized)):
                if original.eng != recovered.eng:
                    print(f"  Original EN : {original.eng}")
                    print(f"  Recovered EN: {recovered.eng}")
                    raise AssertionError(f"English ASS tags not preserved at index {i}")

                if original.chinese != recovered.chinese:
                    print(f"  Original ZH : {original.chinese}")
                    print(f"  Recovered ZH: {recovered.chinese}")
                    raise AssertionError(f"Chinese ASS tags not preserved at index {i}")

            print(f"✓ {fmt.upper()} preserved all ASS tags")

        except Exception as e:
            print(f"✗ {fmt.upper()} failed: {str(e)}")
            return False

    print("\n✓ All ASS tag preservation tests passed!")
    return True


def test_special_characters():
    """Test handling of special characters."""
    print("\n" + "=" * 60)
    print("Test 3: Special Characters")
    print("=" * 60)

    test_pairs = [
        SubtitlePair(
            id=0,
            eng='He said, "Hello!"',
            chinese='他说："你好！"'
        ),
        SubtitlePair(
            id=1,
            eng="Text with = equals and : colons",
            chinese="包含 = 等号和 : 冒号的文本"
        ),
        SubtitlePair(
            id=2,
            eng="Ellipsis... and more...",
            chinese="省略号...和更多..."
        ),
        SubtitlePair(
            id=3,
            eng="Question? Answer!",
            chinese="问题？答案！"
        ),
    ]

    formats = ["json", "xml-pair", "pseudo-toml"]

    for fmt in formats:
        print(f"\nTesting {fmt.upper()} format with special chars:")
        print("-" * 40)

        try:
            serialized = serialize(test_pairs, fmt)
            deserialized = deserialize(serialized, fmt)

            for i, (original, recovered) in enumerate(zip(test_pairs, deserialized)):
                assert original.eng == recovered.eng, \
                    f"English mismatch at {i}: '{original.eng}' != '{recovered.eng}'"
                assert original.chinese == recovered.chinese, \
                    f"Chinese mismatch at {i}: '{original.chinese}' != '{recovered.chinese}'"

            print(f"✓ {fmt.upper()} handled special characters correctly")

        except Exception as e:
            print(f"✗ {fmt.upper()} failed: {str(e)}")
            return False

    print("\n✓ All special character tests passed!")
    return True


def test_edge_cases():
    """Test edge cases (empty strings, single pair, etc.)."""
    print("\n" + "=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)

    # Test 1: Single pair
    single_pair = [SubtitlePair(id=0, eng="Single", chinese="单个")]

    # Test 2: Empty text (valid but unusual)
    empty_text = [SubtitlePair(id=0, eng="", chinese="")]

    # Test 3: Very long text
    long_text = [SubtitlePair(
        id=0,
        eng="This is a very long subtitle that contains many words and characters to test if the serializer can handle long strings properly without breaking.",
        chinese="这是一个非常长的字幕，包含许多单词和字符，用于测试序列化器是否可以正确处理长字符串而不会中断。"
    )]

    test_cases = [
        ("Single pair", single_pair),
        ("Empty text", empty_text),
        ("Long text", long_text),
    ]

    formats = ["json", "xml-pair", "pseudo-toml"]

    for case_name, test_pairs in test_cases:
        print(f"\n{case_name}:")
        print("-" * 40)

        for fmt in formats:
            try:
                serialized = serialize(test_pairs, fmt)
                deserialized = deserialize(serialized, fmt)

                assert len(deserialized) == len(test_pairs)
                for original, recovered in zip(test_pairs, deserialized):
                    assert original.id == recovered.id
                    assert original.eng == recovered.eng
                    assert original.chinese == recovered.chinese

                print(f"  ✓ {fmt.upper()}")

            except Exception as e:
                print(f"  ✗ {fmt.upper()} failed: {str(e)}")
                return False

    print("\n✓ All edge case tests passed!")
    return True


def test_error_handling():
    """Test error handling for malformed input."""
    print("\n" + "=" * 60)
    print("Test 5: Error Handling")
    print("=" * 60)

    malformed_cases = [
        ("JSON", "json", "{invalid json"),
        ("XML-pair", "xml-pair", "<pair>\nID=0\n"),  # Incomplete
        ("Pseudo-TOML", "pseudo-toml", "[pair]\nid = 0\n"),  # Missing fields
    ]

    for name, fmt, malformed_input in malformed_cases:
        print(f"\nTesting {name} error handling:")
        print("-" * 40)

        try:
            deserialize(malformed_input, fmt)
            print(f"  ✗ Should have raised SerializationError")
            return False
        except SerializationError as e:
            print(f"  ✓ Correctly raised SerializationError: {str(e)[:50]}...")
        except Exception as e:
            print(f"  ✗ Raised wrong exception type: {type(e).__name__}")
            return False

    print("\n✓ All error handling tests passed!")
    return True


def test_format_examples_conversion():
    """Test conversion of JSON examples to other formats."""
    print("\n" + "=" * 60)
    print("Test 6: Example Format Conversion")
    print("=" * 60)

    json_example = '''[
  {"id": 1, "eng": "Hello", "chinese": "你好"},
  {"id": 2, "eng": "Goodbye", "chinese": "再见"}
]'''

    print("\nOriginal JSON:")
    print(json_example)

    try:
        from serializers import convert_json_examples_to_format

        # Test XML conversion
        print("\n\nConverted to XML-pair:")
        print("-" * 40)
        xml_result = convert_json_examples_to_format(json_example, "xml-pair")
        print(xml_result)

        # Test pseudo-TOML conversion
        print("\n\nConverted to Pseudo-TOML:")
        print("-" * 40)
        toml_result = convert_json_examples_to_format(json_example, "pseudo-toml")
        print(toml_result)

        # Verify round-trip
        pairs_from_json = deserialize_json(json_example)
        pairs_from_xml = deserialize_xml_pair(xml_result)
        pairs_from_toml = deserialize_pseudo_toml(toml_result)

        for i, (pj, px, pt) in enumerate(zip(pairs_from_json, pairs_from_xml, pairs_from_toml)):
            assert pj.id == px.id == pt.id, f"ID mismatch at {i}"
            assert pj.eng == px.eng == pt.eng, f"English mismatch at {i}"
            assert pj.chinese == px.chinese == pt.chinese, f"Chinese mismatch at {i}"

        print("\n✓ Example conversion and round-trip verified!")
        return True

    except Exception as e:
        print(f"\n✗ Example conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INTERMEDIATE REPRESENTATION SERIALIZER TESTS")
    print("=" * 60)

    tests = [
        test_basic_serialization,
        test_ass_tags_preservation,
        test_special_characters,
        test_edge_cases,
        test_error_handling,
        test_format_examples_conversion,
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
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
