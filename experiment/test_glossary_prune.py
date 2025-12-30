#!/usr/bin/env python3
"""
Regression test for learned glossary pruning vs user glossary.

Ensures:
- Learned glossary entries whose `eng` exists in the user glossary are removed.
- Matching is case-insensitive and whitespace/Unicode-normalized.
"""

import os
import sys
from types import SimpleNamespace

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import GlobalMemory, prune_learned_glossary_against_user_glossary, update_global_memory


def test_prune_function():
    print("=" * 70)
    print("Test 1: prune_learned_glossary_against_user_glossary()")
    print("=" * 70)

    memory = GlobalMemory(
        user_glossary=[
            {"eng": "JAG", "zh": "军法署"},
            {"eng": "Admiral", "zh": "将军"},
            {"eng": "Mac", "zh": "麦可"},
        ],
        glossary=[
            {"eng": "JAG", "zh": "军法署", "type": "acronym", "confidence": 1.0},
            {"eng": "  admiral  ", "zh": "将军", "type": "other", "confidence": 1.0},
            {"eng": "MAC", "zh": "麦可", "type": "person", "confidence": 0.9},
            {"eng": "Watertown", "zh": "水城号", "type": "ship", "confidence": 1.0},
        ],
    )

    removed_count, removed = prune_learned_glossary_against_user_glossary(memory)
    assert removed_count == 3, f"expected 3 removed entries, got {removed_count}"
    assert {e.get("eng", "").strip().casefold() for e in removed} >= {"jag", "admiral", "mac"}
    assert len(memory.glossary) == 1, f"expected 1 learned entry remaining, got {len(memory.glossary)}"
    assert memory.glossary[0]["eng"] == "Watertown"

    print("\n✅ Test 1 PASSED")
    return True


def test_update_global_memory_prunes_even_with_empty_chunk():
    print("\n" + "=" * 70)
    print("Test 2: update_global_memory() prunes on call")
    print("=" * 70)

    memory = GlobalMemory(
        user_glossary=[{"eng": "JAG", "zh": "军法署"}],
        glossary=[{"eng": "JAG", "zh": "军法署", "type": "acronym", "confidence": 1.0}],
    )

    # corrected_pairs empty => terminology extraction short-circuits (no network)
    cfg = SimpleNamespace(glossary_policy="lock", glossary_max_entries=100, verbose=False)
    update_global_memory(memory, corrected_pairs=[], config=cfg)

    assert memory.glossary == [], "expected learned glossary to be pruned by update_global_memory"

    print("\n✅ Test 2 PASSED")
    return True


if __name__ == "__main__":
    try:
        all_passed = True
        all_passed &= test_prune_function()
        all_passed &= test_update_global_memory_prunes_even_with_empty_chunk()

        print("\n" + "=" * 70)
        if all_passed:
            print("✅ ALL TESTS PASSED")
            print("=" * 70)
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 70)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

