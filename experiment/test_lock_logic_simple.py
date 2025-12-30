#!/usr/bin/env python3
"""Simple test of terminology lock logic without heavy imports."""

def test_lock_logic():
    """Test the lock mechanism logic directly."""

    print("Test: Terminology Lock Logic")
    print("=" * 60)

    # Simulate user glossary (what we now populate from template)
    user_glossary = [
        {"eng": "Mac", "zh": "麦可"},
        {"eng": "Harm", "zh": "哈姆"}
    ]

    # Build user map (case-insensitive) - same logic as in memory.py:268
    user_map = {entry.get("eng", "").strip().casefold(): entry for entry in user_glossary}

    print(f"User glossary: {user_glossary}")
    print(f"User map keys: {list(user_map.keys())}")
    print()

    # Simulate learned terms from LLM
    new_terms = [
        {
            "eng": "Mac",
            "zh": "麦",  # Different from user glossary!
            "type": "person",
            "confidence": 1.0
        },
        {
            "eng": "Harm",
            "zh": "哈姆",  # Same as user glossary
            "type": "person",
            "confidence": 1.0
        },
        {
            "eng": "Admiral",
            "zh": "将军",  # Not in user glossary
            "type": "person",
            "confidence": 1.0
        }
    ]

    # Test the lock mechanism (same logic as memory.py:290-299)
    policy = "lock"
    skipped_conflict = 0
    skipped_user_dup = 0
    added = 0
    learned_glossary = []

    for term in new_terms:
        eng = term.get("eng", "").strip()
        zh = term.get("zh", "").strip()
        eng_key = eng.casefold()

        print(f"Processing: '{eng}' -> '{zh}'")
        print(f"  Key: '{eng_key}', In user_map: {eng_key in user_map}")

        if policy == "lock" and eng_key in user_map:
            user_zh = user_map[eng_key].get("zh", "").strip()
            print(f"  Found in user glossary: '{user_zh}'")

            if user_zh and user_zh != zh:
                print(f"  ✓ CONFLICT DETECTED! Skipping.")
                skipped_conflict += 1
            else:
                print(f"  ✓ Duplicate (same translation). Skipping.")
                skipped_user_dup += 1
        else:
            print(f"  ✓ Adding to learned glossary.")
            learned_glossary.append(term)
            added += 1
        print()

    print("=" * 60)
    print("Results:")
    print(f"  Total candidates: {len(new_terms)}")
    print(f"  Added: {added}")
    print(f"  Conflicts: {skipped_conflict}")
    print(f"  Duplicates: {skipped_user_dup}")
    print(f"  User-locked total: {skipped_conflict + skipped_user_dup}")
    print()

    # Verify
    assert added == 1, f"Expected 1 term added (Admiral), got {added}"
    assert skipped_conflict == 1, f"Expected 1 conflict (Mac), got {skipped_conflict}"
    assert skipped_user_dup == 1, f"Expected 1 duplicate (Harm), got {skipped_user_dup}"
    assert len(learned_glossary) == 1, f"Expected 1 term in learned glossary, got {len(learned_glossary)}"
    assert learned_glossary[0]["eng"] == "Admiral", "Expected Admiral to be added"

    print("✅ TEST PASSED!")
    print()
    print("This confirms the lock logic works IF user_glossary is populated.")
    print("The bug was that user_glossary was empty [] at runtime.")
    print("The fix populates it from main_prompt.md template at initialization.")
    print()

    return True

if __name__ == "__main__":
    try:
        test_lock_logic()
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        exit(1)
