#!/usr/bin/env python3
"""
Ad-hoc test for split_user_prompt_and_glossary() behaviour.

Checks:
- HTML comments <!-- ... --> (single-line and multi-line) are removed.
- Section headers introducing glossary blocks are skipped.
- Lines like "* Term -> 术语" are parsed into glossary entries.
- Remaining non-glossary, non-comment lines stay in instructions.
"""

from subretrans.prompts import split_user_prompt_and_glossary


SAMPLE_PROMPT = """You are a professional translator.

<!--
This whole block
should be ignored
-->

Use the following name translations consistently:
* Admiral -> 将军
* JAG -> 军法署
Line kept before inline comment <!-- comment inside line -->
* NCIS -> 海军刑事调查局

Use the following institutional correspondences:
* Navy's Judge Advocate General Corps -> 海军军法署
"""


def main() -> None:
    instructions, glossary = split_user_prompt_and_glossary(SAMPLE_PROMPT)

    print("=== Instructions ===")
    print(instructions)
    print("\n=== Glossary ===")
    for entry in glossary:
        print(entry)

    # Basic assertions
    assert "Admiral -> 将军" not in instructions
    assert "Use the following name translations consistently:" not in instructions
    assert "This whole block" not in instructions
    assert "Line kept before inline comment" in instructions

    eng_terms = {e["eng"] for e in glossary}
    assert "Admiral" in eng_terms
    assert "JAG" in eng_terms
    assert "NCIS" in eng_terms
    assert "Navy's Judge Advocate General Corps" in eng_terms

    print("\n[OK] split_user_prompt_and_glossary behaves as expected.")


if __name__ == "__main__":
    main()
