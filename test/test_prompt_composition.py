from pathlib import Path

from subretrans.config import PromptPaths
from subretrans.memory import GlobalMemory
from subretrans.prompts import (
    build_effective_prompt_glossary,
    inject_memory_into_template,
    load_qa_prompt_template,
    load_refine_prompt_template,
    render_memory_sections,
)


def test_stage_prompts_share_rules_and_keep_distinct_tasks(tmp_path: Path) -> None:
    shared = tmp_path / "shared.md"
    refine = tmp_path / "refine.md"
    qa = tmp_path / "qa.md"
    repair = tmp_path / "repair.md"
    shared.write_text("SHARED RULES\n", encoding="utf-8")
    refine.write_text("REFINE TASK\n", encoding="utf-8")
    qa.write_text("QA TASK\n", encoding="utf-8")
    repair.write_text("REPAIR TASK\n", encoding="utf-8")
    paths = PromptPaths(shared=shared, refine=refine, qa=qa, repair=repair)

    refine_prompt = load_refine_prompt_template(paths)
    qa_prompt = load_qa_prompt_template(paths)

    assert refine_prompt == "SHARED RULES\n\nREFINE TASK\n"
    assert qa_prompt == "SHARED RULES\n\nQA TASK\n"
    assert refine_prompt.count("SHARED RULES") == 1
    assert qa_prompt.count("SHARED RULES") == 1


def test_prompt_and_memory_render_the_same_frozen_effective_glossary() -> None:
    template = """Rules.

### 1. User Terminology (Authoritative Glossary)
- Harm: 哈姆

### 2. Input/Output Format & Constraint
Return JSON.
    """
    memory = GlobalMemory(
        user_glossary=[{"eng": "Harm", "zh": "哈姆"}],
        glossary=[
            {"eng": "Webb", "zh": "韦布", "type": "person", "evidence_ids": [0]},
            {"eng": "旧称", "zh": "罗伯茨", "type": "other", "evidence_ids": [1]},
        ],
        story_description="Webb is questioning a witness.",
    )
    template_glossary = [{"eng": "Harm", "zh": "哈姆"}]
    effective = build_effective_prompt_glossary(
        template_glossary,
        memory,
        episode_replacements=[("罗伯茨", "罗伯特")],
        episode_id="S07E08",
        manifest_hash="m",
        artifact_hash="a",
    )

    rendered = inject_memory_into_template(template, memory, effective=effective)
    memory_sections = render_memory_sections(memory, effective=effective)

    for text in (rendered, memory_sections):
        assert "- Harm: 哈姆" in text
        assert "- 罗伯茨: 罗伯特" in text
        assert "- Webb (person): 韦布" in text
        assert "旧称" not in text
    assert effective.manifest_hash == "m"
