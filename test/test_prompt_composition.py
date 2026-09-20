from pathlib import Path
from types import SimpleNamespace

from subretrans.config import PromptPaths
from subretrans.prompts import load_main_prompt_template, load_qa_prompt_template


def test_stage_prompts_share_rules_and_keep_distinct_tasks(tmp_path: Path) -> None:
    shared = tmp_path / "shared.md"
    refine = tmp_path / "refine.md"
    qa = tmp_path / "qa.md"
    shared.write_text("SHARED RULES\n", encoding="utf-8")
    refine.write_text("REFINE TASK\n", encoding="utf-8")
    qa.write_text("QA TASK\n", encoding="utf-8")
    paths = PromptPaths(shared=shared, refine=refine, qa=qa)

    refine_prompt = load_main_prompt_template(SimpleNamespace(prompt_paths=paths))
    qa_prompt = load_qa_prompt_template(paths)

    assert refine_prompt == "SHARED RULES\n\nREFINE TASK\n"
    assert qa_prompt == "SHARED RULES\n\nQA TASK\n"
    assert refine_prompt.count("SHARED RULES") == 1
    assert qa_prompt.count("SHARED RULES") == 1
