"""Validated repair-agent reports for unresolved human-review work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ass_parser import build_pairs_from_ass_lines, parse_ass_file
from .fsutil import require_exact_fields, sha256_file


REPORT_VERSION = 2
REPORT_SYSTEM_PROMPT = """You are the subtitle repair agent in human-review report mode.
The host supplies the complete set of unresolved issues and verified coverage facts.
Return exactly one JSON object with fields coverage_note, glossary_items, and groups.
Partition every supplied issue ID into review groups: each original ID must appear exactly
once. Merge issues only when they describe the same underlying correction on overlapping
cues. Keep distinct problems separate even when they affect the same cue; when uncertain,
use separate groups. Each group must contain exactly issue_ids, priority, title, summary,
and review_action. Each glossary item must contain exactly eng, priority, summary, and
review_action. priority must be high, medium, or low. Keep titles, summaries, and review
actions concise and in Chinese. Treat QA findings as allegations requiring human
confirmation, not as established errors. Do not return Markdown or commentary."""


def report_path_for_review(review_path: Path) -> Path:
    """Place the human-readable sidecar beside the review subtitle."""

    review = Path(review_path)
    return review.with_name(f"{review.stem}.unresolved.md")


def _load_json(path: Path, location: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{location} must be a JSON object")
    return value


def _session_suggestions(repair_state_path: Path) -> list[dict[str, Any]]:
    state_path = Path(repair_state_path)
    state = _load_json(state_path, "repair state")
    generation_dir = state.get("generation_dir")
    artifacts = state.get("artifacts")
    if type(generation_dir) is not str or not generation_dir:
        raise ValueError("repair state generation_dir must be a non-empty string")
    relative = Path(generation_dir)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("repair state generation_dir must stay inside the session")
    if type(artifacts) is not dict or type(artifacts.get("suggestions.json")) is not str:
        raise ValueError("repair state must hash suggestions.json")
    suggestions_path = state_path.parent / relative / "suggestions.json"
    if sha256_file(suggestions_path) != artifacts["suggestions.json"]:
        raise ValueError("repair state suggestions hash does not match")
    payload = _load_json(suggestions_path, "repair suggestions")
    suggestions = payload.get("suggestions")
    if type(suggestions) is not list or any(type(entry) is not dict for entry in suggestions):
        raise ValueError("repair suggestions must contain an array of objects")
    return suggestions


def _suggestion_map(values: list[dict[str, Any]], location: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(values):
        issue_id = entry.get("issue_id")
        if type(issue_id) is not str or not issue_id:
            raise ValueError(f"{location}[{index}].issue_id must be a non-empty string")
        if issue_id in result:
            raise ValueError(f"{location} contains duplicate issue id: {issue_id}")
        result[issue_id] = entry
    return result


def build_report_source(
    *,
    decision_log_path: Path,
    repair_state_path: Path,
    coverage_path: Path,
    glossary_decisions_path: Path,
    suggestion_pool_path: Path,
    current_artifact_path: Path,
    review_artifact_path: Path,
) -> dict[str, Any]:
    """Bind final unresolved issues and coverage gaps to trusted run artifacts."""

    current_sha256 = sha256_file(current_artifact_path)
    review_sha256 = sha256_file(review_artifact_path)
    if current_sha256 != review_sha256:
        raise ValueError("review artifact does not match the current repair artifact")

    decision_payload = _load_json(decision_log_path, "repair decision log")
    decisions = decision_payload.get("decisions")
    if type(decisions) is not list or any(type(entry) is not dict for entry in decisions):
        raise ValueError("repair decision log must contain a decisions array")
    latest_decisions: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        issue_id = decision.get("issue_id")
        status = decision.get("status")
        reason = decision.get("reason")
        if any(type(value) is not str or not value for value in (issue_id, status, reason)):
            raise ValueError(f"repair decision log decision {index} is invalid")
        latest_decisions[issue_id] = decision

    cumulative = _suggestion_map(
        _session_suggestions(repair_state_path), "repair suggestions"
    )
    pool_payload = _load_json(suggestion_pool_path, "QA suggestion pool")
    pool_values = pool_payload.get("suggestions")
    if type(pool_values) is not list or any(type(entry) is not dict for entry in pool_values):
        raise ValueError("QA suggestion pool must contain a suggestions array")
    latest_pool = _suggestion_map(pool_values, "QA suggestion pool")
    suggestions = {**cumulative, **latest_pool}

    closed = {"resolved", "merged", "dismissed", "kept"}
    unresolved_ids = {
        issue_id
        for issue_id, decision in latest_decisions.items()
        if decision["status"] == "escalated"
    }
    unresolved_ids.update(
        issue_id
        for issue_id in latest_pool
        if issue_id not in latest_decisions or latest_decisions[issue_id]["status"] not in closed
    )
    missing_suggestions = sorted(unresolved_ids - set(suggestions))
    if missing_suggestions:
        raise ValueError(f"unresolved decisions lack suggestions: {missing_suggestions}")

    _, current_lines = parse_ass_file(current_artifact_path)
    current_pairs = {pair.id: pair for pair in build_pairs_from_ass_lines(current_lines)}
    issues: list[dict[str, Any]] = []
    for issue_id in unresolved_ids:
        suggestion = suggestions[issue_id]
        affected_ids = suggestion.get("affected_ids")
        if (
            type(affected_ids) is not list
            or not affected_ids
            or any(type(cue_id) is not int or cue_id not in current_pairs for cue_id in affected_ids)
        ):
            raise ValueError(f"unresolved suggestion {issue_id} has invalid affected_ids")
        kind = suggestion.get("kind")
        diagnosis = suggestion.get("diagnosis")
        if type(kind) is not str or not kind or type(diagnosis) is not str or not diagnosis:
            raise ValueError(f"unresolved suggestion {issue_id} lacks kind or diagnosis")
        decision = latest_decisions.get(issue_id)
        issues.append(
            {
                "issue_id": issue_id,
                "status": decision["status"] if decision is not None else "unreviewed",
                "decision_reason": decision["reason"] if decision is not None else "QA finding not processed before review",
                "affected_ids": affected_ids,
                "kind": kind,
                "diagnosis": diagnosis,
                "evidence": suggestion.get("evidence", []),
                "suggested_translations": suggestion.get("suggested_translations", []),
                "cues": [
                    {
                        "id": cue_id,
                        "english": current_pairs[cue_id].eng,
                        "chinese": current_pairs[cue_id].chinese,
                    }
                    for cue_id in affected_ids
                ],
            }
        )
    issues.sort(key=lambda entry: (entry["affected_ids"][0], entry["issue_id"]))

    coverage_payload = _load_json(coverage_path, "repair coverage")
    coverage = coverage_payload.get("coverage")
    if type(coverage) is not list or any(type(entry) is not dict for entry in coverage):
        raise ValueError("repair coverage must contain a coverage array")
    covered_ids = {
        cue_id
        for entry in coverage
        if entry.get("completed") is True
        for cue_id in entry.get("covered_ids", [])
        if type(cue_id) is int
    }
    all_ids = set(current_pairs)
    if not covered_ids <= all_ids:
        raise ValueError("repair coverage contains unknown cue ids")
    missing_ids = sorted(all_ids - covered_ids)
    glossary_payload = _load_json(glossary_decisions_path, "glossary decisions")
    glossary_unresolved = glossary_payload.get("unresolved")
    if type(glossary_unresolved) is not list or any(
        type(entry) is not dict for entry in glossary_unresolved
    ):
        raise ValueError("glossary decisions must contain an unresolved array")
    glossary_terms: list[dict[str, Any]] = []
    seen_terms: set[str] = set()
    for index, entry in enumerate(glossary_unresolved):
        eng = entry.get("eng")
        reason = entry.get("reason")
        evidence_ids = entry.get("evidence_ids")
        if type(eng) is not str or not eng or eng in seen_terms:
            raise ValueError(f"glossary unresolved entry {index} has an invalid eng")
        if type(reason) is not str or not reason:
            raise ValueError(f"glossary unresolved entry {index} lacks reason")
        if type(evidence_ids) is not list or any(
            type(cue_id) is not int or cue_id not in current_pairs for cue_id in evidence_ids
        ):
            raise ValueError(f"glossary unresolved entry {index} has invalid evidence_ids")
        seen_terms.add(eng)
        glossary_terms.append(
            {
                "eng": eng,
                "candidate_zh": entry.get("zh", ""),
                "reason": reason,
                "source": entry.get("source", "unknown"),
                "evidence_ids": evidence_ids,
                "cues": [
                    {
                        "id": cue_id,
                        "english": current_pairs[cue_id].eng,
                        "chinese": current_pairs[cue_id].chinese,
                    }
                    for cue_id in evidence_ids
                ],
            }
        )
    glossary_terms.sort(key=lambda entry: entry["eng"].casefold())
    return {
        "version": REPORT_VERSION,
        "review": {
            "name": Path(review_artifact_path).name,
            "sha256": review_sha256,
        },
        "current_sha256": current_sha256,
        "unresolved_issues": issues,
        "unresolved_glossary": glossary_terms,
        "coverage": {
            "covered": len(covered_ids),
            "total": len(all_ids),
            "missing_ids": missing_ids,
            "full_sweeps_completed": coverage_payload.get("full_sweeps_completed", 0),
        },
    }


def parse_agent_report(
    value: object,
    issue_cues: dict[str, set[int]],
    expected_glossary_terms: set[str],
) -> dict[str, Any]:
    """Validate an exact, overlap-connected partition of the unresolved issues."""

    payload = require_exact_fields(
        value,
        {"coverage_note", "glossary_items", "groups"},
        location="report response",
    )
    if type(payload["coverage_note"]) is not str or not payload["coverage_note"].strip():
        raise ValueError("report response coverage_note must be a non-empty string")
    groups = payload["groups"]
    if type(groups) is not list:
        raise ValueError("report response groups must be an array")
    normalized_groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(groups):
        group = require_exact_fields(
            value,
            {"issue_ids", "priority", "title", "summary", "review_action"},
            location=f"report response.groups[{index}]",
        )
        if type(group["priority"]) is not str or group["priority"] not in {
            "high",
            "medium",
            "low",
        }:
            raise ValueError(f"report response.groups[{index}].priority is invalid")
        for field in ("title", "summary", "review_action"):
            if type(group[field]) is not str or not group[field].strip():
                raise ValueError(f"report response.groups[{index}].{field} is required")
        issue_ids = group["issue_ids"]
        if (
            type(issue_ids) is not list
            or not issue_ids
            or any(type(issue_id) is not str or not issue_id for issue_id in issue_ids)
        ):
            raise ValueError(f"report response.groups[{index}].issue_ids is invalid")
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError(f"report response.groups[{index}] repeats an issue id")
        unknown = set(issue_ids) - set(issue_cues)
        if unknown:
            raise ValueError(
                f"report response.groups[{index}] contains unknown issue ids: {sorted(unknown)}"
            )
        duplicate = set(issue_ids) & seen
        if duplicate:
            raise ValueError(
                f"report response contains duplicate issue ids: {sorted(duplicate)}"
            )
        if len(issue_ids) > 1:
            reachable = {issue_ids[0]}
            cue_union = set(issue_cues[issue_ids[0]])
            changed = True
            while changed:
                changed = False
                for issue_id in issue_ids:
                    if issue_id not in reachable and cue_union & issue_cues[issue_id]:
                        reachable.add(issue_id)
                        cue_union.update(issue_cues[issue_id])
                        changed = True
            if len(reachable) != len(issue_ids):
                raise ValueError(
                    f"report response.groups[{index}] merges issues without overlapping cues"
                )
        seen.update(issue_ids)
        normalized_groups.append(
            {
                "issue_ids": list(issue_ids),
                "priority": group["priority"].strip(),
                "title": group["title"].strip(),
                "summary": group["summary"].strip(),
                "review_action": group["review_action"].strip(),
            }
        )
    expected_issue_ids = set(issue_cues)
    if seen != expected_issue_ids:
        missing = sorted(expected_issue_ids - seen)
        unknown = sorted(seen - expected_issue_ids)
        raise ValueError(f"report response issue ids differ (missing={missing}; unknown={unknown})")
    glossary_items = payload["glossary_items"]
    if type(glossary_items) is not list:
        raise ValueError("report response glossary_items must be an array")
    normalized_glossary: list[dict[str, str]] = []
    seen_terms: set[str] = set()
    for index, value in enumerate(glossary_items):
        item = require_exact_fields(
            value,
            {"eng", "priority", "summary", "review_action"},
            location=f"report response.glossary_items[{index}]",
        )
        if type(item["priority"]) is not str or item["priority"] not in {
            "high",
            "medium",
            "low",
        }:
            raise ValueError(f"report response.glossary_items[{index}].priority is invalid")
        for field in ("eng", "summary", "review_action"):
            if type(item[field]) is not str or not item[field].strip():
                raise ValueError(f"report response.glossary_items[{index}].{field} is required")
        if item["eng"] in seen_terms:
            raise ValueError(f"report response contains duplicate glossary term: {item['eng']}")
        seen_terms.add(item["eng"])
        normalized_glossary.append({field: item[field].strip() for field in item})
    if seen_terms != expected_glossary_terms:
        missing = sorted(expected_glossary_terms - seen_terms)
        unknown = sorted(seen_terms - expected_glossary_terms)
        raise ValueError(
            f"report response glossary terms differ (missing={missing}; unknown={unknown})"
        )
    return {
        "coverage_note": payload["coverage_note"].strip(),
        "glossary_items": normalized_glossary,
        "groups": normalized_groups,
    }


def _single_line(value: object) -> str:
    return " ".join(str(value).split())


def _ranges(values: list[int]) -> str:
    if not values:
        return "none"
    groups: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(groups)


def render_markdown(source: dict[str, Any], agent_report: dict[str, Any]) -> str:
    """Render trusted facts plus validated repair-agent guidance."""

    issues_by_id = {item["issue_id"]: item for item in source["unresolved_issues"]}
    groups = sorted(
        agent_report["groups"],
        key=lambda group: (
            min(
                cue_id
                for issue_id in group["issue_ids"]
                for cue_id in issues_by_id[issue_id]["affected_ids"]
            ),
            group["title"],
        ),
    )
    glossary_guidance = {item["eng"]: item for item in agent_report["glossary_items"]}
    coverage = source["coverage"]
    lines = [
        "# 未决字幕审核事项",
        "",
        f"- 待审核字幕：`{source['review']['name']}`",
        f"- 字幕 SHA-256：`{source['review']['sha256']}`",
        f"- 原始未决 issue：{len(source['unresolved_issues'])}",
        f"- 合并后审核项：{len(groups)}",
        f"- 未决术语：{len(source['unresolved_glossary'])}",
        f"- 有效覆盖：{coverage['covered']}/{coverage['total']} cues",
        f"- 未覆盖 cue：{len(coverage['missing_ids'])}（{_ranges(coverage['missing_ids'])}）",
        f"- repair 全集巡检计数：{coverage['full_sweeps_completed']}",
        "",
        "> 本报告列出的是尚需人工确认的 QA 指控与覆盖缺口，不代表已经确认的翻译错误。",
        "",
        "## 覆盖说明",
        "",
        _single_line(agent_report["coverage_note"]),
        "",
    ]
    generator = source.get("generator")
    if type(generator) is dict:
        lines.insert(3, f"- 生成角色/模型：`{generator['role']}` / `{generator['model']}`")
    if source["unresolved_glossary"]:
        lines.extend(["## 未决术语", ""])
        for term in source["unresolved_glossary"]:
            item = glossary_guidance[term["eng"]]
            lines.extend(
                [
                    f"### {term['eng']}",
                    "",
                    f"- 优先级：`{item['priority']}`",
                    f"- 未决原因：{_single_line(term['reason'])}",
                    f"- Repair agent 摘要：{_single_line(item['summary'])}",
                    f"- 建议人工动作：{_single_line(item['review_action'])}",
                    f"- 证据 cue：{', '.join(str(value) for value in term['evidence_ids'])}",
                    "",
                ]
            )
            for cue in term["cues"]:
                lines.extend(
                    [
                        f"- `{cue['id']}` EN: {_single_line(cue['english'])}",
                        f"- `{cue['id']}` ZH: {_single_line(cue['chinese'])}",
                    ]
                )
            lines.append("")
    if groups:
        lines.extend(["## 未决字幕问题", ""])
    for index, group in enumerate(groups, start=1):
        issues = [issues_by_id[issue_id] for issue_id in group["issue_ids"]]
        cue_map = {
            cue["id"]: cue for issue in issues for cue in issue["cues"]
        }
        cue_ids = sorted(cue_map)
        kinds = sorted({issue["kind"] for issue in issues})
        statuses = sorted({issue["status"] for issue in issues})
        reasons = sorted({issue["decision_reason"] for issue in issues})
        lines.extend(
            [
                f"## {index}. cues {', '.join(str(value) for value in cue_ids)} · {group['title']}",
                "",
                f"- 原始 Issue IDs：{', '.join(f'`{value}`' for value in group['issue_ids'])}",
                f"- 问题类型：{', '.join(f'`{value}`' for value in kinds)}",
                f"- 状态：{', '.join(f'`{value}`' for value in statuses)}",
                f"- 优先级：`{group['priority']}`",
                f"- 未决原因：{'；'.join(_single_line(value) for value in reasons)}",
                f"- Repair agent 摘要：{_single_line(group['summary'])}",
                f"- 建议人工动作：{_single_line(group['review_action'])}",
                "",
                "### QA 诊断（合并来源）",
                "",
            ]
        )
        for issue in issues:
            lines.append(
                f"- `{issue['issue_id']}`：{_single_line(issue['diagnosis'])}"
            )
        lines.extend(
            [
                "",
                "### 当前字幕",
                "",
            ]
        )
        for cue_id in cue_ids:
            cue = cue_map[cue_id]
            lines.extend(
                [
                    f"- `{cue['id']}` EN: {_single_line(cue['english'])}",
                    f"- `{cue['id']}` ZH: {_single_line(cue['chinese'])}",
                ]
            )
        suggested = [
            (issue["issue_id"], translation)
            for issue in issues
            for translation in issue["suggested_translations"]
            if type(translation) is dict
        ]
        if suggested:
            lines.extend(["", "### QA 建议（未确认）", ""])
            seen_suggestions: set[tuple[object, str]] = set()
            for issue_id, translation in suggested:
                key = (
                    translation.get("id", "?"),
                    _single_line(translation.get("translation", "")),
                )
                if key in seen_suggestions:
                    continue
                seen_suggestions.add(key)
                lines.append(f"- `{issue_id}` · cue `{key[0]}`：{key[1]}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
