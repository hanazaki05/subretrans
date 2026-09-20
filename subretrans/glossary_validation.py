"""Conservative terminology validation and effective-glossary construction.

The extraction model is allowed to propose terminology, but it is not an
authority.  This module keeps the authoritative user glossary and configured
episode replacements separate from learned candidates, validates evidence
against the canonical cue ids, and records every accept/reject decision.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060]")
ASS_TAG_RE = re.compile(r"\{[^}]*\}|</?[^>]+>")
WORD_RE = r"[\w\u4e00-\u9fff]"

GlossarySource = Literal["user_glossary", "episode_replacement", "learned"]


def normalize_glossary_text(value: Any) -> str:
    """Normalize text for glossary keys and source-term matching."""

    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\u00a0", " ").replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value.strip())
    return value.casefold()


def normalize_term_key(value: Any) -> str:
    """Return the canonical case-insensitive key used by every glossary path."""

    return normalize_glossary_text(value)


def normalize_source_text(value: Any) -> str:
    """Normalize English source text while removing ASS/HTML presentation tags."""

    if not isinstance(value, str):
        return ""
    return normalize_glossary_text(ASS_TAG_RE.sub(" ", value))


def term_occurs(term: str, source_text: str) -> bool:
    """Whether a complete term occurs with token boundaries in source text.

    A bare name also matches its English possessive form (``Rabb`` matches
    ``Rabb's``), while a longer phrase is never reduced to a substring of a
    larger token.
    """

    normalized_term = normalize_source_text(term)
    normalized_source = normalize_source_text(source_text)
    if not normalized_term or not normalized_source:
        return False
    pattern = (
        rf"(?<!{WORD_RE}){re.escape(normalized_term)}"
        rf"(?:'s)?(?!{WORD_RE})"
    )
    return re.search(pattern, normalized_source, flags=re.IGNORECASE) is not None


def order_terms_longest_first(terms: Sequence[str]) -> tuple[str, ...]:
    """Order terms for matching/replacement without discarding shorter terms."""

    return tuple(
        sorted(
            (term for term in terms if isinstance(term, str) and term.strip()),
            key=lambda term: (-len(normalize_source_text(term)), -len(term), term.casefold()),
        )
    )


def _matched_authoritative_terms(
    learned_eng: str, authoritative: Sequence[GlossaryTerm]
) -> tuple[GlossaryTerm, ...]:
    """Return longest, non-overlapping authoritative phrases in a learned key."""

    source = normalize_source_text(learned_eng)
    matches: list[tuple[int, int, GlossaryTerm]] = []
    for term in authoritative:
        if term.source != "user_glossary":
            continue
        needle = normalize_source_text(term.eng)
        if not needle:
            continue
        pattern = rf"(?<!{WORD_RE}){re.escape(needle)}(?:'s)?(?!{WORD_RE})"
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match is not None:
            matches.append((match.start(), match.end(), term))
    selected: list[tuple[int, int, GlossaryTerm]] = []
    for start, end, term in sorted(matches, key=lambda value: (-(value[1] - value[0]), value[0])):
        if any(not (end <= chosen_start or start >= chosen_end) for chosen_start, chosen_end, _ in selected):
            continue
        selected.append((start, end, term))
    return tuple(term for _, _, term in sorted(selected, key=lambda value: value[0]))


def matched_authoritative_terms(
    source_text: str, authoritative: Sequence[GlossaryTerm]
) -> tuple[GlossaryTerm, ...]:
    """Return the longest non-overlapping user-authority matches in source text."""

    return _matched_authoritative_terms(source_text, authoritative)


@dataclass(frozen=True)
class GlossaryTerm:
    """A prompt-visible mapping with provenance and evidence binding."""

    eng: str
    zh: str
    source: GlossarySource
    type: str | None = None
    confidence: float | None = None
    evidence_ids: tuple[int, ...] = ()
    episode_id: str | None = None
    manifest_hash: str | None = None
    artifact_hash: str | None = None

    @property
    def key(self) -> str:
        return normalize_term_key(self.eng)

    @property
    def provenance(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "episode_id": self.episode_id,
            "manifest_hash": self.manifest_hash,
            "artifact_hash": self.artifact_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"eng": self.eng, "zh": self.zh, "source": self.source}
        if self.type is not None:
            payload["type"] = self.type
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.evidence_ids:
            payload["evidence_ids"] = list(self.evidence_ids)
        if self.episode_id is not None:
            payload["episode_id"] = self.episode_id
        if self.manifest_hash is not None:
            payload["manifest_hash"] = self.manifest_hash
        if self.artifact_hash is not None:
            payload["artifact_hash"] = self.artifact_hash
        return payload


@dataclass(frozen=True)
class GlossaryDecision:
    """An auditable accept/reject decision for one learned candidate."""

    eng: str
    zh: str
    decision: Literal["accepted", "rejected", "unresolved"]
    reason: str
    source: GlossarySource = "learned"
    evidence_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "eng": self.eng,
            "zh": self.zh,
            "decision": self.decision,
            "reason": self.reason,
            "source": self.source,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class EvidenceIssue:
    """A recoverable evidence error isolated from the rest of an extraction."""

    eng: str
    evidence_id: int | None
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "eng": self.eng,
            "evidence_id": self.evidence_id,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CandidateValidation:
    """Validated learned terms plus isolated decisions and evidence issues."""

    accepted: tuple[GlossaryTerm, ...]
    decisions: tuple[GlossaryDecision, ...]
    unresolved: tuple[GlossaryDecision, ...]
    evidence_issues: tuple[EvidenceIssue, ...]


@dataclass(frozen=True)
class EffectiveGlossary:
    """Frozen prompt/QA glossary with authority, provenance and decisions."""

    authoritative: tuple[GlossaryTerm, ...]
    learned: tuple[GlossaryTerm, ...]
    decisions: tuple[GlossaryDecision, ...] = ()
    unresolved: tuple[GlossaryDecision, ...] = ()
    episode_id: str | None = None
    manifest_hash: str | None = None
    artifact_hash: str | None = None

    @property
    def terms(self) -> tuple[GlossaryTerm, ...]:
        return self.authoritative + self.learned

    @property
    def authoritative_keys(self) -> frozenset[str]:
        return frozenset(term.key for term in self.authoritative if term.key)

    def lookup(self, eng: str) -> GlossaryTerm | None:
        key = normalize_term_key(eng)
        for term in self.terms:
            if term.key == key:
                return term
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "manifest_hash": self.manifest_hash,
            "artifact_hash": self.artifact_hash,
            "authoritative": [term.to_dict() for term in self.authoritative],
            "learned": [term.to_dict() for term in self.learned],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "unresolved": [decision.to_dict() for decision in self.unresolved],
        }


def _mapping_term(
    value: Mapping[str, Any],
    *,
    source: GlossarySource,
    episode_id: str | None,
    manifest_hash: str | None,
    artifact_hash: str | None,
) -> GlossaryTerm | None:
    eng = value.get("eng")
    zh = value.get("zh")
    if not isinstance(eng, str) or not eng.strip() or not isinstance(zh, str) or not zh.strip():
        return None
    evidence = value.get("evidence_ids", ())
    if isinstance(evidence, list):
        evidence = tuple(evidence)
    if not isinstance(evidence, tuple):
        evidence = ()
    return GlossaryTerm(
        eng=eng.strip(),
        zh=zh.strip(),
        source=source,
        type=value.get("type") if isinstance(value.get("type"), str) else None,
        confidence=(
            float(value["confidence"])
            if isinstance(value.get("confidence"), (int, float))
            and not isinstance(value.get("confidence"), bool)
            else None
        ),
        evidence_ids=evidence,
        episode_id=value.get("episode_id", episode_id)
        if isinstance(value.get("episode_id", episode_id), str)
        else episode_id,
        manifest_hash=value.get("manifest_hash", manifest_hash)
        if isinstance(value.get("manifest_hash", manifest_hash), str)
        else manifest_hash,
        artifact_hash=value.get("artifact_hash", artifact_hash)
        if isinstance(value.get("artifact_hash", artifact_hash), str)
        else artifact_hash,
    )


def _as_term(
    value: Any,
    *,
    source: GlossarySource,
    episode_id: str | None,
    manifest_hash: str | None,
    artifact_hash: str | None,
) -> GlossaryTerm | None:
    if isinstance(value, GlossaryTerm):
        return value
    if isinstance(value, Mapping):
        return _mapping_term(
            value,
            source=source,
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
    # ``memory.extract_memory_update`` uses its frozen ``TerminologyEntry``
    # value internally.  Keep this module independent of memory.py while
    # accepting that structured candidate shape before evidence validation.
    if hasattr(value, "eng") and hasattr(value, "zh"):
        eng, zh = getattr(value, "eng"), getattr(value, "zh")
        if not isinstance(eng, str) or not isinstance(zh, str) or not eng.strip() or not zh.strip():
            return None
        evidence = getattr(value, "evidence_ids", ())
        if isinstance(evidence, list):
            evidence = tuple(evidence)
        if not isinstance(evidence, tuple):
            evidence = ()
        confidence = getattr(value, "confidence", None)
        return GlossaryTerm(
            eng=eng.strip(),
            zh=zh.strip(),
            source=source,
            type=getattr(value, "type", None) if isinstance(getattr(value, "type", None), str) else None,
            confidence=confidence if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else None,
            evidence_ids=evidence,
            episode_id=getattr(value, "episode_id", episode_id)
            if isinstance(getattr(value, "episode_id", episode_id), str)
            else episode_id,
            manifest_hash=getattr(value, "manifest_hash", manifest_hash)
            if isinstance(getattr(value, "manifest_hash", manifest_hash), str)
            else manifest_hash,
            artifact_hash=getattr(value, "artifact_hash", artifact_hash)
            if isinstance(getattr(value, "artifact_hash", artifact_hash), str)
            else artifact_hash,
        )
    return None


def _replacement_pair(value: Any) -> tuple[str, str] | None:
    if isinstance(value, Mapping):
        left, right = value.get("from"), value.get("to")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        left, right = value
    else:
        return None
    if not isinstance(left, str) or not left.strip() or not isinstance(right, str):
        return None
    return left.strip(), right.strip()


def build_effective_glossary(
    user_glossary: Sequence[GlossaryTerm | Mapping[str, Any]],
    learned_glossary: Sequence[GlossaryTerm | Mapping[str, Any]] = (),
    episode_replacements: Sequence[Any] = (),
    *,
    episode_id: str | None = None,
    manifest_hash: str | None = None,
    artifact_hash: str | None = None,
) -> EffectiveGlossary:
    """Build a frozen glossary with user and replacement authority."""

    authoritative: list[GlossaryTerm] = []
    decisions: list[GlossaryDecision] = []
    by_key: dict[str, GlossaryTerm] = {}
    replacement_left_keys: set[str] = set()

    for value in user_glossary:
        term = _as_term(
            value,
            source="user_glossary",
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
        if term is None or not term.key:
            continue
        if term.key in by_key:
            decisions.append(
                GlossaryDecision(term.eng, term.zh, "rejected", "duplicate-user-authority", "user_glossary")
            )
            continue
        authoritative.append(term)
        by_key[term.key] = term

    for raw_replacement in episode_replacements:
        pair = _replacement_pair(raw_replacement)
        if pair is None:
            decisions.append(
                GlossaryDecision("", "", "unresolved", "invalid-episode-replacement", "episode_replacement")
            )
            continue
        left, right = pair
        left_key = normalize_term_key(left)
        replacement_left_keys.add(left_key)
        if not left_key:
            continue
        replacement_term = GlossaryTerm(
            eng=left,
            zh=right,
            source="episode_replacement",
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
        if left_key in by_key:
            decisions.append(
                GlossaryDecision(
                    left,
                    right,
                    "rejected",
                    "user-glossary-authority-wins",
                    "episode_replacement",
                )
            )
            continue
        authoritative.append(replacement_term)
        by_key[left_key] = replacement_term

    learned: list[GlossaryTerm] = []
    learned_keys: set[str] = set()
    for value in learned_glossary:
        term = _as_term(
            value,
            source="learned",
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
        if term is None or not term.key:
            continue
        reason: str | None = None
        if term.key in by_key:
            reason = "duplicate-authoritative-key"
        elif term.key in learned_keys:
            reason = "duplicate-learned-key"
        elif any(
            normalize_term_key(authority.zh) not in normalize_term_key(term.zh)
            for authority in _matched_authoritative_terms(term.eng, authoritative)
        ):
            reason = "authoritative-phrase-conflict"
        elif any(left_key in normalize_term_key(term.zh) for left_key in replacement_left_keys):
            reason = "episode-replacement-left-value-conflict"
        if reason is not None:
            decisions.append(
                GlossaryDecision(
                    term.eng,
                    term.zh,
                    "rejected",
                    reason,
                    "learned",
                    term.evidence_ids,
                )
            )
            continue
        learned.append(term)
        learned_keys.add(term.key)

    unresolved = tuple(decision for decision in decisions if decision.decision == "unresolved")
    return EffectiveGlossary(
        authoritative=tuple(authoritative),
        learned=tuple(learned),
        decisions=tuple(decisions),
        unresolved=unresolved,
        episode_id=episode_id,
        manifest_hash=manifest_hash,
        artifact_hash=artifact_hash,
    )


def _cue_id_and_english(cue: Any) -> tuple[int, str] | None:
    if hasattr(cue, "pair_id") and hasattr(cue, "english"):
        pair_id, english = cue.pair_id, cue.english
    elif hasattr(cue, "id") and hasattr(cue, "eng"):
        pair_id, english = cue.id, cue.eng
    else:
        return None
    if type(pair_id) is not int or type(english) is not str:
        return None
    return pair_id, english


def validate_candidate_evidence(
    candidate: GlossaryTerm | Mapping[str, Any],
    cues: Sequence[Any],
) -> tuple[tuple[int, ...], tuple[EvidenceIssue, ...]]:
    """Validate strict evidence ids and complete normalized source-term matches."""

    eng = candidate.eng if isinstance(candidate, GlossaryTerm) else candidate.get("eng", "")
    raw_ids = candidate.evidence_ids if isinstance(candidate, GlossaryTerm) else candidate.get("evidence_ids", ())
    if not isinstance(eng, str):
        return (), (EvidenceIssue("", None, "invalid-term", "eng must be a string"),)
    if not isinstance(raw_ids, (list, tuple)):
        return (), (EvidenceIssue(eng, None, "invalid-evidence-shape", "evidence_ids must be a list"),)

    cue_map: dict[int, str] = {}
    for cue in cues:
        value = _cue_id_and_english(cue)
        if value is not None:
            cue_map[value[0]] = value[1]
    valid: list[int] = []
    issues: list[EvidenceIssue] = []
    for raw_id in raw_ids:
        if type(raw_id) is not int:
            issues.append(EvidenceIssue(eng, None, "invalid-evidence-id", "evidence id must be a strict integer"))
            continue
        if raw_id in valid:
            issues.append(EvidenceIssue(eng, raw_id, "duplicate-evidence-id", "duplicate evidence id"))
            continue
        if raw_id not in cue_map:
            issues.append(EvidenceIssue(eng, raw_id, "unknown-cue", "evidence id is not in the cue manifest"))
            continue
        if not term_occurs(eng, cue_map[raw_id]):
            issues.append(
                EvidenceIssue(
                    eng,
                    raw_id,
                    "term-not-in-source",
                    "complete normalized term does not occur in the English cue",
                )
            )
            continue
        if len(valid) >= 5:
            issues.append(EvidenceIssue(eng, raw_id, "too-many-evidence-ids", "at most five evidence ids are allowed"))
            continue
        valid.append(raw_id)
    return tuple(valid), tuple(issues)


def validate_learned_candidates(
    candidates: Sequence[GlossaryTerm | Mapping[str, Any]],
    cues: Sequence[Any],
    authoritative: EffectiveGlossary,
    *,
    episode_id: str | None = None,
    manifest_hash: str | None = None,
    artifact_hash: str | None = None,
) -> CandidateValidation:
    """Validate candidates before they can enter ``GlobalMemory.glossary``."""

    accepted: list[GlossaryTerm] = []
    decisions: list[GlossaryDecision] = []
    unresolved: list[GlossaryDecision] = []
    issues: list[EvidenceIssue] = []
    existing_keys = {term.key for term in authoritative.learned}
    authoritative_keys = authoritative.authoritative_keys
    replacement_left_keys = {
        normalize_term_key(term.eng)
        for term in authoritative.authoritative
        if term.source == "episode_replacement"
    }

    for raw_candidate in candidates:
        term = _as_term(
            raw_candidate,
            source="learned",
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
        if term is None or not term.key:
            decision = GlossaryDecision("", "", "unresolved", "invalid-candidate", "learned")
            decisions.append(decision)
            unresolved.append(decision)
            continue
        valid_ids, candidate_issues = validate_candidate_evidence(term, cues)
        issues.extend(candidate_issues)
        if not valid_ids:
            decision = GlossaryDecision(
                term.eng,
                term.zh,
                "unresolved",
                "invalid-or-empty-evidence",
                "learned",
                valid_ids,
            )
            decisions.append(decision)
            unresolved.append(decision)
            continue
        term = GlossaryTerm(
            eng=term.eng,
            zh=term.zh,
            source="learned",
            type=term.type,
            confidence=term.confidence,
            evidence_ids=valid_ids,
            episode_id=episode_id,
            manifest_hash=manifest_hash,
            artifact_hash=artifact_hash,
        )
        if term.key in authoritative_keys:
            decisions.append(
                GlossaryDecision(term.eng, term.zh, "rejected", "duplicate-authoritative-key", "learned", valid_ids)
            )
            continue
        if term.key in existing_keys:
            decisions.append(
                GlossaryDecision(term.eng, term.zh, "rejected", "duplicate-learned-key", "learned", valid_ids)
            )
            continue
        if any(
            normalize_term_key(authority.zh) not in normalize_term_key(term.zh)
            for authority in _matched_authoritative_terms(term.eng, authoritative.authoritative)
        ):
            decisions.append(
                GlossaryDecision(
                    term.eng,
                    term.zh,
                    "rejected",
                    "authoritative-phrase-conflict",
                    "learned",
                    valid_ids,
                )
            )
            continue
        if any(left_key in normalize_term_key(term.zh) for left_key in replacement_left_keys):
            decisions.append(
                GlossaryDecision(
                    term.eng,
                    term.zh,
                    "rejected",
                    "episode-replacement-left-value-conflict",
                    "learned",
                    valid_ids,
                )
            )
            continue
        accepted.append(term)
        existing_keys.add(term.key)
        decisions.append(GlossaryDecision(term.eng, term.zh, "accepted", "validated", "learned", valid_ids))

    return CandidateValidation(tuple(accepted), tuple(decisions), tuple(unresolved), tuple(issues))


__all__ = [
    "CandidateValidation",
    "EffectiveGlossary",
    "EvidenceIssue",
    "GlossaryDecision",
    "GlossaryTerm",
    "GlossarySource",
    "build_effective_glossary",
    "normalize_glossary_text",
    "normalize_source_text",
    "normalize_term_key",
    "order_terms_longest_first",
    "matched_authoritative_terms",
    "term_occurs",
    "validate_candidate_evidence",
    "validate_learned_candidates",
]
