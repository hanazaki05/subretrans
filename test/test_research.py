from __future__ import annotations

import json

import pytest

from subretrans.config import ResearchSettings
from subretrans.research import (
    EXA_SEARCH_ENDPOINT,
    EvidenceCitation,
    ExaSearchClient,
    RankCandidate,
    RankFinding,
    ResearchRun,
    SearchHit,
    rank_candidates_from_terms,
)
from subretrans.webfetch import FetchedPage


HASH_A = "a" * 64
HASH_B = "b" * 64


def settings(*, max_requests=4, max_fetches_per_request=3, max_response_bytes=4096):
    return ResearchSettings(
        exa_key_file=None,
        timeout=2,
        max_requests=max_requests,
        max_fetches_per_request=max_fetches_per_request,
        max_response_bytes=max_response_bytes,
    )


class FakeResponse:
    def __init__(self, payload, *, content_type="application/json", content_length=None):
        raw = json.dumps(payload).encode("utf-8") if isinstance(payload, dict) else payload
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.raw = raw
        self.closed = False

    def iter_content(self, *, chunk_size):
        yield self.raw

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.trust_env = True

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def candidate(pair_id=12):
    return RankCandidate(
        (pair_id,),
        "Harmon Rabb",
        "Commander",
        "Commander Harmon Rabb",
        "JAG.S07E07",
        HASH_A,
        HASH_B,
    )


def finding(branch="United States Navy"):
    return RankFinding(
        "Harmon Rabb",
        branch,
        "Commander",
        "哈蒙·拉布",
        "中校",
        True,
        (EvidenceCitation("https://example.test/source", "bounded evidence", "chars:0-16"),),
    )


def fetched_page():
    return FetchedPage(
        "https://example.test/source",
        "https://example.test/source",
        200,
        "text/plain",
        "bounded evidence",
        16,
        "c" * 64,
        (),
        "Source",
        "2026-09-21T00:00:00+00:00",
        False,
        "bounded evidence",
        "bounded evidence",
    )


def run_with(searcher, *, parser=None, max_requests=4):
    contexts = []

    def factory(pass_index):
        token = object()
        contexts.append(token)
        return parser or (lambda item, current_pass, pages: finding())

    run = ResearchRun(
        searcher,
        settings(max_requests=max_requests),
        finding_parser_factory=factory,
        page_fetcher=lambda url: fetched_page(),
    )
    return run, contexts


def test_rank_candidate_prefixes_come_from_user_glossary_not_a_builtin_list():
    cue = type("Cue", (), {"pair_id": 4, "english": "Wing Leader Alice Smith reports."})()

    result = rank_candidates_from_terms(
        [{"eng": "Wing Leader Alice Smith", "zh": "爱丽丝·史密斯联队长", "evidence_ids": [4]}],
        [cue],
        rank_terms=[{"eng": "Wing Leader", "zh": "联队长"}],
        episode_id="episode",
        manifest_hash=HASH_A,
        artifact_hash=HASH_B,
    )

    assert len(result) == 1
    assert result[0].source_rank == "Wing Leader"
    assert result[0].full_name == "Alice Smith"


def test_rank_candidate_keeps_surname_only_and_invalid_term_evidence_for_research_gate():
    cue = type("Cue", (), {"pair_id": 365, "english": "Chaplain Turner reports."})()

    result = rank_candidates_from_terms(
        [{"eng": "Commander Turner", "zh": "特纳指挥官", "evidence_ids": [365]}],
        [cue],
        rank_terms=[{"eng": "Commander", "zh": "中校"}],
        episode_id="episode",
        manifest_hash=HASH_A,
        artifact_hash=HASH_B,
    )

    assert len(result) == 1
    assert result[0].full_name == "Turner"
    assert "Chaplain Turner" in result[0].context


def test_exa_client_uses_fixed_endpoint_bounded_json_and_no_environment_proxy():
    response = FakeResponse(
        {
            "results": [
                {"title": "Official", "url": "https://example.test/source#one", "text": "evidence"},
                {"title": "Duplicate", "url": "https://EXAMPLE.test/source", "text": "duplicate"},
            ]
        }
    )
    session = FakeSession(response)

    client = ExaSearchClient("test-key", settings(), session=session)
    hits = client.search("Harmon Rabb Commander", limit=10)

    assert len(hits) == 1
    assert hits[0].url == "https://example.test/source#one"
    assert session.trust_env is False
    assert session.calls[0][0] == EXA_SEARCH_ENDPOINT
    assert session.calls[0][1]["allow_redirects"] is False
    assert session.calls[0][1]["stream"] is True
    assert session.calls[0][1]["timeout"] == 2
    assert session.calls[0][1]["json"]["numResults"] == 3
    assert session.calls[0][1]["headers"]["x-api-key"] == "test-key"
    assert response.closed


class RecordingSearcher:
    def __init__(self):
        self.queries = []

    def search(self, query, *, limit):
        self.queries.append((query, limit))
        return (SearchHit("source", "https://example.test/source", "bounded evidence"),)


def test_two_independent_passes_agree_and_duplicate_candidate_is_cached():
    searcher = RecordingSearcher()
    run, contexts = run_with(searcher)

    first = run.research_candidate(candidate())
    second = run.research_candidate(candidate())

    assert first is second
    assert first.status == "agree"
    assert first.human_review is False
    assert first.proposal is not None
    assert first.proposal.eng == "Commander Harmon Rabb"
    assert first.proposal.zh == "哈蒙·拉布中校"
    assert len(first.passes) == 2
    assert searcher.queries[0][0] != searcher.queries[1][0]
    assert run.requests_used == 2
    assert run.report().decisions == (first,)
    assert len(contexts) == 2
    assert contexts[0] is not contexts[1]
    assert first.passes[0].pages[0].evidence_snapshot == "bounded evidence"


def test_conflicting_findings_create_human_review_without_proposal():
    run, _ = run_with(
        RecordingSearcher(),
        parser=lambda item, pass_index, pages: finding(
            "United States Navy" if pass_index == 1 else "United States Air Force"
        ),
    )

    decision = run.research_candidate(candidate())

    assert decision.status == "conflict"
    assert decision.proposal is None
    assert decision.human_review is True


def test_context_incompatible_or_unquoted_finding_is_human_review():
    incompatible = RankFinding(
        "Harmon Rabb",
        "United States Navy",
        "Commander",
        "哈蒙·拉布",
        "中校",
        False,
        (EvidenceCitation("https://example.test/source", "bounded evidence", "chars:0-16"),),
    )
    run, _ = run_with(
        RecordingSearcher(), parser=lambda item, pass_index, pages: incompatible
    )

    decision = run.research_candidate(candidate())

    assert decision.status == "insufficient"
    assert decision.human_review is True
    assert all(item.error == "insufficient_context_compatible_evidence" for item in decision.passes)


def test_failed_passes_are_insufficient_human_review_items():
    class FailingSearcher:
        def search(self, query, *, limit):
            raise RuntimeError("network unavailable")

    run, _ = run_with(FailingSearcher())
    decision = run.research_candidate(candidate())

    assert decision.status == "insufficient"
    assert decision.human_review is True
    assert decision.proposal is None
    assert len(decision.passes) == 2
    assert all(item.error.startswith("search_failed:") for item in decision.passes)
    assert run.requests_used == 2


def test_run_budget_is_cumulative_and_exhaustion_is_explicit():
    searcher = RecordingSearcher()
    run, _ = run_with(searcher, max_requests=2)

    first = run.research_candidate(candidate(1))
    second = run.research_candidate(candidate(2))

    assert first.status == "agree"
    assert second.status == "skipped_budget"
    assert second.human_review is True
    assert run.requests_used == 2
    assert len(searcher.queries) == 2


def test_authoritative_full_phrase_skips_research_without_mutating_terms():
    searcher = RecordingSearcher()
    run, _ = run_with(searcher)

    decision = run.research_candidate(
        candidate(),
        authoritative_terms={"Commander Harmon Rabb": "哈蒙·拉布中校"},
    )

    assert decision.status == "skipped_authoritative"
    assert decision.proposal is None
    assert decision.human_review is False
    assert run.requests_used == 0
    assert searcher.queries == []


def test_generic_authoritative_rank_does_not_suppress_full_name_research():
    searcher = RecordingSearcher()
    run, _ = run_with(searcher)

    decision = run.research_candidate(candidate(), authoritative_terms={"Commander": "中校"})

    assert decision.status == "agree"
    assert run.requests_used == 2


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"not_results": []}, "no results"),
        ({"results": [{"title": "missing url"}]}, "no URL"),
    ],
)
def test_exa_response_shape_is_strict(payload, match):
    client = ExaSearchClient("test-key", settings(), session=FakeSession(FakeResponse(payload)))

    with pytest.raises(Exception, match=match):
        client.search("query", limit=1)
    EvidenceCitation,
