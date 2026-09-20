from __future__ import annotations

import pytest

import subretrans.webfetch as webfetch
from subretrans.webfetch import FetchBudget, WebFetchError, fetch_url


class FakeResponse:
    def __init__(self, status_code, *, headers=None, chunks=(), encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = tuple(chunks)
        self.encoding = encoding
        self.closed = False

    def iter_content(self, *, chunk_size):
        self.chunk_size = chunk_size
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def public_resolver(hostname, port):
    return ["93.184.216.34"]


def test_pinned_connection_uses_approved_ip_and_original_tls_hostname(monkeypatch):
    calls = {}

    class Context:
        def wrap_socket(self, sock, *, server_hostname):
            calls["server_hostname"] = server_hostname
            return object()

    monkeypatch.setattr(webfetch.ssl, "create_default_context", lambda: Context())
    monkeypatch.setattr(
        webfetch.socket,
        "create_connection",
        lambda address, timeout, source: calls.update(
            address=address, timeout=timeout, source=source
        )
        or object(),
    )
    connection = webfetch._PinnedHTTPSConnection(
        "evidence.example", "93.184.216.34", 443, 2
    )

    connection.connect()

    assert calls["address"] == ("93.184.216.34", 443)
    assert calls["server_hostname"] == "evidence.example"


def test_fetches_public_text_with_manual_redirects_and_disables_environment_proxy():
    session = FakeSession(
        [
            FakeResponse(302, headers={"Location": "https://example.test/next"}),
            FakeResponse(
                200,
                headers={"Content-Type": "text/html", "Content-Length": "11"},
                chunks=[b"hello ", b"world"],
            ),
        ]
    )

    result = fetch_url(
        "https://example.test/start#fragment",
        FetchBudget(timeout=2, max_bytes=32, max_redirects=1),
        session=session,
        resolver=public_resolver,
    )

    assert result.requested_url == "https://example.test/start"
    assert result.final_url == "https://example.test/next"
    assert result.body == "hello world"
    assert result.redirects == ("https://example.test/next",)
    assert result.bytes_read == 11
    assert session.trust_env is False
    assert all(kwargs["allow_redirects"] is False for _, kwargs in session.calls)
    assert all(kwargs["stream"] is True for _, kwargs in session.calls)
    assert all(kwargs["timeout"] == 2 for _, kwargs in session.calls)
    assert [kwargs["resolved_ip"] for _, kwargs in session.calls] == [
        "93.184.216.34",
        "93.184.216.34",
    ]
    assert all(kwargs["server_hostname"] == "example.test" for _, kwargs in session.calls)
    assert result.fetched_at.endswith("+00:00")
    assert result.truncated is False
    assert result.evidence_snapshot == "hello world"


def test_rejects_non_public_dns_before_request():
    session = FakeSession([])

    with pytest.raises(WebFetchError, match="non-public"):
        fetch_url(
            "https://internal.example/secret",
            FetchBudget(timeout=1, max_bytes=32),
            session=session,
            resolver=lambda hostname, port: ["127.0.0.1"],
        )
    assert session.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/plain",
        "file:///tmp/secret",
        "https://user:pass@example.test/private",
    ],
)
def test_rejects_unsafe_url_forms(url):
    with pytest.raises(WebFetchError):
        fetch_url(url, FetchBudget(timeout=1, max_bytes=32), resolver=public_resolver)


def test_revalidates_redirect_target_and_rejects_http_or_private_target():
    response = FakeResponse(302, headers={"Location": "http://127.0.0.1/admin"})
    session = FakeSession(
        [response]
    )

    with pytest.raises(WebFetchError, match="https"):
        fetch_url(
            "https://example.test/start",
            FetchBudget(timeout=1, max_bytes=32, max_redirects=2),
            session=session,
            resolver=public_resolver,
        )
    assert response.closed


def test_revalidates_https_redirect_dns_target():
    response = FakeResponse(302, headers={"Location": "https://internal.test/admin"})
    session = FakeSession([response])

    def resolver(hostname, port):
        return ["10.0.0.2"] if hostname == "internal.test" else ["93.184.216.34"]

    with pytest.raises(WebFetchError, match="non-public"):
        fetch_url(
            "https://example.test/start",
            FetchBudget(timeout=1, max_bytes=32, max_redirects=2),
            session=session,
            resolver=resolver,
        )
    assert response.closed


@pytest.mark.parametrize(
    "response, budget, match",
    [
        (
            FakeResponse(
                200,
                headers={"Content-Type": "text/plain", "Content-Length": "20"},
                chunks=[b"short"],
            ),
            FetchBudget(timeout=1, max_bytes=8),
            "byte budget",
        ),
        (
            FakeResponse(
                200,
                headers={"Content-Type": "application/octet-stream"},
                chunks=[b"binary"],
            ),
            FetchBudget(timeout=1, max_bytes=32),
            "content type",
        ),
        (
            FakeResponse(
                200,
                headers={"Content-Type": "text/plain"},
                chunks=[b"12345", b"67890"],
            ),
            FetchBudget(timeout=1, max_bytes=8),
            "byte budget",
        ),
    ],
)
def test_enforces_content_type_and_streamed_byte_limit(response, budget, match):
    with pytest.raises(WebFetchError, match=match):
        fetch_url(
            "https://example.test/resource",
            budget,
            session=FakeSession([response]),
            resolver=public_resolver,
        )


def test_enforces_redirect_limit():
    session = FakeSession(
        [
            FakeResponse(302, headers={"Location": "https://example.test/one"}),
            FakeResponse(302, headers={"Location": "https://example.test/two"}),
        ]
    )

    with pytest.raises(WebFetchError, match="redirect limit"):
        fetch_url(
            "https://example.test/start",
            FetchBudget(timeout=1, max_bytes=32, max_redirects=1),
            session=session,
            resolver=public_resolver,
        )
