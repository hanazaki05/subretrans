"""Bounded HTTPS fetching with DNS-to-TLS address pinning.

Research URLs are untrusted. Every hop is resolved once, all returned
addresses must be public, and the TCP socket is opened to one of those exact
addresses. TLS still uses the original hostname for SNI and certificate
verification, closing the DNS-rebinding gap created by validating DNS before a
separate hostname-based HTTP connection.
"""

from __future__ import annotations

import hashlib
import html
import http.client
import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import Message
from typing import Any, Callable, Protocol
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit


class WebFetchError(RuntimeError):
    """Raised when a URL cannot be fetched within the safety contract."""


@dataclass(frozen=True)
class FetchBudget:
    timeout: float
    max_bytes: int
    max_redirects: int = 3

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("fetch timeout must be positive")
        if self.max_bytes <= 0:
            raise ValueError("fetch max_bytes must be positive")
        if self.max_redirects < 0:
            raise ValueError("fetch max_redirects must be non-negative")


@dataclass(frozen=True)
class FetchedPage:
    """Fetched body plus bounded audit evidence safe to persist."""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    body: str
    bytes_read: int
    body_sha256: str
    redirects: tuple[str, ...]
    title: str
    fetched_at: str
    truncated: bool
    body_summary: str
    evidence_snapshot: str

    def audit_dict(self) -> dict[str, object]:
        """Return bounded provenance without persisting the full response body."""

        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "content_type": self.content_type,
            "bytes_read": self.bytes_read,
            "body_sha256": self.body_sha256,
            "redirects": list(self.redirects),
            "title": self.title,
            "fetched_at": self.fetched_at,
            "truncated": self.truncated,
            "body_summary": self.body_summary,
            "evidence_snapshot": self.evidence_snapshot,
        }


Resolver = Callable[[str, int], list[str]]


class ResponseLike(Protocol):
    status_code: int
    headers: Any
    encoding: str | None

    def iter_content(self, *, chunk_size: int): ...
    def close(self) -> None: ...


class PinnedTransport(Protocol):
    """Testable transport contract that cannot discard the approved address."""

    def get(
        self,
        url: str,
        *,
        resolved_ip: str,
        server_hostname: str,
        allow_redirects: bool,
        stream: bool,
        timeout: float,
    ) -> ResponseLike: ...


def _default_resolver(hostname: str, port: int) -> list[str]:
    try:
        entries = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WebFetchError(f"DNS lookup failed for {hostname!r}: {exc}") from exc
    addresses: list[str] = []
    for entry in entries:
        address = entry[4][0]
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise WebFetchError(f"DNS lookup returned no addresses for {hostname!r}")
    return addresses


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise WebFetchError(f"DNS returned an invalid IP address: {value!r}") from exc
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_global


def _validated_addresses(parts: SplitResult, resolver: Resolver) -> tuple[str, ...]:
    hostname = parts.hostname
    if not hostname:
        raise WebFetchError("URL must contain a hostname")
    try:
        port = parts.port or 443
    except ValueError as exc:
        raise WebFetchError("URL contains an invalid port") from exc
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        addresses = resolver(hostname, port)
    else:
        addresses = [hostname]
    if not addresses:
        raise WebFetchError(f"DNS lookup returned no addresses for {hostname!r}")
    if any(not _is_public_address(address) for address in addresses):
        raise WebFetchError(f"URL resolves to a non-public address: {hostname!r}")
    return tuple(addresses)


def _normalise_url(url: str) -> tuple[str, SplitResult]:
    if not isinstance(url, str) or not url.strip():
        raise WebFetchError("URL must be a non-empty string")
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise WebFetchError("URL is malformed") from exc
    if parts.scheme.lower() != "https":
        raise WebFetchError("only https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise WebFetchError("URL credentials are not allowed")
    if not parts.hostname:
        raise WebFetchError("URL must contain a hostname")
    clean = urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))
    return clean, urlsplit(clean)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, resolved_ip: str, port: int, timeout: float) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._resolved_ip = resolved_ip

    def connect(self) -> None:
        try:
            sock = socket.create_connection(
                (self._resolved_ip, self.port), self.timeout, self.source_address
            )
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except OSError as exc:
            raise WebFetchError(
                f"HTTPS connection to approved address failed for {self.host!r}"
            ) from exc


class _HTTPResponseAdapter:
    def __init__(self, connection: _PinnedHTTPSConnection, response: http.client.HTTPResponse):
        self._connection = connection
        self._response = response
        self.status_code = response.status
        self.headers: Message = response.headers
        self.encoding = response.headers.get_content_charset() or "utf-8"

    def iter_content(self, *, chunk_size: int):
        while True:
            chunk = self._response.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        self._response.close()
        self._connection.close()


class _DefaultPinnedTransport:
    def get(
        self,
        url: str,
        *,
        resolved_ip: str,
        server_hostname: str,
        allow_redirects: bool,
        stream: bool,
        timeout: float,
    ) -> ResponseLike:
        del allow_redirects, stream
        parts = urlsplit(url)
        port = parts.port or 443
        target = parts.path or "/"
        if parts.query:
            target += f"?{parts.query}"
        connection = _PinnedHTTPSConnection(server_hostname, resolved_ip, port, timeout)
        host_header = server_hostname if port == 443 else f"{server_hostname}:{port}"
        try:
            connection.request(
                "GET",
                target,
                headers={"Host": host_header, "Accept": "text/html,text/plain,application/json"},
            )
            return _HTTPResponseAdapter(connection, connection.getresponse())
        except Exception:
            connection.close()
            raise


def _content_type(response: ResponseLike) -> str:
    raw = response.headers.get("Content-Type", "")
    return raw.split(";", 1)[0].strip().lower()


def _read_limited(response: ResponseLike, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except (TypeError, ValueError) as exc:
            raise WebFetchError("response Content-Length is invalid") from exc
        if declared < 0 or declared > max_bytes:
            raise WebFetchError("response exceeds the byte budget")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=min(65536, max_bytes + 1)):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise WebFetchError("response exceeds the byte budget")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _audit_text(body: str, content_type: str) -> tuple[str, str, str]:
    title = ""
    text = body
    if content_type in {"text/html", "application/xhtml+xml"}:
        match = _TITLE_RE.search(body)
        if match is not None:
            title = " ".join(html.unescape(match.group(1)).split())[:300]
        text = html.unescape(_TAG_RE.sub(" ", body))
    normalized = " ".join(text.split())
    return title, normalized[:500], normalized[:4_000]


def fetch_url(
    url: str,
    budget: FetchBudget,
    *,
    transport: PinnedTransport | None = None,
    session: PinnedTransport | None = None,
    resolver: Resolver | None = None,
) -> FetchedPage:
    """Fetch one public HTTPS text resource with manual bounded redirects.

    ``session`` is retained as a compatibility alias for the injected pinned
    transport used by tests. It must accept ``resolved_ip`` and
    ``server_hostname``; a normal ``requests.Session`` is intentionally not a
    valid transport because it would re-resolve the hostname.
    """

    if transport is not None and session is not None:
        raise ValueError("provide either transport or session, not both")
    requested_url, _ = _normalise_url(url)
    resolve = resolver or _default_resolver
    client = transport or session or _DefaultPinnedTransport()
    if hasattr(client, "trust_env"):
        client.trust_env = False
    current_url = requested_url
    redirects: list[str] = []

    for redirect_index in range(budget.max_redirects + 1):
        current_url, parts = _normalise_url(current_url)
        approved_ip = _validated_addresses(parts, resolve)[0]
        assert parts.hostname is not None
        try:
            response = client.get(
                current_url,
                resolved_ip=approved_ip,
                server_hostname=parts.hostname,
                allow_redirects=False,
                stream=True,
                timeout=budget.timeout,
            )
        except WebFetchError:
            raise
        except Exception as exc:
            raise WebFetchError(f"HTTPS request failed for {current_url!r}") from exc

        try:
            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                if not location:
                    raise WebFetchError("redirect response has no Location header")
                if redirect_index >= budget.max_redirects:
                    raise WebFetchError("redirect limit exceeded")
                next_url, _ = _normalise_url(urljoin(current_url, location))
                redirects.append(next_url)
                current_url = next_url
                continue
            if not 200 <= response.status_code < 300:
                raise WebFetchError(f"HTTP response was unsuccessful: {response.status_code}")
            content_type = _content_type(response)
            if not (
                content_type.startswith("text/")
                or content_type in {"application/xhtml+xml", "application/json"}
            ):
                raise WebFetchError(f"unsupported content type: {content_type or 'missing'}")
            raw = _read_limited(response, budget.max_bytes)
            encoding = response.encoding or "utf-8"
            try:
                body = raw.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                body = raw.decode("utf-8", errors="replace")
            title, summary, snapshot = _audit_text(body, content_type)
            return FetchedPage(
                requested_url=requested_url,
                final_url=current_url,
                status_code=response.status_code,
                content_type=content_type,
                body=body,
                bytes_read=len(raw),
                body_sha256=hashlib.sha256(raw).hexdigest(),
                redirects=tuple(redirects),
                title=title,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                truncated=False,
                body_summary=summary,
                evidence_snapshot=snapshot,
            )
        finally:
            response.close()

    raise WebFetchError("fetch did not produce a final response")


__all__ = [
    "FetchBudget",
    "FetchedPage",
    "PinnedTransport",
    "WebFetchError",
    "fetch_url",
]
