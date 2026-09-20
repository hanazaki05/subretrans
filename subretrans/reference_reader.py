"""Bounded reader for explicitly selected local subtitle references."""

from __future__ import annotations

import hashlib
import difflib
from dataclasses import dataclass
from pathlib import Path


ALLOWED_REFERENCE_SUFFIXES = frozenset({".ass", ".srt", ".ttml2"})


class ReferenceReadError(ValueError):
    """Raised when a local reference is outside the configured boundary."""


@dataclass(frozen=True)
class ReferenceDocument:
    """A bounded local reference and its provenance."""

    path: Path
    relative_path: str
    text: str
    bytes_read: int
    sha256: str


@dataclass(frozen=True)
class ReferenceResource:
    """One discoverable subtitle reference without its body."""

    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ReferenceMatch:
    """One bounded case-insensitive line match."""

    relative_path: str
    line: int
    text: str
    sha256: str


@dataclass(frozen=True)
class ReferenceContext:
    """A bounded line window from one reference subtitle."""

    relative_path: str
    start_line: int
    end_line: int
    lines: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class ReferenceComparison:
    """A bounded unified diff between two subtitle references."""

    left_path: str
    right_path: str
    diff: tuple[str, ...]
    truncated: bool
    left_sha256: str
    right_sha256: str


class ReferenceReader:
    """Read only subtitle files below one resolved root directory.

    The caller supplies the exact path.  This class does not glob, recurse, or
    interpret a path returned by a model as a URI.  Resolving before the
    containment check also rejects symlinks that escape the root.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_bytes: int = 1_048_576,
        max_lines: int = 20_000,
    ) -> None:
        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise ReferenceReadError(f"reference root is not a directory: {resolved_root}")
        if max_bytes <= 0:
            raise ValueError("reference max_bytes must be positive")
        if max_lines <= 0:
            raise ValueError("reference max_lines must be positive")
        self.root = resolved_root
        self.max_bytes = max_bytes
        self.max_lines = max_lines

    def _resolve_inside_root(self, path: str | Path) -> Path:
        if isinstance(path, str):
            raw = path.strip()
            if not raw:
                raise ReferenceReadError("reference path must be non-empty")
            if raw.lower().startswith(("file:", "http:", "https:", "data:")):
                raise ReferenceReadError("reference path must be a local path, not a URI")
            candidate_input = Path(raw)
        elif isinstance(path, Path):
            candidate_input = path
        else:
            raise TypeError("reference path must be a string or Path")

        candidate = candidate_input.expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ReferenceReadError(f"reference file not found: {candidate}") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ReferenceReadError("reference path escapes the configured root") from exc
        if not resolved.is_file():
            raise ReferenceReadError(f"reference path is not a regular file: {resolved}")
        if resolved.suffix.lower() not in ALLOWED_REFERENCE_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_REFERENCE_SUFFIXES))
            raise ReferenceReadError(f"reference extension is not allowed (use {allowed})")
        return resolved

    def read(self, path: str | Path) -> ReferenceDocument:
        """Read one bounded UTF-8 subtitle file and return its provenance."""

        resolved = self._resolve_inside_root(path)
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            raise ReferenceReadError(f"cannot read reference file: {resolved}") from exc
        if len(raw) > self.max_bytes:
            raise ReferenceReadError("reference file exceeds the byte budget")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ReferenceReadError("reference file is not valid UTF-8") from exc
        if text.count("\n") + 1 > self.max_lines:
            raise ReferenceReadError("reference file exceeds the line budget")
        relative = resolved.relative_to(self.root).as_posix()
        return ReferenceDocument(
            path=resolved,
            relative_path=relative,
            text=text,
            bytes_read=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def list_resources(self, *, limit: int = 200) -> tuple[ReferenceResource, ...]:
        """List bounded subtitle resources below the configured root."""

        if type(limit) is not int or limit <= 0:
            raise ValueError("reference list limit must be a positive integer")
        resources: list[ReferenceResource] = []
        for candidate in sorted(self.root.rglob("*")):
            if len(resources) >= limit:
                break
            if candidate.suffix.lower() not in ALLOWED_REFERENCE_SUFFIXES:
                continue
            try:
                document = self.read(candidate)
            except ReferenceReadError:
                continue
            resources.append(
                ReferenceResource(
                    document.relative_path,
                    document.bytes_read,
                    document.sha256,
                )
            )
        return tuple(resources)

    def search_subtitles(
        self,
        query: str,
        *,
        paths: tuple[str, ...] | None = None,
        max_results: int = 20,
    ) -> tuple[ReferenceMatch, ...]:
        """Search allowed subtitle text and return bounded matching lines."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("reference search query must be non-empty")
        if type(max_results) is not int or max_results <= 0:
            raise ValueError("reference max_results must be a positive integer")
        selected = paths or tuple(resource.relative_path for resource in self.list_resources())
        needle = query.casefold()
        matches: list[ReferenceMatch] = []
        for path in selected:
            document = self.read(path)
            for line_number, line in enumerate(document.text.splitlines(), start=1):
                if needle in line.casefold():
                    matches.append(
                        ReferenceMatch(
                            document.relative_path,
                            line_number,
                            line,
                            document.sha256,
                        )
                    )
                    if len(matches) >= max_results:
                        return tuple(matches)
        return tuple(matches)

    def read_subtitle_context(
        self, path: str | Path, *, line: int, radius: int = 3
    ) -> ReferenceContext:
        """Read a bounded line window around one one-based line number."""

        if type(line) is not int or line <= 0:
            raise ValueError("reference context line must be a positive integer")
        if type(radius) is not int or radius < 0 or radius > 50:
            raise ValueError("reference context radius must be between 0 and 50")
        document = self.read(path)
        lines = document.text.splitlines()
        if line > len(lines):
            raise ReferenceReadError("reference context line is outside the file")
        start = max(1, line - radius)
        end = min(len(lines), line + radius)
        return ReferenceContext(
            document.relative_path,
            start,
            end,
            tuple(lines[start - 1 : end]),
            document.sha256,
        )

    def compare(
        self,
        left: str | Path,
        right: str | Path,
        *,
        max_diff_lines: int = 200,
    ) -> ReferenceComparison:
        """Return a bounded unified diff for two allowed subtitle files."""

        if type(max_diff_lines) is not int or max_diff_lines <= 0:
            raise ValueError("reference max_diff_lines must be a positive integer")
        left_doc = self.read(left)
        right_doc = self.read(right)
        full_diff = list(
            difflib.unified_diff(
                left_doc.text.splitlines(),
                right_doc.text.splitlines(),
                fromfile=left_doc.relative_path,
                tofile=right_doc.relative_path,
                lineterm="",
            )
        )
        return ReferenceComparison(
            left_doc.relative_path,
            right_doc.relative_path,
            tuple(full_diff[:max_diff_lines]),
            len(full_diff) > max_diff_lines,
            left_doc.sha256,
            right_doc.sha256,
        )


__all__ = [
    "ALLOWED_REFERENCE_SUFFIXES",
    "ReferenceDocument",
    "ReferenceResource",
    "ReferenceMatch",
    "ReferenceContext",
    "ReferenceComparison",
    "ReferenceReader",
    "ReferenceReadError",
]
