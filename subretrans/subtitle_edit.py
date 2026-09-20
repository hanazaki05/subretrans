"""Pinned build and invocation wrapper for Subtitle Edit's ``seconv``."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .subtitle_processing import read_srt


_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_REVISION_MARKER = ".subtitle-edit-revision"


@dataclass(frozen=True)
class SubtitleEditSettings:
    repository_url: str
    revision: str
    source_dir: Path
    build_dir: Path
    dotnet_executable: str
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("repository_url", "revision", "dotnet_executable"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if _REVISION_RE.fullmatch(self.revision) is None:
            raise ValueError("revision must be a 40-character hexadecimal commit hash")
        for name in ("source_dir", "build_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        if not isinstance(self.operations, tuple):
            raise TypeError("operations must be a tuple")
        if any(not isinstance(operation, str) or not operation.strip() for operation in self.operations):
            raise ValueError("operations must contain only non-empty strings")


@dataclass(frozen=True)
class SeconvCommand:
    """Exact argv prefix for a framework-dependent ``seconv.dll`` build."""

    argv_prefix: tuple[str, ...]


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=True, capture_output=True, text=True)


def _validate_checkout(settings: SubtitleEditSettings) -> None:
    source = str(settings.source_dir)
    inside = _run(
        ["git", "-C", source, "rev-parse", "--is-inside-work-tree"]
    ).stdout.strip()
    if inside != "true":
        raise ValueError(f"source_dir is not a Git work tree: {settings.source_dir}")

    origin = _run(["git", "-C", source, "remote", "get-url", "origin"]).stdout.strip()
    if origin != settings.repository_url:
        raise ValueError(
            f"source_dir origin is {origin!r}, expected {settings.repository_url!r}"
        )

    head = _run(["git", "-C", source, "rev-parse", "HEAD"]).stdout.strip()
    if head.lower() != settings.revision.lower():
        raise ValueError(
            f"source_dir HEAD is {head!r}, expected {settings.revision!r}"
        )


def _built_command(settings: SubtitleEditSettings) -> Path | SeconvCommand | None:
    executable = settings.build_dir / "seconv"
    if executable.is_file():
        return executable
    assembly = settings.build_dir / "seconv.dll"
    if assembly.is_file():
        return SeconvCommand((settings.dotnet_executable, str(assembly)))
    return None


def ensure_seconv(settings: SubtitleEditSettings) -> Path | SeconvCommand:
    """Ensure the pinned checkout and its corresponding ``seconv`` build exist."""

    if settings.source_dir.exists():
        _validate_checkout(settings)
    else:
        settings.source_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                settings.repository_url,
                str(settings.source_dir),
            ]
        )
        _run(
            [
                "git",
                "-C",
                str(settings.source_dir),
                "checkout",
                "--detach",
                settings.revision,
            ]
        )

    marker = settings.build_dir / _REVISION_MARKER
    command = _built_command(settings)
    if (
        command is not None
        and marker.is_file()
        and marker.read_text(encoding="ascii").strip().lower()
        == settings.revision.lower()
    ):
        return command

    _run(
        [
            settings.dotnet_executable,
            "build",
            str(settings.source_dir / "src/seconv/SeConv.csproj"),
            "-c",
            "Release",
            "--output",
            str(settings.build_dir),
        ]
    )
    command = _built_command(settings)
    if command is None:
        raise FileNotFoundError(
            f"dotnet build produced neither {settings.build_dir / 'seconv'} nor "
            f"{settings.build_dir / 'seconv.dll'}"
        )
    marker.write_text(f"{settings.revision}\n", encoding="ascii")
    return command


def preprocess_with_seconv(
    settings: SubtitleEditSettings, input_path: Path, output_path: Path
) -> Path:
    """Convert one subtitle to strict UTF-8 SRT through the pinned ``seconv``."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")

    command = ensure_seconv(settings)
    argv_prefix = (
        command.argv_prefix if isinstance(command, SeconvCommand) else (str(command),)
    )
    subprocess.run(
        [
            *argv_prefix,
            str(input_path),
            "subrip",
            f"--output-filename:{output_path}",
            "--overwrite",
            "--encoding:utf-8-no-bom",
            "--json",
            *settings.operations,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not output_path.is_file():
        raise FileNotFoundError(f"seconv did not create output file: {output_path}")
    read_srt(output_path)
    return output_path
