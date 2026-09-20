"""Pinned build and invocation wrapper for Subtitle Edit's ``seconv``."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .subtitle_processing import read_srt


logger = logging.getLogger(__name__)

_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
_REVISION_MARKER = ".subtitle-edit-revision"


@dataclass(frozen=True)
class SubtitleEditSettings:
    repository_url: str
    revision: str
    source_dir: Path
    build_dir: Path
    dotnet_executable: str
    settings_file: Path
    multiple_replace_file: Path
    first_pass_operations: tuple[str, ...]
    second_pass_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("repository_url", "revision", "dotnet_executable"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if _REVISION_RE.fullmatch(self.revision) is None:
            raise ValueError("revision must be a 40-character hexadecimal commit hash")
        for name in (
            "source_dir",
            "build_dir",
            "settings_file",
            "multiple_replace_file",
        ):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        for field_name in ("first_pass_operations", "second_pass_operations"):
            operations = getattr(self, field_name)
            if not isinstance(operations, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            if any(
                not isinstance(operation, str) or not operation.strip()
                for operation in operations
            ):
                raise ValueError(
                    f"{field_name} must contain only non-empty strings"
                )


@dataclass(frozen=True)
class SeconvCommand:
    """Exact argv prefix for a framework-dependent ``seconv.dll`` build."""

    argv_prefix: tuple[str, ...]


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "no command output").strip()
        raise RuntimeError(
            f"Command failed with exit code {error.returncode}: {details}"
        ) from error


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
    assembly = settings.build_dir / "seconv.dll"
    if assembly.is_file():
        return SeconvCommand((settings.dotnet_executable, str(assembly)))
    executable = settings.build_dir / "seconv"
    if executable.is_file():
        return executable
    return None


def ensure_seconv(settings: SubtitleEditSettings) -> Path | SeconvCommand:
    """Ensure the pinned checkout and its corresponding ``seconv`` build exist."""

    if settings.source_dir.exists():
        logger.info("Subtitle Edit: validating pinned checkout")
        _validate_checkout(settings)
    else:
        logger.info("Subtitle Edit: cloning %s", settings.repository_url)
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
        logger.info("Subtitle Edit: reusing pinned seconv build")
        return command

    logger.info("Subtitle Edit: building seconv with %s", settings.dotnet_executable)
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


def _convert_with_seconv(
    settings: SubtitleEditSettings,
    input_path: Path,
    output_path: Path,
    *,
    output_format: str,
    operations: tuple[str, ...],
    include_settings: bool,
    multiple_replace: bool,
    action: str,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")

    command = ensure_seconv(settings)
    argv_prefix = (
        command.argv_prefix if isinstance(command, SeconvCommand) else (str(command),)
    )
    argv = [
        *argv_prefix,
        str(input_path),
        output_format,
        f"--output-filename:{output_path}",
        "--overwrite",
        "--encoding:utf-8-no-bom",
        "--json",
    ]
    if include_settings:
        argv.append(f"--settings:{settings.settings_file}")
    argv.extend(operations)
    if multiple_replace:
        argv.append(f"--multiple-replace:{settings.multiple_replace_file}")
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "no seconv output").strip()
        raise RuntimeError(
            f"Subtitle Edit {action} failed with exit code "
            f"{error.returncode}: {details}"
        ) from error
    if not output_path.is_file():
        raise FileNotFoundError(f"seconv did not create output file: {output_path}")
    return output_path


def preprocess_with_seconv(
    settings: SubtitleEditSettings, input_path: Path, output_path: Path
) -> Path:
    """Run the configured two-pass English cleanup into strict UTF-8 SRT."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")
    first_pass_path = output_path.with_name(
        f"{output_path.stem}.subtitle-edit-first-pass{output_path.suffix}"
    )
    first_pass = _convert_with_seconv(
        settings,
        input_path,
        first_pass_path,
        output_format="subrip",
        operations=settings.first_pass_operations,
        include_settings=True,
        multiple_replace=True,
        action="preprocessing",
    )
    read_srt(first_pass)
    logger.info("Subtitle Edit: first English cleanup pass complete: %s", first_pass)
    output = _convert_with_seconv(
        settings,
        first_pass,
        output_path,
        output_format="subrip",
        operations=settings.second_pass_operations,
        include_settings=False,
        multiple_replace=False,
        action="second preprocessing pass",
    )
    read_srt(output)
    logger.info("Subtitle Edit: second English space-cleanup pass complete: %s", output)
    return output
