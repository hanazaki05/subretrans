from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess

import pytest

import subretrans.subtitle_edit as subtitle_edit
from subretrans.subtitle_edit import (
    SeconvCommand,
    SubtitleEditSettings,
    ensure_seconv,
    preprocess_with_seconv,
)


REVISION = "7fca79c1b0f88e6cd59d5800f9c0b49c642a13b9"
REPOSITORY = "https://github.com/SubtitleEdit/subtitleedit.git"


def settings(tmp_path: Path, *, operations: tuple[str, ...] = ()) -> SubtitleEditSettings:
    return SubtitleEditSettings(
        repository_url=REPOSITORY,
        revision=REVISION,
        source_dir=tmp_path / "source",
        build_dir=tmp_path / "build",
        dotnet_executable="dotnet-test",
        settings_file=tmp_path / "subtitle-edit-settings.json",
        multiple_replace_file=tmp_path / "multiple-replace.template",
        operations=operations,
    )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"repository_url": ""}, "repository_url must be a non-empty"),
        ({"revision": "abc"}, "40-character hexadecimal"),
        ({"dotnet_executable": "  "}, "dotnet_executable must be a non-empty"),
        ({"operations": ("--remove-text-for-hearing-impaired", "")}, "operations"),
    ],
)
def test_settings_reject_invalid_values(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "repository_url": REPOSITORY,
        "revision": REVISION,
        "source_dir": tmp_path / "source",
        "build_dir": tmp_path / "build",
        "dotnet_executable": "dotnet",
        "settings_file": tmp_path / "subtitle-edit-settings.json",
        "multiple_replace_file": tmp_path / "multiple-replace.template",
        "operations": (),
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        SubtitleEditSettings(**values)  # type: ignore[arg-type]


def test_settings_are_frozen(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    with pytest.raises(FrozenInstanceError):
        configured.revision = "0" * 40  # type: ignore[misc]


def test_ensure_clones_exact_revision_and_builds_native_apphost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        calls.append(argv)
        if argv[:2] == ["git", "clone"]:
            configured.source_dir.mkdir()
        elif argv[:2] == ["dotnet-test", "build"]:
            configured.build_dir.mkdir()
            (configured.build_dir / "seconv").write_text("apphost", encoding="ascii")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subtitle_edit.subprocess, "run", fake_run)

    assert ensure_seconv(configured) == configured.build_dir / "seconv"
    assert calls == [
        ["git", "clone", REPOSITORY, str(configured.source_dir)],
        [
            "git",
            "-C",
            str(configured.source_dir),
            "checkout",
            "--detach",
            REVISION,
        ],
        [
            "dotnet-test",
            "build",
            str(configured.source_dir / "src/seconv/SeConv.csproj"),
            "-c",
            "Release",
            "--output",
            str(configured.build_dir),
        ],
    ]
    assert (configured.build_dir / ".subtitle-edit-revision").read_text(
        encoding="ascii"
    ) == f"{REVISION}\n"


def test_ensure_validates_existing_pin_and_reuses_dll_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    configured.source_dir.mkdir()
    configured.build_dir.mkdir()
    assembly = configured.build_dir / "seconv.dll"
    assembly.write_text("assembly", encoding="ascii")
    (configured.build_dir / ".subtitle-edit-revision").write_text(
        REVISION, encoding="ascii"
    )
    outputs = iter(("true\n", f"{REPOSITORY}\n", f"{REVISION}\n"))
    calls: list[list[str]] = []

    def fake_run(
        argv: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(subtitle_edit.subprocess, "run", fake_run)

    assert ensure_seconv(configured) == SeconvCommand(
        ("dotnet-test", str(assembly))
    )
    assert calls == [
        ["git", "-C", str(configured.source_dir), "rev-parse", "--is-inside-work-tree"],
        ["git", "-C", str(configured.source_dir), "remote", "get-url", "origin"],
        ["git", "-C", str(configured.source_dir), "rev-parse", "HEAD"],
    ]


def test_ensure_prefers_dll_when_apphost_and_assembly_both_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    configured.source_dir.mkdir()
    configured.build_dir.mkdir()
    (configured.build_dir / "seconv").write_text("apphost", encoding="ascii")
    assembly = configured.build_dir / "seconv.dll"
    assembly.write_text("assembly", encoding="ascii")
    (configured.build_dir / ".subtitle-edit-revision").write_text(
        REVISION, encoding="ascii"
    )
    outputs = iter(("true\n", f"{REPOSITORY}\n", f"{REVISION}\n"))
    monkeypatch.setattr(
        subtitle_edit,
        "_run",
        lambda argv: subprocess.CompletedProcess(
            argv, 0, stdout=next(outputs), stderr=""
        ),
    )

    assert ensure_seconv(configured) == SeconvCommand(
        ("dotnet-test", str(assembly))
    )


def test_preprocess_reports_seconv_error_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    input_path = tmp_path / "episode.srt"
    input_path.write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        subtitle_edit,
        "ensure_seconv",
        lambda value: SeconvCommand(("dotnet-test", "/build/seconv.dll")),
    )

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(131, argv, stderr="runtime not found")

    monkeypatch.setattr(subtitle_edit.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="exit code 131: runtime not found"):
        preprocess_with_seconv(configured, input_path, tmp_path / "output.srt")


@pytest.mark.parametrize(
    "outputs, match",
    [
        (("false\n",), "not a Git work tree"),
        (("true\n", "https://example.invalid/wrong.git\n"), "origin is"),
        (("true\n", f"{REPOSITORY}\n", f"{'0' * 40}\n"), "HEAD is"),
    ],
)
def test_ensure_rejects_mismatched_existing_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outputs: tuple[str, ...],
    match: str,
) -> None:
    configured = settings(tmp_path)
    configured.source_dir.mkdir()
    pending = iter(outputs)

    def fake_run(
        argv: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=next(pending), stderr="")

    monkeypatch.setattr(subtitle_edit.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match=match):
        ensure_seconv(configured)
    assert not configured.build_dir.exists()


def test_preprocess_invokes_exact_command_and_strictly_parses_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(
        tmp_path,
        operations=("--remove-text-for-hearing-impaired", "--fix-common-errors"),
    )
    input_path = tmp_path / "episode.ass"
    output_path = tmp_path / "episode.srt"
    input_path.write_text("[Events]\n", encoding="utf-8")
    command = SeconvCommand(("dotnet-test", "/build/seconv.dll"))
    monkeypatch.setattr(subtitle_edit, "ensure_seconv", lambda value: command)
    seen: list[list[str]] = []

    def fake_run(
        argv: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        seen.append(argv)
        output_path.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nLine\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, stdout='{"success":true}', stderr="")

    monkeypatch.setattr(subtitle_edit.subprocess, "run", fake_run)

    assert preprocess_with_seconv(configured, input_path, output_path) == output_path
    assert seen == [
        [
            "dotnet-test",
            "/build/seconv.dll",
            str(input_path),
            "subrip",
            f"--output-filename:{output_path}",
            "--overwrite",
            "--encoding:utf-8-no-bom",
            "--json",
            f"--settings:{configured.settings_file}",
            "--remove-text-for-hearing-impaired",
            "--fix-common-errors",
            f"--multiple-replace:{configured.multiple_replace_file}",
        ]
    ]


def test_preprocess_rejects_same_path_and_missing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = settings(tmp_path)
    input_path = tmp_path / "episode.ass"
    input_path.write_text("[Events]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        preprocess_with_seconv(configured, input_path, input_path)

    monkeypatch.setattr(
        subtitle_edit, "ensure_seconv", lambda value: tmp_path / "build/seconv"
    )
    monkeypatch.setattr(
        subtitle_edit.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )

    with pytest.raises(FileNotFoundError, match="did not create output"):
        preprocess_with_seconv(configured, input_path, tmp_path / "missing.srt")
