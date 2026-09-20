from __future__ import annotations

import pytest

from subretrans.reference_reader import ReferenceReadError, ReferenceReader


def test_reads_allowed_subtitle_and_records_root_relative_provenance(tmp_path):
    root = tmp_path / "bsub"
    root.mkdir()
    source = root / "JAG.S07E07.en-cn.ass"
    source.write_text("[Events]\nDialogue: text\n", encoding="utf-8")

    document = ReferenceReader(root).read("JAG.S07E07.en-cn.ass")

    assert document.path == source.resolve()
    assert document.relative_path == "JAG.S07E07.en-cn.ass"
    assert document.text.startswith("[Events]")
    assert document.bytes_read == source.stat().st_size
    assert len(document.sha256) == 64


@pytest.mark.parametrize("suffix", [".txt", ".json", ".zip"])
def test_rejects_non_subtitle_extensions(tmp_path, suffix):
    root = tmp_path / "bsub"
    root.mkdir()
    path = root / f"reference{suffix}"
    path.write_text("not a subtitle", encoding="utf-8")

    with pytest.raises(ReferenceReadError, match="extension"):
        ReferenceReader(root).read(path)


def test_rejects_path_traversal_and_absolute_path_outside_root(tmp_path):
    root = tmp_path / "bsub"
    root.mkdir()
    outside = tmp_path / "outside.ass"
    outside.write_text("outside", encoding="utf-8")
    reader = ReferenceReader(root)

    with pytest.raises(ReferenceReadError, match="escapes"):
        reader.read("../outside.ass")
    with pytest.raises(ReferenceReadError, match="escapes"):
        reader.read(outside)


def test_rejects_symlink_escape(tmp_path):
    root = tmp_path / "bsub"
    root.mkdir()
    outside = tmp_path / "outside.srt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "linked.srt"
    link.symlink_to(outside)

    with pytest.raises(ReferenceReadError, match="escapes"):
        ReferenceReader(root).read(link)


def test_rejects_uri_and_file_size_or_line_budget(tmp_path):
    root = tmp_path / "bsub"
    root.mkdir()
    source = root / "episode.srt"
    source.write_text("one\ntwo\nthree\n", encoding="utf-8")

    reader = ReferenceReader(root, max_bytes=4, max_lines=2)
    with pytest.raises(ReferenceReadError, match="URI"):
        reader.read("file:///tmp/episode.srt")
    with pytest.raises(ReferenceReadError, match="byte budget"):
        reader.read(source)

    reader = ReferenceReader(root, max_bytes=1024, max_lines=2)
    with pytest.raises(ReferenceReadError, match="line budget"):
        reader.read(source)


def test_lists_searches_reads_context_and_compares_bounded_resources(tmp_path):
    root = tmp_path / "bsub"
    root.mkdir()
    (root / "a.srt").write_text("1\nCommander Turner\nNext\n", encoding="utf-8")
    (root / "b.srt").write_text("1\n特纳中校\nNext\n", encoding="utf-8")
    (root / "secret.json").write_text('{"key":"hidden"}', encoding="utf-8")
    reader = ReferenceReader(root)

    resources = reader.list_resources(limit=10)
    assert [item.relative_path for item in resources] == ["a.srt", "b.srt"]
    matches = reader.search_subtitles("turner", max_results=5)
    assert [(item.relative_path, item.line) for item in matches] == [("a.srt", 2)]
    context = reader.read_subtitle_context("a.srt", line=2, radius=1)
    assert context.start_line == 1
    assert context.end_line == 3
    assert context.lines == ("1", "Commander Turner", "Next")
    comparison = reader.compare("a.srt", "b.srt", max_diff_lines=4)
    assert comparison.left_path == "a.srt"
    assert len(comparison.diff) == 4
    assert comparison.truncated is True
