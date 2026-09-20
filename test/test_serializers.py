import pytest

from subretrans.pairs import SubtitlePair
from subretrans.serializers import (
    SerializationError,
    convert_json_examples_to_format,
    deserialize,
    deserialize_best_effort,
    extract_from_format_marker,
    serialize,
)


PAIRS = [
    SubtitlePair(id=0, eng="Tonight, on JAG...", chinese="今晚，在《军法署》..."),
    SubtitlePair(id=1, eng="AJ is here.{\\i1} Go.", chinese="AJ在这{\\i1} 走"),
]


@pytest.mark.parametrize("representation", ["json", "xml-pair", "pseudo-toml"])
def test_roundtrip_preserves_ids_text_and_tags(representation: str) -> None:
    restored = deserialize(serialize(PAIRS, representation), representation)

    assert [(p.id, p.eng, p.chinese) for p in restored] == [
        (p.id, p.eng, p.chinese) for p in PAIRS
    ]


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        serialize(PAIRS, "csv")


def test_json_rejects_wrong_shapes() -> None:
    with pytest.raises(SerializationError, match="must be an array"):
        deserialize('{"id": 1}', "json")
    with pytest.raises(SerializationError, match="Missing required fields"):
        deserialize('[{"id": 1, "eng": "x"}]', "json")
    with pytest.raises(SerializationError, match="Invalid ID value"):
        deserialize('[{"id": "abc", "eng": "x", "chinese": "y"}]', "json")


@pytest.mark.parametrize(
    "block",
    [
        "<pair>\nID=5\neng>Hello\nchinese=你好\n</pair>",
        "<pair>\nID: 5\neng = Hello\nchinese | 你好\n</pair>",
    ],
)
def test_xml_pair_accepts_alternate_separators(block: str) -> None:
    (pair,) = deserialize(block, "xml-pair")

    assert (pair.id, pair.eng, pair.chinese) == (5, "Hello", "你好")


def test_xml_pair_strict_errors_and_best_effort_salvage() -> None:
    text = "<pair>\nID=5\neng=Hello\nchinese=你好\n</pair>\n<pair>\nID=6\neng=Bye\n</pair>"

    with pytest.raises(SerializationError):
        deserialize(text, "xml-pair")
    pairs, errors = deserialize_best_effort(text, "xml-pair")

    assert [(p.id, p.chinese) for p in pairs] == [(5, "你好")]
    assert errors == ["pair#2: Missing field 'chinese'"]
    assert deserialize_best_effort("nothing", "xml-pair") == ([], ["No <pair>...</pair> blocks found"])
    assert deserialize_best_effort("[]", "json")[0] == []


def test_format_marker_extraction() -> None:
    assert extract_from_format_marker("chat\n<pair>\nID=1\n</pair>", "xml-pair") == "<pair>\nID=1\n</pair>"
    assert extract_from_format_marker("chat\n[pair]\nid = 1", "pseudo-toml") == "[pair]\nid = 1"
    assert extract_from_format_marker('note [{"id": 1}] end', "json") == '[{"id": 1}]'
    assert extract_from_format_marker("no marker", "json") is None


def test_examples_convert_from_json_with_raw_ass_tags() -> None:
    example = '[\n  {"id": 1, "eng": "Go{\\i1} now", "chinese": "走"}\n]'

    converted = convert_json_examples_to_format(example, "xml-pair")

    assert converted == "<pair>\nID=1\neng=Go{\\i1} now\nchinese=走\n</pair>"
    assert convert_json_examples_to_format(example, "json") == example
    with pytest.raises(SerializationError):
        convert_json_examples_to_format("not json", "pseudo-toml")
