from __future__ import annotations

from agent.protocol import extract_answer, extract_zoom, stable_extra_fields, tag_mask_or_all


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_extract_zoom_primitive_and_json_audit():
    primitive = "<zoom><|ref|>x<|/ref|><|box|>[[1,2,3,4]]<|/box|></zoom>"
    parsed = extract_zoom(primitive)
    assert parsed.parse_ok
    assert parsed.bbox_1024 == [1.0, 2.0, 3.0, 4.0]
    assert parsed.has_ref
    assert parsed.has_box
    assert parsed.primitive_format_ok
    assert parsed.ref_text == "x"
    assert parsed.bbox_format == "primitive"

    parsed = extract_zoom('<think>x</think><zoom>[{"bbox_2d":[5,6,7,8],"label":"a"}]</zoom>')
    assert not parsed.parse_ok
    assert parsed.bbox_1024 == [5.0, 6.0, 7.0, 8.0]
    assert parsed.bbox_format == "json"

    parsed = extract_zoom('<zoom>[{"bbox_2d":[5,6,7,8],"label":"a"}]</zoom>', allow_json=True)
    assert parsed.parse_ok

    malformed = "<zoom><|ref|>x<|/ref|><|box|>[[1,2,3,4]]</zoom>"
    parsed = extract_zoom(malformed)
    assert not parsed.parse_ok
    assert parsed.zoom_text
    assert parsed.bbox_1024 is None


def test_extract_zoom_rejects_compact_by_default_and_allows_audit():
    parsed = extract_zoom("<zoom>[[1,2,3,4]]</zoom>")
    assert not parsed.parse_ok
    assert parsed.bbox_1024 == [1.0, 2.0, 3.0, 4.0]
    assert parsed.bbox_format == "compact"
    assert not parsed.primitive_format_ok

    parsed = extract_zoom("<zoom>[[1,2,3,4]]</zoom>", allow_compact=True)
    assert parsed.parse_ok


def test_malformed_primitive_does_not_compact_fallback():
    parsed = extract_zoom("<zoom>[[1,2,3,4]]<|ref|>x<|/ref|><|box|>[[1,2,3,4]]</zoom>")
    assert not parsed.parse_ok
    assert parsed.bbox_1024 is None


def test_extract_answer_and_stable_schema():
    parsed = extract_answer("<answer>yes</answer>")
    assert parsed.parse_ok
    assert parsed.answer_pred == "yes"
    fields = stable_extra_fields(answer_pred="yes")
    for key in ("zoom_text", "answer_text", "answer_pred", "stage1_raw_text", "stage2_raw_text", "tool_error"):
        assert key in fields


def test_mask_fallback_nonzero_on_format_error():
    token_ids = ids("bad format without tag")
    mask, ok = tag_mask_or_all(CharTokenizer(), token_ids, "bad format without tag", "zoom")
    assert not ok
    assert mask == [1] * len(token_ids)
