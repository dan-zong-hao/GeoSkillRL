from __future__ import annotations

from agent.protocol import tag_mask_or_all


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def test_zoom_mask_focuses_tag_span():
    text = "preface<zoom><|box|>[[1,2,3,4]]<|/box|></zoom>tail"
    mask, ok = tag_mask_or_all(CharTokenizer(), ids(text), text, "zoom")
    assert ok
    assert sum(mask) < len(text)
    assert sum(mask) > 0
    assert mask[0] == 0
    assert mask[-1] == 0


def test_answer_mask_fallback_all_generated_tokens():
    text = "no answer tag"
    mask, ok = tag_mask_or_all(CharTokenizer(), ids(text), text, "answer")
    assert not ok
    assert mask == [1] * len(text)

