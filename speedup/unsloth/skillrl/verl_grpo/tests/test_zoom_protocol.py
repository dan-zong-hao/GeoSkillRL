import unittest

from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import (
    build_zoom_response_mask,
    decode_token_ids,
    find_zoom_token_span,
    parse_zoom_text,
)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    bos_token_id = 2
    all_special_ids = [0, 2, 3]

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)


class ZoomProtocolTest(unittest.TestCase):
    def test_zoom_span_mask_uses_raw_ids(self):
        tok = CharTokenizer()
        text = "prefix<zoom><|ref|>dock<|/ref|><|box|>[[1,2,3,4]]<|/box|></zoom>tail"
        ids = tok.encode(text)
        span = find_zoom_token_span(ids, tok)
        self.assertIsNotNone(span)
        mask, parsed = build_zoom_response_mask(ids, tok)
        self.assertTrue(parsed.zoom_parse_ok)
        self.assertEqual(sum(mask), span.end - span.start)
        self.assertEqual(decode_token_ids(tok, ids), text)

    def test_unclosed_zoom_is_parse_fail_and_masks_non_special(self):
        tok = CharTokenizer()
        text = "<zoom><|box|>[[1,2,3,4]]<|/box|>"
        ids = tok.encode(text)
        mask, parsed = build_zoom_response_mask(ids, tok)
        self.assertFalse(parsed.zoom_parse_ok)
        self.assertEqual(parsed.error, "zoom_unclosed")
        self.assertEqual(sum(mask), len(ids))

    def test_multiple_zoom_rejected(self):
        parsed = parse_zoom_text(
            "<zoom><|box|>[[1,2,3,4]]<|/box|></zoom>"
            "<zoom><|box|>[[5,6,7,8]]<|/box|></zoom>"
        )
        self.assertFalse(parsed.zoom_parse_ok)
        self.assertEqual(parsed.error, "multiple_zoom")


if __name__ == "__main__":
    unittest.main()
