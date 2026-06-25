import unittest

from speedup.unsloth.skillrl.verl_grpo.agent.zoomearth_agent_loop import build_crop_observation_text
from speedup.unsloth.skillrl.verl_grpo.agent.zoom_protocol import build_zoom_response_mask


class CharTokenizer:
    pad_token_id = 0
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class FakeServerAgentLoopTest(unittest.TestCase):
    def test_stage1_tokens_remain_unchanged_when_stage2_is_appended(self):
        tok = CharTokenizer()
        stage1 = "<zoom><|ref|>ship<|/ref|><|box|>[[10,20,30,40]]<|/box|></zoom>"
        stage2 = "<answer>ship</answer>"
        stage1_ids = tok.encode(stage1)
        mask, parsed = build_zoom_response_mask(stage1_ids, tok)
        obs_ids = tok.encode(build_crop_observation_text())
        stage2_ids = tok.encode(stage2)
        response_ids = stage1_ids + obs_ids + stage2_ids
        response_mask = mask + [0] * len(obs_ids) + [0] * len(stage2_ids)

        self.assertTrue(parsed.zoom_parse_ok)
        self.assertEqual(response_ids[: len(stage1_ids)], stage1_ids)
        self.assertEqual(response_mask[: len(stage1_ids)], mask)
        self.assertEqual(sum(response_mask[len(stage1_ids) :]), 0)
        self.assertEqual(len(response_ids), len(response_mask))


if __name__ == "__main__":
    unittest.main()
