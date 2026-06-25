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


class ResponseMaskTest(unittest.TestCase):
    def test_observation_and_answer_are_zero_masked(self):
        tok = CharTokenizer()
        stage1 = "<zoom><|ref|>x<|/ref|><|box|>[[1,2,3,4]]<|/box|></zoom>"
        obs = build_crop_observation_text()
        answer = "<answer>yes</answer>"
        stage1_ids = tok.encode(stage1)
        obs_ids = tok.encode(obs)
        answer_ids = tok.encode(answer)
        mask, parsed = build_zoom_response_mask(stage1_ids, tok)
        self.assertTrue(parsed.zoom_parse_ok)
        full_mask = mask + [0] * len(obs_ids) + [0] * len(answer_ids)
        self.assertEqual(len(full_mask), len(stage1_ids) + len(obs_ids) + len(answer_ids))
        self.assertEqual(sum(full_mask[: len(stage1_ids)]), len(stage1_ids))
        self.assertEqual(sum(full_mask[len(stage1_ids) :]), 0)


if __name__ == "__main__":
    unittest.main()
