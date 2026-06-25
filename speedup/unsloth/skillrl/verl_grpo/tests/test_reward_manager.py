import asyncio
import unittest

import torch

from speedup.unsloth.skillrl.verl_grpo.reward.reward_manager import GeoRewardManager


class NoDecodeTokenizer:
    def decode(self, *_args, **_kwargs):
        raise AssertionError("decode should not be called when zoom_text is supplied")


class RewardManagerTest(unittest.TestCase):
    def test_score_item_prefers_agent_loop_fields(self):
        mgr = GeoRewardManager(tokenizer=NoDecodeTokenizer())
        score = mgr.score_item(
            response_ids=[1, 2, 3],
            non_tensor={
                "zoom_text": "<zoom><|ref|>dock<|/ref|><|box|>[[100,100,200,200]]<|/box|></zoom>",
                "pred_bbox_1024": [100, 100, 200, 200],
                "zoom_parse_ok": True,
                "answer_parse_ok": False,
                "extra_info": {
                    "question_id": "q1",
                    "gt_bbox_1024": [100, 100, 200, 200],
                    "image_size": [1024, 1024],
                    "bbox_coord_mode": "max_side",
                },
                "reward_model": {"ground_truth": "yes"},
            },
        )
        self.assertEqual(score["question_id"], "q1")
        self.assertTrue(score["zoom_parse_ok"])
        self.assertGreater(score["score"], 0.0)

    def test_run_single_uses_tool_extra_fields(self):
        class FakeItem:
            batch = {
                "responses": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            }
            non_tensor_batch = {
                "tool_extra_fields": {
                    "zoom_text": "<zoom><|ref|>dock<|/ref|><|box|>[[100,100,200,200]]<|/box|></zoom>",
                    "pred_bbox_1024": [100, 100, 200, 200],
                    "zoom_parse_ok": True,
                    "answer_parse_ok": False,
                },
                "extra_info": {
                    "question_id": "q1",
                    "gt_bbox_1024": [100, 100, 200, 200],
                    "image_size": [1024, 1024],
                    "bbox_coord_mode": "max_side",
                },
                "reward_model": {"ground_truth": "yes"},
            }

        class FakeData:
            def __getitem__(self, index):
                if isinstance(index, slice):
                    return self
                return FakeItem()

        mgr = GeoRewardManager(tokenizer=NoDecodeTokenizer())
        result = asyncio.run(mgr.run_single(FakeData()))
        self.assertGreater(result["reward_score"], 0.0)
        self.assertTrue(result["reward_extra_info"]["zoom_parse_ok"])


if __name__ == "__main__":
    unittest.main()
