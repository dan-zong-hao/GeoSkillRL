import unittest

from speedup.unsloth.skillrl.verl_grpo.reward.components import answer_match, compute_zoom_reward


class RewardComponentsTest(unittest.TestCase):
    def test_no_does_not_match_north(self):
        self.assertFalse(answer_match("no", "north"))
        self.assertTrue(answer_match("no", "no"))

    def test_invalid_bbox_gets_negative_reward(self):
        score = compute_zoom_reward(
            zoom_text="<zoom><|box|>[[5,5,1,1]]<|/box|></zoom>",
            gt_bbox_1024=[1, 1, 10, 10],
            image_size=(1024, 1024),
        )
        self.assertFalse(score["zoom_parse_ok"])
        self.assertLess(score["score"], 0.0)

    def test_valid_bbox_has_positive_components(self):
        score = compute_zoom_reward(
            zoom_text="<zoom><|ref|>dock<|/ref|><|box|>[[100,100,200,200]]<|/box|></zoom>",
            gt_bbox_1024=[100, 100, 200, 200],
            image_size=(1024, 1024),
        )
        self.assertTrue(score["zoom_parse_ok"])
        self.assertGreater(score["apo_iou"], 0.99)
        self.assertGreater(score["region_guided"], 0.0)


if __name__ == "__main__":
    unittest.main()
