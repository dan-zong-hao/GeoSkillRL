import unittest

from speedup.unsloth.skillrl.verl_grpo.agent.crop_environment import bbox_1024_to_original, expand_fixed512_orig
from speedup.unsloth.skillrl.verl_grpo.reward.components import apo_iou_fixed512


class BBoxCoordinatesTest(unittest.TestCase):
    def test_non_square_modes_are_distinct(self):
        bbox = [256, 256, 512, 512]
        size = (2048, 1024)
        xy = bbox_1024_to_original(bbox, size, coord_mode="xy")
        max_side = bbox_1024_to_original(bbox, size, coord_mode="max_side")
        self.assertEqual(xy, [512.0, 256.0, 1024.0, 512.0])
        self.assertEqual(max_side, [512.0, 512.0, 1024.0, 1024.0])

    def test_fixed512_identical_boxes_iou_one(self):
        bbox = [100, 120, 240, 300]
        self.assertAlmostEqual(apo_iou_fixed512(bbox, bbox, (2048, 1024)), 1.0)

    def test_fixed512_expansion_stays_inside_image(self):
        crop = expand_fixed512_orig([0, 0, 20, 20], (300, 400), size=512)
        self.assertEqual(crop, [0.0, 0.0, 300.0, 400.0])


if __name__ == "__main__":
    unittest.main()
