from __future__ import annotations

from PIL import Image

from agent.crop_environment import bbox_1024_to_orig, cut_image, resize_image


def test_bbox_1024_to_orig_uses_max_side():
    assert bbox_1024_to_orig([0, 0, 512, 512], (2000, 1000)) == [0.0, 0.0, 1000.0, 1000.0]


def test_tiny_crop_returns_whole_image_then_resize():
    image = Image.new("RGB", (800, 600), "white")
    crop = cut_image(image, [10, 10, 20, 20])
    assert crop.size == image.size
    assert resize_image(crop, 512).size == (512, 384)


def test_small_crop_expands_to_legacy_min_size():
    image = Image.new("RGB", (2000, 1000), "white")
    crop = cut_image(image, [100, 100, 200, 180])
    assert crop.size == (512, 512)


def test_edge_crop_shifts_inside_image():
    image = Image.new("RGB", (600, 600), "white")
    crop = cut_image(image, [0, 0, 80, 80])
    assert crop.size == (512, 512)

