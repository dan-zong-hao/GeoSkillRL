#!/usr/bin/env python3
"""Image lookup and legacy crop helpers for GeoSkillRL slime rollouts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

IMAGE_DIRS = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
]


def find_image(name_or_path: str | Path) -> Path | None:
    raw = Path(str(name_or_path))
    if raw.exists():
        return raw
    for directory in IMAGE_DIRS:
        candidate = directory / str(name_or_path)
        if candidate.exists():
            return candidate
    return None


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    width, height = image.size
    scale = max_size / max(width, height)
    if scale < 1:
        return image.resize((int(width * scale), int(height * scale)), Image.BICUBIC)
    return image


def bbox_1024_to_orig(bbox_1024: Iterable[float], image_size: tuple[int, int]) -> list[float]:
    scale = max(float(image_size[0]), float(image_size[1])) / 1024.0
    return [float(v) * scale for v in bbox_1024]


def cut_image(image: Image.Image, bbox_orig: Iterable[float], min_size: int = 512) -> Image.Image:
    width, height = image.width, image.height
    x1, y1, x2, y2 = [
        max(0, min(int(v), bound))
        for v, bound in zip(bbox_orig, [width, height, width, height], strict=True)
    ]
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        return image.copy()
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        nx1 = max(0, cx - min_size // 2)
        ny1 = max(0, cy - min_size // 2)
        nx2 = min(width, nx1 + min_size)
        ny2 = min(height, ny1 + min_size)
        return image.crop((int(nx1), int(ny1), int(nx2), int(ny2)))
    return image.crop((x1, y1, x2, y2))


def make_legacy_crop(
    image_path: str | Path,
    bbox_1024: Iterable[float],
    *,
    max_size: int = 512,
) -> tuple[Image.Image, dict]:
    resolved = find_image(image_path)
    if resolved is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    image = Image.open(resolved).convert("RGB")
    bbox_orig = bbox_1024_to_orig(bbox_1024, image.size)
    crop = resize_image(cut_image(image, bbox_orig), max_size=max_size)
    return crop, {
        "image_path": str(resolved),
        "image_size": [int(image.width), int(image.height)],
        "bbox_orig": [float(v) for v in bbox_orig],
        "crop_size": [int(crop.width), int(crop.height)],
    }

