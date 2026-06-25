"""Image lookup, coordinate mapping, and fixed-512 crop environment."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from .zoom_protocol import validate_bbox_1024


Image.MAX_IMAGE_PIXELS = None

DEFAULT_IMAGE_DIRS = [
    Path("/root/autodl-tmp/dataset/lrs_gro/image"),
    Path("/root/autodl-tmp/dataset/lrs_gro/images"),
    Path("/root/autodl-tmp/dataset/lrs_gro"),
    Path("/root/autodl-tmp/dataset/DIOR-RSVG/JPEGImages"),
]

COORD_MODE_MAX_SIDE = "max_side"
COORD_MODE_XY = "xy"


@dataclass(frozen=True)
class CropResult:
    image_path: str
    image_size: tuple[int, int]
    coord_mode: str
    bbox_1024: list[float]
    bbox_orig: list[float]
    crop_bbox_orig: list[float]
    crop_created: bool

    def to_dict(self) -> dict:
        return asdict(self)


def image_dirs(extra_dirs: Iterable[str | Path] | None = None) -> list[Path]:
    dirs = [Path(p) for p in (extra_dirs or [])]
    dirs.extend(DEFAULT_IMAGE_DIRS)
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_image(name_or_path: str | Path, extra_dirs: Iterable[str | Path] | None = None) -> Path | None:
    value = Path(str(name_or_path))
    if value.is_absolute() and value.exists():
        return value
    if value.exists():
        return value.resolve()
    for directory in image_dirs(extra_dirs):
        candidate = directory / value.name
        if candidate.exists():
            return candidate.resolve()
    return None


def open_rgb_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_to_max_side(image: Image.Image, max_side: int = 512) -> Image.Image:
    width, height = image.size
    scale = float(max_side) / float(max(width, height))
    if scale >= 1.0:
        return image.copy()
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.BICUBIC)


def save_global_cache(image_path: str | Path, output_path: str | Path, max_side: int = 512) -> Path:
    out, _ = save_global_cache_with_size(image_path, output_path, max_side=max_side)
    return out


def _valid_cached_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def save_global_cache_with_size(
    image_path: str | Path,
    output_path: str | Path,
    max_side: int = 512,
) -> tuple[Path, tuple[int, int]]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _valid_cached_image(out):
        with Image.open(image_path) as image:
            return out.resolve(), (int(image.width), int(image.height))
    with open_rgb_image(image_path) as image:
        image_size = (int(image.width), int(image.height))
        resize_to_max_side(image, max_side=max_side).save(out)
    return out.resolve(), image_size


def bbox_1024_to_original(
    bbox_1024: Sequence[float],
    image_size: tuple[int, int],
    *,
    coord_mode: str = COORD_MODE_MAX_SIDE,
) -> list[float]:
    bbox = validate_bbox_1024(bbox_1024, require_in_range=True)
    if bbox is None:
        raise ValueError(f"invalid bbox_1024: {bbox_1024}")
    width, height = image_size
    if coord_mode == COORD_MODE_MAX_SIDE:
        scale = max(float(width), float(height)) / 1024.0
        return [float(v) * scale for v in bbox]
    if coord_mode == COORD_MODE_XY:
        sx = float(width) / 1024.0
        sy = float(height) / 1024.0
        return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
    raise ValueError(f"unsupported coord_mode={coord_mode!r}")


def clamp_bbox_orig(bbox: Sequence[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid original-space bbox after clamp: {bbox}")
    return [x1, y1, x2, y2]


def _shift_inside(x1: float, x2: float, low: float, high: float) -> tuple[float, float]:
    span = x2 - x1
    if high - low <= span:
        return low, high
    if x1 < low:
        x2 += low - x1
        x1 = low
    if x2 > high:
        x1 -= x2 - high
        x2 = high
    return max(low, x1), min(high, x2)


def expand_fixed512_orig(
    bbox_orig: Sequence[float],
    image_size: tuple[int, int],
    *,
    size: int = 512,
) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = clamp_bbox_orig(bbox_orig, image_size)
    if (x2 - x1) < 32.0 or (y2 - y1) < 32.0:
        return [0.0, 0.0, float(width), float(height)]
    side_x = min(float(size), float(width))
    side_y = min(float(size), float(height))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ex1, ex2 = cx - side_x / 2.0, cx + side_x / 2.0
    ey1, ey2 = cy - side_y / 2.0, cy + side_y / 2.0
    ex1, ex2 = _shift_inside(ex1, ex2, 0.0, float(width))
    ey1, ey2 = _shift_inside(ey1, ey2, 0.0, float(height))
    return [ex1, ey1, ex2, ey2]


def crop_from_original(
    image_path: str | Path,
    bbox_1024: Sequence[float],
    *,
    coord_mode: str = COORD_MODE_MAX_SIDE,
    fixed_size: int = 512,
    output_max_side: int = 512,
) -> tuple[Image.Image, CropResult]:
    with open_rgb_image(image_path) as image:
        image_size = image.size
        bbox_orig = bbox_1024_to_original(bbox_1024, image_size, coord_mode=coord_mode)
        crop_box = expand_fixed512_orig(bbox_orig, image_size, size=fixed_size)
        crop = image.crop(tuple(int(round(v)) for v in crop_box))
        crop = resize_to_max_side(crop, max_side=output_max_side)
    return crop, CropResult(
        image_path=str(Path(image_path).resolve()),
        image_size=image_size,
        coord_mode=coord_mode,
        bbox_1024=[float(v) for v in bbox_1024],
        bbox_orig=[float(v) for v in bbox_orig],
        crop_bbox_orig=[float(v) for v in crop_box],
        crop_created=True,
    )


def draw_overlay(
    image_path: str | Path,
    bboxes: dict[str, Sequence[float]],
    output_path: str | Path,
    *,
    colors: dict[str, str] | None = None,
    max_side: int = 1024,
) -> Path:
    colors = colors or {"xy": "lime", "max_side": "red", "gt": "cyan"}
    with open_rgb_image(image_path) as image:
        original_size = image.size
        shown = resize_to_max_side(image, max_side=max_side)
    sx = shown.size[0] / float(original_size[0])
    sy = shown.size[1] / float(original_size[1])
    draw = ImageDraw.Draw(shown)
    for label, bbox in bboxes.items():
        x1, y1, x2, y2 = bbox
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        color = colors.get(label, "yellow")
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), label, fill=color)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shown.save(out)
    return out.resolve()


def sha256_jsonable(obj: object) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def bbox_area_1024(bbox: Sequence[float] | None) -> float:
    valid = validate_bbox_1024(bbox, require_in_range=False)
    if valid is None:
        return 0.0
    return max(0.0, valid[2] - valid[0]) * max(0.0, valid[3] - valid[1])


def is_nearly_square(size: tuple[int, int], *, tolerance: float = 0.05) -> bool:
    width, height = size
    if max(width, height) <= 0:
        return False
    return math.isclose(width / height, 1.0, rel_tol=tolerance)
