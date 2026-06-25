"""Token-in-token-out zoom parsing and masking utilities.

The parser may decode raw response IDs for text inspection, but it never
cleans and re-tokenizes a model completion. Token masks are built from
subsequences in the original response IDs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


ZOOM_OPEN = "<zoom>"
ZOOM_CLOSE = "</zoom>"
BOX_OPEN = "<|box|>"
BOX_CLOSE = "<|/box|>"
REF_OPEN = "<|ref|>"
REF_CLOSE = "<|/ref|>"

BOX_RE = re.compile(
    r"<\|box\|>\s*\[\[\s*(.*?)\s*\]\]\s*<\|/box\|>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ZoomParseResult:
    text: str
    zoom_text: str
    pred_bbox_1024: list[float] | None
    zoom_parse_ok: bool
    error: str | None
    zoom_count: int
    bbox_raw: list[float] | None = None


@dataclass(frozen=True)
class ZoomTokenSpan:
    start: int
    end: int
    error: str | None = None


def normalize_zoom_tags(text: str) -> str:
    """Normalize common malformed closing variants for parser inspection only."""
    return (
        str(text or "")
        .replace("</|box|>", BOX_CLOSE)
        .replace("</|ref|>", REF_CLOSE)
        .replace("<\\|/box\\|>", BOX_CLOSE)
        .replace("<\\|/ref\\|>", REF_CLOSE)
    )


def decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode without skipping special tokens so primitive tags remain visible."""
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def validate_bbox_1024(bbox: Sequence[Any] | None, *, require_in_range: bool = True) -> list[float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if require_in_range and not all(0.0 <= v <= 1024.0 for v in (x1, y1, x2, y2)):
        return None
    return [x1, y1, x2, y2]


def parse_bbox_from_zoom_text(zoom_text: str) -> tuple[list[float] | None, list[float] | None, str | None]:
    match = BOX_RE.search(normalize_zoom_tags(zoom_text))
    if not match:
        return None, None, "bbox_missing"
    parts = [p.strip() for p in match.group(1).split(",")]
    try:
        raw = [float(p) for p in parts]
    except ValueError:
        return None, None, "bbox_non_numeric"
    if len(raw) != 4:
        return None, raw, "bbox_wrong_arity"
    parsed = validate_bbox_1024(raw, require_in_range=True)
    if parsed is None:
        return None, raw, "bbox_invalid_or_out_of_range"
    return parsed, raw, None


def parse_zoom_text(text: str) -> ZoomParseResult:
    normalized = normalize_zoom_tags(text)
    zoom_count = normalized.count(ZOOM_OPEN)
    if zoom_count == 0:
        return ZoomParseResult(normalized, "", None, False, "zoom_missing", 0)
    if zoom_count > 1:
        first = normalized.find(ZOOM_OPEN)
        last_close = normalized.find(ZOOM_CLOSE, first + len(ZOOM_OPEN))
        zoom_text = normalized[first : last_close + len(ZOOM_CLOSE)] if last_close >= 0 else normalized[first:]
        return ZoomParseResult(normalized, zoom_text, None, False, "multiple_zoom", zoom_count)
    start = normalized.find(ZOOM_OPEN)
    end = normalized.find(ZOOM_CLOSE, start + len(ZOOM_OPEN))
    if end < 0:
        return ZoomParseResult(normalized, normalized[start:], None, False, "zoom_unclosed", zoom_count)
    zoom_text = normalized[start : end + len(ZOOM_CLOSE)]
    bbox, raw, error = parse_bbox_from_zoom_text(zoom_text)
    if error:
        return ZoomParseResult(normalized, zoom_text, None, False, error, zoom_count, raw)
    return ZoomParseResult(normalized, zoom_text, bbox, True, None, zoom_count, raw)


def extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else None


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"] if isinstance(encoded, dict) else encoded)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int], *, start: int = 0) -> int:
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    for idx in range(max(0, start), max(0, limit)):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return idx
    return -1


def find_all_subsequences(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    out: list[int] = []
    pos = 0
    while pos < len(haystack):
        idx = find_subsequence(haystack, needle, start=pos)
        if idx < 0:
            break
        out.append(idx)
        pos = idx + max(1, len(needle))
    return out


def find_zoom_token_span(response_ids: Sequence[int], tokenizer: Any) -> ZoomTokenSpan | None:
    """Locate the closed zoom span directly in raw response token IDs."""
    start_ids = _encode_literal(tokenizer, ZOOM_OPEN)
    end_ids = _encode_literal(tokenizer, ZOOM_CLOSE)
    starts = find_all_subsequences(response_ids, start_ids)
    if not starts:
        return None
    if len(starts) > 1:
        return ZoomTokenSpan(starts[0], starts[0] + len(start_ids), "multiple_zoom")
    close = find_subsequence(response_ids, end_ids, start=starts[0] + len(start_ids))
    if close < 0:
        return ZoomTokenSpan(starts[0], len(response_ids), "zoom_unclosed")
    return ZoomTokenSpan(starts[0], close + len(end_ids), None)


def special_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids.add(int(value))
    extra = getattr(tokenizer, "all_special_ids", None)
    if extra:
        ids.update(int(x) for x in extra if x is not None)
    return ids


def non_special_mask(response_ids: Sequence[int], tokenizer: Any) -> list[int]:
    specials = special_token_ids(tokenizer)
    return [0 if int(tok) in specials else 1 for tok in response_ids]


def build_zoom_response_mask(response_ids: Sequence[int], tokenizer: Any) -> tuple[list[int], ZoomParseResult]:
    """Build the policy mask for a completion without modifying raw IDs."""
    ids = list(response_ids)
    decoded = decode_token_ids(tokenizer, ids)
    parsed = parse_zoom_text(decoded)
    span = find_zoom_token_span(ids, tokenizer)
    if parsed.zoom_parse_ok and span is not None and span.error is None:
        mask = [0] * len(ids)
        for i in range(span.start, min(span.end, len(mask))):
            mask[i] = 1
        return mask, parsed
    return non_special_mask(ids, tokenizer), parsed


def ensure_aligned_response(
    response_ids: Sequence[int],
    response_mask: Sequence[int],
    response_logprobs: Sequence[float] | None,
) -> None:
    if len(response_ids) != len(response_mask):
        raise ValueError(f"response_ids/mask length mismatch: {len(response_ids)} != {len(response_mask)}")
    if response_logprobs is not None and len(response_ids) != len(response_logprobs):
        raise ValueError(
            f"response_ids/logprobs length mismatch: {len(response_ids)} != {len(response_logprobs)}"
        )


def count_masked(mask: Iterable[int]) -> int:
    return sum(1 for x in mask if int(x) != 0)
