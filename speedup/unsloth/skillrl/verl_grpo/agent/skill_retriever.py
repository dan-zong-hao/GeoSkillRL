#!/usr/bin/env python3
"""Deterministic SkillBank retrieval using only the question text."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SKILLBANK = Path("/root/autodl-tmp/VQA/speedup/origin/geoskill/skillbank_round0.json")

SEED_SKILLS: list[dict[str, Any]] = [
    {
        "skill_id": "top_extremum",
        "trigger": ["top-most", "uppermost", "upper", "north", "northernmost"],
        "coordinate_prior": "smaller y-center",
        "when_to_apply": "Use for questions asking for the upper or northern instance of an object.",
        "avoid": "Do not select a lower but larger or more visually salient instance.",
        "covered_failure_types": ["top_violated"],
    },
    {
        "skill_id": "bottom_extremum",
        "trigger": ["bottom-most", "lowermost", "lower", "south", "southernmost"],
        "coordinate_prior": "larger y-center",
        "when_to_apply": "Use for questions asking for the lower or southern instance of an object.",
        "avoid": "Do not select an upper but larger or more central instance.",
        "covered_failure_types": ["bottom_violated"],
    },
    {
        "skill_id": "left_extremum",
        "trigger": ["left-most", "leftmost", "left", "west", "westernmost"],
        "coordinate_prior": "smaller x-center",
        "when_to_apply": "Use for questions asking for the left or western instance of an object.",
        "avoid": "Do not select a more central instance if a plausible leftmost candidate exists.",
        "covered_failure_types": ["left_violated"],
    },
    {
        "skill_id": "right_extremum",
        "trigger": ["right-most", "rightmost", "right", "east", "easternmost"],
        "coordinate_prior": "larger x-center",
        "when_to_apply": "Use for questions asking for the right or eastern instance of an object.",
        "avoid": "Do not select a more central instance if a plausible rightmost candidate exists.",
        "covered_failure_types": ["right_violated"],
    },
    {
        "skill_id": "corner_locator",
        "trigger": ["upper-left", "upper right", "lower-left", "lower right", "northwest", "northeast"],
        "coordinate_prior": "satisfy both x and y directions",
        "when_to_apply": "Use when a question combines vertical and horizontal locator words.",
        "avoid": "Do not stop after satisfying only one axis of a corner instruction.",
        "covered_failure_types": ["corner_partial", "top_violated", "bottom_violated", "left_violated", "right_violated"],
    },
]

LOCATOR_PATTERNS = {
    "top": [r"\btop[- ]?most\b", r"\bupper[- ]?most\b", r"\bnorth(?:ern)?[- ]?most\b", r"\btop\b", r"\bupper\b"],
    "bottom": [r"\bbottom[- ]?most\b", r"\blower[- ]?most\b", r"\bsouth(?:ern)?[- ]?most\b", r"\bbottom\b", r"\blower\b"],
    "left": [r"\bleft[- ]?most\b", r"\bwest(?:ern)?[- ]?most\b", r"\bleft\b", r"\bwest(?:ern)?\b"],
    "right": [r"\bright[- ]?most\b", r"\beast(?:ern)?[- ]?most\b", r"\bright\b", r"\beast(?:ern)?\b"],
}


RELATION_RE = re.compile(
    r"\b(?:directly\s+)?(?:to\s+the\s+)?"
    r"(?P<relation>left|right|above|below|top|bottom|north|south|east|west)\s+of\s+the\s+"
    r"(?P<referent>.+?)(?:\?|$)",
    re.IGNORECASE,
)


def parse_locator(question: str) -> dict[str, Any]:
    q = (question or "").lower()
    axes = [name for name, patterns in LOCATOR_PATTERNS.items() if any(re.search(p, q) for p in patterns)]
    vertical = [x for x in axes if x in {"top", "bottom"}]
    horizontal = [x for x in axes if x in {"left", "right"}]
    family = "corner" if vertical and horizontal else (axes[0] if axes else "none")
    return {"has_locator": bool(axes), "family": family, "axes": axes}


def _clean_phrase(value: str) -> str:
    phrase = " ".join(str(value or "").strip(" ?.!,").split())
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.split(
        r"\s+\b(?:is|are|was|were|appears?|appear|arranged|designed|used|covered|made|located|"
        r"mainly|primarily|mostly|a|an|or)\b",
        phrase,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(phrase.strip(" ?.!,").split())


def parse_stage2_relation(question: str) -> str:
    match = RELATION_RE.search(question or "")
    if not match:
        return ""
    relation = match.group("relation").lower()
    aliases = {"east": "right", "west": "left", "north": "above", "south": "below", "top": "above", "bottom": "below"}
    relation = aliases.get(relation, relation)
    return f"{relation} of the referent"


def extract_referent_phrase(question: str) -> str:
    text = " ".join(str(question or "").strip().split())
    relation_match = RELATION_RE.search(text)
    if relation_match:
        return _clean_phrase(relation_match.group("referent"))

    of_match = re.search(r"\b(?:of|for|around|inside|within|from)\s+the\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
    if of_match:
        phrase = _clean_phrase(of_match.group(1))
        if phrase:
            return phrase

    locator_spans: list[tuple[int, int]] = []
    for patterns in LOCATOR_PATTERNS.values():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                locator_spans.append(match.span())
    if locator_spans:
        start = min(span[0] for span in locator_spans)
        return _clean_phrase(text[start:])

    subject_match = re.search(r"\b(?:is|are|does|do|what|which)\s+the\s+(.+?)(?:\?|$)", text, re.IGNORECASE)
    if subject_match:
        return _clean_phrase(subject_match.group(1))
    return ""


def load_skillbank(path: str | Path | None = None) -> list[dict[str, Any]]:
    requested = Path(path) if path else DEFAULT_SKILLBANK
    if not requested.exists():
        return list(SEED_SKILLS)
    data = json.loads(requested.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("skills"), list):
            return data["skills"]
        skills: list[dict[str, Any]] = []
        for key in ("general_skills", "task_specific_skills", "common_mistakes"):
            value = data.get(key)
            if isinstance(value, list):
                skills.extend(x for x in value if isinstance(x, dict))
        return skills or list(SEED_SKILLS)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return list(SEED_SKILLS)


def _trigger_hit(question: str, trigger: str) -> bool:
    pattern = re.escape(trigger.lower()).replace("\\-", "[- ]?")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", question.lower()) is not None


def retrieve_skills(question: str, skillbank: list[dict[str, Any]], max_skills: int = 2) -> list[dict[str, Any]]:
    locator = parse_locator(question)
    wanted = set(locator.get("axes") or [])
    if locator.get("family") == "corner":
        wanted.add("corner")
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for skill in skillbank:
        triggers = [str(x) for x in skill.get("trigger", [])]
        covered = " ".join(str(x) for x in skill.get("covered_failure_types", []))
        skill_id = str(skill.get("skill_id") or skill.get("id") or skill.get("name") or "skill")
        if skill_id == "corner_locator" and locator.get("family") != "corner":
            continue
        hit = any(_trigger_hit(question or "", t) for t in triggers)
        axis_hit = any(axis in covered or axis in skill_id for axis in wanted)
        if hit or axis_hit:
            corner_bonus = 3 if skill_id == "corner_locator" and locator.get("family") == "corner" else 0
            scored.append(((2 if hit else 0) + (1 if axis_hit else 0) + corner_bonus, skill_id, skill))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [skill for _, _, skill in scored[:max_skills]]


def skill_ids(skills: list[dict[str, Any]]) -> list[str]:
    return [str(s.get("skill_id") or s.get("id") or s.get("name") or f"skill_{i}") for i, s in enumerate(skills)]


def format_skill_block(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ""
    lines = ["Relevant spatial grounding skills:"]
    for skill in skills:
        lines.append(
            "- "
            + str(skill.get("skill_id") or skill.get("id") or skill.get("name") or "skill")
            + ": prior="
            + str(skill.get("coordinate_prior", ""))
            + "; apply="
            + str(skill.get("when_to_apply", skill.get("principle", "")))
            + "; avoid="
            + str(skill.get("avoid", ""))
        )
    lines.append("Use these only to choose the first <zoom> bbox; keep the required XML-like protocol.")
    return "\n".join(lines)


class SkillRetriever:
    def __init__(self, path: str | Path | None = None, max_skills: int = 2, enabled: bool = True) -> None:
        self.path = Path(path) if path else DEFAULT_SKILLBANK
        self.max_skills = max_skills
        self.enabled = enabled
        self.skillbank = load_skillbank(self.path) if enabled else []

    def build(self, question: str, category: str | None = None) -> dict[str, Any]:
        normalized_question = " ".join(str(question or "").split())
        retrieval_key = hashlib.sha1(normalized_question.lower().encode("utf-8")).hexdigest()[:16]
        skills = retrieve_skills(normalized_question, self.skillbank, self.max_skills) if self.enabled else []
        referent = extract_referent_phrase(normalized_question)
        locator = parse_locator(referent or normalized_question)
        return {
            "skill_block": format_skill_block(skills),
            "retrieved_skill_ids": skill_ids(skills),
            "retrieved_skills": skills,
            "skill_retrieval_key": retrieval_key,
            "referent_phrase": referent,
            "stage2_relation": parse_stage2_relation(normalized_question),
            "stage1_locator_axes": list(locator.get("axes") or []),
            "category": str(category or ""),
        }
