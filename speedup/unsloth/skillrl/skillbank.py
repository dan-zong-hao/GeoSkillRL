"""Spatial Grounding SkillBank V0 for SkillRL.

The retriever is deliberately rule/template based. It uses only question and
target label text, not generated global captions or previous reasoning.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SEED_SKILLBANK: dict[str, Any] = {
    "version": "spatial_grounding_v0",
    "general_skills": [
        {
            "skill_id": "gen_coordinate_frame",
            "title": "Use image coordinate frame correctly",
            "principle": "BBox coordinates are in the global image frame: x increases from left to right, y increases from top to bottom. Top means smaller y; bottom means larger y; left means smaller x; right means larger x.",
            "when_to_apply": "Always apply before choosing the first <zoom> bbox.",
            "avoid": "Do not reverse the y-axis or choose objects by visual salience before applying the locator.",
        },
        {
            "skill_id": "gen_candidate_set_first",
            "title": "Build candidate set before applying locator",
            "principle": "First identify all plausible candidates of the referred class or parent object, then apply the spatial word to rank or choose among them.",
            "when_to_apply": "Use whenever the question contains a directional, ordinal, size, distance, or relative-location phrase.",
            "avoid": "Do not jump to the largest, most central, or most salient object before checking the locator.",
        },
        {
            "skill_id": "gen_crop_sanity",
            "title": "Keep crop useful and non-degenerate",
            "principle": "The first <zoom> bbox should cover the referred object or anchor plus enough surrounding context needed by the question.",
            "when_to_apply": "Use for every <zoom> decision.",
            "avoid": "Do not output boxes outside the image, tiny boxes with no context, or full-image boxes when a local referent is required.",
        },
    ],
    "task_specific_skills": {
        "directional_grounding": [
            {
                "skill_id": "dir_top_extremum_instance",
                "title": "Choose the uppermost instance",
                "trigger": ["top-most", "topmost", "uppermost", "northernmost", "top cluster", "upper cluster"],
                "principle": "When several same-class instances are plausible, choose the one with the smallest y-center after forming the candidate set.",
                "when_to_apply": "Use only when the directional word selects one whole instance among multiple same-class candidates.",
                "avoid": "Do not apply when 'upper' describes a subpart of one object.",
                "covered_failure_types": ["top_violated"],
                "priority": 40,
            },
            {
                "skill_id": "dir_bottom_extremum_instance",
                "title": "Choose the lowermost instance",
                "trigger": ["bottom-most", "bottommost", "lowermost", "southernmost", "bottom cluster", "lower cluster"],
                "principle": "When several same-class instances are plausible, choose the one with the largest y-center after forming the candidate set.",
                "when_to_apply": "Use only when the directional word selects one whole instance among multiple same-class candidates.",
                "avoid": "Do not apply when 'lower' describes a subpart of one object.",
                "covered_failure_types": ["bottom_violated"],
                "priority": 40,
            },
            {
                "skill_id": "dir_left_extremum_instance",
                "title": "Choose the leftmost instance",
                "trigger": ["left-most", "leftmost", "westernmost", "far-left", "left cluster"],
                "principle": "When several same-class instances are plausible, choose the one with the smallest x-center after forming the candidate set.",
                "when_to_apply": "Use only when the question asks for the left/western instance of an object class.",
                "avoid": "Do not apply when 'left' means a region relative to an anchor object or a side-specific subpart.",
                "covered_failure_types": ["left_violated"],
                "priority": 40,
            },
            {
                "skill_id": "dir_right_extremum_instance",
                "title": "Choose the rightmost instance",
                "trigger": ["right-most", "rightmost", "easternmost", "far-right", "right cluster"],
                "principle": "When several same-class instances are plausible, choose the one with the largest x-center after forming the candidate set.",
                "when_to_apply": "Use only when the question asks for the right/eastern instance of an object class.",
                "avoid": "Do not apply when 'right' means a region relative to an anchor object or a side-specific subpart.",
                "covered_failure_types": ["right_violated"],
                "priority": 40,
            },
            {
                "skill_id": "dir_corner_instance",
                "title": "Choose the instance in a specified corner",
                "trigger": ["upper-left", "upper right", "upper-right", "lower-left", "lower right", "lower-right", "northwest", "northeast", "southwest", "southeast", "top-left", "top-right", "bottom-left", "bottom-right"],
                "principle": "For corner locators, satisfy both axes together: upper/top means smaller y, lower/bottom means larger y, left/west means smaller x, right/east means larger x.",
                "when_to_apply": "Use when the question or target label explicitly combines vertical and horizontal locators.",
                "avoid": "Do not trigger from global image descriptions; trigger only from the question or target label.",
                "covered_failure_types": ["corner_partial", "top_violated", "bottom_violated", "left_violated", "right_violated"],
                "priority": 60,
            },
        ],
        "subpart_grounding": [
            {
                "skill_id": "subpart_side_locator",
                "title": "Locate a side-specific subpart of a parent object",
                "trigger": ["right-side", "left-side", "upper side", "lower side", "top side", "bottom side", "entrance", "gate", "edge", "end", "corner of", "side of"],
                "principle": "When a directional word modifies a subpart, first locate the parent object, then locate the named subpart on the specified side of that parent. The <zoom> label and bbox should target the subpart, not only the whole parent.",
                "when_to_apply": "Use for phrases like 'right-side stadium entrance', 'left side of the terminal', 'top edge of the field', or 'bottom end of the runway'.",
                "avoid": "Do not replace the subpart with the entire parent object. Do not interpret side-specific subparts as rightmost/leftmost instances.",
                "covered_failure_types": ["subpart_dropped", "subpart_as_extremum"],
                "priority": 100,
            }
        ],
        "relative_grounding": [
            {
                "skill_id": "rel_left_of_anchor",
                "title": "Find object left of an anchor",
                "trigger": ["left of", "to the left of", "west of", "western side of"],
                "principle": "First locate the anchor object after 'of', then inspect the region immediately to its left. The target bbox should cover the object satisfying the relation, not the anchor itself.",
                "when_to_apply": "Use when the question asks for an object or area located left/west of a reference object.",
                "avoid": "Do not choose the leftmost object in the whole image unless it is also left of the specified anchor.",
                "covered_failure_types": ["anchor_relative_left"],
                "priority": 90,
            },
            {
                "skill_id": "rel_right_of_anchor",
                "title": "Find object right of an anchor",
                "trigger": ["right of", "to the right of", "east of", "eastern side of"],
                "principle": "First locate the anchor object after 'of', then inspect the region immediately to its right. The target bbox should cover the object satisfying the relation, not the anchor itself.",
                "when_to_apply": "Use when the question asks for an object or area located right/east of a reference object.",
                "avoid": "Do not choose the rightmost object in the whole image unless it is also right of the specified anchor.",
                "covered_failure_types": ["anchor_relative_right"],
                "priority": 90,
            },
            {
                "skill_id": "rel_above_below_anchor",
                "title": "Find object above or below an anchor",
                "trigger": ["above", "below", "north of", "south of", "over", "under", "beneath"],
                "principle": "First locate the anchor object, then inspect the region above or below it according to the relation. Above/north means smaller y; below/south means larger y.",
                "when_to_apply": "Use when the question asks for an object or area located above/below a reference object.",
                "avoid": "Do not choose the globally topmost or bottommost object without checking the anchor relation.",
                "covered_failure_types": ["anchor_relative_vertical"],
                "priority": 90,
            },
            {
                "skill_id": "rel_front_context",
                "title": "Include front-facing context",
                "trigger": ["in front of", "directly in front of", "front of", "facing", "outside the entrance", "before the gate"],
                "principle": "For front-of questions, locate the anchor such as an entrance, gate, facade, or runway end, then include the immediate outward-facing area in the crop because the answer is usually outside the anchor.",
                "when_to_apply": "Use when the target answer is an object or area directly in front of an anchor.",
                "avoid": "Do not crop only the anchor. Do not answer using objects beside or behind the anchor.",
                "covered_failure_types": ["front_context_missing"],
                "priority": 80,
            },
            {
                "skill_id": "rel_adjacent_context",
                "title": "Include adjacent context",
                "trigger": ["next to", "adjacent to", "beside", "near", "around", "surrounding", "bordering"],
                "principle": "For adjacency questions, locate the anchor and include the immediate neighboring area on the described side or all around if no side is specified.",
                "when_to_apply": "Use when the answer depends on what is next to or around a known object.",
                "avoid": "Do not crop so tightly that neighboring objects are excluded.",
                "covered_failure_types": ["adjacent_context_missing"],
                "priority": 70,
            },
        ],
        "rank_grounding": [
            {
                "skill_id": "rank_largest_instance",
                "title": "Choose the largest instance",
                "trigger": ["largest", "biggest", "widest", "largest cluster"],
                "principle": "After forming the candidate set, compare visible spatial extent rather than salience or centrality. Choose the candidate with the largest apparent area.",
                "when_to_apply": "Use when the question explicitly asks for the largest or biggest instance.",
                "avoid": "Do not select the most visually salient object unless it is also the largest among candidates.",
                "covered_failure_types": ["largest_backlog"],
                "priority": 50,
            },
            {
                "skill_id": "rank_nearest_to_anchor",
                "title": "Choose nearest instance to an anchor",
                "trigger": ["nearest", "closest", "closest to", "nearest to"],
                "principle": "First locate the anchor object, then rank candidate objects by distance between centers or boundaries. Choose the candidate with the shortest distance to the anchor.",
                "when_to_apply": "Use when the question asks for the nearest/closest object to a reference object.",
                "avoid": "Do not choose a central candidate without locating the anchor first.",
                "covered_failure_types": ["nearest_backlog"],
                "priority": 50,
            },
        ],
    },
    "common_mistakes": [
        {
            "mistake_id": "err_subpart_as_extremum",
            "description": "Treating a side-specific subpart such as 'right-side entrance' as a whole-image rightmost object.",
            "how_to_avoid": "Use subpart_side_locator before directional extremum skills when the phrase contains entrance, gate, side, edge, end, or corner of a parent object.",
            "trigger": ["right-side", "left-side", "entrance", "gate", "edge", "end", "side of"],
            "priority": 30,
        },
        {
            "mistake_id": "err_anchor_selected_as_target",
            "description": "Cropping the anchor instead of the object satisfying a left/right/above/below relation.",
            "how_to_avoid": "First locate the anchor, then crop the object or area satisfying the relation.",
            "trigger": ["left of", "right of", "above", "below", "under", "beneath"],
            "priority": 30,
        },
        {
            "mistake_id": "err_too_tight_crop",
            "description": "Cropping only the anchor and excluding needed neighboring context.",
            "how_to_avoid": "For front-of, adjacent, surrounding, or context questions, include the anchor plus immediate surrounding region.",
            "trigger": ["in front of", "adjacent", "next to", "near", "around", "surrounding"],
            "priority": 20,
        },
    ],
}


def flatten_skills(bank: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(bank, list):
        return list(bank)
    out: list[dict[str, Any]] = []
    out.extend(dict(s, skill_group="general") for s in bank.get("general_skills", []))
    for group, skills in (bank.get("task_specific_skills") or {}).items():
        out.extend(dict(s, skill_group=group) for s in skills)
    return out


def count_skills(bank: dict[str, Any] | list[dict[str, Any]]) -> int:
    return len(flatten_skills(bank))


def load_skillbank(path: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
    if not path:
        return json.loads(json.dumps(SEED_SKILLBANK))
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SkillBank not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and ("task_specific_skills" in data or "general_skills" in data):
        return data
    if isinstance(data, dict) and "skills" in data:
        return data["skills"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported SkillBank format: {path}")


def save_skillbank(path: str, skillbank: dict[str, Any] | list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] | list[dict[str, Any]]
    if isinstance(skillbank, list):
        payload = {"version": "legacy_flat", "skills": skillbank}
    else:
        payload = skillbank
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _trigger_regex(trigger: str) -> str:
    escaped = re.escape(trigger.lower()).replace("\\-", "[- ]")
    return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"


def _trigger_hit(text: str, trigger: str) -> bool:
    return re.search(_trigger_regex(trigger), text) is not None


def _hits(text: str, skill: dict[str, Any]) -> list[str]:
    return [t for t in skill.get("trigger", []) if _trigger_hit(text, str(t))]


def retrieve_skills(
    question: str,
    skillbank: dict[str, Any] | list[dict[str, Any]],
    *,
    label: str = "",
    include_general: bool = False,
    max_general: int = 0,
    max_task: int = 2,
    max_mistakes: int = 0,
) -> list[dict[str, Any]]:
    """Template retrieval over question + target label only.

    Priority intentionally prevents broad extremum skills from shadowing
    subpart and anchor-relative skills.
    """
    text = _normalize(" ".join(x for x in [question, label] if x))
    if not text:
        return []

    bank = skillbank if isinstance(skillbank, dict) else {"general_skills": [], "task_specific_skills": {"legacy": skillbank}, "common_mistakes": []}
    general = [dict(s, skill_group="general", matched_triggers=[]) for s in bank.get("general_skills", [])] if include_general else []

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for skill in flatten_skills(bank):
        if skill.get("skill_group") == "general":
            continue
        matched = _hits(text, skill)
        if not matched:
            continue
        score = int(skill.get("priority", 0)) + 5 * len(matched)
        item = dict(skill, matched_triggers=matched)
        scored.append((score, str(skill.get("skill_id", "")), item))
    scored.sort(key=lambda x: (-x[0], x[1]))

    selected_task = [item for _, _, item in scored[:max_task]]
    selected_general = general[:max_general] if selected_task or include_general else []

    mistakes: list[dict[str, Any]] = []
    mistake_scores: list[tuple[int, str, dict[str, Any]]] = []
    for mistake in bank.get("common_mistakes", []):
        matched = _hits(text, mistake)
        if matched:
            item = dict(mistake, skill_id=mistake.get("mistake_id"), skill_group="common_mistake", matched_triggers=matched)
            mistake_scores.append((int(mistake.get("priority", 0)) + 5 * len(matched), str(item.get("skill_id", "")), item))
    mistake_scores.sort(key=lambda x: (-x[0], x[1]))
    if max_mistakes > 0:
        for _, _, mistake in mistake_scores:
            mistakes.append(mistake)
            if len(mistakes) >= max_mistakes:
                break

    return selected_general + selected_task + mistakes


def format_skill_block(skills: list[dict[str, Any]]) -> str:
    if not skills:
        return ""
    compact_rules = {
        "dir_top_extremum_instance": ("Choose the candidate with the smallest y-center among same-class candidates.", "Do not use for upper-side subparts."),
        "dir_bottom_extremum_instance": ("Choose the candidate with the largest y-center among same-class candidates.", "Do not use for lower-side subparts."),
        "dir_left_extremum_instance": ("Choose the candidate with the smallest x-center among same-class candidates.", "Do not use for left-of-anchor relations or left-side subparts."),
        "dir_right_extremum_instance": ("Choose the candidate with the largest x-center among same-class candidates.", "Do not use for right-of-anchor relations or right-side subparts."),
        "dir_corner_instance": ("Choose the candidate satisfying both corner axes: top/bottom by y and left/right by x.", "Do not trigger from global captions."),
        "subpart_side_locator": ("First locate the parent object, then crop the named subpart on the specified side.", "Do not replace the subpart with the whole parent."),
        "rel_left_of_anchor": ("First locate the anchor, then inspect the immediate region to its left; crop the target, not the anchor.", "Do not choose the global leftmost object."),
        "rel_right_of_anchor": ("First locate the anchor, then inspect the immediate region to its right; crop the target, not the anchor.", "Do not choose the global rightmost object."),
        "rel_above_below_anchor": ("First locate the anchor, then inspect above/below it; above means smaller y and below means larger y.", "Do not choose the global topmost/bottommost object."),
        "rel_front_context": ("Locate the anchor entrance/gate/facade, then include the immediate front-facing area.", "Do not crop only the anchor."),
        "rel_adjacent_context": ("Locate the anchor and include the immediate neighboring area needed by the question.", "Do not crop so tightly that neighbors are excluded."),
        "rank_largest_instance": ("After forming candidates, choose the one with the largest visible area.", "Do not choose by salience alone."),
        "rank_nearest_to_anchor": ("First locate the anchor, then choose the candidate with the shortest distance to it.", "Do not choose a central candidate without locating the anchor."),
    }
    header = "Relevant grounding skill:" if len(skills) == 1 else "Relevant grounding skills:"
    lines = [header]
    for skill in skills:
        sid = str(skill.get("skill_id", "skill"))
        group = str(skill.get("skill_group", "skill"))
        if group == "common_mistake":
            principle = str(skill.get("how_to_avoid") or "")
            lines.append(f"- avoid {sid}: {principle}")
        else:
            rule, warning = compact_rules.get(sid, (str(skill.get("principle") or ""), str(skill.get("avoid") or "")))
            line = f"- {sid}: {rule}"
            if warning:
                line += f" Avoid: {warning}"
            lines.append(line)
    lines.append("Use these only to choose the first <zoom> bbox. Do not mention skill names in the final answer.")
    return "\n".join(lines)
