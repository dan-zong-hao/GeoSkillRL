"""
Rewrite Think 2 (zoom analysis) and Think 3 (seg analysis) in the SFT JSONL
from generic/mask-description style to concise causal reasoning that bridges
visual evidence to the answer.

Think 2: analyze what the crop reveals about the question → justify the seg prompt
Think 3: what the mask reveals → causal link → answer (no mask% / spatial talk)
"""

import json
import re
from pathlib import Path

SRC = Path("/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_level/sft-00000-of-00001.zoom_seg.think.jsonl")
DST = Path("/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_causal_reasoning/sft-00000-of-00001.zoom_seg.think.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj(r):
    """Best object name: prefer seg_prompt, fall back to label_cleaned."""
    sp = r.get("seg_prompt", "").strip()
    if sp:
        return sp
    return r.get("label_cleaned", "")


def _gt(r):
    """Clean ground truth: strip leading article for relative qtype."""
    gt = r["ground_truth"]
    if r.get("qtype") == "relative":
        for prefix in ("another ", "a ", "an ", "the "):
            if gt.startswith(prefix):
                gt = gt[len(prefix):]
                break
    return gt


def _is_yn(gt):
    return str(gt).lower() in ("yes", "no")


# ---------------------------------------------------------------------------
# Think 2 generators — per qtype
# ---------------------------------------------------------------------------

def think2_shape(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "narrow" in rl or "wide" in rl:
        cue = "its proportions and outline"
    elif "curved" in rl or "straight" in rl:
        cue = "its path and curvature"
    elif "flat" in rl or "domed" in rl:
        cue = "the shading and contour of the top surface"
    elif "sharp" in rl or "angular" in rl:
        cue = "its edge profile and silhouette"
    else:
        cue = "its outline and surface shape"
    return f'The crop shows the {obj} up close, revealing {cue}. Segmenting "{obj}" isolates it for shape assessment.'


def think2_category(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "deck" in rl and "container" in rl:
        cue = "its deck layout and cargo configuration"
    elif "hull" in rl or ("ship" in rl and "vessel" in rl):
        cue = "its hull form and deck structure"
    elif "wing" in rl or "aircraft" in rl or "plane" in rl:
        cue = "its wing configuration and fuselage shape"
    elif "truck" in rl or "vehicle" in rl or "car" in rl:
        cue = "its size, shape, and cargo area"
    elif "building" in rl or "roof" in rl:
        cue = "its roof design and structural layout"
    elif "tank" in rl or "storage" in rl:
        cue = "its shape and surrounding infrastructure"
    else:
        cue = "its key distinguishing features"
    return f'The crop reveals the {obj} with {cue} visible. Segmenting "{obj}" isolates it for classification.'


def think2_counting(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "breakwater" in rl:
        cue = "distinct linear structures extending into the water"
    elif "truck" in rl or "vehicle" in rl:
        cue = "individual vehicle shapes"
    elif "building" in rl:
        cue = "separate building footprints"
    elif "ship" in rl or "vessel" in rl:
        cue = "each distinct outline"
    else:
        cue = "each distinct instance"
    return f'The crop shows {cue} within the area. Segmenting "{obj}" marks each instance for counting.'


def think2_color(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "roof" in rl:
        cue = "the roof surfaces under ambient lighting"
    elif "water" in rl:
        cue = "the water surface under current light conditions"
    elif "vegetation" in rl:
        cue = "the vegetation cover"
    else:
        cue = "its surface under the given lighting"
    return f'The crop shows the {obj} with {cue}. Segmenting "{obj}" isolates the surface for accurate color identification.'


def think2_material(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "reflect" in rl or "shine" in rl or "glint" in rl:
        cue = "its surface reflectance and texture"
    elif "rust" in rl or "corrosion" in rl:
        cue = "surface wear and oxidation patterns"
    elif "paint" in rl or "coating" in rl:
        cue = "surface finish and coating appearance"
    elif "wood" in rl or "metal" in rl:
        cue = "its surface texture and reflectivity"
    else:
        cue = "its surface texture and reflectance"
    return f'The crop shows the {obj} with {cue} apparent. Segmenting "{obj}" isolates it for material assessment.'


def think2_relative(r):
    obj = _obj(r)
    return f'The crop shows the {obj} within its surroundings. Segmenting "{obj}" anchors it for spatial reasoning.'


def think2_existence(r):
    obj = _obj(r)
    return f'The crop shows the relevant area. Segmenting "{obj}" confirms what is present.'


def think2_function(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "parking" in rl or "vehicle" in rl:
        cue = "the types and arrangement of vehicles"
    elif "storage" in rl or "container" in rl:
        cue = "the layout of stored materials and access paths"
    elif "dock" in rl or "loading" in rl:
        cue = "the loading infrastructure and vessel positioning"
    else:
        cue = "the structures and activity patterns within it"
    return f'The crop shows the {obj} with {cue}. Segmenting "{obj}" isolates the region to infer its primary use.'


def think2_activity(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "wake" in rl:
        cue = "the presence or absence of a wake"
    elif "moving" in rl or "stationary" in rl:
        cue = "visual indicators of movement versus rest"
    else:
        cue = "its position and surrounding activity indicators"
    return f'The crop shows the {obj} with {cue}. Segmenting "{obj}" isolates it for activity assessment.'


def think2_motion(r):
    return think2_activity(r)


def think2_construction(r):
    building = r.get("label_cleaned", "") or _obj(r)
    evidence = r.get("seg_prompt", "")
    if r.get("seg_purpose") == "evidence":
        return f'The crop shows the {building}. Segmenting "{evidence}" checks for signs of ongoing construction.'
    return f'The crop shows the {building} in detail. Segmenting "{evidence}" isolates it for assessment.'


def think2_arrangement(r):
    obj = _obj(r)
    reasoning = r.get("stage_2_reasoning", "")
    rl = reasoning.lower()
    if "highway" in rl or "road" in rl:
        cue = "its path and curvature across the frame"
    elif "building" in rl:
        cue = "the spatial distribution of structures"
    else:
        cue = "its spatial layout and organization"
    return f'The crop shows the {obj} across the frame, with {cue} visible. Segmenting "{obj}" isolates it to assess arrangement.'


def think2_referent(r):
    obj = _obj(r)
    return f'The crop shows the {obj} in detail. Segmenting "{obj}" isolates the relevant region for answering the question.'


THINK2_GEN = {
    "shape": think2_shape,
    "category": think2_category,
    "counting": think2_counting,
    "color": think2_color,
    "material": think2_material,
    "relative": think2_relative,
    "existence": think2_existence,
    "function": think2_function,
    "activity": think2_activity,
    "motion": think2_motion,
    "construction": think2_construction,
    "arrangement": think2_arrangement,
    "referent": think2_referent,
}


# ---------------------------------------------------------------------------
# Think 3 generators — causal bridge: mask reveals → conclusion → answer
# ---------------------------------------------------------------------------

def think3_shape(r):
    obj = _obj(r)
    gt = _gt(r)
    if gt in ("domed", "flat"):
        return f"The mask isolates the {obj}, whose surface shading confirms a {gt} profile."
    if gt in ("narrow", "wide"):
        return f"The mask isolates the {obj}, whose proportions confirm it is {gt} in shape."
    if gt in ("curved", "straight"):
        return f"The mask isolates the {obj}, whose path confirms a {gt} alignment."
    return f"The mask isolates the {obj}, revealing a {gt} shape."


def think3_category(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask isolates the {obj}, whose visual features are consistent with {gt}."


def think3_counting(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask marks {gt} distinct {obj}, confirming the count."


def think3_color(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask isolates the {obj} surface, which appears {gt}."


def think3_material(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask isolates the {obj}, whose surface texture and reflectance indicate {gt} material."


def think3_relative(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask anchors the {obj}; from its position, the adjacent {gt} is clearly identifiable."


def think3_existence(r):
    obj = _obj(r)
    gt = _gt(r)
    # existence catches many subtypes — handle by ground truth format
    if _is_yn(gt):
        verb = "confirms" if str(gt).lower() == "yes" else "rules out"
        return f"The mask isolates the {obj}, which {verb} the condition asked about."
    if str(gt).isdigit():
        return f"The mask marks {gt} distinct {obj}, confirming the count."
    return f"The mask isolates the {obj}, confirming {gt}."


def think3_function(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask isolates the {obj}, whose contents and layout indicate its primary function is {gt}."


def think3_activity(r):
    obj = _obj(r)
    gt = _gt(r)
    if _is_yn(gt):
        verb = "confirming" if str(gt).lower() == "yes" else "ruling out"
        return f"The mask isolates the {obj}, with visual indicators {verb} the condition."
    return f"The mask isolates the {obj}, whose visual indicators are consistent with {gt}."


def think3_motion(r):
    return think3_activity(r)


def think3_construction(r):
    building = r.get("label_cleaned", "") or _obj(r)
    gt = _gt(r)
    if r.get("seg_purpose") == "evidence":
        return f"The mask assesses the {building} for construction activity — it is {gt}."
    return f"The mask isolates the {building}, confirming {gt}."


def think3_arrangement(r):
    obj = _obj(r)
    gt = _gt(r)
    return f"The mask isolates the {obj}, whose spatial layout confirms a {gt} arrangement."


def think3_referent(r):
    obj = _obj(r)
    gt = _gt(r)
    if _is_yn(gt):
        return f"The mask isolates the {obj}, confirming the answer is {gt}."
    return f"The mask isolates the {obj}, confirming {gt}."


THINK3_GEN = {
    "shape": think3_shape,
    "category": think3_category,
    "counting": think3_counting,
    "color": think3_color,
    "material": think3_material,
    "relative": think3_relative,
    "existence": think3_existence,
    "function": think3_function,
    "activity": think3_activity,
    "motion": think3_motion,
    "construction": think3_construction,
    "arrangement": think3_arrangement,
    "referent": think3_referent,
}


# ---------------------------------------------------------------------------
# Main rewrite logic
# ---------------------------------------------------------------------------

def rewrite_assistant_text(r):
    text = r["assistant_text"]
    qtype = r.get("qtype", "referent")

    if "</zoom>" not in text or "</seg>" not in text:
        return text

    gen2 = THINK2_GEN.get(qtype, think2_referent)
    gen3 = THINK3_GEN.get(qtype, think3_referent)

    new_think2 = gen2(r)
    new_think3 = gen3(r)

    # Replace Think 2: between </zoom> and <seg>
    pat_think2 = re.compile(r'(</zoom>\n)<think>.*?</think>(\n<seg>)', re.DOTALL)
    text = pat_think2.sub(f'\\1<think>{new_think2}</think>\\2', text, count=1)

    # Replace Think 3: between </seg> and <answer>
    pat_think3 = re.compile(r'(</seg>\n)<think>.*?</think>(\n<answer>)', re.DOTALL)
    text = pat_think3.sub(f'\\1<think>{new_think3}</think>\\2', text, count=1)

    return text


def main():
    with open(SRC) as f:
        records = [json.loads(line) for line in f]

    modified = 0
    skipped = 0
    out_records = []

    for r in records:
        new_text = rewrite_assistant_text(r)
        if new_text != r["assistant_text"]:
            modified += 1
        else:
            skipped += 1
        r_out = dict(r)
        r_out["assistant_text"] = new_text
        out_records.append(r_out)

    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done. Modified: {modified}, Skipped (no zoom/seg): {skipped}, Total: {len(out_records)}")
    return out_records


if __name__ == "__main__":
    main()
