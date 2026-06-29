from __future__ import annotations

from agent.skill_retriever import SEED_SKILLS, SkillRetriever, extract_referent_phrase, parse_stage2_relation, retrieve_skills, skill_ids


def test_relative_relation_does_not_pollute_stage1_skill():
    question = "What is located directly to the right of the top-most military transport aircraft?"
    info = SkillRetriever(enabled=True).build(question, "Object relative position / context")
    assert info["referent_phrase"] == "top-most military transport aircraft"
    assert info["stage2_relation"] == "right of the referent"
    assert "top_extremum" in info["retrieved_skill_ids"]
    assert "right_extremum" not in info["retrieved_skill_ids"]


def test_corner_locator_requires_true_corner_phrase():
    skills = retrieve_skills("left-most sports field", SEED_SKILLS, max_skills=3)
    assert "left_extremum" in skill_ids(skills)
    assert "corner_locator" not in skill_ids(skills)

    skills = retrieve_skills("upper-right sports field", SEED_SKILLS, max_skills=3)
    assert "corner_locator" in skill_ids(skills)


def test_referent_and_relation_helpers():
    question = "Is there a parking lot to the left of the lower football field?"
    assert extract_referent_phrase(question) == "lower football field"
    assert parse_stage2_relation(question) == "left of the referent"
    assert extract_referent_phrase("What color is the surface of the left-most vertical breakwater?") == "left-most vertical breakwater"
    assert extract_referent_phrase("Is the left-most sports field a tennis court or a basketball court?") == "left-most sports field"
