import tempfile
import unittest
from pathlib import Path

from PIL import Image

from speedup.unsloth.skillrl.verl_grpo.data.prepare_zoomearth_parquet import build_prompt, convert_split


class DataSchemaTest(unittest.TestCase):
    def test_prompt_contains_single_image_placeholder(self):
        prompt = build_prompt("What is visible?", "Retrieved grounding skills:\n- s")
        self.assertEqual(sum(m["content"].count("<image>") for m in prompt), 1)

    def test_convert_split_uses_question_only_skill_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.png"
            Image.new("RGB", (64, 64), color="white").save(image_path)
            skillbank = {
                "general_skills": [
                    {"skill_id": "gen_skill_first_grounding", "title": "g", "principle": "p"},
                    {"skill_id": "format_primitive_bbox", "title": "f", "principle": "p"},
                ],
                "task_specific_skills": {},
            }
            records, stats = convert_split(
                [
                    {
                        "question_id": "q1",
                        "image_name": "img.png",
                        "question": "Is there a port?",
                        "label": "SECRET_LABEL_SHOULD_NOT_APPEAR",
                        "ground_truth": "yes",
                        "bbox": [1, 2, 30, 40],
                        "type": "object",
                    }
                ],
                split="train",
                output_dir=root / "out",
                image_dirs=[str(root)],
                skillbank=skillbank,
                coord_mode="max_side",
            )
            self.assertEqual(stats["kept_rows"], 1)
            text = "\n".join(m["content"] for m in records[0]["prompt"])
            self.assertNotIn("SECRET_LABEL_SHOULD_NOT_APPEAR", text)
            self.assertTrue(Path(records[0]["images"][0]).exists())


if __name__ == "__main__":
    unittest.main()
