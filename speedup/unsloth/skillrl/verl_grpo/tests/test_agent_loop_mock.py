from __future__ import annotations

import types
import asyncio
from pathlib import Path

from PIL import Image

from agent.zoomearth_agent_loop import ZoomEarthAgentLoop, build_stage2_messages


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i < 256)


def enc(text: str) -> list[int]:
    return [ord(c) for c in text]


class FakeServer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.images = []
        self.sampling_params = []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt_ids"])
        self.images.append(kwargs.get("image_data"))
        self.sampling_params.append(kwargs["sampling_params"])
        return self.outputs.pop(0)


def output(text: str):
    return types.SimpleNamespace(token_ids=enc(text), log_probs=None, routed_experts=None)


def test_agent_loop_preserves_stage1_tokens_and_masks_observation(tmp_path: Path):
    img = tmp_path / "img.png"
    Image.new("RGB", (1000, 1000), "white").save(img)
    stage1 = "<zoom><|ref|>target<|/ref|><|box|>[[10,10,200,220]]<|/box|></zoom>"
    stage2 = "<answer>white</answer>"

    loop = ZoomEarthAgentLoop.__new__(ZoomEarthAgentLoop)
    loop.prompt_length = 4096
    loop.response_length = 4096
    loop.stage2_observation_role = "user"
    loop.crop_max_size = 512
    loop.stage1_max_tokens = 512
    loop.stage2_max_tokens = 1024
    loop.tokenizer = CharTokenizer()
    loop.processor = types.SimpleNamespace(image_token_id=999, video_token_id=1000)
    loop.server_manager = FakeServer(
        [
            types.SimpleNamespace(token_ids=[999] + enc(stage1) + [999], log_probs=None, routed_experts=None),
            output(stage2),
        ]
    )

    async def process_multi_modal_info(messages):
        return {"images": ["global-image"]}

    async def apply_chat_template(messages, **kwargs):
        if kwargs.get("remove_system_prompt"):
            return [300, 301, 302]
        return [100, 101]

    loop.process_multi_modal_info = process_multi_modal_info
    loop.apply_chat_template = apply_chat_template
    loop._get_mm_processor_kwargs = lambda audios=None: {}

    result = asyncio.run(
        loop.run(
            {"temperature": 0},
            raw_prompt=[{"role": "user", "content": "<image>\nq"}],
            extra_info={"original_image_path": str(img), "retrieved_skill_ids": ["left_extremum"]},
        )
    )
    response_ids = result.response_ids
    response_mask = result.response_mask
    assert 999 not in response_ids
    assert result.extra_fields["stage1_mm_tokens_stripped"] == 2
    assert loop.server_manager.sampling_params[0]["max_tokens"] == 512
    assert loop.server_manager.sampling_params[1]["max_tokens"] == 1024
    assert loop.server_manager.prompts[1][: 2 + len(enc(stage1))] == [100, 101] + enc(stage1)
    assert loop.server_manager.images[0] == ["global-image"]
    assert len(loop.server_manager.images[1]) == 1
    assert result.multi_modal_data["images"][0] == "global-image"
    assert len(result.multi_modal_data["images"]) == 2
    obs_start = len(enc(stage1))
    assert response_ids[obs_start : obs_start + 3] == [300, 301, 302]
    assert response_mask[obs_start : obs_start + 3] == [0, 0, 0]
    assert result.extra_fields["zoom_text"]
    assert result.extra_fields["answer_text"]
    assert result.extra_fields["answer_pred"] == "white"


def test_stage2_role_builders_exist():
    for role in ("tool", "user", "assistant+user"):
        assert build_stage2_messages(role)
