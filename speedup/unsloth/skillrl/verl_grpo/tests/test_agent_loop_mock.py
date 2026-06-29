from __future__ import annotations

import types
import asyncio
from pathlib import Path

import torch
from PIL import Image

from agent.zoomearth_agent_loop import ZoomEarthAgentLoop, build_stage2_messages


class CharTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return enc(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i < 256)


def enc(text: str) -> list[int]:
    return [ord(c) for c in text]


def contains_subsequence(values: list[int], subsequence: list[int]) -> bool:
    if not subsequence:
        return True
    return any(values[i : i + len(subsequence)] == subsequence for i in range(len(values) - len(subsequence) + 1))


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
    assistant_end = "<|im_end|>"
    stage1_trailing = "<answer>too early</answer>"
    stage2_trailing = "<zoom>second zoom[[1,1,2,2]]</zoom>"

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
            types.SimpleNamespace(
                token_ids=[999] + enc(stage1 + stage1_trailing) + [999],
                log_probs=None,
                routed_experts=None,
            ),
            types.SimpleNamespace(
                token_ids=enc(stage2 + stage2_trailing),
                log_probs=None,
                routed_experts=torch.ones((2, 1, 1), dtype=torch.int64),
            ),
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
    assert result.extra_fields["stage1_trailing_tokens_stripped"] == len(enc(stage1_trailing))
    assert result.extra_fields["stage2_trailing_tokens_stripped"] == len(enc(stage2_trailing))
    assert loop.server_manager.sampling_params[0]["max_tokens"] == 512
    assert loop.server_manager.sampling_params[1]["max_tokens"] == 1024
    assert loop.server_manager.sampling_params[0]["stop"] == ["</zoom>", "<|/zoom>", "<|/zoom|>"]
    assert loop.server_manager.sampling_params[1]["stop"] == ["</answer>"]
    assert loop.server_manager.sampling_params[0]["include_stop_str_in_output"] is True
    assert loop.server_manager.sampling_params[1]["include_stop_str_in_output"] is True
    assert "<answer>" in loop.server_manager.sampling_params[0]["bad_words"]
    assert "<image>" in loop.server_manager.sampling_params[0]["bad_words"]
    assert "</image>" in loop.server_manager.sampling_params[0]["bad_words"]
    assert "<zoom>" in loop.server_manager.sampling_params[1]["bad_words"]
    assert loop.server_manager.prompts[1][: 2 + len(enc(stage1))] == [100, 101] + enc(stage1)
    assert loop.server_manager.prompts[1][2 + len(enc(stage1)) : 2 + len(enc(stage1 + assistant_end))] == enc(
        assistant_end
    )
    assert not contains_subsequence(loop.server_manager.prompts[1], enc(stage1_trailing))
    assert loop.server_manager.images[0] == ["global-image"]
    assert len(loop.server_manager.images[1]) == 2
    assert loop.server_manager.images[1][0] == "global-image"
    assert result.multi_modal_data["images"][0] == "global-image"
    assert len(result.multi_modal_data["images"]) == 2
    assert result.extra_fields["assistant_end_tokens"] == len(enc(assistant_end))
    assert result.num_turns == 4
    boundary_start = len(enc(stage1))
    obs_start = boundary_start + len(enc(assistant_end))
    assert response_ids[boundary_start:obs_start] == enc(assistant_end)
    assert response_mask[boundary_start:obs_start] == [0] * len(enc(assistant_end))
    assert response_ids[obs_start : obs_start + 3] == [300, 301, 302]
    assert not contains_subsequence(response_ids, enc(stage1_trailing))
    assert not contains_subsequence(response_ids, enc(stage2_trailing))
    assert response_mask[obs_start : obs_start + 3] == [0, 0, 0]
    assert result.extra_fields["zoom_text"]
    assert result.extra_fields["answer_text"]
    assert result.extra_fields["answer_pred"] == "white"
    assert result.routed_experts.shape == (2, 1, 1)


def test_stage2_role_builders_exist():
    for role in ("tool", "user", "assistant+user"):
        assert build_stage2_messages(role)


def test_agent_loop_stage1_only_reports_two_turns(tmp_path: Path):
    img = tmp_path / "img.png"
    Image.new("RGB", (1000, 1000), "white").save(img)

    loop = ZoomEarthAgentLoop.__new__(ZoomEarthAgentLoop)
    loop.prompt_length = 4096
    loop.response_length = 4096
    loop.stage2_observation_role = "user"
    loop.crop_max_size = 512
    loop.stage1_max_tokens = 512
    loop.stage2_max_tokens = 1024
    loop.tokenizer = CharTokenizer()
    loop.processor = types.SimpleNamespace(image_token_id=999, video_token_id=1000)
    loop.server_manager = FakeServer([output("<zoom>bad box</zoom>")])

    async def process_multi_modal_info(messages):
        return {"images": ["global-image"]}

    async def apply_chat_template(messages, **kwargs):
        return [100, 101]

    loop.process_multi_modal_info = process_multi_modal_info
    loop.apply_chat_template = apply_chat_template
    loop._get_mm_processor_kwargs = lambda audios=None: {}

    result = asyncio.run(
        loop.run(
            {"temperature": 0},
            raw_prompt=[{"role": "user", "content": "<image>\nq"}],
            extra_info={"original_image_path": str(img)},
        )
    )
    assert result.num_turns == 2
    assert result.extra_fields["zoom_parse_ok"] is False
    assert len(loop.server_manager.prompts) == 1


def test_agent_loop_compact_zoom_does_not_create_crop(tmp_path: Path):
    img = tmp_path / "img.png"
    Image.new("RGB", (1000, 1000), "white").save(img)

    loop = ZoomEarthAgentLoop.__new__(ZoomEarthAgentLoop)
    loop.prompt_length = 4096
    loop.response_length = 4096
    loop.stage2_observation_role = "user"
    loop.crop_max_size = 512
    loop.stage1_max_tokens = 512
    loop.stage2_max_tokens = 1024
    loop.tokenizer = CharTokenizer()
    loop.processor = types.SimpleNamespace(image_token_id=999, video_token_id=1000)
    loop.server_manager = FakeServer([output("<zoom>[[10,10,200,220]]</zoom>")])

    async def process_multi_modal_info(messages):
        return {"images": ["global-image"]}

    async def apply_chat_template(messages, **kwargs):
        return [100, 101]

    loop.process_multi_modal_info = process_multi_modal_info
    loop.apply_chat_template = apply_chat_template
    loop._get_mm_processor_kwargs = lambda audios=None: {}

    result = asyncio.run(
        loop.run(
            {"temperature": 0},
            raw_prompt=[{"role": "user", "content": "<image>\nq"}],
            extra_info={"original_image_path": str(img)},
        )
    )
    assert result.num_turns == 2
    assert result.extra_fields["zoom_parse_ok"] is False
    assert result.extra_fields["zoom_bbox_format"] == "compact"
    assert result.extra_fields["crop_created"] is False
    assert len(loop.server_manager.prompts) == 1
    assert result.response_mask == [1] * len(result.response_ids)
