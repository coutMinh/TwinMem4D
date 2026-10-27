"""VLM baselines (plan B1): Qwen2.5-VL answers with a mark id, from the current frame or k history frames.

Candidates are the front-end masks at t_q whose DAM category matches the query, the same
candidate pool the memory chooses from. History frames are unmarked (marks would leak track IDs).
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..data.episode import Episode
from ..frontend.store import FrontendResult
from ..queries.templates import Query
from .som import draw_marks, history_frames

PROMPT = (
    "These images come from a fixed camera watching a robot arm on a table, in chronological order. "
    "The last image is the current frame; candidate objects in it are outlined and numbered. "
    "Which numbered object is {text}? Answer with the number only."
)


@dataclass
class VLMConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 8
    mark_scale: int = 2


def candidate_rows(frontend: FrontendResult, query: Query) -> List[int]:
    return [r for r in frontend.rows_at(query.t_q)
            if frontend.tracks[frontend.track_id[r]].category == query.category]


class QwenSoM:
    def __init__(self, cfg: VLMConfig):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_name, torch_dtype=getattr(torch, cfg.dtype), device_map=cfg.device
        ).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_name)

    def _generate(self, content: List[Dict], images: List[Image.Image]):
        import torch

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return reply, int(inputs.input_ids.shape[1])

    def answer(self, episode: Episode, frontend: FrontendResult, query: Query, k: int) -> Dict:
        rows = candidate_rows(frontend, query)
        base = {"qid": query.qid, "k": k, "n_candidates": len(rows), "mask_row": None, "reply": "", "tokens": 0}
        if not rows:
            return {**base, "latency_s": 0.0, "reason": "no candidates"}
        content, images = [], []
        for t in history_frames(query.t_q, k):
            content += [{"type": "text", "text": f"t = {t / episode.fps:.1f} s"}, {"type": "image"}]
            images.append(Image.fromarray(episode.rgb(t)).resize(
                (episode.camera.width * self.cfg.mark_scale, episode.camera.height * self.cfg.mark_scale)))
        marked = draw_marks(episode.rgb(query.t_q), [frontend.mask(r) for r in rows], self.cfg.mark_scale)
        content += [{"type": "text", "text": f"t = {query.t_q / episode.fps:.1f} s (current)"}, {"type": "image"},
                    {"type": "text", "text": PROMPT.format(text=query.text)}]
        images.append(Image.fromarray(marked))

        t0 = time.perf_counter()
        reply, tokens = self._generate(content, images)
        latency = time.perf_counter() - t0
        mark = parse_mark(reply, len(rows))
        return {**base, "reply": reply, "tokens": tokens, "latency_s": latency,
                "mask_row": rows[mark - 1] if mark else None, "reason": "" if mark else "unparsable reply"}


def parse_mark(reply: str, n: int) -> Optional[int]:
    m = re.search(r"\d+", reply)
    if not m:
        return None
    k = int(m.group())
    return k if 1 <= k <= n else None


def random_baseline(frontend: FrontendResult, query: Query, rng: np.random.Generator) -> Optional[int]:
    """Chance level: a uniformly random same-category candidate."""
    rows = candidate_rows(frontend, query)
    return int(rng.choice(rows)) if rows else None
