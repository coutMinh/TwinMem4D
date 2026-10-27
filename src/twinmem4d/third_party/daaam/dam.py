# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/query_manager/dam/services.py  (DAMAgentPanoptic.__init__, query_multi_image_multi_mask)
#   src/daaam/grounding/utils.py             (prepare_masks_from_tracks)
#   src/daaam/grounding/workers/dam_grounding.py (query text and sampling parameters)
# Modifications (TwinMem4D): the gradio / OpenAI / LangChain imports and debug prints of the
# upstream agent are removed; query_single is expressed through the multi-image call.
"""Describe Anything Model (DAM) with DAAAM's multi-image, multi-mask batching."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Prompt and decoding settings used by DAAAM's DAMGroundingWorkerMultiImage
DAM_QUERY = "Describe what you see in this region."
DAM_SAMPLING = {"temperature": 0.2, "top_p": 0.9, "num_beams": 1}
DAM_MAX_NEW_TOKENS = 512


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


class DAMAgent:
    """Thin wrapper around ``dam.DescribeAnythingModel`` from DAAAM's describe-anything fork."""

    def __init__(self, model_path: str = "nvidia/DAM-3B", conv_mode: str = "v1", prompt_mode: str = "focal_prompt"):
        from dam import DescribeAnythingModel, disable_torch_init

        disable_torch_init()
        self.model = DescribeAnythingModel(
            model_path=model_path,
            conv_mode=conv_mode,
            prompt_mode=f"full+{prompt_mode.replace('_prompt', '_crop')}" if "prompt" in prompt_mode else prompt_mode,
        )

    def describe(
        self,
        image_mask_pairs: Sequence[Tuple[np.ndarray, Sequence[np.ndarray]]],
        query: str = DAM_QUERY,
        batch_size: Optional[int] = None,
        auto_batch: bool = True,
        max_new_tokens: int = DAM_MAX_NEW_TOKENS,
    ) -> List[List[str]]:
        """One description per mask, grouped per image, in a single batched DAM pass."""
        if not image_mask_pairs:
            return []
        pairs = [
            (Image.fromarray(rgb).convert("RGB"), [mask_to_pil(m) for m in masks])
            for rgb, masks in image_mask_pairs
        ]
        if "<image>" not in query:
            query = f"<image>\n{query}"
        return self.model.get_multi_image_descriptions(
            image_mask_pairs=pairs,
            query=query,
            streaming=False,
            batch_size=batch_size,
            auto_batch=auto_batch,
            max_new_tokens=max_new_tokens,
            **DAM_SAMPLING,
        )
