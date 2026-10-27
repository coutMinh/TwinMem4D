# Copyright (c) 2026, Massachusetts Institute of Technology. BSD 3-Clause (see LICENSE).
# Vendored from DAAAM:
#   src/daaam/utils/embedding.py  (CLIPHandler [OpenCLIP backend], SentenceEmbeddingHandler,
#                                  get_combined_embedding)
#   src/daaam/grounding/workers/dam_grounding.py (description prefix/postfix for sentence features)
# Modifications (TwinMem4D): the Perception-Encoder backend, thread locks and provenance
# helpers are removed; logging uses the standard library.
"""CLIP crop features and sentence-T5 description features, as DAAAM attaches to each fragment."""

from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# DAAAM embeds each DAM description as: an object that looks like '<description>'
DESCRIPTION_PREFIX = "an object that looks like '"
DESCRIPTION_POSTFIX = "'"


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class CLIPHandler:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cuda"):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def image_features(self, crops: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """L2-normalised features of RGB crops (H, W, 3)."""
        import torch

        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        feats = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops[i:i + batch_size]])
                feats.append(self.model.encode_image(batch.to(self.device)).float().cpu().numpy())
        return _l2(np.vstack(feats))

    def text_features(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        import torch

        feats = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[i:i + batch_size]).to(self.device)
                feats.append(self.model.encode_text(tokens).float().cpu().numpy())
        return _l2(np.vstack(feats))


class SentenceEmbeddingHandler:
    def __init__(self, model_name: str = "sentence-transformers/sentence-t5-xl", device: str = "cuda",
                 model_kwargs: Optional[Dict] = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs or {})

    def embed(self, texts: List[str], batch_size: int = 64, prefix: str = "", postfix: str = "") -> np.ndarray:
        formatted = [f"{prefix}{t}{postfix}" for t in texts]
        emb = self.model.encode(formatted, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = emb[None]
        return _l2(emb)

    def embed_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """Object-side features exactly as DAAAM's DAM grounding worker computes them."""
        return self.embed(descriptions, prefix=DESCRIPTION_PREFIX, postfix=DESCRIPTION_POSTFIX)


def get_combined_embedding(
    clip_embedding: Optional[np.ndarray],
    sentence_embedding: Optional[np.ndarray],
    clip_weight: float = 0.5,
    sentence_weight: float = 0.5,
) -> np.ndarray:
    """Weighted concatenation of CLIP and sentence features (DAAAM's "CLIP+DAM" retrieval feature)."""
    if clip_embedding is None and sentence_embedding is None:
        raise ValueError("At least one of clip_embedding or sentence_embedding must be provided")
    if clip_embedding is None:
        return _l2(sentence_embedding)
    if sentence_embedding is None:
        return _l2(clip_embedding)
    combined = np.concatenate(
        [clip_embedding * np.sqrt(clip_weight), sentence_embedding * np.sqrt(sentence_weight)], axis=-1
    )
    return _l2(combined)
