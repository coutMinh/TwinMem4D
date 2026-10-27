"""R3: DAAAM Table 4 refCOCOg (umd) retrieval row: CLIP / DAM+ST / CLIP+DAM, Top-1/5/10.

Protocol (paper: val, 2573 subjects): every subject gets a CLIP ViT-L/14 crop feature and a
DAM-3B description embedded with sentence-T5-xl; each GT expression is a query ranked against
all subjects. Guessed details, recorded in the run metadata: crop = GT box without padding;
DAM prompt = GT mask; query side = CLIP text feature and sentence-T5 of the raw expression
(as DAAAM's ``get_matching_subjects`` tool); fusion = ``get_combined_embedding`` with 0.5/0.5.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..io_utils import PathLike
from ..third_party.daaam.embedding import get_combined_embedding

COCO_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
REPORTED = {  # DAAAM paper Table 4, refCOCOg rows (Top-1 / Top-5 / Top-10, %)
    "CLIP": (19.59, 41.12, 52.51),
    "DAM+ST": (18.07, 39.53, 51.18),
    "CLIP+DAM": (25.11, 50.10, 62.96),
}
PROTOCOL_GUESSES = {
    "crop": "GT bbox, no padding",
    "dam_prompt": "GT segmentation mask (polygons rasterised)",
    "query_side": "CLIP text(expression) and sentence-T5(expression), no prefix",
    "object_side_sentence": "sentence-T5(\"an object that looks like '<DAM description>'\")",
    "fusion": "get_combined_embedding(clip, sentence, 0.5, 0.5)",
}


@dataclass
class Subject:
    ref_id: int
    image_id: int
    bbox: List[float]                 # COCO xywh
    polygons: List[List[float]]
    expressions: List[str]


def load_refcocog_umd(refs_pickle: PathLike, instances_json: PathLike, split: str = "val") -> List[Subject]:
    """From the ``refer`` release files ``refs(umd).p`` and ``instances.json``."""
    with open(refs_pickle, "rb") as f:
        refs = pickle.load(f)
    anns = {a["id"]: a for a in json.loads(Path(instances_json).read_text())["annotations"]}
    subjects = []
    for r in refs:
        if r["split"] != split:
            continue
        ann = anns[r["ann_id"]]
        polys = ann["segmentation"] if isinstance(ann["segmentation"], list) else []
        subjects.append(Subject(r["ref_id"], r["image_id"], ann["bbox"], polys, [s["sent"] for s in r["sentences"]]))
    return subjects


def polygons_to_mask(polygons: List[List[float]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [np.asarray(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)], 1)
    return mask.astype(bool)


def crop_box(rgb: np.ndarray, bbox_xywh: List[float]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    x1, y1 = max(int(np.floor(x)), 0), max(int(np.floor(y)), 0)
    x2, y2 = min(int(np.ceil(x + w)), rgb.shape[1]), min(int(np.ceil(y + h)), rgb.shape[0])
    return rgb[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]


def topk_recall(sim: np.ndarray, target: np.ndarray, ks=(1, 5, 10)) -> Dict[str, float]:
    """``sim`` (queries, subjects); ``target`` subject index per query -> Top-k accuracy in %."""
    target_score = sim[np.arange(len(target)), target]
    rank = (sim > target_score[:, None]).sum(axis=1)  # 0 = best
    return {f"top{k}": float(100 * np.mean(rank < k)) for k in ks}


def retrieval_scores(subject_clip: np.ndarray, subject_sent: np.ndarray, query_clip: np.ndarray,
                     query_sent: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    fused_s = get_combined_embedding(subject_clip, subject_sent)
    fused_q = get_combined_embedding(query_clip, query_sent)
    return {
        "CLIP": topk_recall(query_clip @ subject_clip.T, target),
        "DAM+ST": topk_recall(query_sent @ subject_sent.T, target),
        "CLIP+DAM": topk_recall(fused_q @ fused_s.T, target),
    }


def describe_subjects(dam, subjects: List[Subject], image_loader, cache_path: Optional[PathLike] = None,
                      masks_per_call: int = 32) -> Dict[int, str]:
    """DAM descriptions per ref_id, batched over images, resumable through a JSON cache."""
    cache: Dict[int, str] = {}
    if cache_path and Path(cache_path).exists():
        cache = {int(k): v for k, v in json.loads(Path(cache_path).read_text()).items()}
    todo: Dict[int, List[Subject]] = {}
    for s in subjects:
        if s.ref_id in cache:
            continue
        if not s.polygons:  # RLE (crowd) annotations have no polygon mask to prompt DAM with
            cache[s.ref_id] = ""
            continue
        todo.setdefault(s.image_id, []).append(s)
    batch, keys = [], []
    image_ids = list(todo)
    for n, image_id in enumerate(image_ids):
        rgb = image_loader(image_id)
        h, w = rgb.shape[:2]
        batch.append((rgb, [polygons_to_mask(s.polygons, h, w) for s in todo[image_id]]))
        keys += [s.ref_id for s in todo[image_id]]
        if len(keys) >= masks_per_call or n == len(image_ids) - 1:
            descs = [d for per_image in dam.describe(batch) for d in per_image]
            cache.update({k: d.strip() for k, d in zip(keys, descs)})
            batch, keys = [], []
            if cache_path:
                Path(cache_path).write_text(json.dumps(cache))
    return cache
