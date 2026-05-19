"""
postprocess.py
--------------
Post-processing for object detection predictions.

Rules applied in order:
1. Score threshold filter.
2. Same-class containment NMS: remove smaller box if >= containment_thr of
   its area is covered by a larger box of the same class.
3. Text box merging: iteratively merge text boxes that overlap >= merge_thr,
   then add padding to each merged text box.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _overlap_of_small(small: list[float], large: list[float]) -> float:
    """Fraction of *small* bbox [x,y,w,h] covered by *large* bbox."""
    ax, ay, aw, ah = small
    bx, by, bw, bh = large
    if aw * ah == 0:
        return 0.0
    ix  = max(ax, bx);  iy  = max(ay, by)
    ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    return (ix2 - ix) * (iy2 - iy) / (aw * ah)


def _iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a;  bx, by, bw, bh = b
    ix  = max(ax, bx);  iy  = max(ay, by)
    ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    return inter / (aw * ah + bw * bh - inter)


def _merge_two(a: list[float], b: list[float]) -> list[float]:
    """Return the union bounding box of two [x,y,w,h] bboxes."""
    x1 = min(a[0], b[0]);  y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    return [x1, y1, x2 - x1, y2 - y1]


# ── Step 2a: same-class overlap merge ────────────────────────────────────────

def merge_same_class_overlapping(
    predictions: list[dict[str, Any]],
    merge_iou_thr: float = 0.3,
    exclude_cat_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Iteratively merge same-class boxes that overlap (IoU >= merge_iou_thr)
    into their union bounding box. Excluded categories (e.g. text) are skipped.
    """
    exclude_cat_ids = exclude_cat_ids or set()

    by_cls: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        by_cls[p["category_id"]].append(p)

    result: list[dict] = []
    for cls_id, boxes in by_cls.items():
        if cls_id in exclude_cat_ids:
            result.extend(boxes)
            continue

        changed = True
        while changed:
            changed = False
            merged_flags = [False] * len(boxes)
            new_boxes: list[dict] = []
            for i in range(len(boxes)):
                if merged_flags[i]:
                    continue
                base = copy.deepcopy(boxes[i])
                for j in range(i + 1, len(boxes)):
                    if merged_flags[j]:
                        continue
                    if _iou(base["bbox"], boxes[j]["bbox"]) >= merge_iou_thr:
                        base["bbox"]  = _merge_two(base["bbox"], boxes[j]["bbox"])
                        base["score"] = max(base["score"], boxes[j]["score"])
                        merged_flags[j] = True
                        changed = True
                new_boxes.append(base)
            boxes = new_boxes
        result.extend(boxes)
    return result


# ── Step 2b: same-class containment NMS ──────────────────────────────────────

def filter_same_class_containment(
    predictions: list[dict[str, Any]],
    containment_thr: float = 0.8,
) -> list[dict[str, Any]]:
    """Remove smaller boxes >= containment_thr contained inside a larger same-class box."""
    by_cls: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        by_cls[p["category_id"]].append(p)

    kept: list[dict] = []
    for cls_id, boxes in by_cls.items():
        boxes = sorted(boxes, key=lambda p: p["bbox"][2] * p["bbox"][3], reverse=True)
        suppressed = [False] * len(boxes)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            kept.append(boxes[i])
            for j in range(i + 1, len(boxes)):
                if not suppressed[j]:
                    if _overlap_of_small(boxes[j]["bbox"], boxes[i]["bbox"]) >= containment_thr:
                        suppressed[j] = True
    return kept


# ── Step 3: text box merging + padding ───────────────────────────────────────

def merge_text_boxes(
    predictions: list[dict[str, Any]],
    text_cat_id: int,
    merge_thr: float = 0.2,
    padding: float = 5.0,
) -> list[dict[str, Any]]:
    """Iteratively merge text boxes that overlap >= merge_thr of the smaller box,
    then expand each merged text box by *padding* pixels on all sides.

    Non-text predictions are returned unchanged.
    """
    text_boxes  = [copy.deepcopy(p) for p in predictions if p["category_id"] == text_cat_id]
    other_boxes = [p for p in predictions if p["category_id"] != text_cat_id]

    # Iterative union-find merge
    changed = True
    while changed:
        changed = False
        merged_flags = [False] * len(text_boxes)
        new_boxes: list[dict] = []
        for i in range(len(text_boxes)):
            if merged_flags[i]:
                continue
            base = copy.deepcopy(text_boxes[i])
            for j in range(i + 1, len(text_boxes)):
                if merged_flags[j]:
                    continue
                ov_i = _overlap_of_small(text_boxes[i]["bbox"], text_boxes[j]["bbox"])
                ov_j = _overlap_of_small(text_boxes[j]["bbox"], text_boxes[i]["bbox"])
                if max(ov_i, ov_j) >= merge_thr:
                    base["bbox"]  = _merge_two(base["bbox"], text_boxes[j]["bbox"])
                    base["score"] = max(base["score"], text_boxes[j]["score"])
                    merged_flags[j] = True
                    changed = True
            new_boxes.append(base)
        text_boxes = new_boxes

    # Add padding
    for p in text_boxes:
        x, y, w, h = p["bbox"]
        p["bbox"] = [
            x - padding,
            y - padding,
            w + 2 * padding,
            h + 2 * padding,
        ]

    return other_boxes + text_boxes


# ── Full pipeline ─────────────────────────────────────────────────────────────

def apply_postprocess(
    predictions: list[dict[str, Any]],
    cat_name_to_id: dict[str, int] | None = None,
    score_thr: float = 0.5,
    containment_thr: float = 0.8,
    merge_iou_thr: float = 0.3,
    text_merge_thr: float = 0.2,
    text_padding: float = 5.0,
) -> list[dict[str, Any]]:
    """Full post-processing pipeline.

    1. Score threshold filter.
    2. Same-class overlap merge: boxes with IoU >= merge_iou_thr → union box
       (skips text, which is handled separately in step 4).
    3. Same-class containment NMS: remove smaller box if >= containment_thr covered.
    4. Text box merging + padding.
    """
    predictions = [p for p in predictions if p["score"] >= score_thr]

    text_cat_id = cat_name_to_id.get("text") if cat_name_to_id else None
    exclude = {text_cat_id} if text_cat_id is not None else set()

    predictions = merge_same_class_overlapping(predictions, merge_iou_thr, exclude_cat_ids=exclude)
    predictions = filter_same_class_containment(predictions, containment_thr)

    if text_cat_id is not None:
        predictions = merge_text_boxes(
            predictions,
            text_cat_id=text_cat_id,
            merge_thr=text_merge_thr,
            padding=text_padding,
        )
    return predictions
