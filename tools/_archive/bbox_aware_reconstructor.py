#!/usr/bin/env python3
"""
BBoxAwareReconstructor
======================
Integrates Detectron2 bounding-box predictions with DXFParser text_coords
to produce a much more accurate Markdown reconstruction.

Classes detected by the model:
  0 = text
  1 = table
  2 = title_block
  3 = diagram

Usage:
  from dxf_parser import DXFParser
  from bbox_aware_reconstructor import BBoxAwareReconstructor

  parser = DXFParser("drawing.dxf")
  parser.parse()

  # bboxes from detectron2: list of {"label": str, "bbox": [x1,y1,x2,y2], "score": float}
  reconstructor = BBoxAwareReconstructor(parser.text_coords, bboxes)
  md = reconstructor.to_markdown()
  Path("drawing_bbox.md").write_text(md, encoding="utf-8")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class DetectedBox:
    label: str           # "text" | "table" | "title_block" | "diagram"
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


# TextItem: (value, x, y) — same format as DXFParser.text_coords
TextItem = tuple[str, float, float]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _point_in_box(x: float, y: float, box: DetectedBox, margin: float = 2.0) -> bool:
    return (box.x1 - margin <= x <= box.x2 + margin and
            box.y1 - margin <= y <= box.y2 + margin)


def _group_rows(items: list[TextItem], y_tol: float = 5.0) -> list[list[TextItem]]:
    """Group text items into Y-rows (top→bottom, left→right within row)."""
    sorted_items = sorted(items, key=lambda t: (-t[2], t[1]))
    rows: list[list[TextItem]] = []
    cur: list[TextItem] = []
    cur_y: float | None = None
    for item in sorted_items:
        if cur_y is None or abs(item[2] - cur_y) <= y_tol:
            cur.append(item)
            if cur_y is None:
                cur_y = item[2]
        else:
            if cur:
                rows.append(sorted(cur, key=lambda t: t[1]))
            cur = [item]
            cur_y = item[2]
    if cur:
        rows.append(sorted(cur, key=lambda t: t[1]))
    return rows


def _cluster_xs(items: list[TextItem], gap_ratio: float = 0.04) -> list[float]:
    """Return column boundary X positions based on X-gap analysis."""
    xs = sorted(set(round(t[1], 1) for t in items))
    if len(xs) < 2:
        return []
    x_span = xs[-1] - xs[0]
    boundaries: list[float] = []
    for i in range(len(xs) - 1):
        if xs[i + 1] - xs[i] > x_span * gap_ratio:
            boundaries.append((xs[i] + xs[i + 1]) / 2)
    return boundaries


# ── Table reconstructor ────────────────────────────────────────────────────────

class TableReconstructor:
    """
    Reconstruct a table from text items inside a table bounding box.

    Strategy:
      1. Group items into Y-rows (rows of the table).
      2. Within each row, cluster items by X proximity → columns.
      3. Align columns across rows by matching column boundaries.
      4. Emit Markdown table.
    """

    Y_TOL = 6.0    # units: items within 6 DXF units are in same row
    X_GAP = 15.0   # units: gap >= 15 between items = new column

    def __init__(self, items: list[TextItem]):
        self.items = items

    def reconstruct(self) -> list[str]:
        if not self.items:
            return []

        rows = _group_rows(self.items, y_tol=self.Y_TOL)
        if not rows:
            return []

        # Determine global column boundaries from ALL items
        all_xs = sorted(t[1] for t in self.items)
        col_boundaries: list[float] = []
        for i in range(len(all_xs) - 1):
            if all_xs[i + 1] - all_xs[i] >= self.X_GAP:
                col_boundaries.append((all_xs[i] + all_xs[i + 1]) / 2)

        n_cols = len(col_boundaries) + 1

        def _assign_col(x: float) -> int:
            for ci, boundary in enumerate(col_boundaries):
                if x < boundary:
                    return ci
            return n_cols - 1

        # Build grid: row × col → list of texts
        grid: list[list[list[str]]] = [
            [[] for _ in range(n_cols)] for _ in rows
        ]
        for ri, row in enumerate(rows):
            for text, x, _ in row:
                ci = _assign_col(x)
                grid[ri][ci].append(text.strip())

        # Convert grid to Markdown
        md: list[str] = []
        for ri, row_cells in enumerate(grid):
            cells = [" ".join(cell) if cell else "" for cell in row_cells]
            md.append("| " + " | ".join(cells) + " |")
            if ri == 0:
                md.append("|" + "---|" * n_cols)

        return md


# ── Spec / text region reconstructor ──────────────────────────────────────────

_SECTION_HDR = re.compile(
    r'^(\d+\.\s*.{1,20}|[一-龥ぁ-んァ-ン]{2,12}'
    r'(概要|仕様|工事|方針|区分|写真|書等|金額)?)\s*$'
)
_NUMBERED = re.compile(r'^\s*(\d+[\.\．]|[①-⑳]|（\d+）|\(\d+\))\s+')


class TextRegionReconstructor:
    """
    Reconstruct a free-text region (multi-column spec or general text).
    Uses column gap detection and section-header heuristics.
    """

    Y_TOL = 4.0
    X_PROXIMITY = 30.0

    def __init__(self, items: list[TextItem]):
        self.items = items

    def reconstruct(self) -> list[str]:
        if not self.items:
            return []

        # Find column boundaries
        boundaries_xs = _cluster_xs(self.items, gap_ratio=0.04)
        x_min = min(t[1] for t in self.items)
        x_max = max(t[1] for t in self.items)
        all_boundaries = [x_min - 1.0] + boundaries_xs + [x_max + 1.0]

        # Split items into columns
        columns: list[list[TextItem]] = []
        for i in range(len(all_boundaries) - 1):
            lo, hi = all_boundaries[i], all_boundaries[i + 1]
            col = [t for t in self.items if lo < t[1] <= hi]
            if col:
                columns.append(col)

        md: list[str] = []
        prev_was_table = False

        for ci, col in enumerate(columns):
            if len(columns) > 1:
                # Column separator
                col_title = sorted(col, key=lambda t: -t[2])
                title_txt = next(
                    (t[0].strip() for t in col_title if len(t[0].strip()) > 2), f"列 {ci + 1}"
                )
                md += ["", "---", "", f"### {title_txt[:30]}", ""]
                prev_was_table = False

            rows = _group_rows(col, y_tol=self.Y_TOL)

            for row in rows:
                items_in_row = [(t[0].strip(), t[1]) for t in row if t[0].strip()]
                if not items_in_row:
                    continue

                # Sub-group by X proximity
                sub_groups: list[list[str]] = []
                cur_grp: list[str] = []
                prev_x: float | None = None
                for txt, x in sorted(items_in_row, key=lambda r: r[1]):
                    if prev_x is None or x - prev_x <= self.X_PROXIMITY:
                        cur_grp.append(txt)
                    else:
                        if cur_grp:
                            sub_groups.append(cur_grp)
                        cur_grp = [txt]
                    prev_x = x
                if cur_grp:
                    sub_groups.append(cur_grp)

                flat = [t for sg in sub_groups for t in sg]

                # Table row: ≥4 short items
                if len(flat) >= 4 and all(len(t) <= 12 for t in flat):
                    if not prev_was_table:
                        md.append("| " + " | ".join(flat) + " |")
                        md.append("|" + "---|" * len(flat))
                        prev_was_table = True
                    else:
                        md.append("| " + " | ".join(flat) + " |")
                    continue
                else:
                    if prev_was_table:
                        md.append("")
                    prev_was_table = False

                for sg in sub_groups:
                    if not sg:
                        continue
                    if len(sg) == 1:
                        t = sg[0]
                        if _SECTION_HDR.match(t) and len(t) <= 25:
                            md += ["", f"**{t}**"]
                        elif _NUMBERED.match(t):
                            md.append(f"- {t}")
                        else:
                            md.append(f"  {t}")
                    else:
                        first = sg[0]
                        if len(first) <= 15 and _SECTION_HDR.match(first):
                            md += ["", f"**{first}**", f"  {'　'.join(sg[1:])}"]
                        else:
                            md.append(f"  {'　'.join(sg)}")

        return md


# ── Main class ─────────────────────────────────────────────────────────────────

class BBoxAwareReconstructor:
    """
    Use detectron2 bounding boxes to guide DXF text reconstruction.

    Parameters
    ----------
    text_coords : list of (text, x, y)
        From DXFParser.text_coords
    bboxes : list of dict with keys:
        "label" : "text" | "table" | "title_block" | "diagram"
        "bbox"  : [x1, y1, x2, y2]  (DXF coordinate space)
        "score" : float (confidence)
    score_threshold : float
        Minimum confidence score to use a bbox.
    """

    LABEL_TEXT        = "text"
    LABEL_TABLE       = "table"
    LABEL_TITLE_BLOCK = "title_block"
    LABEL_DIAGRAM     = "diagram"

    def __init__(
        self,
        text_coords: list[TextItem],
        bboxes: list[dict],
        score_threshold: float = 0.3,
    ):
        self.text_coords = text_coords
        self.boxes: list[DetectedBox] = []
        for b in bboxes:
            score = b.get("score", 1.0)
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = b["bbox"]
            self.boxes.append(DetectedBox(
                label=b["label"],
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                score=float(score),
            ))

        # Assign each text item to a box region
        self._assigned: dict[int, DetectedBox] = {}   # idx → box
        self._unassigned: list[int] = []

        self._assign_texts()

    # ── Assignment ────────────────────────────────────────────────────────────

    def _assign_texts(self) -> None:
        """Assign each text item to the highest-confidence box it falls in."""
        for idx, (text, x, y) in enumerate(self.text_coords):
            best_box: DetectedBox | None = None
            best_score = -1.0
            for box in self.boxes:
                if _point_in_box(x, y, box):
                    if box.score > best_score:
                        best_score = box.score
                        best_box = box
            if best_box is not None:
                self._assigned[idx] = best_box
            else:
                self._unassigned.append(idx)

    def _items_in_box(self, box: DetectedBox) -> list[TextItem]:
        return [
            self.text_coords[idx]
            for idx, b in self._assigned.items()
            if b is box
        ]

    def _unassigned_items(self) -> list[TextItem]:
        return [self.text_coords[idx] for idx in self._unassigned]

    # ── Per-region reconstruction ──────────────────────────────────────────────

    def _reconstruct_table_box(self, box: DetectedBox) -> list[str]:
        items = self._items_in_box(box)
        if not items:
            return []
        rec = TableReconstructor(items)
        lines = rec.reconstruct()
        return ["", "### Table", ""] + lines if lines else []

    def _reconstruct_text_box(self, box: DetectedBox) -> list[str]:
        items = self._items_in_box(box)
        if not items:
            return []
        rec = TextRegionReconstructor(items)
        lines = rec.reconstruct()
        return lines if lines else []

    def _reconstruct_title_block(self, box: DetectedBox) -> list[str]:
        items = self._items_in_box(box)
        if not items:
            return []
        lines = ["", "## 表題欄 (Title Block)", ""]
        for text, _, _ in sorted(items, key=lambda t: (-t[2], t[1])):
            if text.strip():
                lines.append(f"- {text.strip()}")
        return lines

    def _reconstruct_diagram_labels(self, box: DetectedBox) -> list[str]:
        """Keep diagram text as a collapsible note (labels only)."""
        items = self._items_in_box(box)
        if not items:
            return []
        labels = [t[0].strip() for t in sorted(items, key=lambda t: (-t[2], t[1]))
                  if t[0].strip()]
        if not labels:
            return []
        lines = ["", "### Diagram Labels", ""]
        for lb in labels:
            lines.append(f"- {lb}")
        return lines

    # ── Main export ────────────────────────────────────────────────────────────

    def to_markdown(self, include_diagrams: bool = False) -> str:
        """
        Build Markdown output using bbox-guided reconstruction.

        Layout order:
          1. Title block(s)
          2. Text regions (top-down, left-right by box position)
          3. Table regions
          4. Unassigned text (items outside any detected box)
          5. Diagram labels (optional)
        """
        md: list[str] = []

        # Sort boxes top→bottom, left→right (by y1 desc then x1 asc)
        ordered_boxes = sorted(self.boxes, key=lambda b: (-b.y1, b.x1))

        title_blocks = [b for b in ordered_boxes if b.label == self.LABEL_TITLE_BLOCK]
        table_boxes  = [b for b in ordered_boxes if b.label == self.LABEL_TABLE]
        text_boxes   = [b for b in ordered_boxes if b.label == self.LABEL_TEXT]
        diagram_boxes = [b for b in ordered_boxes if b.label == self.LABEL_DIAGRAM]

        # 1. Title blocks
        for box in title_blocks:
            md += self._reconstruct_title_block(box)

        # 2. Text regions
        if text_boxes:
            md += ["", "## Content (テキスト)", ""]
            for box in text_boxes:
                md += self._reconstruct_text_box(box)

        # 3. Tables
        for i, box in enumerate(table_boxes, 1):
            md += ["", f"## Table {i}", ""]
            md += self._reconstruct_table_box(box)

        # 4. Unassigned text (outside all detected boxes)
        unassigned = self._unassigned_items()
        if unassigned:
            md += ["", "## Other Text (未分類)", ""]
            rec = TextRegionReconstructor(unassigned)
            md += rec.reconstruct()

        # 5. Diagram labels (opt-in)
        if include_diagrams:
            for i, box in enumerate(diagram_boxes, 1):
                md += ["", f"## Diagram {i}", ""]
                md += self._reconstruct_diagram_labels(box)

        return "\n".join(md) + "\n"

    def to_markdown_file(self, out_path: Path | str,
                         include_diagrams: bool = False) -> Path:
        out = Path(out_path)
        out.write_text(self.to_markdown(include_diagrams), encoding="utf-8")
        return out

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        label_counts: dict[str, int] = {}
        for box in self.boxes:
            label_counts[box.label] = label_counts.get(box.label, 0) + 1

        assigned_count = len(self._assigned)
        total = len(self.text_coords)
        return {
            "total_texts": total,
            "assigned_texts": assigned_count,
            "unassigned_texts": total - assigned_count,
            "boxes_by_label": label_counts,
        }


# ── Coordinate converter ───────────────────────────────────────────────────────

def pixel_bbox_to_dxf(
    pixel_boxes: list[dict],
    dxf_text_coords: list[TextItem],
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    Convert pixel-space bboxes (from detectron2 on a rendered PNG)
    into DXF coordinate space using text_coords as calibration points.

    Parameters
    ----------
    pixel_boxes : detectron2 output boxes in pixel space
        [{"label": ..., "bbox": [px1,py1,px2,py2], "score": ...}]
    dxf_text_coords : DXFParser.text_coords
        (text, dxf_x, dxf_y) for all text items
    image_width, image_height : rendered PNG dimensions

    Returns
    -------
    Same structure with bbox in DXF coordinate space.
    """
    if not dxf_text_coords:
        return pixel_boxes

    dxf_xs = [t[1] for t in dxf_text_coords]
    dxf_ys = [t[2] for t in dxf_text_coords]

    dxf_x_min, dxf_x_max = min(dxf_xs), max(dxf_xs)
    dxf_y_min, dxf_y_max = min(dxf_ys), max(dxf_ys)

    # DXF Y increases upward; pixel Y increases downward → flip Y
    def px_to_dxf(px: float, py: float) -> tuple[float, float]:
        rx = px / image_width
        ry = py / image_height
        dxf_x = dxf_x_min + rx * (dxf_x_max - dxf_x_min)
        dxf_y = dxf_y_max - ry * (dxf_y_max - dxf_y_min)  # flip Y
        return dxf_x, dxf_y

    converted = []
    for b in pixel_boxes:
        px1, py1, px2, py2 = b["bbox"]
        x1, y1 = px_to_dxf(px1, py1)
        x2, y2 = px_to_dxf(px2, py2)
        # Ensure x1<x2, y1<y2 after Y flip
        converted.append({
            **b,
            "bbox": [
                min(x1, x2), min(y1, y2),
                max(x1, x2), max(y1, y2),
            ]
        })
    return converted


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dxf_parser import DXFParser

    parser = argparse.ArgumentParser(
        description="BBox-guided DXF reconstruction"
    )
    parser.add_argument("dxf",   help="Path to .dxf file")
    parser.add_argument("bboxes", help="Path to JSON file with bbox predictions")
    parser.add_argument("-o", "--output", help="Output .md path")
    parser.add_argument("--diagrams", action="store_true",
                        help="Include diagram label sections")
    parser.add_argument("--score", type=float, default=0.3,
                        help="Minimum confidence score (default: 0.3)")
    parser.add_argument("--img-w", type=int, default=0,
                        help="Image width (px) for pixel→DXF conversion")
    parser.add_argument("--img-h", type=int, default=0,
                        help="Image height (px) for pixel→DXF conversion")
    args = parser.parse_args()

    dxf_path = Path(args.dxf)
    dxf_parser = DXFParser(str(dxf_path))
    dxf_parser.parse()

    with open(args.bboxes) as f:
        raw_bboxes = json.load(f)

    # Convert pixel→DXF if image dimensions provided
    if args.img_w > 0 and args.img_h > 0:
        raw_bboxes = pixel_bbox_to_dxf(
            raw_bboxes, dxf_parser.text_coords,
            args.img_w, args.img_h
        )

    rec = BBoxAwareReconstructor(
        dxf_parser.text_coords,
        raw_bboxes,
        score_threshold=args.score,
    )

    print("\n📊 Stats:", rec.stats())

    out_path = Path(args.output) if args.output else dxf_path.with_suffix(".bbox.md")
    rec.to_markdown_file(out_path, include_diagrams=args.diagrams)
    print(f"\n✅ Saved: {out_path}")
