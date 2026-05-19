"""grid_engine.py — Rule-based CAD grid detection + R-tree spatial indexing.

Pipeline:
  1. Extract H/V lines (+ LWPOLYLINE segments)
  2. Cluster similar coordinates → canonical row/col boundaries
  3. Build R-tree index over cells
  4. Assign each TEXT to its cell in O(n log n)
  5. Output structured grid → list[list[str]]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import ezdxf
from rtree import index as rtree_index


# ──────────────────────────────────────────────────────────────────────────────
# Snap / cluster helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cluster(values: list[float], tol: float) -> list[float]:
    """Merge values within *tol* of each other → sorted unique cluster centres."""
    if not values:
        return []
    vals = sorted(values)
    clusters: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _snap(value: float, grid: list[float]) -> float:
    """Return the nearest value in *grid*."""
    return min(grid, key=lambda g: abs(g - value))


# ──────────────────────────────────────────────────────────────────────────────
# Geometry extraction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextItem:
    text:   str
    x:      float
    y:      float
    height: float
    layer:  str


def _extract_lines(msp) -> tuple[list[tuple[float,float,float,float]],
                                  list[tuple[float,float,float,float]]]:
    """Return (h_lines, v_lines) where each entry is (x_lo, y, x_hi, y) / (x, y_lo, x, y_hi)."""
    TOL = 1.0          # max deviation to call a line "horizontal" or "vertical"

    h_segs: list[tuple[float,float,float,float]] = []
    v_segs: list[tuple[float,float,float,float]] = []

    def _classify(x1: float, y1: float, x2: float, y2: float) -> None:
        if abs(y1 - y2) <= TOL:                          # horizontal
            h_segs.append((min(x1,x2), (y1+y2)/2, max(x1,x2), (y1+y2)/2))
        elif abs(x1 - x2) <= TOL:                        # vertical
            v_segs.append(((x1+x2)/2, min(y1,y2), (x1+x2)/2, max(y1,y2)))

    for e in msp:
        t = e.dxftype()
        if t == "LINE":
            _classify(e.dxf.start.x, e.dxf.start.y, e.dxf.end.x, e.dxf.end.y)
        elif t == "LWPOLYLINE":
            pts = list(e.get_points("xy"))
            if e.is_closed:
                pts.append(pts[0])
            for i in range(len(pts) - 1):
                _classify(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

    return h_segs, v_segs


def _collect_texts_from_entities(
    entities,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    items: list | None = None,
    circles: list | None = None,
) -> tuple[list, list]:
    """Collect TEXT/MTEXT/CIRCLE from an entity iterator with an optional offset.

    Used internally by _extract_texts to handle both msp entities and
    exploded INSERT (block reference) entities.
    """
    if items is None:
        items = []
    if circles is None:
        circles = []

    def _normalize_text(raw: str) -> str:
        # Drop white circle U+25CB completely (do not normalize to another symbol).
        return raw.replace("○", "").strip()

    for e in entities:
        t = e.dxftype()
        if t == "TEXT":
            txt = _normalize_text(e.dxf.text)
            if txt:
                items.append(TextItem(txt,
                    float(e.dxf.insert.x) + offset_x,
                    float(e.dxf.insert.y) + offset_y,
                    float(e.dxf.height), e.dxf.layer))
        elif t == "MTEXT":
            txt = _normalize_text(e.plain_text())
            if txt:
                items.append(TextItem(txt,
                    float(e.dxf.insert.x) + offset_x,
                    float(e.dxf.insert.y) + offset_y,
                    float(e.dxf.char_height), e.dxf.layer))
        elif t == "CIRCLE":
            circles.append((
                float(e.dxf.center.x) + offset_x,
                float(e.dxf.center.y) + offset_y,
                float(e.dxf.radius),
                e.dxf.layer,
            ))
    return items, circles


def _extract_texts(msp,
                   circle_symbol: str = "〇",
                   circle_max_r: float | None = None,
                   explode_inserts: bool = True) -> list[TextItem]:
    """Extract TEXT, MTEXT, and optionally CIRCLE entities as text items.

    When *explode_inserts* is True (default), also extracts text from INSERT
    (block reference) entities by resolving them against their block definition.
    This is required to capture title-block content stored in named blocks
    (e.g. 図面枠, 建築者氏名).

    CIRCLE entities whose radius ≤ circle_max_r are treated as checkbox marks
    and represented by *circle_symbol* (default "○") at their centre position.
    Pass circle_max_r=None (default) to auto-detect from the median circle radius.
    """
    items: list[TextItem] = []
    circles: list[tuple[float, float, float, str]] = []

    # Collect from model space directly
    _collect_texts_from_entities(msp, 0.0, 0.0, items, circles)

    # Explode INSERT block references
    if explode_inserts:
        try:
            doc = msp.doc  # ezdxf >= 0.17: msp.doc is the Drawing object
        except AttributeError:
            doc = None
        if doc is not None:
            for e in msp:
                if e.dxftype() != "INSERT":
                    continue
                block_name = e.dxf.name
                ix = float(e.dxf.insert.x)
                iy = float(e.dxf.insert.y)
                try:
                    block_def = doc.blocks[block_name]
                    _collect_texts_from_entities(block_def, ix, iy, items, circles)
                except (KeyError, Exception):
                    pass

    if circles:
        radii = sorted(c[2] for c in circles)
        if circle_max_r is None:
            idx = max(0, int(len(radii) * 0.75) - 1)
            circle_max_r = radii[idx] * 1.5

        for cx, cy, r, layer in circles:
            if r <= circle_max_r:
                items.append(TextItem(circle_symbol, cx, cy, r * 2, layer))

    return items


# ──────────────────────────────────────────────────────────────────────────────
# Grid detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Grid:
    """Detected grid with row/col boundaries and R-tree index over cells."""
    row_ys:   list[float]   # sorted Y positions of horizontal lines (bottom→top)
    col_xs:   list[float]   # sorted X positions of vertical lines (left→right)
    # Derived
    rows:     int = field(init=False)
    cols:     int = field(init=False)
    _idx:     object = field(init=False, repr=False)
    _cells:   list[tuple[int,int]] = field(init=False, repr=False)  # (ri, ci) per R-tree id

    def __post_init__(self):
        self.rows = max(0, len(self.row_ys) - 1)
        self.cols = max(0, len(self.col_xs) - 1)
        self._build_rtree()

    def _build_rtree(self):
        """Index each cell rectangle in the R-tree."""
        self._idx   = rtree_index.Index()
        self._cells = []
        cell_id = 0
        for ri in range(self.rows):
            y_lo = self.row_ys[ri]
            y_hi = self.row_ys[ri + 1]
            for ci in range(self.cols):
                x_lo = self.col_xs[ci]
                x_hi = self.col_xs[ci + 1]
                # rtree bbox: (left, bottom, right, top)
                self._idx.insert(cell_id, (x_lo, y_lo, x_hi, y_hi))
                self._cells.append((ri, ci))
                cell_id += 1

    def cell_at(self, x: float, y: float) -> tuple[int,int] | None:
        """Return (row_idx, col_idx) for point (x,y), or None if outside grid."""
        hits = list(self._idx.intersection((x, y, x, y)))
        if not hits:
            return None
        # Pick the tightest (smallest area) containing cell
        best = min(hits, key=lambda h: self._cell_area(h))
        return self._cells[best]

    def _cell_area(self, cell_id: int) -> float:
        ri, ci = self._cells[cell_id]
        return ((self.col_xs[ci+1] - self.col_xs[ci]) *
                (self.row_ys[ri+1] - self.row_ys[ri]))


def _row_boundaries_from_texts(text_ys: list[float],
                               y_lo: float, y_hi: float,
                               cluster_tol: float) -> list[float]:
    """Given a list of text Y positions, produce sorted row-boundary Y values.

    Boundaries are placed at the midpoint between consecutive text-row centres,
    with the overall extent clamped to [y_lo, y_hi].
    """
    centres = _cluster(text_ys, cluster_tol)
    if not centres:
        return [y_lo, y_hi]

    boundaries: list[float] = [y_lo]
    for i in range(len(centres) - 1):
        boundaries.append((centres[i] + centres[i + 1]) / 2.0)
    boundaries.append(y_hi)

    # Ensure monotonically increasing (cluster may already guarantee this)
    return sorted(set(boundaries))


def build_grid(h_segs: list, v_segs: list,
               cluster_tol: float = 1.5) -> Grid:
    """Cluster line positions → canonical boundaries → Grid.

    When H-segs (or V-segs) are absent or don't span the full extent of the
    opposite-axis segments, synthetic boundary lines are inferred from segment
    endpoints so that a valid grid can still be formed.
    """
    raw_ys = [float(y) for _, y, _, _ in h_segs]
    raw_xs = [float(x) for x, _, _, _ in v_segs]

    # Infer missing H-boundaries from V-segment endpoints
    if v_segs:
        v_y_lo = min(min(s[1], s[3]) for s in v_segs)
        v_y_hi = max(max(s[1], s[3]) for s in v_segs)
        if not raw_ys or min(raw_ys) - v_y_lo > cluster_tol:
            raw_ys.append(v_y_lo)
        if not raw_ys or v_y_hi - max(raw_ys) > cluster_tol:
            raw_ys.append(v_y_hi)

    # Infer missing V-boundaries from H-segment endpoints
    if h_segs:
        h_x_lo = min(min(s[0], s[2]) for s in h_segs)
        h_x_hi = max(max(s[0], s[2]) for s in h_segs)
        if not raw_xs or min(raw_xs) - h_x_lo > cluster_tol:
            raw_xs.append(h_x_lo)
        if not raw_xs or h_x_hi - max(raw_xs) > cluster_tol:
            raw_xs.append(h_x_hi)

    row_ys = sorted(_cluster(raw_ys, cluster_tol))
    col_xs = sorted(_cluster(raw_xs, cluster_tol))

    return Grid(row_ys=row_ys, col_xs=col_xs)


def _col_boundaries_from_texts(text_xs: list[float],
                               x_lo: float, x_hi: float,
                               cluster_tol: float) -> list[float]:
    """Given text X positions, produce sorted col-boundary X values."""
    centres = _cluster(text_xs, cluster_tol)
    if not centres:
        return [x_lo, x_hi]
    boundaries: list[float] = [x_lo]
    for i in range(len(centres) - 1):
        boundaries.append((centres[i] + centres[i + 1]) / 2.0)
    boundaries.append(x_hi)
    return sorted(set(boundaries))


def build_grid_text_aware(
    h_segs: list,
    v_segs: list,
    texts: list,                   # list[TextItem]
    cluster_tol: float = 1.5,
    text_cluster_tol: float = 3.0,
    sparse_ratio: float = 3.0,
) -> Grid:
    """Build a grid, using text-position clustering as fallback for missing lines.

    Handles four cases:
      1. Both H and V lines present   → standard grid + sparse-row fix
      2. Only H lines                 → cols inferred from text X-clusters
      3. Only V lines                 → rows inferred from text Y-clusters
      4. No lines at all              → both rows and cols from text clusters

    Args:
        h_segs, v_segs:   raw line segments
        texts:            TextItem list for the region
        cluster_tol:      tolerance for line clustering (mm)
        text_cluster_tol: tolerance for grouping text positions into rows/cols (mm)
        sparse_ratio:     if (# text Y-clusters) / (# H-line rows) ≥ this,
                          the H-lines are considered sparse → use text rows
    """
    MIN_COL = 8.0   # mm — minimum meaningful column width

    grid = build_grid(h_segs, v_segs, cluster_tol)

    if not texts:
        return grid

    all_xs = [t.x for t in texts]
    all_ys = [t.y for t in texts]

    # ── Case 4: No lines at all → build everything from texts ──
    if not h_segs and not v_segs:
        x_lo = min(all_xs) - text_cluster_tol
        x_hi = max(all_xs) + text_cluster_tol
        y_lo = min(all_ys) - text_cluster_tol
        y_hi = max(all_ys) + text_cluster_tol
        row_ys = _row_boundaries_from_texts(all_ys, y_lo, y_hi, text_cluster_tol)
        col_xs = _col_boundaries_from_texts(all_xs, x_lo, x_hi, text_cluster_tol)
        return Grid(row_ys=row_ys, col_xs=col_xs)

    # ── Case 2: Only H lines → infer cols from text X-clusters ──
    if h_segs and not v_segs:
        # Use grid row_ys only if they form at least 1 row (≥2 boundaries)
        if len(grid.row_ys) >= 2:
            row_ys = list(grid.row_ys)
        else:
            y_lo = min(min(s[1], s[3]) for s in h_segs)
            y_hi = max(max(s[1], s[3]) for s in h_segs)
            row_ys = _row_boundaries_from_texts(all_ys, y_lo, y_hi, text_cluster_tol)
        x_lo = min(min(s[0], s[2]) for s in h_segs)
        x_hi = max(max(s[0], s[2]) for s in h_segs)
        col_xs = _col_boundaries_from_texts(all_xs, x_lo, x_hi, text_cluster_tol)
        return Grid(row_ys=row_ys, col_xs=col_xs)

    # ── Case 3: Only V lines → rows from text Y-clusters, cols from text X-clusters ──
    # V-line positions alone cover only part of the x-range; use text-based x-clusters
    # to capture the full column structure (V lines may only mark a sub-region).
    if v_segs and not h_segs:
        v_x_lo = min(min(s[0], s[2]) for s in v_segs)
        v_x_hi = max(max(s[0], s[2]) for s in v_segs)
        x_lo = min(v_x_lo, min(all_xs)) if all_xs else v_x_lo
        x_hi = max(v_x_hi, max(all_xs)) if all_xs else v_x_hi
        col_xs = _col_boundaries_from_texts(
            all_xs, x_lo - text_cluster_tol, x_hi + text_cluster_tol, text_cluster_tol)
        y_lo = min(min(s[1], s[3]) for s in v_segs)
        y_hi = max(max(s[1], s[3]) for s in v_segs)
        row_ys = _row_boundaries_from_texts(all_ys, y_lo, y_hi, text_cluster_tol)
        return Grid(row_ys=row_ys, col_xs=col_xs)

    # ── Case 1: Both H and V lines → standard grid + enhancements ──
    if grid.cols == 0 or not grid.col_xs:
        return grid

    y_lo_bound = min(grid.row_ys) if grid.row_ys else None
    y_hi_bound = max(grid.row_ys) if grid.row_ys else None
    inner_ys = [
        t.y for t in texts
        if (y_lo_bound is None or y_lo_bound - cluster_tol <= t.y <= y_hi_bound + cluster_tol)
    ]
    if not inner_ys:
        return grid

    text_centres = _cluster(inner_ys, text_cluster_tol)
    n_text_rows  = len(text_centres)
    n_grid_rows  = max(grid.rows, 1)

    # Extend col boundaries to capture texts outside V-line span
    col_xs = list(grid.col_xs)
    if col_xs:
        text_x_lo = min(t.x for t in texts)
        text_x_hi = max(t.x for t in texts)
        if col_xs[0] - text_x_lo > MIN_COL:
            col_xs = [text_x_lo] + col_xs
        if text_x_hi - col_xs[-1] > MIN_COL:
            col_xs = col_xs + [text_x_hi]

    # Split wide columns with bimodal X distribution
    final_xs: list[float] = []
    if len(col_xs) >= 2:
        for i in range(len(col_xs) - 1):
            x_lo_col, x_hi_col = col_xs[i], col_xs[i + 1]
            col_w = x_hi_col - x_lo_col
            if col_w < MIN_COL * 2:
                final_xs.append(x_lo_col)
                continue
            col_texts = [t for t in texts if x_lo_col <= t.x < x_hi_col]
            if len(col_texts) < 3:
                final_xs.append(x_lo_col)
                continue
            col_text_xs = sorted(t.x for t in col_texts)
            gaps = [(col_text_xs[j+1] - col_text_xs[j], j)
                    for j in range(len(col_text_xs)-1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > max(MIN_COL, col_w * 0.2):
                split_x = (col_text_xs[gap_idx] + col_text_xs[gap_idx+1]) / 2
                final_xs.append(x_lo_col)
                final_xs.append(split_x)
                continue
            final_xs.append(x_lo_col)
        final_xs.append(col_xs[-1])
        col_xs = final_xs

    if col_xs != list(grid.col_xs):
        grid = Grid(row_ys=grid.row_ys, col_xs=col_xs)

    if n_text_rows < sparse_ratio * n_grid_rows:
        return grid

    # Sparse H-lines → rebuild row_ys from text Y-clusters
    y_lo = min(grid.row_ys) if grid.row_ys else min(inner_ys) - text_cluster_tol
    y_hi = max(grid.row_ys) if grid.row_ys else max(inner_ys) + text_cluster_tol
    new_row_ys = _row_boundaries_from_texts(inner_ys, y_lo, y_hi, text_cluster_tol)
    return Grid(row_ys=new_row_ys, col_xs=col_xs)


# ──────────────────────────────────────────────────────────────────────────────
# Text → cell assignment
# ──────────────────────────────────────────────────────────────────────────────

def assign_texts(grid: Grid,
                 texts: list[TextItem]) -> list[list[list[str]]]:
    """Return matrix[row][col] = list of text strings in that cell.

    Row 0 = BOTTOM row (as stored in DXF; flip if needed for display).
    """
    matrix: list[list[list[str]]] = [
        [[] for _ in range(grid.cols)]
        for _ in range(grid.rows)
    ]
    unassigned: list[TextItem] = []

    for item in texts:
        cell = grid.cell_at(item.x, item.y)
        if cell is not None:
            ri, ci = cell
            matrix[ri][ci].append(item.text)
        else:
            unassigned.append(item)

    return matrix, unassigned


def snap_circles_to_labels(
    circle_cells: dict[tuple[int,int], bool],
    matrix: list[list[list[str]]],
    label_cols: list[int],
    section_row_range: tuple[int, int],
) -> dict[tuple[int,int], tuple[int,int]]:
    """Map circle cells to the nearest labeled row within section range.

    circle_cells : {(mat_ri, ci): True, ...}   — circles in matrix coordinates
    label_cols   : column indices that hold row labels (e.g. [16, 24])
    section_row_range : (start_mat_ri, end_mat_ri) in matrix coordinates

    Returns {(mat_ri_circle, ci): (mat_ri_label, ci)} mapping each circle
    to its best-matching labeled row.  If already in a labeled row, maps
    to itself.
    """
    start_ri, end_ri = section_row_range
    # Pre-build: for each label_col, sorted list of mat_ri that have a label
    labeled_rows: dict[int, list[int]] = {}
    for lc in label_cols:
        rows = [r for r in range(start_ri, end_ri)
                if r < len(matrix) and lc < len(matrix[r]) and matrix[r][lc]]
        labeled_rows[lc] = sorted(rows)

    result: dict[tuple[int,int], tuple[int,int]] = {}
    for (mat_ri, ci) in circle_cells:
        if not (start_ri <= mat_ri < end_ri):
            result[(mat_ri, ci)] = (mat_ri, ci)
            continue
        # Find nearest label_col for this ci
        best_lc = min(label_cols, key=lambda lc: abs(lc - ci)) if label_cols else ci
        rows = labeled_rows.get(best_lc, [])
        if not rows:
            result[(mat_ri, ci)] = (mat_ri, ci)
            continue
        # Already has a label in this row?
        if mat_ri in rows:
            result[(mat_ri, ci)] = (mat_ri, ci)
        else:
            # Snap to nearest labeled row
            nearest = min(rows, key=lambda r: abs(r - mat_ri))
            result[(mat_ri, ci)] = (nearest, ci)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_grid_from_dxf(
    dxf_path: str | Path,
    cluster_tol: float = 1.5,
) -> tuple[Grid, list[list[list[str]]], list[TextItem]]:
    """Full pipeline: DXF → (grid, matrix, unassigned_texts).

    matrix[row][col] = list[str]   (row 0 = bottom of drawing)
    """
    doc  = ezdxf.readfile(str(dxf_path))
    msp  = doc.modelspace()
    h_segs, v_segs = _extract_lines(msp)
    texts           = _extract_texts(msp)
    grid            = build_grid(h_segs, v_segs, cluster_tol)
    matrix, unassigned = assign_texts(grid, texts)
    return grid, matrix, unassigned


# ──────────────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python grid_engine.py <file.dxf>")
        sys.exit(1)

    grid, matrix, unassigned = parse_grid_from_dxf(path)

    print(f"Grid: {grid.rows} rows × {grid.cols} cols")
    print(f"Unassigned texts: {len(unassigned)}")
    print()

    # Print top 20 rows (display order = top to bottom → reverse)
    display = list(reversed(matrix))
    for ri, row in enumerate(display[:20]):
        cells = [" | ".join(c) if c else "" for c in row]
        print(f"  [{ri:3d}] {' │ '.join(cells[:8])}")
    if grid.rows > 20:
        print(f"  ... +{grid.rows - 20} more rows")
