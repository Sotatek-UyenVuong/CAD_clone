"""rich_grid.py — Enrich a DXF grid with per-cell metadata.

Pipeline
--------
  grid_engine  →  RichGrid.from_dxf()  →  CellGrid
                                          ├─ Cell(texts, marks, borders, span)
                                          ├─ detect_structure()  → 'table' | 'checklist' | 'spec' | 'diagram'
                                          └─ to_markdown()       → str

Each Cell knows
  - texts    : list[TextItem] with height/layer (not just strings)
  - marks    : bool  — ○/× or any small circle inside the cell
  - borders  : (top, right, bottom, left)  — which sides have lines
  - col_span : int   — how many grid columns this cell occupies
  - row_span : int   — how many grid rows this cell occupies
  - kind     : 'header' | 'data' | 'mark' | 'empty'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import ezdxf

from grid_engine import (
    Grid,
    TextItem,
    _cluster,
    _extract_lines,
    _extract_texts,
    build_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cell
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Cell:
    row:      int
    col:      int
    texts:    list[TextItem] = field(default_factory=list)
    marks:    list[tuple[float, float]] = field(default_factory=list)  # (cx, cy)
    borders:  tuple[bool, bool, bool, bool] = (True, True, True, True)  # T R B L
    col_span: int = 1
    row_span: int = 1

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """All text joined, sorted top→bottom left→right."""
        return " / ".join(
            t.text for t in sorted(self.texts, key=lambda t: (-t.y, t.x))
        )

    @property
    def is_empty(self) -> bool:
        return not self.texts and not self.marks

    @property
    def has_mark(self) -> bool:
        return bool(self.marks)

    @property
    def kind(self) -> str:
        """Classify cell role based on content."""
        if self.is_empty:
            return "empty"
        if not self.texts and self.marks:
            return "mark"
        # Header heuristic: large font (h > 3.0) or ALL-CAPS short text
        heights = [t.height for t in self.texts if t.height > 0]
        if heights and max(heights) > 3.0 and len(self.text) <= 20:
            return "header"
        return "data"

    def text_by_height(self, min_h: float = 0.0, max_h: float = 999.0) -> str:
        return " / ".join(
            t.text for t in sorted(self.texts, key=lambda t: (-t.y, t.x))
            if min_h <= t.height <= max_h
        )


# ─────────────────────────────────────────────────────────────────────────────
# CellGrid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellGrid:
    """2-D grid of enriched cells + structure-detection + Markdown rendering."""

    cells:    list[list[Cell]]   # [row][col], row 0 = TOP (display order)
    grid:     Grid               # underlying geometry grid
    unassigned: list[TextItem]   # texts that fell outside all cells

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def cell(self, ri: int, ci: int) -> Cell:
        return self.cells[ri][ci]

    def iter_rows(self) -> Iterator[list[Cell]]:
        yield from self.cells

    # ── Structure detection ───────────────────────────────────────────────────

    def detect_structure(self) -> str:
        """Return 'checklist' | 'table' | 'spec' | 'diagram' | 'empty'.

        Heuristics (order matters):
          checklist  : mark cells represent ≥5% of non-empty cells (AND > 5 marks)
          table      : ≥3 rows with ≥2 non-empty cells
          spec       : multi-column, sparse header rows alternate with dense data rows
          diagram    : very sparse fill (<6%)
        """
        total = self.n_rows * self.n_cols
        if total == 0:
            return "empty"

        n_marks  = sum(1 for row in self.cells for c in row if c.kind == "mark")
        n_filled = sum(1 for row in self.cells for c in row if not c.is_empty)
        fill_ratio = n_filled / total if total else 0

        # Checklist: many mark cells relative to filled content
        if n_marks > 5 and n_filled > 0 and n_marks / n_filled > 0.05:
            return "checklist"

        if fill_ratio < 0.06:
            return "diagram"

        # Count rows with ≥2 non-empty cells
        dense_rows = sum(
            1 for row in self.cells
            if sum(1 for c in row if not c.is_empty) >= 2
        )
        if dense_rows / self.n_rows > 0.3:
            return "table"

        return "spec"

    # ── Markdown rendering ────────────────────────────────────────────────────

    def to_markdown(self, structure: str | None = None) -> str:
        """Convert to Markdown using the detected (or specified) structure."""
        s = structure or self.detect_structure()
        if s == "checklist":
            return self._md_checklist()
        if s == "table":
            return self._md_table()
        if s == "spec":
            return self._md_spec()
        return self._md_list()  # diagram / fallback

    def _active_cols(self) -> list[int]:
        """Return column indices that have at least one non-empty cell."""
        return [
            ci for ci in range(self.n_cols)
            if any(not row[ci].is_empty for row in self.cells)
        ]

    def _row_to_text(self, row: list[Cell], cols: list[int],
                     mark_as: str = "○", empty_as: str = "") -> list[str]:
        """Extract text from selected columns, replacing marks."""
        result = []
        for ci in cols:
            c = row[ci]
            if c.kind == "mark":
                result.append(mark_as)
            elif c.is_empty:
                result.append(empty_as)
            else:
                result.append(c.text.replace("|", "｜"))
        return result

    def _md_table(self) -> str:
        """Output as Markdown table, skipping all-empty rows and columns."""
        cols = self._active_cols()
        if not cols:
            return ""
        lines: list[str] = []
        header_done = False
        for row in self.cells:
            cells_text = self._row_to_text(row, cols)
            if all(t == "" for t in cells_text):
                continue
            lines.append("| " + " | ".join(cells_text) + " |")
            if not header_done:
                lines.append("|" + "---|" * len(cols))
                header_done = True
        return "\n".join(lines) + "\n"

    def _md_checklist(self) -> str:
        """Detect section-header rows → `### heading`, then checklist table.

        A "section-header row" has exactly one non-empty cell which is a header
        AND spans multiple consecutive (otherwise empty) rows below it.
        We split the grid at each header row and render each block as a table.
        """
        cols = self._active_cols()
        if not cols:
            return ""

        # Find rows that are "section dividers": single header cell, rest empty
        is_divider = [
            sum(1 for ci in cols if not self.cells[ri][ci].is_empty) == 1
            and any(self.cells[ri][ci].kind == "header" for ci in cols)
            for ri in range(self.n_rows)
        ]

        out: list[str] = []
        block_start = 0

        def flush_block(start: int, end: int, heading: str) -> None:
            if heading:
                out.append(f"\n### {heading}\n")
            block_rows = self.cells[start:end]
            block_cols = [ci for ci in cols
                          if any(not row[ci].is_empty for row in block_rows)]
            if not block_cols:
                return
            hdr_done = False
            for row in block_rows:
                cells_text = self._row_to_text(row, block_cols, mark_as="○", empty_as="·")
                if all(t == "·" for t in cells_text):
                    continue
                out.append("| " + " | ".join(cells_text) + " |")
                if not hdr_done:
                    out.append("|" + "---|" * len(block_cols))
                    hdr_done = True

        heading = ""
        for ri, div in enumerate(is_divider):
            if div:
                if ri > block_start:
                    flush_block(block_start, ri, heading)
                heading = next(
                    self.cells[ri][ci].text for ci in cols
                    if self.cells[ri][ci].kind == "header"
                )
                block_start = ri + 1
        flush_block(block_start, self.n_rows, heading)
        return "\n".join(out) + "\n"

    def _md_spec(self) -> str:
        """Output as structured spec document: section headers + body text."""
        _HDR_RE = re.compile(r"^\d+[.．]")
        cols = self._active_cols()
        lines: list[str] = []
        for row in self.cells:
            non_empty = [row[ci] for ci in cols if not row[ci].is_empty]
            if not non_empty:
                continue
            first = non_empty[0]
            txt = first.text
            if first.kind == "header" or _HDR_RE.match(txt):
                lines.append(f"\n### {txt}")
                rest = " ".join(c.text for c in non_empty[1:] if c.text)
                if rest:
                    lines.append(rest)
            else:
                if len(non_empty) == 1:
                    lines.append(f"- {txt}")
                else:
                    label = non_empty[0].text
                    value = " ".join(c.text for c in non_empty[1:] if c.text)
                    lines.append(
                        f"**{label}**: {value}" if len(label) <= 15 else f"- {label} {value}"
                    )
        return "\n".join(lines) + "\n"

    def _md_list(self) -> str:
        """Output as flat list — diagram or sparse grid."""
        cols = self._active_cols()
        lines: list[str] = []
        for row in self.cells:
            texts = [row[ci].text for ci in cols if row[ci].text]
            if texts:
                lines.append("- " + " / ".join(texts))
        return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def _extract_circles(msp) -> list[tuple[float, float, float]]:
    """Return (cx, cy, radius) for small CIRCLE entities (marks/symbols)."""
    circles = []
    for e in msp:
        if e.dxftype() == "CIRCLE" and e.dxf.radius < 8:
            circles.append((e.dxf.center.x, e.dxf.center.y, e.dxf.radius))
    return circles


def _border_map(
    h_segs: list[tuple], v_segs: list[tuple],
    row_ys: list[float], col_xs: list[float],
    tol: float = 2.0,
) -> dict[tuple[int, int], tuple[bool, bool, bool, bool]]:
    """For each (ri, ci) return (top, right, bottom, left) booleans.

    A border exists when a line segment lies within *tol* of the cell boundary
    AND spans at least 50% of the cell's width/height.
    """
    borders: dict[tuple[int, int], tuple[bool, bool, bool, bool]] = {}
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1

    for ri in range(n_rows):
        y_lo = row_ys[ri]
        y_hi = row_ys[ri + 1]
        cell_h = y_hi - y_lo
        for ci in range(n_cols):
            x_lo = col_xs[ci]
            x_hi = col_xs[ci + 1]
            cell_w = x_hi - x_lo

            # Bottom border: H-seg near y_lo spanning ≥50% of cell width
            has_bot = any(
                abs(seg[1] - y_lo) < tol
                and seg[0] < x_hi - cell_w * 0.3
                and seg[2] > x_lo + cell_w * 0.3
                for seg in h_segs
            )
            # Top border: H-seg near y_hi
            has_top = any(
                abs(seg[1] - y_hi) < tol
                and seg[0] < x_hi - cell_w * 0.3
                and seg[2] > x_lo + cell_w * 0.3
                for seg in h_segs
            )
            # Left border: V-seg near x_lo spanning ≥50% of cell height
            has_left = any(
                abs(seg[0] - x_lo) < tol
                and seg[1] < y_hi - cell_h * 0.3
                and seg[3] > y_lo + cell_h * 0.3
                for seg in v_segs
            )
            # Right border: V-seg near x_hi
            has_right = any(
                abs(seg[0] - x_hi) < tol
                and seg[1] < y_hi - cell_h * 0.3
                and seg[3] > y_lo + cell_h * 0.3
                for seg in v_segs
            )
            borders[(ri, ci)] = (has_top, has_right, has_bot, has_left)

    return borders


def build_rich_grid(
    dxf_path: str | Path,
    cluster_tol: float = 1.5,
    border_tol: float = 2.0,
) -> CellGrid:
    """Full pipeline: DXF → CellGrid.

    Args:
        dxf_path:    Path to DXF file.
        cluster_tol: Tolerance for merging nearby grid lines (default 1.5 units).
        border_tol:  Tolerance for matching a line segment to a cell edge.

    Returns:
        CellGrid with per-cell texts, marks, and borders.
    """
    doc  = ezdxf.readfile(str(dxf_path))
    msp  = doc.modelspace()

    h_segs, v_segs = _extract_lines(msp)
    texts           = _extract_texts(msp)
    circles         = _extract_circles(msp)
    grid            = build_grid(h_segs, v_segs, cluster_tol)

    n_rows = grid.rows
    n_cols = grid.cols

    if n_rows == 0 or n_cols == 0:
        return CellGrid(cells=[], grid=grid, unassigned=texts)

    # ── Assign texts to cells (preserve TextItem, not just string) ────────────
    cell_texts: list[list[list[TextItem]]] = [
        [[] for _ in range(n_cols)] for _ in range(n_rows)
    ]
    unassigned: list[TextItem] = []
    for item in texts:
        cell = grid.cell_at(item.x, item.y)
        if cell is not None:
            ri, ci = cell
            cell_texts[ri][ci].append(item)
        else:
            unassigned.append(item)

    # ── Assign circles to cells ────────────────────────────────────────────────
    cell_marks: list[list[list[tuple[float, float]]]] = [
        [[] for _ in range(n_cols)] for _ in range(n_rows)
    ]
    for cx, cy, _ in circles:
        cell = grid.cell_at(cx, cy)
        if cell is not None:
            ri, ci = cell
            cell_marks[ri][ci].append((cx, cy))

    # ── Compute borders ────────────────────────────────────────────────────────
    borders = _border_map(h_segs, v_segs, grid.row_ys, grid.col_xs, border_tol)

    # ── Assemble Cell objects (flip row order: row 0 = top of drawing) ─────────
    display_cells: list[list[Cell]] = []
    for display_ri in range(n_rows):
        mat_ri = n_rows - 1 - display_ri   # DXF row 0 = bottom → flip
        row: list[Cell] = []
        for ci in range(n_cols):
            row.append(Cell(
                row=display_ri,
                col=ci,
                texts=sorted(cell_texts[mat_ri][ci], key=lambda t: (-t.y, t.x)),
                marks=cell_marks[mat_ri][ci],
                borders=borders.get((mat_ri, ci), (True, True, True, True)),
            ))
        display_cells.append(row)

    return CellGrid(cells=display_cells, grid=grid, unassigned=unassigned)


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python rich_grid.py <file.dxf> [cluster_tol]")
        sys.exit(1)

    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    cg = build_rich_grid(path, cluster_tol=tol)

    structure = cg.detect_structure()
    print(f"Grid     : {cg.n_rows}r × {cg.n_cols}c")
    print(f"Structure: {structure}")
    print(f"Fill     : {sum(1 for r in cg.cells for c in r if not c.is_empty)}"
          f" / {cg.n_rows * cg.n_cols} cells")
    print(f"Marks    : {sum(1 for r in cg.cells for c in r if c.has_mark)}")
    print(f"Unassigned texts: {len(cg.unassigned)}")
    print()

    # Show non-empty rows (first 20)
    shown = 0
    for ri, row in enumerate(cg.cells):
        non_empty = [(ci, c) for ci, c in enumerate(row) if not c.is_empty]
        if not non_empty:
            continue
        parts = [f"[{ci}]({c.kind}) {c.text[:35]!r}" for ci, c in non_empty[:5]]
        print(f"  row {ri:3d}: " + "  |  ".join(parts))
        shown += 1
        if shown >= 20:
            print(f"  ... +{cg.n_rows - ri - 1} more rows")
            break

    print()
    print("── Markdown preview (first 40 lines) ──")
    md = cg.to_markdown(structure)
    for line in md.splitlines()[:40]:
        print(line)
