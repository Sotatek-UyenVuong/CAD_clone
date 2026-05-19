#!/usr/bin/env python3
"""gen_grid_md.py — Generate structured Markdown from DXF files with table structure.

Approach: purely geometric, cell-based extraction.
  1. Build grid from table lines (H/V segments), excluding page-frame lines.
  2. Detect column zones by large X gaps between V-lines.
  3. Within each zone split rows into sections by Y gaps.
  4. Render each section as a compact Markdown table (skip empty rows/cols).
  5. Preserve ○ circle marks; join multi-text cells cleanly.

Usage:
  python3 gen_grid_md.py <file.dxf>            # write <file>_grid.md
  python3 gen_grid_md.py <file.dxf> -o out.md  # explicit output path
  python3 gen_grid_md.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ezdxf

from grid_engine import (
    _extract_texts,
    _extract_lines,
    build_grid,
    assign_texts,
)


def generate(dxf_path: str | Path) -> str:
    """Generate Markdown string from the given DXF file.

    Purely geometric: detects table structure from H/V line segments and
    extracts text content cell-by-cell without any document-type heuristics.

    Returns:
        Markdown string (multiple tables separated by ``---``).
    """
    p = Path(dxf_path)
    if not p.exists():
        raise FileNotFoundError(p)
    return _gen_cell_based(p)


# ─────────────────────────────────────────────────────────────────────────────
# Cell-based generator
# ─────────────────────────────────────────────────────────────────────────────

def _gen_cell_based(dxf_path: Path) -> str:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    h_segs, v_segs = _extract_lines(msp)
    all_texts = _extract_texts(msp)

    if not all_texts:
        return "_No content found._\n"

    # ── Main content Y-band (exclude stray entities far from the main body) ──
    all_y = [t.y for t in all_texts]
    sorted_y = sorted(all_y)
    _n = len(sorted_y)
    q1, q3 = sorted_y[_n // 4], sorted_y[3 * _n // 4]
    iqr = max(q3 - q1, 1.0)
    band_lo = q1 - 1.5 * iqr
    band_hi = q3 + 1.5 * iqr

    # Document-scale dimensions (for relative thresholds)
    all_x = [t.x for t in all_texts]
    page_w = max(all_x) - min(all_x) if len(all_x) > 1 else 1.0
    page_h = max(all_y) - min(all_y) if len(all_y) > 1 else 1.0

    # Scale-relative thresholds
    COL_GAP = page_w * 0.06    # ~6% of page width → separates distinct column zones
    ROW_GAP = page_h * 0.025   # ~2.5% of page height → separates row sections

    # Circle marks ○ — cap radius filter to avoid huge elements
    circ_max_r = min(page_w * 0.008, 20.0)
    circles: list[tuple[float, float]] = [
        (e.dxf.center.x, e.dxf.center.y)
        for e in msp
        if e.dxftype() == "CIRCLE" and e.dxf.radius < circ_max_r
    ]

    # Exclude page-frame H-segs (spanning >75% of page width) and stray H-segs
    table_h = [
        s for s in h_segs
        if abs(s[0] - s[2]) < page_w * 0.75
        and band_lo <= min(s[1], s[3]) <= band_hi
    ]

    # cluster_tol=1.5 works well across typical CAD coordinate scales
    grid = build_grid(table_h, v_segs, cluster_tol=1.5)
    matrix, unassigned = assign_texts(grid, all_texts)

    n_rows, n_cols = grid.rows, grid.cols
    if n_rows == 0 or n_cols == 0:
        return "_No table structure detected._\n"

    x_lo_grid = min(grid.col_xs)
    x_hi_grid = max(grid.col_xs)
    max_row_y = max(grid.row_ys)

    # Snap unassigned texts to nearest border cell.
    # Texts outside the grid's X range (e.g. labels in a column that has no
    # V-line on its outer edge) are clipped to the nearest column.
    if unassigned:
        for t in unassigned:
            # Only snap texts inside the main content Y-band
            if not (band_lo <= t.y <= band_hi):
                continue
            # Find the grid row by Y
            cell_by_y = grid.cell_at(
                x_lo_grid + (x_hi_grid - x_lo_grid) / 2,  # centre X
                t.y,
            )
            if cell_by_y is None:
                continue
            ri = cell_by_y[0]
            # Determine column: clip to leftmost or rightmost
            if t.x < x_lo_grid:
                ci = 0
                # Prepend so left-side labels appear before in-cell values
                matrix[ri][ci].insert(0, t.text)
            else:
                ci = n_cols - 1
                matrix[ri][ci].append(t.text)

    # Collect texts that sit just above the grid's top boundary — these are
    # section/table titles.  Use a proximity threshold so we don't pick up
    # far-away sheet-title-block text (e.g. 100 units above the table).
    title_proximity = ROW_GAP * 4  # ~4 row-heights above the top H-line
    above_grid_texts = sorted(
        [
            t for t in all_texts
            if band_lo <= t.y <= band_hi
            and max_row_y < t.y <= max_row_y + title_proximity
            and grid.cell_at(t.x, t.y) is None
        ],
        key=lambda t: t.y,  # bottom-to-top → rendered top-to-bottom in MD
    )

    # Assign circles to cells
    circle_cells: dict[tuple[int, int], int] = {}
    for cx, cy in circles:
        cell = grid.cell_at(cx, cy)
        if cell:
            circle_cells[cell] = circle_cells.get(cell, 0) + 1

    def cell_val(ri: int, ci: int) -> str:
        texts = [t.strip() for t in matrix[ri][ci] if t.strip()]
        n_circ = circle_cells.get((ri, ci), 0)
        if n_circ:
            texts = ["○" * n_circ] + texts
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]
        total = sum(len(t) for t in texts)
        return (" / ".join(texts) if total < 50 else " ".join(texts))

    # DXF row 0 = bottom of page; reverse to get top-to-bottom display order
    def row_y_center(ri: int) -> float:
        return (grid.row_ys[ri] + grid.row_ys[ri + 1]) / 2

    display_rows = [
        ri for ri in range(n_rows - 1, -1, -1)
        if band_lo <= row_y_center(ri) <= band_hi
    ]

    # ── Detect column zones by large X gaps between V-lines ──────────────────
    sorted_xs = sorted(grid.col_xs)
    zone_splits: list[int] = [0]
    for i in range(1, len(sorted_xs)):
        if sorted_xs[i] - sorted_xs[i - 1] > COL_GAP:
            zone_splits.append(i)
    zone_splits.append(n_cols)

    zones: list[tuple[int, int]] = [
        (zone_splits[i], zone_splits[i + 1])
        for i in range(len(zone_splits) - 1)
        if zone_splits[i + 1] > zone_splits[i]
    ]

    # ── Split rows into sections by Y gaps ───────────────────────────────────
    def split_into_sections(rows: list[int]) -> list[list[int]]:
        if not rows:
            return []
        sections: list[list[int]] = [[rows[0]]]
        for prev_ri, cur_ri in zip(rows, rows[1:]):
            if abs(row_y_center(prev_ri) - row_y_center(cur_ri)) > ROW_GAP:
                sections.append([])
            sections[-1].append(cur_ri)
        return [s for s in sections if s]

    # ── Render helpers ────────────────────────────────────────────────────────
    def render_section(rows: list[int], active_cols: list[int]) -> str:
        def row_md(ri: int) -> str:
            return "| " + " | ".join(cell_val(ri, ci) for ci in active_cols) + " |"
        hdr = row_md(rows[0])
        sep = "|" + "---|" * len(active_cols)
        body = [row_md(ri) for ri in rows[1:]]
        return "\n".join([hdr, sep] + body)

    def render_zone(col_start: int, col_end: int) -> str | None:
        col_range = range(col_start, col_end)
        zone_rows = [ri for ri in display_rows
                     if any(cell_val(ri, ci) for ci in col_range)]

        # Page-level titles sitting above the grid for this zone's X range.
        # For zone 0 we also catch titles whose X falls left of the grid.
        zone_x_lo = sorted_xs[col_start]
        zone_x_hi = sorted_xs[col_end] if col_end < len(sorted_xs) else float("inf")
        zone_titles = [
            t.text.strip()
            for t in above_grid_texts
            if (zone_x_lo <= t.x < zone_x_hi)
            or (col_start == 0 and t.x < zone_x_lo)
        ]

        if not zone_rows and not zone_titles:
            return None

        active_cols = [ci for ci in col_range
                       if any(cell_val(ri, ci) for ri in zone_rows)]

        rendered: list[str] = []

        if not active_cols:
            # No grid content — emit titles as a lone single-column table
            if zone_titles:
                title_text = " / ".join(zone_titles)
                rendered.append(f"| {title_text} |\n|---|")
            return "\n\n".join(rendered) if rendered else None

        sections = split_into_sections(zone_rows)
        title_text = " / ".join(zone_titles) if zone_titles else None

        for i, sec in enumerate(sections):
            sec_cols = [ci for ci in active_cols
                        if any(cell_val(ri, ci) for ri in sec)]
            if not sec_cols:
                continue
            if i == 0 and title_text:
                # Merge the page title as the very first (header) row of the
                # first section so it belongs to the same Markdown table.
                def row_md(ri: int) -> str:
                    return "| " + " | ".join(cell_val(ri, ci) for ci in sec_cols) + " |"
                hdr = "| " + title_text + " |" + " |" * (len(sec_cols) - 1)
                sep = "|" + "---|" * len(sec_cols)
                body = [row_md(ri) for ri in sec]
                rendered.append("\n".join([hdr, sep] + body))
            else:
                rendered.append(render_section(sec, sec_cols))
        return "\n\n".join(rendered) if rendered else None

    parts: list[str] = []
    for z_start, z_end in zones:
        zone_md = render_zone(z_start, z_end)
        if zone_md:
            parts.append(zone_md)

    if not parts:
        return "_No table content found._\n"
    return "\n\n---\n\n".join(parts) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown from DXF table structure (cell-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dxf", help="Input DXF file")
    parser.add_argument("-o", "--output", help="Output .md file (default: <stem>_grid.md)")
    args = parser.parse_args()

    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        print(f"Error: file not found: {dxf_path}", file=sys.stderr)
        sys.exit(1)

    out_path = (Path(args.output) if args.output
                else dxf_path.with_name(dxf_path.stem + "_grid.md"))

    print(f"Reading : {dxf_path}")
    md = generate(dxf_path)
    out_path.write_text(md, encoding="utf-8")
    print(f"Written : {out_path}  ({len(md):,} chars)")


if __name__ == "__main__":
    main()
