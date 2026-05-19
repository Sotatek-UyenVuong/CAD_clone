#!/usr/bin/env python3
"""
render_symbol_boxes.py
Render một trang DXF ra PNG và vẽ bounding box màu cho từng INSERT symbol,
kèm tên label từ symbols.json.

Usage:
    python3 tools/render_symbol_boxes.py <dxf_file> [--page 0] [--out output.png]
    python3 tools/render_symbol_boxes.py \
        "dxf_output/.../D109·110_A棟B1階平面詳細図-240229.dxf"
"""

import argparse
import json
import sys
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import numpy as np
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# ── colour palette (cycles through for different labels) ──────────────────────
PALETTE = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
    "#ff5722","#607d8b","#795548","#cddc39","#673ab7",
]

BASE_DIR  = Path(__file__).resolve().parent.parent
SYMDB     = BASE_DIR / "symbol_db" / "symbols.json"
IDX_FILE  = BASE_DIR / "symbol_db" / "block_name_index.json"


def load_dbs():
    db  = json.loads(SYMDB.read_text("utf-8")) if SYMDB.exists() else {}
    idx = json.loads(IDX_FILE.read_text("utf-8")) if IDX_FILE.exists() else {}
    return db, idx


def get_label(block_name: str, db: dict, idx: dict) -> str:
    h = idx.get(block_name)
    if h and h in db:
        lbl = db[h].get("label", "?") or "?"
        if lbl != "?":
            return lbl
    return block_name[:16] if block_name else "?"


def collect_inserts(layout) -> list[dict]:
    """Return list of {name, x, y, sx, sy, rot} for all INSERT entities."""
    result = []
    for e in layout:
        if e.dxftype() != "INSERT":
            continue
        try:
            ins = e.dxf
            x, y = ins.insert.x, ins.insert.y
            sx = getattr(ins, "xscale", 1.0) or 1.0
            sy = getattr(ins, "yscale", 1.0) or 1.0
            rot = getattr(ins, "rotation", 0.0)
            result.append(dict(name=ins.name, x=x, y=y, sx=sx, sy=sy, rot=rot))
        except Exception:
            continue
    return result


def block_local_bbox(doc, name: str) -> tuple[float, float, float, float] | None:
    """Tight bbox of a block in its own coordinate system."""
    try:
        blk = doc.blocks.get(name)
        if blk is None:
            return None
        xs, ys = [], []
        for e in blk:
            t = e.dxftype()
            try:
                if t in ("LINE",):
                    xs += [e.dxf.start.x, e.dxf.end.x]
                    ys += [e.dxf.start.y, e.dxf.end.y]
                elif t in ("CIRCLE", "ARC"):
                    r = e.dxf.radius
                    cx, cy = e.dxf.center.x, e.dxf.center.y
                    xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
                elif t in ("TEXT", "MTEXT"):
                    p = e.dxf.insert
                    xs.append(p.x); ys.append(p.y)
                elif t in ("INSERT",):
                    p = e.dxf.insert
                    xs.append(p.x); ys.append(p.y)
                elif hasattr(e.dxf, "insert"):
                    p = e.dxf.insert
                    xs.append(p.x); ys.append(p.y)
                elif hasattr(e.dxf, "start"):
                    xs.append(e.dxf.start.x); ys.append(e.dxf.start.y)
            except Exception:
                continue
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def world_bbox(ins: dict, doc) -> tuple[float, float, float, float] | None:
    """Transform block bbox to world coordinates."""
    bb = block_local_bbox(doc, ins["name"])
    if bb is None:
        # Fallback: 1-unit box around insert point
        x, y = ins["x"], ins["y"]
        return (x - 0.5, y - 0.5, x + 0.5, y + 0.5)
    lx0, ly0, lx1, ly1 = bb
    # Apply scale
    sx, sy = ins["sx"], ins["sy"]
    # Apply rotation
    import math
    rad = math.radians(ins["rot"])
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    corners_local = [(lx0, ly0), (lx1, ly0), (lx1, ly1), (lx0, ly1)]
    wx, wy = [], []
    for lx, ly in corners_local:
        rx = lx * sx * cos_r - ly * sy * sin_r + ins["x"]
        ry = lx * sx * sin_r + ly * sy * cos_r + ins["y"]
        wx.append(rx); wy.append(ry)
    return min(wx), min(wy), max(wx), max(wy)


def render(dxf_path: str, page_idx: int, out_path: str,
           min_count: int = 1, highlight_labels: list[str] | None = None):
    print(f"Loading {dxf_path} …")
    doc = ezdxf.readfile(dxf_path)
    db, idx = load_dbs()

    # Choose layout — prefer the one with the most INSERT entities
    msp = doc.modelspace()
    paper_layouts = [l for l in doc.layouts if l.name != "Model"]
    candidates = paper_layouts if paper_layouts else [msp]
    # Count INSERTs in each candidate
    def count_inserts(l):
        return sum(1 for e in l if e.dxftype() == "INSERT")
    msp_inserts = count_inserts(msp)
    best_paper  = max(candidates, key=count_inserts) if candidates else None
    best_paper_n = count_inserts(best_paper) if best_paper else 0
    # Use model space if it has significantly more INSERTs
    if msp_inserts > best_paper_n * 3:
        layout = msp
    else:
        if page_idx < len(candidates):
            layout = candidates[page_idx]
        else:
            layout = best_paper or msp
    print(f"  Layout: '{layout.name}'  entities: {len(list(layout))}")

    # ── render DXF to raster via ezdxf drawing addon ──────────────────────────
    fig = plt.figure(figsize=(24, 17), dpi=150)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    ax.axis("off")

    ctx     = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    frontend = Frontend(ctx, backend)
    frontend.draw_layout(layout, finalize=True)

    # Get world extents from current axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    world_w = xlim[1] - xlim[0]
    world_h = ylim[1] - ylim[0]

    # ── collect INSERTs and draw boxes ────────────────────────────────────────
    inserts = collect_inserts(layout)
    print(f"  Found {len(inserts)} INSERT entities")

    label_colors: dict[str, str] = {}
    color_idx = 0
    legend_handles = []

    for ins in inserts:
        label = get_label(ins["name"], db, idx)

        # Filter by highlight_labels if specified
        if highlight_labels and not any(
            hl.lower() in label.lower() for hl in highlight_labels
        ):
            continue

        # Assign colour per label
        if label not in label_colors:
            label_colors[label] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1

        color = label_colors[label]
        bb = world_bbox(ins, doc)
        if bb is None:
            continue
        bx0, by0, bx1, by1 = bb

        # Skip degenerate (zero area) boxes – likely text or single point
        if abs(bx1 - bx0) < 1e-3 and abs(by1 - by0) < 1e-3:
            # Draw small dot instead
            ax.plot(ins["x"], ins["y"], "o", color=color,
                    markersize=4, alpha=0.8, zorder=5)
        else:
            rect = mpatches.FancyBboxPatch(
                (bx0, by0), bx1 - bx0, by1 - by0,
                boxstyle="square,pad=0",
                linewidth=1.0, edgecolor=color,
                facecolor=(*to_rgba(color)[:3], 0.08),
                zorder=4,
            )
            ax.add_patch(rect)

        # Label text (only if box large enough to be visible)
        box_w_px = (bx1 - bx0) / world_w * fig.get_figwidth() * fig.get_dpi()
        if box_w_px > 20:
            ax.text(bx0, by1, label, fontsize=3.5, color=color,
                    va="bottom", ha="left", zorder=6,
                    clip_on=True)

    # ── legend ────────────────────────────────────────────────────────────────
    if label_colors:
        legend_handles = [
            mpatches.Patch(color=c, label=f"{l} ({sum(1 for i in inserts if get_label(i['name'],db,idx)==l)})")
            for l, c in list(label_colors.items())[:30]
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=5, framealpha=0.85,
                  ncol=max(1, len(legend_handles)//15 + 1))

    plt.savefig(out_path, bbox_inches="tight", dpi=150, pad_inches=0)
    plt.close(fig)
    print(f"  Saved → {out_path}")
    print(f"  Labels used: {len(label_colors)}")


def main():
    ap = argparse.ArgumentParser(description="Render DXF with symbol bounding boxes")
    ap.add_argument("dxf", help="Path to DXF file")
    ap.add_argument("--page", type=int, default=0,
                    help="Paper-space layout index (default 0 = first sheet)")
    ap.add_argument("--out", default="", help="Output PNG path")
    ap.add_argument("--label", action="append", default=[],
                    help="Highlight only these labels (repeatable). E.g. --label toilet --label door")
    args = ap.parse_args()

    dxf_path = args.dxf
    out = args.out or (Path(dxf_path).stem + "_symbol_boxes.png")
    render(dxf_path, args.page, out, highlight_labels=args.label or None)


if __name__ == "__main__":
    main()
