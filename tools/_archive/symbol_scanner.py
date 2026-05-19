#!/usr/bin/env python3
"""
symbol_scanner.py  –  Scan DXF files for unique block symbols,
render review grids, and query symbol counts/positions.

COMMANDS
--------
scan    Scan a folder of DXF files and build the symbol database.
        python tools/symbol_scanner.py scan <dxf_dir> [--out <db_dir>]

review  Open the latest grid images in the db folder (informational).
        python tools/symbol_scanner.py review [--db <db_dir>]

query   Count and locate symbols by label in one or more DXF files.
        python tools/symbol_scanner.py query <dxf_file_or_dir> --label toilet sink ...

WORKFLOW
--------
1.  python tools/symbol_scanner.py scan dxf_output/竣工図…/
        → symbol_db/symbols.json      (edit this to add labels)
        → symbol_db/images/<hash>.png (one thumbnail per unique shape)
        → symbol_db/grids/grid_NNN.png (review grids, 8 per row)

2.  Open symbol_db/grids/grid_001.png, look at each cell.
    Edit symbol_db/symbols.json:  change  "label": "?"  →  "label": "toilet"

3.  python tools/symbol_scanner.py query path/to/file.dxf --label toilet
        → prints count + (x, y) positions in DXF units
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ezdxf


# ── geometry ──────────────────────────────────────────────────────────────────

def _block_keypoints(block_def) -> list[tuple[float, float]]:
    """Extract representative (x, y) keypoints from a block definition."""
    pts: list[tuple[float, float]] = []
    for e in block_def:
        t = e.dxftype()
        try:
            if t == "LINE":
                pts += [(e.dxf.start.x, e.dxf.start.y),
                        (e.dxf.end.x,   e.dxf.end.y)]
            elif t in ("ARC", "CIRCLE"):
                pts.append((e.dxf.center.x, e.dxf.center.y))
                if t == "ARC":
                    r = e.dxf.radius
                    for ang in (e.dxf.start_angle, e.dxf.end_angle):
                        a = math.radians(ang)
                        pts.append((e.dxf.center.x + r * math.cos(a),
                                    e.dxf.center.y + r * math.sin(a)))
            elif t == "LWPOLYLINE":
                pts += [(p[0], p[1]) for p in e.get_points()]
            elif t == "POLYLINE":
                pts += [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            elif t == "ELLIPSE":
                cx, cy = e.dxf.center.x, e.dxf.center.y
                pts.append((cx, cy))
                mx, my = e.dxf.major_axis.x, e.dxf.major_axis.y
                pts += [(cx + mx, cy + my), (cx - mx, cy - my)]
            elif t == "SPLINE":
                pts += [(p[0], p[1]) for p in e.control_points]
        except Exception:
            pass
    return pts


def geometry_hash(block_def) -> str:
    """
    Rotation- and scale-invariant hash of a block's geometry.

    Strategy: compute sorted pairwise distances between keypoints,
    normalised so the largest distance = 1.0.  This is invariant to
    translation, rotation, and uniform scaling.
    """
    entity_type_sig = ",".join(
        sorted(e.dxftype() for e in block_def
               if not e.dxftype().startswith("ATTDEF"))
    )
    if not entity_type_sig:
        return hashlib.md5(b"empty").hexdigest()[:8]

    pts = _block_keypoints(block_def)

    if len(pts) < 2:
        return hashlib.md5(entity_type_sig.encode()).hexdigest()[:8]

    # Cap to avoid O(n²) blow-up on large blocks; sample evenly
    MAX_PTS = 40
    if len(pts) > MAX_PTS:
        step = len(pts) / MAX_PTS
        pts = [pts[int(i * step)] for i in range(MAX_PTS)]

    # Pairwise distances (rotation-invariant)
    dists: list[float] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dists.append(math.sqrt(dx * dx + dy * dy))

    dists.sort()
    max_d = dists[-1]
    if max_d > 0:
        dists = [round(d / max_d, 4) for d in dists]

    payload = entity_type_sig + "|" + ",".join(f"{d:.4f}" for d in dists)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


# ── rendering ─────────────────────────────────────────────────────────────────

def render_block(block_def, size: int = 128) -> np.ndarray:
    """Render a block definition to a (size×size, RGB) numpy array."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    drawn = False
    for e in block_def:
        t = e.dxftype()
        try:
            if t == "LINE":
                ax.plot([e.dxf.start.x, e.dxf.end.x],
                        [e.dxf.start.y, e.dxf.end.y],
                        color="black", linewidth=0.8, solid_capstyle="round")
                drawn = True
            elif t == "ARC":
                arc = mpatches.Arc(
                    (e.dxf.center.x, e.dxf.center.y),
                    2 * e.dxf.radius, 2 * e.dxf.radius,
                    angle=0,
                    theta1=e.dxf.start_angle, theta2=e.dxf.end_angle,
                    color="black", linewidth=0.8)
                ax.add_patch(arc)
                drawn = True
            elif t == "CIRCLE":
                c = mpatches.Circle(
                    (e.dxf.center.x, e.dxf.center.y), e.dxf.radius,
                    fill=False, color="black", linewidth=0.8)
                ax.add_patch(c)
                drawn = True
            elif t == "ELLIPSE":
                major = e.dxf.major_axis
                ratio = e.dxf.ratio
                ang = math.degrees(math.atan2(major.y, major.x))
                a = math.sqrt(major.x ** 2 + major.y ** 2)
                b = a * ratio
                el = mpatches.Ellipse(
                    (e.dxf.center.x, e.dxf.center.y),
                    2 * a, 2 * b, angle=ang,
                    fill=False, color="black", linewidth=0.8)
                ax.add_patch(el)
                drawn = True
            elif t == "LWPOLYLINE":
                pts = list(e.get_points())
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if e.closed:
                        xs.append(xs[0]); ys.append(ys[0])
                    ax.plot(xs, ys, color="black", linewidth=0.8)
                    drawn = True
            elif t == "POLYLINE":
                verts = list(e.vertices)
                if verts:
                    xs = [v.dxf.location.x for v in verts]
                    ys = [v.dxf.location.y for v in verts]
                    if e.is_closed:
                        xs.append(xs[0]); ys.append(ys[0])
                    ax.plot(xs, ys, color="black", linewidth=0.8)
                    drawn = True
            elif t == "SPLINE":
                pts = list(e.control_points)
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    ax.plot(xs, ys, color="black", linewidth=0.5, linestyle="--")
                    drawn = True
        except Exception:
            pass

    if drawn:
        ax.autoscale_view()
        # Add a little padding
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        pw = (xlim[1] - xlim[0]) * 0.1 or 0.1
        ph = (ylim[1] - ylim[0]) * 0.1 or 0.1
        ax.set_xlim(xlim[0] - pw, xlim[1] + pw)
        ax.set_ylim(ylim[0] - ph, ylim[1] + ph)
    else:
        ax.text(0.5, 0.5, "?", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 3)
    plt.close(fig)
    return img


# ── grid builder ──────────────────────────────────────────────────────────────

THUMB = 128     # thumbnail size in pixels
LABEL_H = 28    # height of label strip below each thumb
COLS = 8        # thumbnails per row

def _make_label_strip(text: str, width: int, height: int,
                      bg: tuple, fg: tuple) -> Image.Image:
    strip = Image.new("RGB", (width, height), bg)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=fg, font=font)
    return strip


def build_grid(entries: list[dict], out_path: Path) -> None:
    """
    entries: list of {"hash": str, "label": str, "count": int, "image": np.ndarray}
    """
    cols = COLS
    rows = math.ceil(len(entries) / cols)
    cell_h = THUMB + LABEL_H * 2   # top label (hash) + bottom label (count/label)
    cell_w = THUMB

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (240, 240, 240))

    for idx, entry in enumerate(entries):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = row * cell_h

        # Thumbnail
        thumb = Image.fromarray(entry["image"]).resize((THUMB, THUMB), Image.LANCZOS)
        canvas.paste(thumb, (x0, y0 + LABEL_H))

        # Top strip: hash + index
        top_text = f"#{idx + 1}  {entry['hash']}"
        top_strip = _make_label_strip(top_text, THUMB, LABEL_H, (60, 60, 80), (255, 255, 255))
        canvas.paste(top_strip, (x0, y0))

        # Bottom strip: current label + count
        lbl   = entry["label"] if entry["label"] and entry["label"] != "?" else "—未ラベル—"
        color = (40, 160, 40) if entry["label"] not in ("?", "", None) else (180, 60, 60)
        bot_text = f"{lbl}  ×{entry['count']}"
        bot_strip = _make_label_strip(bot_text, THUMB, LABEL_H, color, (255, 255, 255))
        canvas.paste(bot_strip, (x0, y0 + LABEL_H + THUMB))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))


# ── scan command ──────────────────────────────────────────────────────────────

SKIP_PREFIXES = ("*Model", "*Paper", "*Block")

def _should_skip(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def cmd_scan(dxf_dir: Path, db_dir: Path) -> None:
    dxf_files = sorted(dxf_dir.rglob("*.dxf"))
    if not dxf_files:
        print(f"No DXF files found in {dxf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(dxf_files)} DXF files in {dxf_dir} …")

    # hash → {count, label, files, block_name, block_doc_path}
    db: dict[str, dict] = {}
    # We keep one representative block per hash for rendering
    block_cache: dict[str, object] = {}   # hash → block_def object

    for file_idx, dxf_path in enumerate(dxf_files, 1):
        print(f"  [{file_idx:3d}/{len(dxf_files)}] {dxf_path.name}", flush=True)
        try:
            doc  = ezdxf.readfile(str(dxf_path))
            msp  = doc.modelspace()

            # Enumerate INSERT entities → find unique block hashes
            seen_in_file: set[str] = set()
            for e in msp:
                if e.dxftype() != "INSERT":
                    continue
                bname = e.dxf.name
                if _should_skip(bname):
                    continue
                try:
                    bdef = doc.blocks[bname]
                except Exception:
                    continue

                # Skip completely empty blocks
                entity_types = [x.dxftype() for x in bdef
                                if x.dxftype() not in ("ATTDEF", "ATTRIB", "SEQEND")]
                if not entity_types:
                    continue

                h = geometry_hash(bdef)

                if h not in db:
                    db[h] = {
                        "hash":        h,
                        "label":       "?",
                        "count":       0,
                        "files":       [],
                        "block_names": [],
                    }
                    block_cache[h] = bdef   # save for rendering

                db[h]["count"] += 1
                rel = str(dxf_path.relative_to(dxf_dir))
                if rel not in db[h]["files"]:
                    db[h]["files"].append(rel)
                if bname not in db[h]["block_names"]:
                    db[h]["block_names"].append(bname)

        except Exception as ex:
            print(f"  WARN {dxf_path.name}: {ex}", file=sys.stderr)

    print(f"Found {len(db)} unique block shapes across all files.")

    # Load existing labels if db already exists (preserve user labels)
    symbols_json = db_dir / "symbols.json"
    if symbols_json.exists():
        existing = json.loads(symbols_json.read_text(encoding="utf-8"))
        for h, info in existing.items():
            if h in db and info.get("label") and info["label"] != "?":
                db[h]["label"] = info["label"]
        print(f"  Preserved {sum(1 for v in existing.values() if v.get('label','?')!='?')} existing labels.")

    # Render thumbnails
    images_dir = db_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    print("Rendering thumbnails …")
    rendered: dict[str, np.ndarray] = {}
    for h, bdef in block_cache.items():
        png_path = images_dir / f"{h}.png"
        if png_path.exists():
            # Reuse existing render
            rendered[h] = np.array(Image.open(png_path).convert("RGB"))
        else:
            img = render_block(bdef)
            rendered[h] = img
            Image.fromarray(img).save(str(png_path))

    # Build review grids (sorted: unlabeled first, then by count desc)
    entries = sorted(db.values(),
                     key=lambda v: (v["label"] != "?", -v["count"]))
    grid_entries = [
        {**e, "image": rendered.get(e["hash"],
                                    np.full((THUMB, THUMB, 3), 200, np.uint8))}
        for e in entries
    ]

    grids_dir = db_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    PAGE = COLS * 8   # symbols per grid page
    for page_idx, start in enumerate(range(0, len(grid_entries), PAGE)):
        chunk    = grid_entries[start: start + PAGE]
        out_path = grids_dir / f"grid_{page_idx + 1:03d}.png"
        build_grid(chunk, out_path)
        print(f"  → {out_path}  ({len(chunk)} symbols)")

    # Write JSON (without image data)
    out_db = {h: {k: v for k, v in info.items() if k != "image"}
              for h, info in db.items()}
    symbols_json.write_text(json.dumps(out_db, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # Write reverse index: block_name → hash (for fast query)
    block_name_index: dict[str, str] = {}
    for h, info in out_db.items():
        for bn in info.get("block_names", []):
            block_name_index[bn] = h
    index_path = db_dir / "block_name_index.json"
    index_path.write_text(json.dumps(block_name_index, ensure_ascii=False, indent=2),
                          encoding="utf-8")

    print(f"\nDone.  Edit labels in:\n  {symbols_json}\nReview grids in:\n  {grids_dir}/")
    print(f"Block name index: {len(block_name_index)} entries → {index_path}")


# ── query command ─────────────────────────────────────────────────────────────

def cmd_query(targets: list[Path], labels: list[str], db_dir: Path) -> None:
    symbols_json = db_dir / "symbols.json"
    if not symbols_json.exists():
        print(f"Symbol DB not found: {symbols_json}\nRun 'scan' first.", file=sys.stderr)
        sys.exit(1)

    db = json.loads(symbols_json.read_text(encoding="utf-8"))
    labels_lower = [l.lower() for l in labels]

    # hash → label mapping (only labeled entries)
    hash_to_label = {h: info["label"]
                     for h, info in db.items()
                     if info.get("label") and info["label"] not in ("?", "")}

    # Fast path: block_name → hash lookup (built during scan)
    index_path = db_dir / "block_name_index.json"
    bn_index: dict[str, str] = {}
    if index_path.exists():
        bn_index = json.loads(index_path.read_text(encoding="utf-8"))

    # Collect DXF files
    dxf_files: list[Path] = []
    for t in targets:
        if t.is_dir():
            dxf_files.extend(sorted(t.rglob("*.dxf")))
        elif t.suffix.lower() == ".dxf":
            dxf_files.append(t)

    if not dxf_files:
        print("No DXF files found.", file=sys.stderr)
        sys.exit(1)

    # Build set of files that *might* contain our target labels
    # (from DB "files" lists) — skip files that can't possibly match
    if labels_lower:
        candidate_files: set[str] = set()
        for h, info in db.items():
            lbl = info.get("label", "?")
            if lbl != "?" and lbl.lower() in labels_lower:
                candidate_files.update(info.get("files", []))
        # Filter dxf_files to only candidates (relative paths stored in DB)
        if candidate_files:
            dxf_dir_for_filter = targets[0] if len(targets) == 1 and targets[0].is_dir() else None
            if dxf_dir_for_filter:
                rel_to_dir = dxf_dir_for_filter
                filtered = [f for f in dxf_files
                            if any(str(f).endswith(cf) or cf in str(f)
                                   for cf in candidate_files)]
                if filtered:
                    print(f"Scanning {len(filtered)} / {len(dxf_files)} files "
                          f"(pre-filtered by DB).")
                    dxf_files = filtered

    total: dict[str, int] = defaultdict(int)

    for dxf_path in dxf_files:
        try:
            doc = ezdxf.readfile(str(dxf_path))
            msp = doc.modelspace()
        except Exception as ex:
            print(f"ERROR {dxf_path.name}: {ex}")
            continue

        found: dict[str, list[tuple[float, float]]] = defaultdict(list)
        unknown_blocks: set[str] = set()

        for e in msp:
            if e.dxftype() != "INSERT":
                continue
            bname = e.dxf.name
            if _should_skip(bname):
                continue

            # Fast lookup in index first
            h = bn_index.get(bname)
            if h is None:
                # Block not seen during scan → compute hash on the fly
                if bname in unknown_blocks:
                    continue
                try:
                    bdef = doc.blocks[bname]
                    h = geometry_hash(bdef)
                    bn_index[bname] = h   # cache for this session
                except Exception:
                    unknown_blocks.add(bname)
                    continue

            label = hash_to_label.get(h)
            if label is None:
                continue
            if labels_lower and label.lower() not in labels_lower:
                continue
            found[label].append((e.dxf.insert.x, e.dxf.insert.y))

        if found:
            print(f"\n{'─'*60}")
            print(f"File: {dxf_path.name}")
            for label, positions in sorted(found.items()):
                print(f"  {label}: {len(positions)} 個")
                for i, (x, y) in enumerate(positions, 1):
                    print(f"    #{i:3d}  x={x:9.1f}  y={y:9.1f}")
                total[label] += len(positions)

    if total:
        print(f"\n{'═'*60}")
        print("TOTAL across all files:")
        for label, cnt in sorted(total.items()):
            print(f"  {label}: {cnt} 個")
    else:
        print("(no matching symbols found)")

    # Save updated index back (in case new blocks were hashed)
    if index_path.exists():
        index_path.write_text(json.dumps(bn_index, ensure_ascii=False, indent=2),
                              encoding="utf-8")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="DXF symbol scanner — build dictionary + query counts/positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # scan
    s = sub.add_parser("scan", help="Scan DXF folder and build symbol DB")
    s.add_argument("dxf_dir", type=Path)
    s.add_argument("--out", dest="db_dir", type=Path,
                   default=Path("symbol_db"), metavar="DB_DIR")

    # query
    q = sub.add_parser("query", help="Count/locate symbols by label")
    q.add_argument("targets", type=Path, nargs="+",
                   help="DXF file(s) or folder(s)")
    q.add_argument("--label", "-l", dest="labels", nargs="+", default=[],
                   metavar="LABEL",
                   help="Labels to search (omit to show all labeled symbols)")
    q.add_argument("--db", dest="db_dir", type=Path,
                   default=Path("symbol_db"), metavar="DB_DIR")

    args = p.parse_args()

    if args.cmd == "scan":
        cmd_scan(args.dxf_dir.resolve(), args.db_dir.resolve())
    elif args.cmd == "query":
        cmd_query([t.resolve() for t in args.targets],
                  args.labels,
                  args.db_dir.resolve())


if __name__ == "__main__":
    main()
