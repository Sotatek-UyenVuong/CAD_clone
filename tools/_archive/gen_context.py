#!/usr/bin/env python3
"""
gen_context.py  —  Generate context-crop images for each symbol.

Renders a small window around the INSERT point of a symbol in its source DXF,
with recursive block expansion so nested INSERTs are visible.
Saves to  symbol_db/context/<hash>.png

Usage:
    python3 tools/gen_context.py              # generate all missing
    python3 tools/gen_context.py --hash ab12  # one hash only
    python3 tools/gen_context.py --force      # regenerate all
"""

import argparse
import json
import math
import sys
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE    = Path(__file__).resolve().parent.parent
DB_F    = BASE / "symbol_db" / "symbols.json"
IDX_F   = BASE / "symbol_db" / "block_name_index.json"
CTX_D   = BASE / "symbol_db" / "context"
DXF_OUT = BASE / "dxf_output"
CTX_D.mkdir(exist_ok=True)

CTX_SCALE  = 8     # context window = block_size * CTX_SCALE (each side)
MAX_DEPTH  = 4     # max recursion depth for nested INSERT expansion
MAX_ENTS   = 8000  # limit entities drawn for performance


# ── file index ────────────────────────────────────────────────────────────────
_FILE_INDEX: dict[str, Path] | None = None

def file_index() -> dict[str, Path]:
    global _FILE_INDEX
    if _FILE_INDEX is None:
        _FILE_INDEX = {p.name: p for p in DXF_OUT.rglob("*.dxf")}
    return _FILE_INDEX

def resolve_dxf(fname: str) -> Path | None:
    p = Path(fname)
    if p.is_absolute() and p.exists():
        return p
    return file_index().get(p.name)


# ── geometry helpers ───────────────────────────────────────────────────────────

def transform_pts(pts, sx, sy, rot_rad, tx, ty):
    """Apply scale + rotation + translation to list of (x,y)."""
    c, s = math.cos(rot_rad), math.sin(rot_rad)
    out = []
    for x, y in pts:
        rx = x * sx * c - y * sy * s + tx
        ry = x * sx * s + y * sy * c + ty
        out.append((rx, ry))
    return out


def block_size(doc, name: str, sx=1.0, sy=1.0, depth=0) -> float:
    """Estimate effective radius (half-diagonal) of a block in world units."""
    if depth > 2:
        return 10.0
    try:
        blk = doc.blocks.get(name)
        if blk is None:
            return 10.0
        xs, ys = [], []
        for e in blk:
            t = e.dxftype()
            try:
                if t == "LINE":
                    xs += [e.dxf.start.x, e.dxf.end.x]
                    ys += [e.dxf.start.y, e.dxf.end.y]
                elif t in ("CIRCLE", "ARC"):
                    r  = e.dxf.radius
                    cx, cy = e.dxf.center.x, e.dxf.center.y
                    xs += [cx-r, cx+r]; ys += [cy-r, cy+r]
                elif t == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if pts:
                        xs += [p[0] for p in pts]
                        ys += [p[1] for p in pts]
                elif t == "INSERT":
                    r = block_size(doc, e.dxf.name,
                                   getattr(e.dxf,"xscale",1)*sx,
                                   getattr(e.dxf,"yscale",1)*sy,
                                   depth+1)
                    cx, cy = e.dxf.insert.x, e.dxf.insert.y
                    xs += [cx-r, cx+r]; ys += [cy-r, cy+r]
                elif hasattr(e.dxf, "insert"):
                    xs.append(e.dxf.insert.x); ys.append(e.dxf.insert.y)
            except Exception:
                pass
        if not xs:
            return 10.0
        w = (max(xs)-min(xs)) * sx
        h = (max(ys)-min(ys)) * sy
        return max(math.sqrt(w**2+h**2)/2, 1.0)
    except Exception:
        return 10.0


# ── recursive entity drawer ───────────────────────────────────────────────────

_ent_count = 0

def draw_entities(ax, doc, layout_or_block, wx0, wy0, wx1, wy1,
                  sx=1.0, sy=1.0, rot=0.0, tx=0.0, ty=0.0,
                  depth=0, color="#444444", lw=0.35):
    """Recursively draw entities, expanding INSERT references."""
    global _ent_count
    if depth > MAX_DEPTH or _ent_count > MAX_ENTS:
        return

    c, s = math.cos(rot), math.sin(rot)

    def to_world(lx, ly):
        rx = lx * sx * c - ly * sy * s + tx
        ry = lx * sx * s + ly * sy * c + ty
        return rx, ry

    def in_window(wxs, wys, margin=0):
        return not (max(wxs) < wx0-margin or min(wxs) > wx1+margin or
                    max(wys) < wy0-margin or min(wys) > wy1+margin)

    for e in layout_or_block:
        if _ent_count > MAX_ENTS:
            break
        t = e.dxftype()
        try:
            if t == "LINE":
                p1 = to_world(e.dxf.start.x, e.dxf.start.y)
                p2 = to_world(e.dxf.end.x,   e.dxf.end.y)
                if in_window([p1[0],p2[0]], [p1[1],p2[1]]):
                    ax.plot([p1[0],p2[0]], [p1[1],p2[1]],
                            color=color, lw=lw, solid_capstyle="round",
                            zorder=2)
                    _ent_count += 1

            elif t == "LWPOLYLINE":
                pts = list(e.get_points("xy"))
                if not pts:
                    continue
                wpx, wpy = zip(*(to_world(x,y) for x,y in pts))
                wpx, wpy = list(wpx), list(wpy)
                if e.closed:
                    wpx.append(wpx[0]); wpy.append(wpy[0])
                if in_window(wpx, wpy):
                    ax.plot(wpx, wpy, color=color, lw=lw, zorder=2)
                    _ent_count += 1

            elif t == "CIRCLE":
                cx, cy = to_world(e.dxf.center.x, e.dxf.center.y)
                r = e.dxf.radius * max(abs(sx), abs(sy))
                if in_window([cx-r,cx+r],[cy-r,cy+r]):
                    circle = plt.Circle((cx,cy), r, fill=False,
                                        edgecolor=color, lw=lw, zorder=2)
                    ax.add_patch(circle)
                    _ent_count += 1

            elif t == "ARC":
                cx, cy = to_world(e.dxf.center.x, e.dxf.center.y)
                r = e.dxf.radius * max(abs(sx), abs(sy))
                if in_window([cx-r,cx+r],[cy-r,cy+r]):
                    a1 = math.radians(e.dxf.start_angle) + rot
                    a2 = math.radians(e.dxf.end_angle)   + rot
                    if a2 <= a1:
                        a2 += 2*math.pi
                    angles = np.linspace(a1, a2, 64)
                    ax.plot(cx + r*np.cos(angles), cy + r*np.sin(angles),
                            color=color, lw=lw, zorder=2)
                    _ent_count += 1

            elif t == "INSERT" and depth < MAX_DEPTH:
                ix = e.dxf.insert.x; iy = e.dxf.insert.y
                isx = getattr(e.dxf, "xscale", 1.0) or 1.0
                isy = getattr(e.dxf, "yscale", 1.0) or 1.0
                irot = math.radians(getattr(e.dxf, "rotation", 0.0))
                # World insert point
                wix, wiy = to_world(ix, iy)
                # Combined transform: new_sx = sx*isx, etc.
                new_sx = sx * isx
                new_sy = sy * isy
                new_rot = rot + irot
                # Check if anywhere near window (rough radius check)
                r_est = block_size(doc, e.dxf.name, new_sx, new_sy) * 1.5
                if in_window([wix-r_est,wix+r_est],[wiy-r_est,wiy+r_est],
                             margin=r_est):
                    blk = doc.blocks.get(e.dxf.name)
                    if blk:
                        draw_entities(ax, doc, blk, wx0, wy0, wx1, wy1,
                                      new_sx, new_sy, new_rot, wix, wiy,
                                      depth+1, color, lw)
        except Exception:
            continue


# ── main generator ────────────────────────────────────────────────────────────

def generate_context(hash_id: str, db: dict, idx: dict, force=False) -> bool:
    global _ent_count
    out = CTX_D / f"{hash_id}.png"
    if out.exists() and not force:
        return True

    info = db.get(hash_id)
    if info is None:
        return False

    files       = info.get("files", [])
    block_names = info.get("block_names", [])
    if not files or not block_names:
        return False

    target_bname = block_names[0]

    for raw in files[:3]:
        dxf_path = resolve_dxf(raw)
        if dxf_path is None:
            continue
        try:
            doc = ezdxf.readfile(str(dxf_path))
            msp = doc.modelspace()

            # Find INSERT entity
            ins_ent = None
            for e in msp:
                if e.dxftype() == "INSERT" and e.dxf.name == target_bname:
                    ins_ent = e
                    break
            if ins_ent is None:
                continue

            ix = ins_ent.dxf.insert.x
            iy = ins_ent.dxf.insert.y
            isx = getattr(ins_ent.dxf, "xscale", 1.0) or 1.0
            isy = getattr(ins_ent.dxf, "yscale", 1.0) or 1.0
            irot = math.radians(getattr(ins_ent.dxf, "rotation", 0.0))

            # Estimate block world size (radius) for setting context window
            rad = block_size(doc, target_bname, abs(isx), abs(isy))
            pad = max(rad * CTX_SCALE, 5.0)

            wx0, wy0 = ix - pad, iy - pad
            wx1, wy1 = ix + pad, iy + pad

            # ── render ─────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
            ax.set_xlim(wx0, wx1)
            ax.set_ylim(wy0, wy1)
            ax.set_aspect("equal")
            ax.set_facecolor("#f8f8f8")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            _ent_count = 0
            draw_entities(ax, doc, msp, wx0, wy0, wx1, wy1)

            # ── highlight: draw a red box around the block's world bbox ──────
            # Compute the axis-aligned bbox of the block in world coordinates
            def block_world_pts(doc, bname, tx, ty, sx, sy, rot, depth=0):
                """Collect world-coord points from a block (for bbox)."""
                pts = []
                if depth > 2:
                    return pts
                try:
                    blk = doc.blocks.get(bname)
                    if blk is None:
                        return pts
                    c, s = math.cos(rot), math.sin(rot)
                    def tw(lx, ly):
                        return (lx*sx*c - ly*sy*s + tx,
                                lx*sx*s + ly*sy*c + ty)
                    for e in blk:
                        t = e.dxftype()
                        try:
                            if t == "LINE":
                                pts.append(tw(e.dxf.start.x, e.dxf.start.y))
                                pts.append(tw(e.dxf.end.x,   e.dxf.end.y))
                            elif t == "LWPOLYLINE":
                                for px, py in e.get_points("xy"):
                                    pts.append(tw(px, py))
                            elif t in ("CIRCLE", "ARC"):
                                cx, cy = e.dxf.center.x, e.dxf.center.y
                                r = e.dxf.radius * max(abs(sx), abs(sy))
                                wc = tw(cx, cy)
                                pts += [(wc[0]-r, wc[1]-r),
                                        (wc[0]+r, wc[1]+r)]
                            elif t == "INSERT":
                                nisx = getattr(e.dxf,"xscale",1)*sx
                                nisy = getattr(e.dxf,"yscale",1)*sy
                                nrot = rot + math.radians(
                                    getattr(e.dxf,"rotation",0))
                                wix, wiy = tw(e.dxf.insert.x, e.dxf.insert.y)
                                pts += block_world_pts(doc, e.dxf.name,
                                                       wix, wiy,
                                                       nisx, nisy, nrot,
                                                       depth+1)
                        except Exception:
                            pass
                except Exception:
                    pass
                return pts

            bbox_pts = block_world_pts(doc, target_bname,
                                       ix, iy, isx, isy, irot)

            if bbox_pts:
                bxs = [p[0] for p in bbox_pts]
                bys = [p[1] for p in bbox_pts]
                bx0b, bx1b = min(bxs), max(bxs)
                by0b, by1b = min(bys), max(bys)
                # Add small margin
                mg = max((bx1b-bx0b), (by1b-by0b), pad*0.05) * 0.1
                bx0b -= mg; bx1b += mg
                by0b -= mg; by1b += mg
            else:
                # Fallback: small box around INSERT point
                mg = pad * 0.1
                bx0b, bx1b = ix-mg, ix+mg
                by0b, by1b = iy-mg, iy+mg

            # Red dashed bounding box
            rect = mpatches.FancyBboxPatch(
                (bx0b, by0b), bx1b-bx0b, by1b-by0b,
                boxstyle="square,pad=0",
                linewidth=1.8, edgecolor="#e74c3c",
                facecolor=(1, 0, 0, 0.0),
                linestyle="--",
                zorder=8,
            )
            ax.add_patch(rect)
            # Small dot at center
            ax.plot(ix, iy, "o", color="#e74c3c",
                    markersize=5, markeredgewidth=0,
                    zorder=9, alpha=0.85)

            label = info.get("label","?") or "?"
            ax.set_title(f"{label}  ·  {target_bname}  ×{info.get('count',0)}",
                         fontsize=7, pad=3, color="#333333")

            plt.savefig(str(out), dpi=130, bbox_inches="tight",
                        pad_inches=0.05)
            plt.close(fig)
            return True

        except Exception as e:
            print(f"  [warn] {hash_id} / {raw}: {e}", file=sys.stderr)
            continue

    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash",  default="", help="Single hash to generate")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db  = json.loads(DB_F.read_text("utf-8"))
    idx = json.loads(IDX_F.read_text("utf-8"))

    if args.hash:
        ok = generate_context(args.hash, db, idx, force=args.force)
        print("OK" if ok else "FAILED")
        return

    targets = list(db.keys())
    if not args.force:
        targets = [h for h in targets
                   if not (CTX_D / f"{h}.png").exists()]
    if args.limit:
        targets = targets[:args.limit]

    print(f"Generating {len(targets)} context images …")
    ok = fail = 0
    for i, h in enumerate(targets, 1):
        if generate_context(h, db, idx, force=args.force):
            ok += 1
        else:
            fail += 1
        if i % 100 == 0:
            print(f"  {i}/{len(targets)}  ok={ok} fail={fail}")

    print(f"Done: ok={ok}  fail={fail}")


if __name__ == "__main__":
    main()
