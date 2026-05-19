#!/usr/bin/env python3
"""
gen_page_thumb.py  —  Generate a quick overview thumbnail of a DXF page,
marking the positions of a given symbol hash with red dots.

Saves to  symbol_db/pages/<file_stem>_<hash>.png

Called from the UI on demand, or in batch:
    python3 tools/gen_page_thumb.py --hash ab12cd34
    python3 tools/gen_page_thumb.py --limit 50
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
import numpy as np

BASE    = Path(__file__).resolve().parent.parent
DB_F    = BASE / "symbol_db" / "symbols.json"
PAGES_D = BASE / "symbol_db" / "pages"
DXF_OUT = BASE / "dxf_output"
PAGES_D.mkdir(exist_ok=True)

THUMB_DPI   = 300
THUMB_SIZE  = (22, 16)   # inches
LINE_COLOR  = "#333333"
LINE_LW     = 0.2

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


def _draw_block_flat(ax, doc, bname: str,
                     tx: float, ty: float,
                     sx: float, sy: float, rot: float,
                     depth: int = 0):
    """Draw a block's geometry flatly (no recursive INSERT expansion for speed)."""
    if depth > 1:
        return
    try:
        blk = doc.blocks.get(bname)
        if blk is None:
            return
        c, s = math.cos(rot), math.sin(rot)
        for e in blk:
            t = e.dxftype()
            try:
                if t == "LINE":
                    def tw(lx, ly):
                        return lx*sx*c - ly*sy*s + tx, lx*sx*s + ly*sy*c + ty
                    p1 = tw(e.dxf.start.x, e.dxf.start.y)
                    p2 = tw(e.dxf.end.x,   e.dxf.end.y)
                    ax.plot([p1[0],p2[0]], [p1[1],p2[1]],
                            color=LINE_COLOR, lw=LINE_LW, zorder=1)
                elif t == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if not pts:
                        continue
                    def tw(lx, ly):
                        return lx*sx*c - ly*sy*s + tx, lx*sx*s + ly*sy*c + ty
                    wpx, wpy = zip(*(tw(x,y) for x,y in pts))
                    wpx, wpy = list(wpx), list(wpy)
                    if e.closed:
                        wpx.append(wpx[0]); wpy.append(wpy[0])
                    ax.plot(wpx, wpy, color=LINE_COLOR, lw=LINE_LW, zorder=1)
                elif t == "INSERT" and depth == 0:
                    _draw_block_flat(
                        ax, doc, e.dxf.name,
                        tx + e.dxf.insert.x, ty + e.dxf.insert.y,
                        sx * getattr(e.dxf,"xscale",1),
                        sy * getattr(e.dxf,"yscale",1),
                        rot + math.radians(getattr(e.dxf,"rotation",0)),
                        depth+1
                    )
            except Exception:
                pass
    except Exception:
        pass


def generate_page_thumb(hash_id: str, db: dict, force: bool = False) -> Path | None:
    info = db.get(hash_id)
    if info is None:
        return None

    files = info.get("files", [])
    if not files:
        return None

    dxf_path = None
    for raw in files[:3]:
        p = resolve_dxf(raw)
        if p:
            dxf_path = p
            break
    if dxf_path is None:
        return None

    out = PAGES_D / f"{dxf_path.stem}_{hash_id}.png"
    if out.exists() and not force:
        return out

    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        # ── pass 1: collect all coordinates to find content bbox ─────────────
        all_x, all_y = [], []

        def collect(layout_or_block, tx=0.0, ty=0.0,
                    sx=1.0, sy=1.0, rot=0.0, depth=0):
            c, s = math.cos(rot), math.sin(rot)
            def tw(lx, ly):
                return lx*sx*c - ly*sy*s + tx, lx*sx*s + ly*sy*c + ty
            for e in layout_or_block:
                t = e.dxftype()
                try:
                    if t == "LINE":
                        for p in (e.dxf.start, e.dxf.end):
                            wx, wy = tw(p.x, p.y)
                            all_x.append(wx); all_y.append(wy)
                    elif t == "LWPOLYLINE":
                        for px, py in e.get_points("xy"):
                            wx, wy = tw(px, py)
                            all_x.append(wx); all_y.append(wy)
                    elif t == "INSERT" and depth < 2:
                        ix, iy = e.dxf.insert.x, e.dxf.insert.y
                        isx = getattr(e.dxf, "xscale", 1.0)
                        isy = getattr(e.dxf, "yscale", 1.0)
                        irot = math.radians(getattr(e.dxf, "rotation", 0.0))
                        wix, wiy = tw(ix, iy)
                        blk = doc.blocks.get(e.dxf.name)
                        if blk:
                            collect(blk, wix, wiy,
                                    sx*isx, sy*isy, rot+irot, depth+1)
                except Exception:
                    pass

        collect(msp)

        if not all_x:
            return None

        # ── outlier filter: keep points within 3 IQR of median ───────────────
        import numpy as np
        ax_arr = np.array(all_x)
        ay_arr = np.array(all_y)

        def iqr_mask(arr, k=3.0):
            q1, q3 = np.percentile(arr, 10), np.percentile(arr, 90)
            iqr = q3 - q1 or 1
            return (arr >= q1 - k*iqr) & (arr <= q3 + k*iqr)

        mask = iqr_mask(ax_arr) & iqr_mask(ay_arr)
        if mask.sum() > 10:
            ax_arr = ax_arr[mask]
            ay_arr = ay_arr[mask]

        xmin, xmax = float(ax_arr.min()), float(ax_arr.max())
        ymin, ymax = float(ay_arr.min()), float(ay_arr.max())
        cw = xmax - xmin or 1
        ch = ymax - ymin or 1
        pad_x = cw * 0.02
        pad_y = ch * 0.02

        # ── pass 2: render ────────────────────────────────────────────────────
        # Adjust figsize to match aspect ratio of content
        aspect = cw / ch if ch else 1
        fig_w = min(max(aspect * 10, 8), 28)
        fig_h = min(max(10 / aspect if aspect > 1 else 10, 6), 20)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=THUMB_DPI)
        ax.set_aspect("equal")
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        fig.patch.set_facecolor("#ffffff")
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

        for e in msp:
            t = e.dxftype()
            try:
                if t == "LINE":
                    ax.plot([e.dxf.start.x, e.dxf.end.x],
                            [e.dxf.start.y, e.dxf.end.y],
                            color=LINE_COLOR, lw=LINE_LW, zorder=1)
                elif t == "LWPOLYLINE":
                    pts = list(e.get_points("xy"))
                    if pts:
                        px = [p[0] for p in pts]
                        py = [p[1] for p in pts]
                        if e.closed:
                            px.append(px[0]); py.append(py[0])
                        ax.plot(px, py, color=LINE_COLOR, lw=LINE_LW, zorder=1)
                elif t == "INSERT":
                    _draw_block_flat(
                        ax, doc, e.dxf.name,
                        e.dxf.insert.x, e.dxf.insert.y,
                        getattr(e.dxf, "xscale", 1.0),
                        getattr(e.dxf, "yscale", 1.0),
                        math.radians(getattr(e.dxf, "rotation", 0.0)),
                    )
            except Exception:
                pass

        plt.savefig(str(out), dpi=THUMB_DPI, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return out

    except Exception as ex:
        print(f"  [page_thumb] {hash_id}: {ex}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash",  default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db = json.loads(DB_F.read_text("utf-8"))

    if args.hash:
        p = generate_page_thumb(args.hash, db, force=args.force)
        print(f"{'OK: '+str(p) if p else 'FAILED'}")
        return

    targets = [h for h in db
               if db[h].get("label","?") not in ("?","",None)]
    targets.sort(key=lambda h: -db[h].get("count",0))
    if args.limit:
        targets = targets[:args.limit]

    print(f"Generating {len(targets)} page thumbs …")
    ok = fail = 0
    for i, h in enumerate(targets, 1):
        p = generate_page_thumb(h, db, force=args.force)
        if p:
            ok += 1
        else:
            fail += 1
        if i % 50 == 0:
            print(f"  {i}/{len(targets)}  ok={ok} fail={fail}")
    print(f"Done: ok={ok}  fail={fail}")


if __name__ == "__main__":
    main()
