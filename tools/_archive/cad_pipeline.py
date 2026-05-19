#!/usr/bin/env python3
"""
cad_pipeline.py — End-to-end CAD analysis pipeline
====================================================

Steps:
  1. DWG → DXF         (ODA FileConverter)
  2. PDF → PNG 300dpi  (fitz/PyMuPDF)
  3. PDF + DXF → align (pdf_dxf_aligner: text-anchor affine transform)
  4. Detectron2 layout detect on PNG → bbox pixel
  5. bbox pixel → DXF coords (via aligner)
  6. Query DXF entities per region (text, blocks, layers)
  7. Export structured JSON + Markdown

Usage:
  # Full pipeline on one drawing
  python tools/cad_pipeline.py \
      --pdf  "260316_.../竣工図（...建築意匠図）.pdf" \
      --dxf  "dxf_output/.../D063_配置図.dxf" \
      --page 62 \
      --out  output/D063

  # Skip Detectron2 (use DXF query only)
  python tools/cad_pipeline.py --pdf ... --dxf ... --no-detect

  # Batch: run on all pages mapped to their DXF
  python tools/cad_pipeline.py --map page_map.json --out output/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: DWG → DXF ─────────────────────────────────────────────────────────

def step_dwg_to_dxf(dwg_path: str, out_dir: str) -> str | None:
    """Convert DWG → DXF using ODA. Returns DXF path or None if already DXF."""
    import shutil, subprocess, tempfile
    from pathlib import Path as P

    src = P(dwg_path)
    if src.suffix.lower() == ".dxf":
        return str(src)

    oda = "/usr/bin/ODAFileConverter"
    if not P(oda).exists():
        _log("⚠️  ODA not found — skipping DWG→DXF conversion")
        return None

    xvfb = shutil.which("xvfb-run") or ""
    out_dxf = P(out_dir) / (src.stem + ".dxf")
    if out_dxf.exists():
        _log(f"  DXF already exists: {out_dxf.name}")
        return str(out_dxf)

    P(out_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_in  = P(tmp) / "in";  tmp_in.mkdir()
        tmp_out = P(tmp) / "out"; tmp_out.mkdir()
        shutil.copy2(src, tmp_in / src.name)

        cmd = [oda, str(tmp_in), str(tmp_out), "ACAD2018", "ACAD2018_DXF", "0", "1"]
        if xvfb:
            cmd = [xvfb, "-a", "--"] + cmd

        try:
            subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        except Exception as e:
            _log(f"  ODA failed: {e}")
            return None

        dxfs = list(tmp_out.glob("*.dxf"))
        if not dxfs:
            _log("  ODA produced no DXF")
            return None

        shutil.copy2(dxfs[0], out_dxf)

    _log(f"  ✅ DXF: {out_dxf.name}")
    return str(out_dxf)


# ── Step 2: PDF → PNG ─────────────────────────────────────────────────────────

def step_pdf_to_png(pdf_path: str, page: int, out_dir: str,
                    dpi: int = 300) -> tuple[str, int, int]:
    """Render one PDF page → PNG. Returns (png_path, width_px, height_px)."""
    import fitz
    from pathlib import Path as P

    P(out_dir).mkdir(parents=True, exist_ok=True)
    stem = P(pdf_path).stem + f"_p{page}"
    png_path = P(out_dir) / f"{stem}.png"

    doc  = fitz.open(pdf_path)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = doc[page].get_pixmap(matrix=mat, alpha=False)
    pix.save(str(png_path))

    _log(f"  ✅ PNG: {png_path.name}  ({pix.width}×{pix.height}px @ {dpi}dpi)")
    return str(png_path), pix.width, pix.height


# ── Step 3: PDF + DXF → alignment ────────────────────────────────────────────

def step_align(pdf_path: str, dxf_path: str, page: int,
               out_dir: str, min_len: int = 3):
    """Fit affine transform PDF↔DXF. Returns PdfDxfAligner."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pdf_dxf_aligner import PdfDxfAligner
    from pathlib import Path as P

    align_json = P(out_dir) / (P(dxf_path).stem + "_align.json")

    if align_json.exists():
        _log(f"  Loading cached alignment: {align_json.name}")
        return PdfDxfAligner.load(str(align_json))

    aligner = PdfDxfAligner(pdf_path, dxf_path, page=page, min_anchor_len=min_len)
    aligner.fit()
    aligner.save(str(align_json))
    return aligner


# ── Step 4: Detectron2 inference ─────────────────────────────────────────────

def step_detect(png_path: str, model_cfg: str, model_weights: str,
                score_thresh: float = 0.4) -> list[dict]:
    """
    Run Detectron2 on PNG. Returns list of:
      {"label": str, "bbox": [x1,y1,x2,y2], "score": float}
    """
    try:
        import torch
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.data import MetadataCatalog
        import cv2
    except ImportError:
        _log("⚠️  Detectron2 not available — skipping layout detection")
        return []

    cfg = get_cfg()
    cfg.merge_from_file(model_cfg)
    cfg.MODEL.WEIGHTS   = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    img = cv2.imread(png_path)
    out = predictor(img)

    instances = out["instances"].to("cpu")
    classes   = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    # Try to get class names from metadata
    try:
        meta   = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        names  = meta.thing_classes
    except Exception:
        names  = ["text", "table", "title_block", "diagram"]

    results = []
    for i in range(len(instances)):
        box   = instances.pred_boxes.tensor[i].tolist()
        cls   = int(instances.pred_classes[i])
        score = float(instances.scores[i])
        label = names[cls] if cls < len(names) else f"class_{cls}"
        results.append({
            "label": label,
            "bbox":  [box[0], box[1], box[2], box[3]],
            "score": round(score, 4),
        })

    _log(f"  ✅ Detected {len(results)} regions")
    return results


# ── Step 5+6: Map bboxes → DXF query ─────────────────────────────────────────

def step_query(aligner, bboxes: list[dict], dpi: int = 300) -> list[dict]:
    """Map each bbox → DXF coords → query entities."""
    import ezdxf
    from collections import Counter

    doc = ezdxf.readfile(aligner.dxf_path)
    msp = doc.modelspace()

    # Pre-collect all entities once (faster than iterating msp per bbox)
    all_texts:  list[tuple[str, str, float, float]] = []  # (text, layer, x, y)
    all_inserts: list[tuple[str, str, float, float]] = []  # (name, layer, x, y)

    for e in msp:
        et = e.dxftype()
        layer = e.dxf.get("layer", "")

        if et == "TEXT":
            pt = e.dxf.get("insert"); t = e.dxf.get("text", "").strip()
            if pt and t:
                all_texts.append((t, layer, float(pt.x), float(pt.y)))

        elif et == "MTEXT":
            pt = e.dxf.get("insert")
            t  = e.plain_mtext().strip() if hasattr(e, "plain_mtext") else ""
            if pt and t:
                all_texts.append((t, layer, float(pt.x), float(pt.y)))

        elif et == "INSERT":
            pt = e.dxf.get("insert")
            if pt:
                all_inserts.append((
                    e.dxf.get("name", "?"), layer,
                    float(pt.x), float(pt.y)
                ))

    results = []
    for b in bboxes:
        mx1, my1, mx2, my2 = aligner.bbox_to_dxf(b["bbox"], dpi=dpi)

        def _in(x, y): return mx1 <= x <= mx2 and my1 <= y <= my2

        region_texts  = [(t, l) for t, l, x, y in all_texts   if _in(x, y)]
        region_blocks = [(n, l) for n, l, x, y in all_inserts  if _in(x, y)]

        block_counts = Counter(n for n, _ in region_blocks)
        layers       = sorted({l for _, l in region_texts + region_blocks} - {""})

        # Semantic block categorisation
        STAIR_KW  = {"STAIR","階段","EGR","ST"}
        DOOR_KW   = {"DOOR","扉","DR","建具","TATEGU"}
        EV_KW     = {"EV","ELEV","エレベータ","昇降機","LEV"}
        COL_KW    = {"COL","柱","COLUMN"}
        TOILET_KW = {"WC","TOILET","トイレ","便所"}

        def _cat(name):
            u = name.upper()
            if any(k in u for k in STAIR_KW):  return "stairs"
            if any(k in u for k in DOOR_KW):   return "doors"
            if any(k in u for k in EV_KW):     return "elevators"
            if any(k in u for k in COL_KW):    return "columns"
            if any(k in u for k in TOILET_KW): return "toilets"
            return "other"

        by_cat: dict[str, dict[str, int]] = {}
        for name, cnt in block_counts.items():
            cat = _cat(name)
            by_cat.setdefault(cat, {})[name] = cnt

        results.append({
            "label":      b["label"],
            "score":      b.get("score", 1.0),
            "bbox_pixel": b["bbox"],
            "bbox_dxf":   [round(mx1,1), round(my1,1),
                           round(mx2,1), round(my2,1)],
            "layers":     layers,
            "texts": {
                "count":  len(region_texts),
                "sample": [t for t, _ in region_texts[:10]],
            },
            "blocks": {
                "total":       sum(block_counts.values()),
                "by_category": {k: v for k, v in by_cat.items() if k != "other"},
                "all":         dict(block_counts.most_common(20)),
            },
        })

    return results


# ── Step 7: Export ────────────────────────────────────────────────────────────

def step_export(results: list[dict], aligner,
                out_dir: str, stem: str) -> None:
    """Save JSON + Markdown summary."""
    from pathlib import Path as P

    P(out_dir).mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = P(out_dir) / f"{stem}_result.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Markdown
    md_path = P(out_dir) / f"{stem}_result.md"
    lines   = [f"# CAD Analysis: {stem}", ""]

    t_info = aligner._transform or {}
    lines += [
        "## Alignment",
        f"- Anchors: {t_info.get('n_anchors','?')}",
        f"- Mean error: {t_info.get('residual_mean_dxf_units',0):.3f} DXF units",
        "",
        "## Detected Regions",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"### {i}. {r['label'].upper()}  (score={r['score']:.2f})",
            f"- Layers: {', '.join(r['layers']) or '—'}",
            f"- Texts:  {r['texts']['count']}  →  "
            f"{r['texts']['sample'][:5]}",
        ]
        bd = r["blocks"]
        if bd["total"]:
            lines.append(f"- Blocks: {bd['total']} total")
            for cat, items in bd["by_category"].items():
                total = sum(items.values())
                lines.append(f"  - **{cat}**: {total}  {items}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"  ✅ JSON → {json_path.name}")
    _log(f"  ✅ MD   → {md_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    dxf_path: str,
    page:     int,
    out_dir:  str,
    dpi:      int  = 300,
    detect:   bool = True,
    model_cfg:     str = "",
    model_weights: str = "",
    score_thresh:  float = 0.4,
    dwg_path: str = "",
) -> dict:
    from pathlib import Path as P

    stem = P(dxf_path).stem
    _log(f"▶ Pipeline: {stem}  (page {page})")

    # Step 1: DWG → DXF (optional)
    if dwg_path:
        _log("Step 1: DWG → DXF")
        dxf_path = step_dwg_to_dxf(dwg_path, out_dir) or dxf_path

    # Step 2: PDF → PNG
    _log("Step 2: PDF → PNG")
    png_path, img_w, img_h = step_pdf_to_png(pdf_path, page, out_dir, dpi)

    # Step 3: Alignment
    _log("Step 3: PDF+DXF alignment")
    aligner = step_align(pdf_path, dxf_path, page, out_dir)

    # Step 4: Detectron2
    bboxes: list[dict] = []
    if detect and model_cfg and model_weights:
        _log("Step 4: Detectron2 layout detection")
        bboxes = step_detect(png_path, model_cfg, model_weights, score_thresh)
    else:
        _log("Step 4: Detectron2 skipped — using full-page bbox")
        bboxes = [{"label": "full_page", "bbox": [0, 0, img_w, img_h], "score": 1.0}]

    # Steps 5+6: Map → query
    _log("Step 5-6: Map bboxes → DXF query")
    results = step_query(aligner, bboxes, dpi=dpi)

    # Step 7: Export
    _log("Step 7: Export")
    step_export(results, aligner, out_dir, stem)

    summary = {
        "stem":          stem,
        "page":          page,
        "png":           png_path,
        "n_regions":     len(results),
        "results":       results,
    }
    _log(f"✅ Done: {stem}  →  {out_dir}/")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CAD pipeline: PDF+DXF → layout detect → DXF query"
    )
    ap.add_argument("--pdf",     required=True, help="PDF file (AutoCAD export)")
    ap.add_argument("--dxf",     required=True, help="DXF file")
    ap.add_argument("--page",    type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--out",     default="output", help="Output directory")
    ap.add_argument("--dpi",     type=int, default=300)
    ap.add_argument("--no-detect", action="store_true",
                    help="Skip Detectron2, query full DXF only")
    ap.add_argument("--model-cfg",     default="",
                    help="Detectron2 config yaml")
    ap.add_argument("--model-weights", default="",
                    help="Detectron2 model weights (.pth)")
    ap.add_argument("--score",   type=float, default=0.4,
                    help="Detectron2 score threshold")
    ap.add_argument("--dwg",     default="",
                    help="DWG input (will convert to DXF first)")
    args = ap.parse_args()

    result = run_pipeline(
        pdf_path       = args.pdf,
        dxf_path       = args.dxf,
        page           = args.page,
        out_dir        = args.out,
        dpi            = args.dpi,
        detect         = not args.no_detect,
        model_cfg      = args.model_cfg,
        model_weights  = args.model_weights,
        score_thresh   = args.score,
        dwg_path       = args.dwg,
    )

    print(f"\n{'─'*60}")
    print(f"Regions detected: {result['n_regions']}")
    for r in result["results"]:
        bd = r["blocks"]
        cats = bd["by_category"]
        print(f"  [{r['label']}]  texts={r['texts']['count']}  "
              f"blocks={bd['total']}  cats={list(cats.keys())}")


if __name__ == "__main__":
    main()
