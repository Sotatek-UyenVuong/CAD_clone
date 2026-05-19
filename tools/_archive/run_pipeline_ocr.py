"""run_pipeline_ocr.py — Layout detect + DXF context + Marker OCR pipeline.

Steps:
  1. Detect layout boxes (table / text / title_block / diagram) from PNG.
  2. Suppress overlapping boxes + sort column-major + merge adjacent boxes.
  3. [Optional] If DXF provided: extract context for each box:
       - Drawing title (from title block INSERT)
       - Layer names present in the box region
       - Key texts near / inside the box
       - Geometry signature (entity type ratios → diagram type hint)
  4. Crop each region, send to Marker API → poll for markdown.
  5. Write combined *_ocr.md with DXF context prepended per box.

Usage:
  # Image-only (no DXF):
  python tools/run_pipeline_ocr.py <image.png>

  # With DXF context:
  python tools/run_pipeline_ocr.py <image.png> --dxf <file.dxf>
"""
from __future__ import annotations

import argparse
import io
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

# ── env ─────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
_MARKER_KEY = os.environ.get("MARKER_API_KEY", "")

_MARKER_URL = "https://www.datalab.to/api/v1/marker"

# ── detection weights (same as layout_pipeline) ───────────────────────────────
_HERE = Path(__file__).parent
WEIGHTS_DEFAULT = _HERE.parent / "layout_detect/models/checkpoints/cad_layout_v7_swapsplit/model_final.pth"

# label-id mapping used in training
LABEL_MAP = {0: "text", 1: "table", 2: "title_block", 3: "diagram", 4: "image"}


# ── DXF context extraction ───────────────────────────────────────────────────

# Layer-name keywords → diagram type hint
_LAYER_HINTS: list[tuple[list[str], str]] = [
    (["KUTAI", "区体"],          "工事区分表 (construction trade assignment matrix)"),
    (["FLOOR", "FL", "平面"],    "平面図 (floor plan)"),
    (["HATCH", "仕上"],          "仕上表 (finish schedule)"),
    (["KAIDAN", "階段"],         "階段詳細図 (stair detail)"),
    (["TOORISIN", "通芯", "芯"], "軸組図 / 通り芯 (structural grid)"),
    (["SETUBI", "設備"],         "設備図 (MEP drawing)"),
    (["KOZO", "構造"],           "構造図 (structural drawing)"),
    (["STAND", "断面"],          "断面図 (section)"),
    (["DETAIL", "詳細"],         "詳細図 (detail drawing)"),
]


def _layer_type_hint(layers: list[str]) -> str | None:
    """Return a diagram-type hint from layer names, or None."""
    combined = " ".join(layers).upper()
    for keywords, hint in _LAYER_HINTS:
        if any(kw.upper() in combined for kw in keywords):
            return hint
    return None


def _geometry_signature(counts: dict[str, int]) -> str:
    """Describe entity composition in plain text."""
    total = sum(counts.values()) or 1
    parts = []
    for etype, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        if pct >= 5:
            parts.append(f"{etype}×{cnt}({pct:.0f}%)")
    return ", ".join(parts)


def extract_dxf_context(dxf_path: Path,
                        img_shape: tuple[int, int],
                        boxes: list[dict],
                        transform_fn=None) -> list[dict]:
    """Extract DXF metadata for each detected box.

    Args:
        dxf_path:      Path to the DXF file.
        img_shape:     (height, width) of the rendered image.
        boxes:         List of detected boxes with 'px_box' key.
        transform_fn:  Optional callable (px_x1,px_y1,px_x2,py_y2) → (dxf_x1,dxf_y1,dxf_x2,dxf_y2).
                       If None, uses a simple proportional transform based on DXF extents.

    Returns:
        Same list with 'dxf_context' dict added to each box.
    """
    try:
        import ezdxf  # type: ignore
    except ImportError:
        return [{**b, "dxf_context": None} for b in boxes]

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:
        return [{**b, "dxf_context": None} for b in boxes]

    msp = doc.modelspace()

    # ── Drawing title from title block ──────────────────────────────────────
    drawing_title = dxf_path.stem  # fallback: filename stem
    for entity in msp.query("INSERT"):
        bname = entity.dxf.name
        if "枠" in bname or "FRAME" in bname.upper():
            # Look for WAKU-MOJI texts nearby
            ix, iy = entity.dxf.insert.x, entity.dxf.insert.y
            for e2 in msp.query("TEXT MTEXT"):
                try:
                    pos = e2.dxf.insert
                    if abs(pos.x - ix) < 2000 and abs(pos.y - iy) < 500:
                        txt = (e2.dxf.text if e2.dxftype() == "TEXT"
                               else e2.plain_text()).strip()
                        if txt and len(txt) > 2:
                            drawing_title = txt
                            break
                except Exception:
                    continue

    # ── Collect all entities with positions ─────────────────────────────────
    all_entities: list[dict] = []
    for e in msp:
        try:
            pos = e.dxf.insert if hasattr(e.dxf, "insert") else None
            if pos is None and hasattr(e.dxf, "start"):
                pos = e.dxf.start
            if pos is None:
                continue
            entry: dict = {
                "type": e.dxftype(),
                "layer": e.dxf.layer if hasattr(e.dxf, "layer") else "0",
                "x": float(pos.x),
                "y": float(pos.y),
                "text": "",
            }
            if e.dxftype() == "TEXT":
                entry["text"] = e.dxf.text or ""
            elif e.dxftype() == "MTEXT":
                entry["text"] = e.plain_text() or ""
            all_entities.append(entry)
        except Exception:
            continue

    # ── DXF extents for coordinate transform ────────────────────────────────
    xs = [en["x"] for en in all_entities if all_entities]
    ys = [en["y"] for en in all_entities if all_entities]
    if not xs:
        return [{**b, "dxf_context": None} for b in boxes]

    dxf_x0, dxf_x1 = min(xs), max(xs)
    dxf_y0, dxf_y1 = min(ys), max(ys)
    img_h, img_w = img_shape

    def px_to_dxf(px1: float, py1: float, px2: float, py2: float):
        """Simple proportional transform px → DXF coords."""
        if transform_fn:
            return transform_fn(px1, py1, px2, py2)
        dxf_w = max(dxf_x1 - dxf_x0, 1)
        dxf_h = max(dxf_y1 - dxf_y0, 1)
        # DXF Y increases upward; image Y increases downward
        x1d = dxf_x0 + (px1 / img_w) * dxf_w
        x2d = dxf_x0 + (px2 / img_w) * dxf_w
        # flip Y
        y2d = dxf_y0 + ((img_h - py1) / img_h) * dxf_h
        y1d = dxf_y0 + ((img_h - py2) / img_h) * dxf_h
        return x1d, y1d, x2d, y2d

    # ── Per-box context ──────────────────────────────────────────────────────
    result = []
    for box in boxes:
        bx1, by1, bx2, by2 = box["px_box"]
        dx1, dy1, dx2, dy2 = px_to_dxf(bx1, by1, bx2, by2)

        # Margin: 5% of box size
        mx = (dx2 - dx1) * 0.05
        my = (dy2 - dy1) * 0.05

        in_box = [
            en for en in all_entities
            if dx1 - mx <= en["x"] <= dx2 + mx
            and dy1 - my <= en["y"] <= dy2 + my
        ]

        layers_in_box = sorted(set(en["layer"] for en in in_box))
        type_counts = Counter(en["type"] for en in in_box)
        texts_in_box = [
            en["text"].strip() for en in in_box
            if en["text"].strip() and en["type"] in ("TEXT", "MTEXT")
        ]
        # Deduplicate while preserving order
        seen: set = set()
        unique_texts: list[str] = []
        for t in texts_in_box:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)

        diagram_hint = _layer_type_hint(layers_in_box)
        geo_sig = _geometry_signature(dict(type_counts))

        ctx: dict = {
            "drawing_title": drawing_title,
            "dxf_file": dxf_path.name,
            "layers": layers_in_box,
            "diagram_type_hint": diagram_hint,
            "geometry": geo_sig,
            "texts": unique_texts[:60],  # cap at 60 text items
        }
        result.append({**box, "dxf_context": ctx})

    return result


def format_dxf_context_md(ctx: dict | None, label: str) -> str:
    """Render DXF context as a markdown comment block for LLM consumption."""
    if not ctx:
        return ""
    lines = [
        "<!-- DXF context",
        f"  drawing : {ctx['drawing_title']}",
        f"  file    : {ctx['dxf_file']}",
        f"  region  : {label}",
    ]
    if ctx["diagram_type_hint"]:
        lines.append(f"  type    : {ctx['diagram_type_hint']}")
    if ctx["layers"]:
        lines.append(f"  layers  : {', '.join(ctx['layers'])}")
    if ctx["geometry"]:
        lines.append(f"  geometry: {ctx['geometry']}")
    if ctx["texts"]:
        lines.append(f"  texts   : {' | '.join(ctx['texts'][:20])}")
    lines.append("-->")
    return "\n".join(lines)


# ── Detectron2 detection ─────────────────────────────────────────────────────

def load_predictor(weights: Path, score_thr: float):
    """Load Detectron2 predictor (lazy import)."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.model_zoo import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


# ── box suppression helpers ──────────────────────────────────────────────────

def _box_area(b: tuple) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _contain_fraction(inner: tuple, outer: tuple) -> float:
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = _box_area(inner)
    return inter / area if area > 0 else 0.0


def _suppress(detections: list[dict],
              contain_thr: float = 0.90,
              iou_thr: float = 0.70) -> list[dict]:
    """Containment suppression + per-label NMS."""
    keep = list(detections)
    to_remove: set[int] = set()
    for i, d in enumerate(keep):
        if i in to_remove:
            continue
        for j, other in enumerate(keep):
            if i == j or j in to_remove:
                continue
            if _contain_fraction(d["px_box"], other["px_box"]) >= contain_thr:
                if d["score"] <= other["score"]:
                    to_remove.add(i)
                    break
    keep = [d for i, d in enumerate(keep) if i not in to_remove]

    by_label: dict[str, list[dict]] = {}
    for d in keep:
        by_label.setdefault(d["label"], []).append(d)
    result = []
    for label_dets in by_label.values():
        label_dets.sort(key=lambda d: -d["score"])
        suppressed: set[int] = set()
        for i, d in enumerate(label_dets):
            if i in suppressed:
                continue
            result.append(d)
            for j in range(i + 1, len(label_dets)):
                if _iou(d["px_box"], label_dets[j]["px_box"]) >= iou_thr:
                    suppressed.add(j)
    return result


# ── column merge ─────────────────────────────────────────────────────────────

def _merge_columnwise(detections: list[dict],
                      col_band_px: int,
                      y_gap_px: float = 120.0,
                      x_overlap_min: float = 0.35) -> list[dict]:
    """Merge consecutive text/table boxes in the same column into one box.

    Boxes must already be sorted column-major top-to-bottom.
    Merged label is 'table' if any constituent was a table, else 'text'.
    """
    if not detections:
        return []

    mergeable = {"text", "table"}

    def _col(d: dict) -> int:
        return int(d["px_box"][0] // max(1, col_band_px))

    def _x_overlap(a: tuple, b: tuple) -> float:
        inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))

    out: list[dict] = []
    cur = detections[0]
    cur_labels: set[str] = {cur["label"]}

    for nxt in detections[1:]:
        can_merge = (
            cur["label"] in mergeable
            and nxt["label"] in mergeable
            and _col(cur) == _col(nxt)
            and _x_overlap(cur["px_box"], nxt["px_box"]) >= x_overlap_min
        )
        if can_merge:
            gap = max(0.0, nxt["px_box"][1] - cur["px_box"][3])
            if gap <= y_gap_px:
                x1 = min(cur["px_box"][0], nxt["px_box"][0])
                y1 = min(cur["px_box"][1], nxt["px_box"][1])
                x2 = max(cur["px_box"][2], nxt["px_box"][2])
                y2 = max(cur["px_box"][3], nxt["px_box"][3])
                cur_labels.add(nxt["label"])
                cur = {
                    "label": "table" if "table" in cur_labels else "text",
                    "score": max(cur["score"], nxt["score"]),
                    "px_box": (x1, y1, x2, y2),
                }
                continue
        out.append(cur)
        cur = nxt
        cur_labels = {cur["label"]}

    out.append(cur)
    return out


def detect_boxes(img_bgr: np.ndarray, predictor,
                 target_labels: set[str],
                 contain_thr: float = 0.90,
                 iou_thr: float = 0.70,
                 merge_y_gap: float | None = None,
                 merge_x_overlap: float = 0.35) -> list[dict]:
    """Run detection → suppress → sort → merge.

    Returns list of {label, score, px_box}.
    """
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    raw_boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    raw: list[dict] = []
    for box, score, cls in zip(raw_boxes, scores, classes):
        label = LABEL_MAP.get(int(cls), "unknown")
        if label not in target_labels:
            continue
        x1, y1, x2, y2 = box.tolist()
        raw.append({"label": label, "score": float(score),
                    "px_box": (x1, y1, x2, y2)})

    kept = _suppress(raw, contain_thr=contain_thr, iou_thr=iou_thr)

    pw = img_bgr.shape[1]
    ph = img_bgr.shape[0]
    col_band = pw // 5
    kept.sort(key=lambda d: (int(d["px_box"][0] // col_band), d["px_box"][1]))

    y_gap = merge_y_gap if merge_y_gap is not None else max(80.0, ph * 0.03)
    kept = _merge_columnwise(kept,
                             col_band_px=col_band,
                             y_gap_px=y_gap,
                             x_overlap_min=merge_x_overlap)
    return kept


# ── Marker OCR ───────────────────────────────────────────────────────────────

def _bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def ocr_crop(img_bgr: np.ndarray,
             api_key: str,
             langs: str = "ja",
             poll_interval: float = 2.0,
             max_polls: int = 150) -> str:
    """Upload a cropped image to Marker API and return markdown.

    Submits the PNG bytes as multipart form upload (no public URL needed).
    Polls until status == 'complete' or timeout.
    """
    png_bytes = _bgr_to_png_bytes(img_bgr)
    headers = {"X-API-Key": api_key}

    payload = {
        "langs": langs,
        "force_ocr": "true",
        "format_lines": "false",
        "paginate": "false",
        "strip_existing_ocr": "false",
        "disable_image_extraction": "true",
        "disable_ocr_math": "true",
        "use_llm": "false",
        "mode": "fast",
        "output_format": "markdown",
        "skip_cache": "false",
    }

    files = {"file": ("crop.png", png_bytes, "image/png")}
    resp = requests.post(_MARKER_URL, data=payload, files=files,
                         headers=headers, timeout=60)
    resp.raise_for_status()
    resp_json = resp.json()

    if "request_check_url" not in resp_json:
        raise RuntimeError(f"Marker submit failed: {resp_json}")

    check_url = resp_json["request_check_url"]

    for _ in range(max_polls):
        time.sleep(poll_interval)
        poll = requests.get(check_url, headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        if data.get("status") == "complete":
            md = data.get("markdown") or ""
            return md.strip()
        if data.get("status") == "error":
            raise RuntimeError(f"Marker job error: {data}")

    raise TimeoutError(f"Marker job did not complete after {max_polls * poll_interval:.0f}s")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(img_path: Path,
        out_path: Path,
        dxf_path: Path | None = None,
        weights: Path = WEIGHTS_DEFAULT,
        score_thr: float = 0.5,
        labels: set[str] | None = None,
        pad: int = 8,
        save_crops: bool = False,
        contain_thr: float = 0.90,
        iou_thr: float = 0.70,
        merge_y_gap: float | None = None,
        merge_x_overlap: float = 0.35,
        langs: str = "ja") -> None:
    if not _MARKER_KEY:
        raise RuntimeError("MARKER_API_KEY not set in .env")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    h, w = img_bgr.shape[:2]

    target_labels = labels or {"table", "title_block", "text", "diagram"}

    print(f"Image: {img_path.name}  ({w}×{h}px)")
    print("Running layout detection …")
    predictor = load_predictor(weights, score_thr)
    boxes = detect_boxes(img_bgr, predictor, target_labels,
                         contain_thr=contain_thr, iou_thr=iou_thr,
                         merge_y_gap=merge_y_gap,
                         merge_x_overlap=merge_x_overlap)
    print(f"Detected {len(boxes)} boxes: " +
          ", ".join(f"{b['label']}({b['score']:.2f})" for b in boxes))

    # ── DXF context (optional) ───────────────────────────────────────────────
    if dxf_path and dxf_path.exists():
        print(f"Extracting DXF context from {dxf_path.name} …")
        boxes = extract_dxf_context(dxf_path, (h, w), boxes)
    else:
        boxes = [{**b, "dxf_context": None} for b in boxes]

    crops_dir = img_path.parent / f"{img_path.stem}_crops"
    if save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    dxf_info = f" | DXF: {dxf_path.name}" if dxf_path else ""
    output_lines: list[str] = [
        f"# {img_path.stem} — OCR output{dxf_info}\n",
        f"Source: `{img_path}`\n",
    ]

    for i, box in enumerate(boxes, 1):
        label = box["label"]
        score = box["score"]
        x1, y1, x2, y2 = (int(v) for v in box["px_box"])
        ctx = box.get("dxf_context")

        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        crop = img_bgr[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            print(f"  Box #{i} {label}: empty crop, skipped")
            continue

        if save_crops:
            cv2.imwrite(str(crops_dir / f"box{i:02d}_{label}.png"), crop)

        print(f"  Box #{i} {label} ({score:.2f}) {x2-x1}×{y2-y1}px … ",
              end="", flush=True)
        try:
            md = ocr_crop(crop, _MARKER_KEY, langs=langs)
            print("OK")
        except Exception as e:
            md = f"_OCR error: {e}_"
            print(f"ERROR: {e}")

        output_lines.append(f"## Box #{i} — {label} (score={score:.2f})")
        output_lines.append(f"<!-- px: ({x1},{y1})→({x2},{y2}) -->")
        # DXF context block (for LLM downstream)
        dxf_md = format_dxf_context_md(ctx, label)
        if dxf_md:
            output_lines.append(dxf_md)
        output_lines.append("")
        output_lines.append(md if md else "_（no content）_")
        output_lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nSaved → {out_path}")


# ── batch helper ──────────────────────────────────────────────────────────────

def run_batch(img_dir: Path,
              out_dir: Path,
              weights: Path = WEIGHTS_DEFAULT,
              score_thr: float = 0.5,
              labels: set[str] | None = None,
              pad: int = 8,
              glob: str = "*.png",
              langs: str = "ja") -> None:
    imgs = sorted(img_dir.glob(glob))
    print(f"Batch: {len(imgs)} images in {img_dir}")
    for img_path in imgs:
        out_path = out_dir / f"{img_path.stem}_ocr.md"
        try:
            run(img_path, out_path, weights=weights,
                score_thr=score_thr, labels=labels, pad=pad, langs=langs)
        except Exception as e:
            print(f"  FAILED {img_path.name}: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Layout detect → crop → Marker OCR → markdown")
    p.add_argument("image", help="Rendered page PNG (or dir for batch)")
    p.add_argument("--dxf", help="DXF file to extract context from (optional)")
    p.add_argument("--out", help="Output .md path (default: <stem>_ocr.md next to image)")
    p.add_argument("--score", type=float, default=0.5,
                   help="Detection score threshold (default: 0.5)")
    p.add_argument("--labels", default="table,title_block,text",
                   help="Comma-separated labels to process (default: table,title_block,text)")
    p.add_argument("--pad", type=int, default=8,
                   help="Pixel padding around each crop (default: 8)")
    p.add_argument("--save-crops", action="store_true",
                   help="Save individual crop PNGs alongside the output")
    p.add_argument("--weights", default=str(WEIGHTS_DEFAULT),
                   help="Detectron2 weights path")
    p.add_argument("--contain-thr", type=float, default=0.90,
                   help="Containment suppression threshold (default: 0.90)")
    p.add_argument("--iou-thr", type=float, default=0.70,
                   help="IoU NMS threshold per label (default: 0.70)")
    p.add_argument("--merge-y-gap", type=float, default=None,
                   help="Max vertical gap (px) to merge adjacent column boxes "
                        "(default: max(80, image_height×0.03))")
    p.add_argument("--merge-x-overlap", type=float, default=0.35,
                   help="Min horizontal overlap ratio for column merge (default: 0.35)")
    p.add_argument("--langs", default="ja",
                   help="OCR language hint for Marker (default: ja)")
    return p.parse_args()


def main() -> int:
    args = _parse()
    img_path = Path(args.image).expanduser().resolve()
    labels = {v.strip() for v in args.labels.split(",") if v.strip()}
    weights = Path(args.weights).expanduser().resolve()

    if img_path.is_dir():
        out_dir = (Path(args.out).expanduser().resolve()
                   if args.out else img_path)
        run_batch(img_path, out_dir, weights=weights,
                  score_thr=args.score, labels=labels, pad=args.pad,
                  langs=args.langs)
    else:
        out = (Path(args.out).expanduser().resolve()
               if args.out
               else img_path.with_name(f"{img_path.stem}_ocr.md"))
        dxf = Path(args.dxf).expanduser().resolve() if args.dxf else None
        run(img_path, out, dxf_path=dxf, weights=weights,
            score_thr=args.score, labels=labels, pad=args.pad,
            save_crops=args.save_crops,
            contain_thr=args.contain_thr, iou_thr=args.iou_thr,
            merge_y_gap=args.merge_y_gap,
            merge_x_overlap=args.merge_x_overlap,
            langs=args.langs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
