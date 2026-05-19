from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_labels(raw: str) -> set[str]:
    labels = {v.strip() for v in raw.split(",") if v.strip()}
    return labels or {"table", "title_block", "text", "diagram"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run layout pipeline for any DXF/image and write *_pipeline.md output."
    )
    p.add_argument("dxf", help="Path to input DXF file")
    p.add_argument("image", help="Path to rendered page image (PNG/JPG)")
    p.add_argument(
        "--out",
        help="Output markdown path (default: <dxf_stem>_pipeline.md next to DXF)",
    )
    p.add_argument(
        "--score",
        type=float,
        default=0.5,
        help="Detection score threshold (default: 0.5)",
    )
    p.add_argument(
        "--labels",
        default="table,title_block,text,diagram",
        help="Comma-separated labels to keep",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    base_dir = Path(__file__).resolve().parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from layout_pipeline import run_pipeline

    dxf_path = Path(args.dxf).expanduser().resolve()
    img_path = Path(args.image).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dxf_path.with_name(f"{dxf_path.stem}_pipeline.md")
    )

    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF not found: {dxf_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    target_labels = _parse_labels(args.labels)
    results = run_pipeline(
        dxf_path,
        img_path,
        score_thr=args.score,
        target_labels=target_labels,
    )

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = [int(v) for v in r["px_box"]]
        lines.append(f'## Box #{i} — {r["label"]} (score={r["score"]:.2f})')
        lines.append(f"<!-- px: ({x1},{y1})→({x2},{y2}) -->")
        lines.append("")
        lines.append(r["markdown"] or "_（no content）_")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Boxes: {len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
