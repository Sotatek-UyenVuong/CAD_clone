#!/usr/bin/env python3
"""collect_layers.py — Scan all DXF files, collect every INSERT block/symbol.

Output:
  symbols_all.csv    — one row per (block_name, dxf_file), with count + layers
  symbols_unique.md  — deduplicated block names across all files, sorted by frequency
  symbols_agg.json   — full aggregated data

Usage:
  python3 collect_layers.py                  # scan CAD_ROOT, output to cwd
  python3 collect_layers.py --root /path/to  # custom root
  python3 collect_layers.py --workers 8      # parallel (default: 6)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ezdxf

# ── Config ─────────────────────────────────────────────────────────────────────

CAD_ROOT = Path("/home/sotatek/Documents/Uyen/cad/260316_新綱島スクエア竣工図CAD")
OUT_DIR  = Path("/home/sotatek/Documents/Uyen/cad")


# ── Per-file worker (runs in subprocess) ──────────────────────────────────────

def _scan_file(dxf_path: str) -> dict:
    """Return INSERT/block stats for one DXF file. Called in worker process."""
    p = Path(dxf_path)
    try:
        doc = ezdxf.readfile(str(p))
        msp = doc.modelspace()

        # block_name → { "count": int, "layers": {layer: count} }
        block_stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "layers": defaultdict(int)}
        )

        for e in msp:
            if e.dxftype() != "INSERT":
                continue
            block_name = getattr(e.dxf, "name", "?")
            layer      = getattr(e.dxf, "layer", "0")
            block_stats[block_name]["count"] += 1
            block_stats[block_name]["layers"][layer] += 1

        # Freeze inner defaultdicts
        return {
            "file":   str(p),
            "stem":   p.stem,
            "folder": p.parent.name,
            "blocks": {
                k: {"count": v["count"], "layers": dict(v["layers"])}
                for k, v in block_stats.items()
            },
            "error": None,
        }
    except Exception as exc:
        return {
            "file": str(p), "stem": p.stem, "folder": p.parent.name,
            "blocks": {}, "error": str(exc),
        }


# ── Aggregator ─────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Build:
      by_block: block_name → {
          files, total_count,
          layers: {layer_name: total_count},
          example_files, folders
      }
    """
    by_block: dict[str, dict] = defaultdict(lambda: {
        "files":         0,
        "total_count":   0,
        "layers":        defaultdict(int),
        "example_files": [],
        "folders":       set(),
    })
    errors: list[str] = []

    for r in results:
        if r["error"]:
            errors.append(f"{r['stem']}: {r['error']}")
            continue
        for block_name, stats in r["blocks"].items():
            entry = by_block[block_name]
            entry["files"]       += 1
            entry["total_count"] += stats["count"]
            for layer, cnt in stats["layers"].items():
                entry["layers"][layer] += cnt
            if len(entry["example_files"]) < 3:
                entry["example_files"].append(r["stem"])
            entry["folders"].add(r["folder"])

    # Freeze
    for entry in by_block.values():
        entry["folders"] = sorted(entry["folders"])
        entry["layers"]  = dict(
            sorted(entry["layers"].items(), key=lambda x: -x[1])
        )

    return {"by_block": dict(by_block), "errors": errors}


# ── Writers ────────────────────────────────────────────────────────────────────

def write_csv(results: list[dict], out_path: Path) -> None:
    """One row per (block_name, file)."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_name", "file_stem", "folder", "count", "layers_json",
        ])
        for r in results:
            if r["error"]:
                continue
            for block_name, stats in sorted(r["blocks"].items()):
                writer.writerow([
                    block_name,
                    r["stem"],
                    r["folder"],
                    stats["count"],
                    json.dumps(stats["layers"], ensure_ascii=False),
                ])
    print(f"  CSV written: {out_path}")


def write_unique_md(agg: dict, out_path: Path) -> None:
    """Deduplicated block/symbol report sorted by file frequency."""
    by_block = agg["by_block"]
    errors   = agg["errors"]

    sorted_blocks = sorted(
        by_block.items(),
        key=lambda x: (-x[1]["files"], -x[1]["total_count"], x[0]),
    )

    lines: list[str] = [
        "# Symbol / Block Catalog — 全DXFファイル横断\n",
        f"> ユニークブロック数: **{len(sorted_blocks)}**  \n",
        f"> スキャン成功ファイル: **{sum(1 for r in sorted_blocks)}**  \n",
    ]
    if errors:
        lines.append(f"> エラー: {len(errors)} ファイル\n")

    lines += [
        "\n## ブロック一覧（出現ファイル数順）\n",
        "| ブロック名 | ファイル数 | 総挿入数 | 主なレイヤー | フォルダ | サンプルファイル |",
        "|-----------|-----------|---------|------------|---------|----------------|",
    ]

    for block_name, entry in sorted_blocks:
        top_layers = list(entry["layers"].items())[:3]
        top_str    = ", ".join(f"{l}({c})" for l, c in top_layers) if top_layers else "—"
        folders_str  = ", ".join(entry["folders"])[:60]
        examples_str = ", ".join(entry["example_files"][:2])[:60]
        safe_name    = block_name.replace("|", "｜")

        lines.append(
            f"| `{safe_name}` | {entry['files']} | {entry['total_count']:,} "
            f"| {top_str} | {folders_str} | {examples_str} |"
        )

    if errors:
        lines += ["\n## エラーファイル\n"]
        for e in errors:
            lines.append(f"- {e}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Markdown written: {out_path}")


def write_json(agg: dict, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(agg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  JSON written: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all INSERT blocks/symbols across DXF files")
    parser.add_argument("--root",    default=str(CAD_ROOT), help="Root directory to scan")
    parser.add_argument("--out",     default=str(OUT_DIR),  help="Output directory")
    parser.add_argument("--workers", type=int, default=6,   help="Parallel worker count")
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = sorted(root.rglob("*.dxf"))
    print(f"Found {len(dxf_files)} DXF files under {root}")
    print(f"Scanning INSERT entities only, {args.workers} workers …\n")

    results: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_scan_file, str(f)): f for f in dxf_files}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            status = "✅" if not r["error"] else f"❌ {r['error'][:40]}"
            print(f"  [{done:4d}/{len(dxf_files)}] {status}  {r['stem'][:60]}", end="\r")

    print(f"\n\nDone. Aggregating …")

    agg = aggregate(results)
    unique_blocks = len(agg["by_block"])
    print(f"  Unique block names found: {unique_blocks:,}")
    print(f"  Errors: {len(agg['errors'])}")

    write_csv(results,     out_dir / "symbols_all.csv")
    write_unique_md(agg,   out_dir / "symbols_unique.md")
    write_json(agg,        out_dir / "symbols_agg.json")

    # Quick top-20 summary
    by_block = agg["by_block"]
    top20 = sorted(by_block.items(), key=lambda x: -x[1]["total_count"])[:20]
    print("\n── Top 20 most-inserted blocks ─────────────────────────────────────")
    print(f"  {'Block name':<50} {'Files':>6}  {'Inserts':>9}")
    print(f"  {'-'*50} {'------':>6}  {'---------':>9}")
    for name, entry in top20:
        print(f"  {name:<50} {entry['files']:>6,}  {entry['total_count']:>9,}")


if __name__ == "__main__":
    main()
