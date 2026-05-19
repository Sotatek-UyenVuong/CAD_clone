#!/usr/bin/env python3
"""
DWG → DXF Converter using ODA File Converter
=============================================
Requires: ODAFileConverter installed at /usr/bin/ODAFileConverter
          xvfb-run  (sudo apt install xvfb)

Usage:
  Single file : uv run python tools/dwg_to_dxf_converter.py drawing.dwg
  Batch folder: uv run python tools/dwg_to_dxf_converter.py /dwg_folder/ -o /output/
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ODA_BIN = "/usr/bin/ODAFileConverter"


def _find_xvfb() -> str | None:
    return shutil.which("xvfb-run")


def convert(dwg: Path, out_dir: Path) -> Path | None:
    if not dwg.exists():
        print(f"  ❌  Not found: {dwg}")
        return None
    if not Path(ODA_BIN).exists():
        print(f"  ❌  ODAFileConverter not found at {ODA_BIN}")
        print("       Install: sudo dpkg -i ODAFileConverter_*.deb")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Converting: {dwg.name}")

    xvfb = _find_xvfb()

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_in_path = Path(tmp_in)
        # Copy file to temp dir (handles non-ASCII filenames)
        shutil.copy2(dwg, tmp_in_path / (dwg.stem + ".DWG"))

        # ODA CLI: InputDir OutputDir OutputVersion OutputType Recurse Audit [filter]
        cmd = [ODA_BIN, str(tmp_in_path), str(out_dir),
               "ACAD2018", "DXF", "0", "1", "*.DWG"]

        if xvfb:
            cmd = [xvfb, "-a", "--"] + cmd

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}
            )
            if result.returncode != 0 and result.stderr:
                print(f"  ⚠  ODA stderr: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("  ❌  Conversion timed out")
            return None

    for ext in (".dxf", ".DXF"):
        dxf = out_dir / (dwg.stem + ext)
        if dxf.exists():
            print(f"  ✅  → {dxf}")
            return dxf

    print("  ❌  DXF output not found after conversion")
    return None


def batch_convert(input_dir: Path, out_dir: Path) -> tuple[int, int]:
    dwg_files = sorted(input_dir.rglob("*.dwg")) + sorted(input_dir.rglob("*.DWG"))
    if not dwg_files:
        print(f"  No .dwg files found in {input_dir}")
        return 0, 0

    print(f"\n  Found {len(dwg_files)} DWG file(s) in {input_dir}\n")
    ok = fail = 0
    for dwg in dwg_files:
        rel = dwg.relative_to(input_dir)
        target_dir = out_dir / rel.parent
        if convert(dwg, target_dir):
            ok += 1
        else:
            fail += 1

    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DWG → DXF converter using ODA File Converter"
    )
    parser.add_argument("input", help="DWG file or folder containing DWG files")
    parser.add_argument("-o", "--output", help="Output folder (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else None

    if input_path.is_dir():
        target = out_dir or input_path
        ok, fail = batch_convert(input_path, target)
        print(f"\n  Batch done — ✅ {ok} converted  ❌ {fail} failed")
        sys.exit(0 if fail == 0 else 1)

    elif input_path.suffix.lower() == ".dwg":
        target = out_dir or input_path.parent
        dxf = convert(input_path, target)
        sys.exit(0 if dxf else 1)

    else:
        print(f"  ❌  Expected a .dwg file or directory, got: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
