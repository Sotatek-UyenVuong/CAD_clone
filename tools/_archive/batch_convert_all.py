#!/usr/bin/env python3
"""Batch convert all DWG → DXF, skipping files that already have a .dxf counterpart.

Usage:
  python3 batch_convert_all.py                 # converts all remaining DWG files
  python3 batch_convert_all.py --status        # show counts only
  python3 batch_convert_all.py --workers 4     # parallel workers (default: 3)
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

CAD_ROOT = Path("/home/sotatek/Documents/Uyen/cad/260316_新綱島スクエア竣工図CAD")
ODA_BIN  = "/usr/bin/ODAFileConverter"

# ── Helpers ────────────────────────────────────────────────────────────────────

_disp_lock = threading.Lock()
_disp_counter = 300  # start from :300 to avoid conflicts


def _next_display() -> int:
    global _disp_counter
    with _disp_lock:
        num = _disp_counter
        _disp_counter += 1
    return num


def _auto_click(display: str) -> None:
    """Background: dismiss ODA 'done' dialog via xdotool."""
    xdotool = shutil.which("xdotool")
    if not xdotool:
        return

    def _click():
        env = {**os.environ, "DISPLAY": display}
        for _ in range(30):
            time.sleep(1)
            r = subprocess.run(
                [xdotool, "search", "--name", "ODA"],
                capture_output=True, text=True, env=env,
            )
            wids = r.stdout.strip().split()
            if wids:
                for wid in wids:
                    subprocess.run([xdotool, "windowfocus", "--sync", wid],
                                   env=env, capture_output=True)
                    time.sleep(0.2)
                    subprocess.run([xdotool, "key", "Return"], env=env, capture_output=True)
                return

    threading.Thread(target=_click, daemon=True).start()


def convert_one(dwg: Path) -> tuple[Path, bool, str]:
    """Convert a single DWG file. Returns (dwg, success, message)."""
    dxf_out = dwg.with_suffix(".dxf")
    if dxf_out.exists():
        return dwg, True, "already exists"

    disp_num = _next_display()
    display  = f":{disp_num}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Copy DWG to temp dir (safe name)
        safe_name = "input.DWG"
        shutil.copy2(dwg, tmp_path / safe_name)

        # Start Xvfb
        xvfb = shutil.which("Xvfb")
        xvfb_proc = None
        if xvfb:
            lock = Path(f"/tmp/.X{disp_num}-lock")
            if lock.exists():
                disp_num = _next_display()
                display = f":{disp_num}"
            xvfb_proc = subprocess.Popen(
                [xvfb, display, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)

        try:
            _auto_click(display)
            env = {**os.environ, "DISPLAY": display}
            result = subprocess.run(
                [ODA_BIN, str(tmp_path), str(tmp_path), "ACAD2018", "DXF", "0", "1", "*.DWG"],
                capture_output=True, text=True, timeout=90, env=env,
            )
        except subprocess.TimeoutExpired:
            return dwg, False, "timeout"
        finally:
            if xvfb_proc:
                xvfb_proc.terminate()
                try:
                    xvfb_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    xvfb_proc.kill()

        # Find output DXF in temp dir
        dxf_candidates = list(tmp_path.glob("*.dxf")) + list(tmp_path.glob("*.DXF"))
        if dxf_candidates:
            shutil.copy2(dxf_candidates[0], dxf_out)
            return dwg, True, "ok"

        err = (result.stderr or result.stdout or "no output").strip()[:120]
        return dwg, False, err


# ── Main ────────────────────────────────────────────────────────────────────────

def find_pending(root: Path) -> list[Path]:
    pending: list[Path] = []
    for dwg in sorted(root.rglob("*.dwg")):
        if not dwg.with_suffix(".dxf").exists():
            pending.append(dwg)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DWG → DXF converter")
    parser.add_argument("--status",  action="store_true", help="Show counts and exit")
    parser.add_argument("--workers", type=int, default=3,  help="Parallel workers (default: 3)")
    args = parser.parse_args()

    all_dwg = list(CAD_ROOT.rglob("*.dwg"))
    all_dxf = list(CAD_ROOT.rglob("*.dxf"))
    pending = find_pending(CAD_ROOT)

    print(f"DWG total : {len(all_dwg)}")
    print(f"DXF done  : {len(all_dxf)}")
    print(f"Pending   : {len(pending)}")

    if args.status or not pending:
        if not pending:
            print("All files already converted!")
        return

    print(f"\nStarting conversion with {args.workers} workers …\n")

    ok = fail = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, dwg): dwg for dwg in pending}
        done = 0
        for future in as_completed(futures):
            dwg, success, msg = future.result()
            done += 1
            elapsed = time.time() - start
            eta_s   = (elapsed / done) * (len(pending) - done) if done < len(pending) else 0
            status  = "✅" if success else "❌"
            if success:
                ok += 1
            else:
                fail += 1
            print(
                f"[{done:4d}/{len(pending)}] {status} {dwg.name[:60]:<60}"
                f"  ({msg})  ETA {eta_s/60:.1f}min"
            )

    total_t = time.time() - start
    print(f"\n{'='*60}")
    print(f"Done in {total_t/60:.1f} min  — ✅ {ok} converted, ❌ {fail} failed")


if __name__ == "__main__":
    main()
