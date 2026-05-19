"""build_unit_catalog.py
Build unit_room_catalog.json from apt_unit_* block definitions in D049.

Output: symbol_db/unit_room_catalog.json
  {
    "apt_unit_70A": {
      "label": "apt_unit_70A",
      "block_name": "70A_タイプ",
      "total_tatami": 42.7,
      "room_count": 12,
      "rooms": [
        {"name": "居間・食堂", "tatami": 11.9, "pos": [1258, 2684]},
        {"name": "洋室（1）",  "tatami": 6.0,  "pos": [7985, 2199]},
        ...
      ]
    },
    ...
  }
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import ezdxf

ROOT     = Path(__file__).resolve().parent.parent
DXF_PATH = ROOT / "dxf_output" / "竣工図（新綱島スクエア　建築意匠図）" / "D049_住戸専有面積算定図 - 1.dxf"
SYM_DB   = ROOT / "symbol_db" / "symbols.json"
OUT_PATH = ROOT / "symbol_db" / "unit_room_catalog.json"

# Short all-caps codes that are NOT room names
SKIP_LABELS = {
    "CL", "MB", "PS", "EPS", "DS", "SS", "IP", "UP", "DN",
    "SP", "ST", "EP", "SB", "DP", "HP", "VP", "MP", "SIC", "WIC", "UB",
}


def _jloc(x: float, y: float) -> list[float]:
    return [round(x, 1), round(y, 1)]


def extract_unit_rooms(blk) -> list[dict]:
    """Extract room names + tatami areas from a block definition."""
    names: list[dict] = []
    areas: list[dict] = []

    for e in blk:
        if e.dxftype() != "TEXT":
            continue
        t = e.dxf.text.strip()
        if not t or len(t) > 20:
            continue

        if e.dxf.layer == "MOJI1":
            if t in SKIP_LABELS:
                continue
            if t.isupper() and len(t) <= 3:
                continue
            names.append({"name": t, "x": e.dxf.insert.x, "y": e.dxf.insert.y})

        elif e.dxf.layer == "MOJI3":
            m = re.match(r"^(\d+\.?\d*)J$", t)
            if m:
                areas.append({"tatami": float(m.group(1)),
                               "x": e.dxf.insert.x, "y": e.dxf.insert.y})

    # Match each room name to its nearest area label (within 4 m = 4000 mm)
    rooms: list[dict] = []
    for nm in names:
        best_area: dict | None = None
        best_d = float("inf")
        for ar in areas:
            d = (nm["x"] - ar["x"]) ** 2 + (nm["y"] - ar["y"]) ** 2
            if d < best_d and d < 4000 ** 2:
                best_d = d
                best_area = ar
        room: dict = {"name": nm["name"], "pos": _jloc(nm["x"], nm["y"])}
        if best_area:
            room["tatami"] = best_area["tatami"]
        rooms.append(room)

    # Sort top-to-bottom, left-to-right (Y descending, X ascending)
    rooms.sort(key=lambda r: (-r["pos"][1], r["pos"][0]))
    return rooms


def main() -> None:
    print(f"Loading {DXF_PATH.name} …")
    doc = ezdxf.readfile(str(DXF_PATH))

    with open(SYM_DB, encoding="utf-8") as f:
        syms: dict = json.load(f)

    catalog: dict = {}
    missing: list[str] = []

    for info in sorted(syms.values(), key=lambda v: v["label"]):
        label: str = info["label"]
        if not label.startswith("apt_unit_"):
            continue
        block_name: str = info["block_names"][0] if info["block_names"] else ""
        if not block_name:
            missing.append(label)
            continue
        blk = doc.blocks.get(block_name)
        if not blk:
            missing.append(label)
            continue

        rooms = extract_unit_rooms(blk)
        total_tatami = round(sum(r.get("tatami", 0.0) for r in rooms), 1)

        catalog[label] = {
            "label": label,
            "block_name": block_name,
            "total_tatami": total_tatami,
            "room_count": len(rooms),
            "rooms": rooms,
        }

    OUT_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Written {len(catalog)} unit types → {OUT_PATH}")
    if missing:
        print(f"Missing blocks (not in D049): {missing}")

    # Preview
    print()
    for label, info in list(catalog.items())[:3]:
        print(f"=== {label}  ({info['total_tatami']}畳 total, {info['room_count']} rooms) ===")
        for r in info["rooms"]:
            a = f"{r['tatami']}畳" if "tatami" in r else "—"
            print(f"  {r['name']:<15} {a}")
        print()


if __name__ == "__main__":
    main()
