"""room_finder.py – xác định vị trí phòng từ file DXF.

Hai chế độ phát hiện phòng:
  A. MENSEKI mode (bản vẽ diện tích như D060-D062):
     Đọc polygon đóng từ layer MENSEKI / MENSEKI_専有 / 面積_専有 / 面積_共用.
     Tên phòng từ TEXT bên trong polygon. Diện tích bằng Shoelace formula.

  B. TEXT-label mode (bản vẽ mặt bằng chi tiết như D111-D114):
     Không có polygon MENSEKI → dùng TEXT "室名" làm tâm phòng.
     Gán INSERT entity vào phòng gần nhất (Voronoi proximity).
     Diện tích ước tính từ bounding box của các entity trong phòng.

CLI examples:
  python tools/room_finder.py D060.dxf list-rooms
  python tools/room_finder.py D060.dxf query 1560000 92000
  python tools/room_finder.py D060.dxf count-symbols D113.dxf
  python tools/room_finder.py D113.dxf list-rooms --text-mode
  python tools/room_finder.py D113.dxf count-symbols D113.dxf --text-mode \\
      --symbol-db symbol_db/symbols.json \\
      --block-index symbol_db/block_name_index.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import ezdxf

# ---------------------------------------------------------------------------
# Layer name patterns that encode room boundary polygons
# ---------------------------------------------------------------------------
_ROOM_LAYERS: tuple[str, ...] = (
    "MENSEKI",        # 面積 – general room area
    "MENSEKI_専有",   # 専有面積 – exclusive area
    "面積_専有",      # 住戸専有面積
    "面積_共用",      # 共用面積
    "HEYA",           # 部屋 – room (some drawings use this)
    "ROOM",
)

# Layer name patterns for room-name text
_NAME_LAYERS: tuple[str, ...] = (
    "MOJI1",          # common room-name layer in area drawings
    "MOJI4",
    "MOJI5",
    "WAKU-MOJI",
    "HEYA_MOJI",
    "ROOM_NAME",
    "文字_室名",
    "MOJI",
    "SETUBI-MOJI",
    "TEN-MOJI",
    "DR-MOJI",
)

# Layer name patterns for pre-computed area text labels
_AREA_TEXT_LAYERS: tuple[str, ...] = (
    "MENSEKI_text",
    "MENSEKI_TEXT",
    "AREA_TEXT",
)

# Minimum polygon size to consider as a room (mm²)
_MIN_AREA_MM2 = 0.1e6  # 0.1 m²


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _shoelace(pts: Sequence[tuple[float, float]]) -> float:
    """Signed area via Shoelace formula. Positive = CCW."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return s / 2.0


def _area_mm2(pts: Sequence[tuple[float, float]]) -> float:
    return abs(_shoelace(pts))


def _centroid(pts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _point_in_polygon(x: float, y: float, pts: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting algorithm."""
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Room:
    polygon: list[tuple[float, float]]  # DXF units (usually mm)
    layer: str
    name: str = ""
    area_m2: float = 0.0
    area_text: str = ""  # raw string from MENSEKI_text layer

    def centroid(self) -> tuple[float, float]:
        return _centroid(self.polygon)

    def contains(self, x: float, y: float) -> bool:
        return _point_in_polygon(x, y, self.polygon)

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> dict:
        cx, cy = self.centroid()
        return {
            "name": self.name,
            "layer": self.layer,
            "area_m2": round(self.area_m2, 4),
            "area_text": self.area_text,
            "centroid": [round(cx, 1), round(cy, 1)],
            "bbox": [round(v, 1) for v in self.bbox()],
            "polygon_pts": len(self.polygon),
        }


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

class RoomFinder:
    """Loads room polygons from a DXF file and supports spatial queries."""

    def __init__(self, dxf_path: str | Path, min_area_m2: float = 0.1) -> None:
        self.dxf_path = Path(dxf_path)
        self.min_area_m2 = min_area_m2
        self.rooms: list[Room] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        doc = ezdxf.readfile(str(self.dxf_path))
        msp = doc.modelspace()

        polygons = self._collect_polygons(msp)
        texts = self._collect_texts(msp)
        area_texts = self._collect_area_texts(msp)

        for layer, pts in polygons:
            area = _area_mm2(pts) / 1e6  # mm² → m²
            if area < self.min_area_m2:
                continue
            room = Room(polygon=pts, layer=layer, area_m2=round(area, 4))
            room.name = self._assign_name(room, texts)
            room.area_text = self._assign_area_text(room, area_texts)
            self.rooms.append(room)

        # Sort by area descending (largest rooms first)
        self.rooms.sort(key=lambda r: r.area_m2, reverse=True)

    def _collect_polygons(self, msp) -> list[tuple[str, list[tuple[float, float]]]]:
        """Returns (layer, pts) for closed LWPOLYLINE on room layers."""
        result: list[tuple[str, list[tuple[float, float]]]] = []

        for ent in msp.query("LWPOLYLINE"):
            lyr = ent.dxf.layer
            if not any(lyr.upper().startswith(k.upper()) for k in _ROOM_LAYERS):
                continue
            if not ent.is_closed and len(list(ent.get_points())) < 3:
                continue
            pts = [(p[0], p[1]) for p in ent.get_points()]
            if len(pts) >= 3:
                result.append((lyr, pts))

        # Fallback: HATCH with polyline boundary paths on room layers
        for ent in msp.query("HATCH"):
            lyr = ent.dxf.layer
            if not any(lyr.upper().startswith(k.upper()) for k in _ROOM_LAYERS):
                continue
            try:
                for path in ent.paths:
                    if hasattr(path, "vertices") and len(path.vertices) >= 3:
                        pts = [(v[0], v[1]) for v in path.vertices]
                        result.append((lyr, pts))
            except Exception:
                pass

        return result

    def _collect_texts(self, msp) -> list[tuple[float, float, str]]:
        """Returns (x, y, text) for room-name text entities."""
        result: list[tuple[float, float, str]] = []

        for ent in msp.query("TEXT MTEXT"):
            lyr = ent.dxf.layer
            is_name_layer = any(lyr.upper().startswith(k.upper()) for k in _NAME_LAYERS)
            # Also accept TEXT with short non-numeric content even on unknown layers
            try:
                if ent.dxftype() == "TEXT":
                    text = ent.dxf.text.strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
                else:
                    text = ent.plain_mtext().strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
            except Exception:
                continue

            if not text:
                continue
            if is_name_layer:
                result.append((x, y, text))
            elif len(text) <= 15 and not text.replace(".", "").replace("-", "").isnumeric():
                result.append((x, y, text))

        return result

    def _collect_area_texts(self, msp) -> list[tuple[float, float, str]]:
        """Returns (x, y, text) for pre-computed area label text entities."""
        result: list[tuple[float, float, str]] = []
        for ent in msp.query("TEXT MTEXT"):
            lyr = ent.dxf.layer
            if not any(lyr.upper().startswith(k.upper()) for k in _AREA_TEXT_LAYERS):
                continue
            try:
                if ent.dxftype() == "TEXT":
                    text = ent.dxf.text.strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
                else:
                    text = ent.plain_mtext().strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
            except Exception:
                continue
            if text:
                result.append((x, y, text))
        return result

    def _assign_name(
        self, room: Room, texts: list[tuple[float, float, str]]
    ) -> str:
        """Find the best room name: text inside polygon, or nearest to centroid."""
        inside = [(x, y, t) for x, y, t in texts if room.contains(x, y)]
        if inside:
            # Prefer shorter text (actual room name, not dimension annotations)
            inside.sort(key=lambda v: len(v[2]))
            return inside[0][2]
        # Fallback: nearest text within 10m (10000 mm)
        cx, cy = room.centroid()
        nearest: tuple[float, str] | None = None
        for x, y, t in texts:
            d = _dist2(cx, cy, x, y)
            if d < 10000**2:
                if nearest is None or d < nearest[0]:
                    nearest = (d, t)
        return nearest[1] if nearest else ""

    def _assign_area_text(
        self, room: Room, area_texts: list[tuple[float, float, str]]
    ) -> str:
        """Find pre-computed area label contained in or nearest to this room."""
        inside = [(x, y, t) for x, y, t in area_texts if room.contains(x, y)]
        if inside:
            return inside[0][2]
        cx, cy = room.centroid()
        nearest: tuple[float, str] | None = None
        for x, y, t in area_texts:
            d = _dist2(cx, cy, x, y)
            if d < 5000**2:
                if nearest is None or d < nearest[0]:
                    nearest = (d, t)
        return nearest[1] if nearest else ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_room(self, x: float, y: float) -> Room | None:
        """Return the smallest room that contains point (x, y)."""
        candidates = [r for r in self.rooms if r.contains(x, y)]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.area_m2)

    def rooms_for_bbox(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> list[Room]:
        """Return rooms that overlap with the given bounding box."""
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        result = []
        for r in self.rooms:
            rx0, ry0, rx1, ry1 = r.bbox()
            # Quick AABB check first
            if rx1 < xmin or rx0 > xmax or ry1 < ymin or ry0 > ymax:
                continue
            # Then precise centroid-in-polygon
            if r.contains(cx, cy) or _point_in_polygon(xmin, ymin, r.polygon):
                result.append(r)
        return result

    # ------------------------------------------------------------------
    # Symbol counting
    # ------------------------------------------------------------------

    def count_symbols(
        self,
        floor_dxf_path: str | Path,
        symbol_db_path: str | Path | None = None,
        block_index_path: str | Path | None = None,
    ) -> list[dict]:
        """Count INSERT symbols per room in floor_dxf_path.

        Returns list of:
          {room_name, area_m2, symbols: [{label, block_name, count}]}
        """
        floor_dxf_path = Path(floor_dxf_path)
        doc = ezdxf.readfile(str(floor_dxf_path))
        msp = doc.modelspace()

        # Load symbol_db if provided
        sym_lookup: dict[str, str] = {}  # block_name → label
        if block_index_path and symbol_db_path:
            with open(block_index_path) as f:
                block_index: dict[str, str] = json.load(f)
            with open(symbol_db_path) as f:
                symbols: dict[str, dict] = json.load(f)
            for bname, hash_ in block_index.items():
                if hash_ in symbols:
                    sym_lookup[bname] = symbols[hash_]["label"]

        # Bucket: room_idx → {block_name: count}
        room_counts: list[dict[str, int]] = [{} for _ in self.rooms]
        unmatched: dict[str, int] = {}

        for ent in msp.query("INSERT"):
            try:
                ix, iy = ent.dxf.insert.x, ent.dxf.insert.y
                bname = ent.dxf.name
            except Exception:
                continue
            room = self.find_room(ix, iy)
            if room is None:
                unmatched[bname] = unmatched.get(bname, 0) + 1
                continue
            idx = self.rooms.index(room)
            room_counts[idx][bname] = room_counts[idx].get(bname, 0) + 1

        results = []
        for i, room in enumerate(self.rooms):
            buckets = room_counts[i]
            if not buckets:
                continue
            syms_out = []
            for bname, cnt in sorted(buckets.items(), key=lambda x: -x[1]):
                label = sym_lookup.get(bname, bname)
                syms_out.append({"label": label, "block_name": bname, "count": cnt})
            results.append({
                "room_name": room.name or f"(layer:{room.layer})",
                "area_m2": room.area_m2,
                "centroid": list(room.centroid()),
                "symbols": syms_out,
            })

        if unmatched:
            results.append({
                "room_name": "(outside all rooms)",
                "area_m2": None,
                "centroid": None,
                "symbols": [
                    {"label": sym_lookup.get(b, b), "block_name": b, "count": c}
                    for b, c in sorted(unmatched.items(), key=lambda x: -x[1])
                ],
            })

        return results


# ---------------------------------------------------------------------------
# TEXT-label based room finder (for floor plans without MENSEKI layer)
# ---------------------------------------------------------------------------

# Layers to ignore as room names (dimensions, grid lines, notes)
_IGNORE_TEXT_LAYERS: tuple[str, ...] = (
    "SUNPOU",      # 寸法 – dimensions
    "KIJUN",       # 基準 – grid reference
    "Defpoints",
    "DEFPOINTS",
    "HOJO",        # auxiliary lines
    "45-SUNPOU",
    "H_TAKASA",    # height dimensions
    "H_TEXT",
    "H_HEN",
)

# Patterns to skip in TEXT content (not room names)
_SKIP_TEXT_PATTERNS: tuple[str, ...] = (
    r"^CH[\d=]",           # ceiling height: CH=2700, CH2500
    r"^\$?\\M\+[0-9A-F]{5}",  # raw MTEXT encoding artifacts
    r"^W-\d",              # wall type: W-3, W-5
    r"^\d+FL$",            # floor level: 2FL
    r"^[XY][AB]\d",        # grid: XA1, YB2
    r"^[+-]?\d+(\.\d+)?$", # pure numbers
    r"^(GL|FL|SL|CL)[=+\-]",  # level references
    r"^########",           # unreadable
    r"^[A-Z]{1,2}=\d",     # dimension codes: A=100
)

# Short non-numeric strings typically used as room names
_MAX_ROOM_NAME_LEN = 20


def _is_room_label(text: str) -> bool:
    """True if text looks like a room name rather than a dimension or code."""
    t = text.strip()
    if not t or len(t) > _MAX_ROOM_NAME_LEN:
        return False
    for pat in _SKIP_TEXT_PATTERNS:
        if re.match(pat, t):
            return False
    return True


@dataclass
class RoomLabel:
    """A room identified only by its name label position (TEXT mode)."""
    name: str
    x: float
    y: float
    layer: str
    symbols: dict[str, int] = field(default_factory=dict)  # block_name → count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "label_pos": [round(self.x, 1), round(self.y, 1)],
            "symbol_count": sum(self.symbols.values()),
        }


class RoomFinderText:
    """Room finder based on TEXT labels (Voronoi proximity assignment).

    Use this for detailed floor plan DXF files that have no explicit
    MENSEKI boundary polygons.
    """

    def __init__(self, dxf_path: str | Path) -> None:
        self.dxf_path = Path(dxf_path)
        self.labels: list[RoomLabel] = []
        self._load()

    def _load(self) -> None:
        doc = ezdxf.readfile(str(self.dxf_path))
        msp = doc.modelspace()
        seen: set[str] = set()

        for ent in msp.query("TEXT MTEXT"):
            lyr = ent.dxf.layer
            # Skip ignored layers
            if any(lyr.upper().startswith(k.upper()) for k in _IGNORE_TEXT_LAYERS):
                continue
            try:
                if ent.dxftype() == "TEXT":
                    text = ent.dxf.text.strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
                else:
                    text = ent.plain_mtext().strip()
                    x, y = ent.dxf.insert.x, ent.dxf.insert.y
            except Exception:
                continue

            if not _is_room_label(text):
                continue
            # Skip duplicates at nearly the same position
            key = f"{text}_{x:.0f}_{y:.0f}"
            if key in seen:
                continue
            seen.add(key)
            self.labels.append(RoomLabel(name=text, x=x, y=y, layer=lyr))

    def find_nearest(self, x: float, y: float) -> RoomLabel | None:
        """Return the label whose position is nearest to (x, y)."""
        if not self.labels:
            return None
        best: RoomLabel | None = None
        best_d = float("inf")
        for lbl in self.labels:
            d = _dist2(x, y, lbl.x, lbl.y)
            if d < best_d:
                best_d = d
                best = lbl
        return best

    def count_symbols(
        self,
        floor_dxf_path: str | Path,
        symbol_db_path: str | Path | None = None,
        block_index_path: str | Path | None = None,
        max_dist_mm: float = 15000.0,
    ) -> list[dict]:
        """Count INSERT symbols per room label in floor_dxf_path.

        max_dist_mm: maximum distance from INSERT to nearest label to be
                     considered as belonging to that room (in DXF units/mm).
        """
        floor_dxf_path = Path(floor_dxf_path)
        doc = ezdxf.readfile(str(floor_dxf_path))
        msp = doc.modelspace()

        sym_lookup: dict[str, str] = {}
        if block_index_path and symbol_db_path:
            with open(block_index_path) as f:
                block_index: dict[str, str] = json.load(f)
            with open(symbol_db_path) as f:
                symbols: dict[str, dict] = json.load(f)
            for bname, hash_ in block_index.items():
                if hash_ in symbols:
                    sym_lookup[bname] = symbols[hash_]["label"]

        room_syms: dict[int, dict[str, int]] = {i: {} for i in range(len(self.labels))}
        unmatched: dict[str, int] = {}
        max_d2 = max_dist_mm ** 2

        for ent in msp.query("INSERT"):
            try:
                ix, iy = ent.dxf.insert.x, ent.dxf.insert.y
                bname = ent.dxf.name
            except Exception:
                continue
            # Find nearest label within max_dist_mm
            best_idx: int | None = None
            best_d = float("inf")
            for i, lbl in enumerate(self.labels):
                d = _dist2(ix, iy, lbl.x, lbl.y)
                if d < best_d and d <= max_d2:
                    best_d = d
                    best_idx = i
            if best_idx is None:
                unmatched[bname] = unmatched.get(bname, 0) + 1
                continue
            room_syms[best_idx][bname] = room_syms[best_idx].get(bname, 0) + 1

        results = []
        for i, lbl in enumerate(self.labels):
            buckets = room_syms[i]
            if not buckets:
                continue
            syms_out = [
                {"label": sym_lookup.get(b, b), "block_name": b, "count": c}
                for b, c in sorted(buckets.items(), key=lambda x: -x[1])
            ]
            results.append({
                "room_name": lbl.name,
                "label_pos": [round(lbl.x, 1), round(lbl.y, 1)],
                "layer": lbl.layer,
                "symbols": syms_out,
            })

        if unmatched:
            results.append({
                "room_name": "(too far / outside)",
                "label_pos": None,
                "layer": None,
                "symbols": [
                    {"label": sym_lookup.get(b, b), "block_name": b, "count": c}
                    for b, c in sorted(unmatched.items(), key=lambda x: -x[1])
                ],
            })

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="room_finder – xác định vị trí phòng từ DXF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dxf", help="File DXF chứa polygon phòng (ví dụ D060.dxf)")
    p.add_argument("--min-area", type=float, default=0.1, metavar="M2",
                   help="Diện tích tối thiểu tính là phòng (m², mặc định 0.1)")
    p.add_argument("--text-mode", action="store_true",
                   help="Dùng TEXT label thay vì polygon MENSEKI (cho bản vẽ mặt bằng)")

    sub = p.add_subparsers(dest="cmd")

    # list-rooms
    lr = sub.add_parser("list-rooms", help="Liệt kê tất cả phòng tìm được")
    lr.add_argument("--json", action="store_true", help="Xuất JSON")

    # query
    q = sub.add_parser("query", help="Tìm phòng chứa điểm (x, y)")
    q.add_argument("x", type=float, help="Tọa độ X (DXF units)")
    q.add_argument("y", type=float, help="Tọa độ Y (DXF units)")

    # count-symbols
    cs = sub.add_parser("count-symbols",
                        help="Đếm symbol (INSERT) trong từng phòng")
    cs.add_argument("floor_dxf", help="File DXF mặt bằng chứa INSERT")
    cs.add_argument("--symbol-db", default="symbol_db/symbols.json",
                    metavar="PATH")
    cs.add_argument("--block-index", default="symbol_db/block_name_index.json",
                    metavar="PATH")
    cs.add_argument("--max-dist", type=float, default=15000.0, metavar="MM",
                    help="Khoảng cách tối đa (mm) gán symbol vào phòng (text-mode, default 15000)")
    cs.add_argument("--json", action="store_true", help="Xuất JSON")

    return p


def _fmt_rooms(rooms: list[Room], as_json: bool) -> None:
    if as_json:
        print(json.dumps([r.to_dict() for r in rooms], ensure_ascii=False, indent=2))
        return
    print(f"{'Room name':<30} {'Layer':<20} {'Area m²':>10} {'Area text':>12}  Centroid")
    print("-" * 95)
    for r in rooms:
        cx, cy = r.centroid()
        print(
            f"{r.name[:30]:<30} {r.layer[:20]:<20} {r.area_m2:>10.3f}"
            f" {r.area_text[:12]:>12}  ({cx:.0f}, {cy:.0f})"
        )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    root = Path(__file__).parent.parent
    dxf_dir = root / "dxf_output" / "竣工図（新綱島スクエア　建築意匠図）"

    dxf_path = Path(args.dxf)
    if not dxf_path.exists():
        dxf_path = dxf_dir / args.dxf
    if not dxf_path.exists():
        sys.exit(f"[ERROR] Cannot find DXF: {args.dxf}")

    text_mode: bool = getattr(args, "text_mode", False)

    if text_mode:
        print(f"[room_finder] TEXT mode – loading labels from: {dxf_path.name}", file=sys.stderr)
        finder_t = RoomFinderText(dxf_path)
        print(f"[room_finder] Found {len(finder_t.labels)} room labels", file=sys.stderr)

        if args.cmd == "list-rooms":
            as_json = getattr(args, "json", False)
            if as_json:
                print(json.dumps([l.to_dict() for l in finder_t.labels],
                                 ensure_ascii=False, indent=2))
            else:
                print(f"{'Room name':<30} {'Layer':<35}  Label position")
                print("-" * 80)
                for lbl in finder_t.labels:
                    print(f"{lbl.name[:30]:<30} {lbl.layer[:35]:<35}  ({lbl.x:.0f}, {lbl.y:.0f})")

        elif args.cmd == "query":
            lbl = finder_t.find_nearest(args.x, args.y)
            if lbl is None:
                print("No labels found")
            else:
                print(json.dumps(lbl.to_dict(), ensure_ascii=False, indent=2))

        elif args.cmd == "count-symbols":
            floor_path = Path(args.floor_dxf)
            if not floor_path.exists():
                floor_path = dxf_dir / args.floor_dxf
            if not floor_path.exists():
                sys.exit(f"[ERROR] Cannot find floor DXF: {args.floor_dxf}")

            sym_db = Path(args.symbol_db)
            blk_idx = Path(args.block_index)
            if not sym_db.is_absolute():
                sym_db = root / sym_db
            if not blk_idx.is_absolute():
                blk_idx = root / blk_idx

            results = finder_t.count_symbols(
                floor_path,
                symbol_db_path=sym_db if sym_db.exists() else None,
                block_index_path=blk_idx if blk_idx.exists() else None,
                max_dist_mm=getattr(args, "max_dist", 15000.0),
            )
            if getattr(args, "json", False):
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                for r in results:
                    pos = f"({r['label_pos'][0]:.0f}, {r['label_pos'][1]:.0f})" \
                          if r["label_pos"] else "—"
                    print(f"\n=== {r['room_name']} @ {pos} ===")
                    for s in r["symbols"]:
                        print(f"  {s['count']:>4}x  {s['label']}")
        return

    # --- MENSEKI polygon mode ---
    print(f"[room_finder] Loading rooms from: {dxf_path.name}", file=sys.stderr)
    finder = RoomFinder(dxf_path, min_area_m2=args.min_area)
    print(f"[room_finder] Found {len(finder.rooms)} rooms", file=sys.stderr)

    if args.cmd == "list-rooms":
        _fmt_rooms(finder.rooms, getattr(args, "json", False))

    elif args.cmd == "query":
        room = finder.find_room(args.x, args.y)
        if room is None:
            print(f"No room found at ({args.x}, {args.y})")
        else:
            print(json.dumps(room.to_dict(), ensure_ascii=False, indent=2))

    elif args.cmd == "count-symbols":
        floor_path = Path(args.floor_dxf)
        if not floor_path.exists():
            floor_path = dxf_dir / args.floor_dxf
        if not floor_path.exists():
            sys.exit(f"[ERROR] Cannot find floor DXF: {args.floor_dxf}")

        sym_db = Path(args.symbol_db)
        blk_idx = Path(args.block_index)
        if not sym_db.is_absolute():
            sym_db = root / sym_db
        if not blk_idx.is_absolute():
            blk_idx = root / blk_idx

        results = finder.count_symbols(
            floor_path,
            symbol_db_path=sym_db if sym_db.exists() else None,
            block_index_path=blk_idx if blk_idx.exists() else None,
        )

        if getattr(args, "json", False):
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                area_str = f"{r['area_m2']:.2f}m²" if r["area_m2"] is not None else "—"
                print(f"\n=== {r['room_name']} [{area_str}] ===")
                for s in r["symbols"]:
                    print(f"  {s['count']:>4}x  {s['label']}")


if __name__ == "__main__":
    main()
