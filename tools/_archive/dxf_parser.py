#!/usr/bin/env python3
"""
DXF Parser — extract layers, blocks, text, dimensions, entity counts
Supports Japanese DXF files (ANSI_932 / Shift-JIS encoding)
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


# ── Encoding detection ─────────────────────────────────────────────────────────

def detect_encoding(path: Path) -> str:
    """
    Detect DXF encoding by reading actual content bytes.
    Note: ODA File Converter outputs UTF-8 even when $DWGCODEPAGE says ANSI_932,
    so we verify by trying UTF-8 strict on a large chunk before trusting the header.
    """
    codepage_map = {
        "ANSI_932":  "cp932",
        "ANSI_936":  "gbk",
        "ANSI_949":  "euc_kr",
        "ANSI_950":  "big5",
        "ANSI_1250": "cp1250",
        "ANSI_1251": "cp1251",
        "ANSI_1252": "cp1252",
        "UTF-8":     "utf-8",
    }
    try:
        with open(path, "rb") as f:
            raw = f.read()          # read full file for accurate detection
        if raw.startswith(b'\xef\xbb\xbf'):
            return "utf-8-sig"
        # Try UTF-8 strict on the whole file first
        try:
            raw.decode("utf-8", errors="strict")
            return "utf-8"
        except UnicodeDecodeError:
            pass
        # Fallback: trust $DWGCODEPAGE header
        header = raw[:4096].decode("ascii", errors="ignore")
        m = re.search(r'\$DWGCODEPAGE\s+\d+\s+(\S+)', header)
        if m:
            enc = codepage_map.get(m.group(1).strip())
            if enc:
                return enc
    except Exception:
        pass
    return "cp932"   # safest default for Japanese DWG files


# ── DXF Parser ─────────────────────────────────────────────────────────────────

class DXFParser:
    def __init__(self, dxf_path: str):
        self.path = Path(dxf_path)
        self.encoding = detect_encoding(self.path)
        self.layers: list[str] = []
        self.blocks: list[str] = []
        self.texts: list[str] = []
        # (text_value, x, y) — used for table reconstruction
        self.text_coords: list[tuple[str, float, float]] = []
        self.dimensions: list[str] = []
        self.areas: list[str] = []
        self.entities: defaultdict[str, int] = defaultdict(int)
        self._content = ""

    def parse(self) -> None:
        with open(self.path, encoding=self.encoding, errors="replace") as f:
            self._content = f.read()
        self._parse_layers()
        self._parse_blocks()
        self._parse_texts()
        self._parse_dimensions()
        self._parse_entities()

    # ── Layers ────────────────────────────────────────────────────────────────

    def _parse_layers(self) -> None:
        print("\n🎨 LAYERS (レイヤー):")
        print("-" * 70)

        # Correct parsing: find each 0\nLAYER entity inside the LAYER TABLE section
        # Structure: 0\nTABLE\n2\nLAYER\n...<0\nLAYER\n2\n<name>>...\n0\nENDTAB
        layers: set[str] = set()

        # Find the LAYER TABLE block
        m = re.search(
            r'  0\nTABLE\n  2\nLAYER\n(.*?)  0\nENDTAB',
            self._content, re.DOTALL
        )
        if m:
            table_block = m.group(1)
            # Each LAYER entry spans multiple lines; extract name (group code 2)
            # after AcDbLayerTableRecord to avoid picking up other code-2 values
            for entry in re.findall(r'  0\nLAYER\n(.*?)(?=  0\n(?:LAYER|ENDTAB))',
                                    table_block, re.DOTALL):
                # The layer name is the first group-code-2 value in the entry
                nm = re.search(r'  2\n([^\n]+)', entry)
                if nm:
                    name = nm.group(1).strip()
                    if name:
                        layers.add(name)

        self.layers = sorted(layers)
        if self.layers:
            for i, layer in enumerate(self.layers, 1):
                print(f"   {i:3d}. {layer}")
        else:
            print("   (No user-defined layers found — may be a simple cover sheet)")
        print(f"\n   ✅ Total layers: {len(self.layers)}")

    # ── Blocks ────────────────────────────────────────────────────────────────

    def _parse_blocks(self) -> None:
        print("\n📦 BLOCKS / SYMBOLS (ブロック):")
        print("-" * 70)
        m = re.search(r'BLOCKS\n(.*?)ENDSEC', self._content, re.DOTALL)
        if m:
            raw = re.findall(r'BLOCK\n.*?\n  2\n([^\n]+)', m.group(1), re.DOTALL)
            self.blocks = sorted(b for b in set(raw) if not b.startswith('*'))
            for i, block in enumerate(self.blocks, 1):
                print(f"   {i:3d}. {block}")
        print(f"\n   ✅ Total blocks: {len(self.blocks)}")

    # ── Texts ─────────────────────────────────────────────────────────────────

    def _parse_texts_with_coords(self) -> list[tuple[str, float, float]]:
        """Parse TEXT/MTEXT entities, returning (value, x, y) for each."""
        lines = self._content.splitlines()
        n = len(lines)
        result: list[tuple[str, float, float]] = []
        i = 0
        while i < n - 1:
            c = lines[i].strip()
            v = lines[i + 1].strip()
            if c == '0' and v in ('TEXT', 'MTEXT'):
                text_val: str | None = None
                x = y = 0.0
                i += 2
                while i < n - 1:
                    c = lines[i].strip()
                    v = lines[i + 1].strip() if i + 1 < n else ''
                    if c == '0':
                        break
                    if c == '1' and text_val is None:
                        text_val = v
                    elif c == '10':
                        try:
                            x = float(v)
                        except ValueError:
                            pass
                    elif c == '20':
                        try:
                            y = float(v)
                        except ValueError:
                            pass
                    i += 2
                if text_val and len(text_val.strip()) > 1:
                    result.append((text_val.strip(), x, y))
            else:
                i += 2
        return result

    def _parse_texts(self) -> None:
        print("\n📝 TEXT CONTENT (テキスト):")
        print("-" * 70)
        self.text_coords = self._parse_texts_with_coords()
        self.texts = [t[0] for t in self.text_coords]

        dim_labels, labels = [], []
        for t in self.texts[:60]:
            if re.search(r'\d[\d.,]*\s*(mm|m|m2|㎡|㎡)', t, re.IGNORECASE):
                dim_labels.append(t)
            else:
                labels.append(t)

        if dim_labels:
            print("\n   📏 Dimension labels:")
            for i, t in enumerate(dim_labels[:20], 1):
                print(f"      {i:2d}. {t}")

        if labels:
            print("\n   🏷  Labels / annotations:")
            for i, t in enumerate(labels[:20], 1):
                print(f"      {i:2d}. {t}")

        print(f"\n   ✅ Total text objects: {len(self.texts)}")

    # ── Dimensions ────────────────────────────────────────────────────────────

    def _parse_dimensions(self) -> None:
        print("\n📏 DIMENSIONS & AREAS (寸法・面積):")
        print("-" * 70)
        dim_objs = re.findall(r'\nDIMENSION\n.*?\n  1\n([^\n]+)', self._content, re.DOTALL)
        area_patterns = [
            r'(\d[\d,]*\.?\d*)\s*㎡',
            r'(\d[\d,]*\.?\d*)\s*m2',
            r'(\d[\d,]*\.?\d*)\s*m²',
        ]
        areas: set[str] = set()
        for pat in area_patterns:
            areas.update(re.findall(pat, self._content, re.IGNORECASE))
        self.areas = sorted(areas, key=lambda x: float(x.replace(',', '')))
        if self.areas:
            print("\n   📐 Areas found:")
            for i, a in enumerate(self.areas, 1):
                print(f"      {i:2d}. {a} ㎡")
        print(f"\n   ✅ Dimension objects: {len(dim_objs)}")
        print(f"   ✅ Area values found: {len(self.areas)}")

    # ── Entities ──────────────────────────────────────────────────────────────

    def _parse_entities(self) -> None:
        print("\n🔢 ENTITY COUNTS (エンティティ数):")
        print("-" * 70)
        entity_types = [
            "LINE", "CIRCLE", "ARC", "ELLIPSE",
            "INSERT", "TEXT", "MTEXT", "DIMENSION",
            "POLYLINE", "LWPOLYLINE", "SPLINE", "HATCH",
        ]
        for etype in entity_types:
            count = len(re.findall(rf'\n{etype}\n', self._content))
            if count:
                self.entities[etype] = count
                print(f"   • {etype:<15} {count:6d}")

        # Block reference breakdown
        insert_names = re.findall(r'\nINSERT\n.*?\n  2\n([^\n]+)', self._content, re.DOTALL)
        sym_counts: defaultdict[str, int] = defaultdict(int)
        for name in insert_names:
            sym_counts[name.strip()] += 1
        if sym_counts:
            print("\n   🚪 Block references (symbols):")
            for sym, cnt in sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f"      • {sym}: {cnt}×")

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"📄 File    : {self.path.name}")
        print(f"🔤 Encoding: {self.encoding}")
        print(f"📁 Layers  : {len(self.layers)}")
        print(f"📦 Blocks  : {len(self.blocks)}")
        print(f"📝 Texts   : {len(self.texts)}")
        print(f"📐 Areas   : {len(self.areas)}")
        print(f"🔢 Entities: {sum(self.entities.values())}")
        if self.areas:
            try:
                total = sum(float(a.replace(',', '')) for a in self.areas)
                print(f"📊 Total area: {total:,.2f} ㎡")
            except ValueError:
                pass
        print("=" * 70)

    # ── Table reconstruction ──────────────────────────────────────────────────

    _SCALE_RE = re.compile(r'^\d+/\s*[\d・/\s]+$')
    _NUM_RE   = re.compile(r'^\d{1,4}$')
    _HDR_RE   = re.compile(r'^(図\s*面\s*名\s*称|縮\s*尺|NO\.|記\s*事|月\s*日|設計NO\.|図番)$')

    def _reconstruct_drawing_table(self) -> list[dict[str, str]] | None:
        """Group text objects by Y-row and reconstruct drawing list rows.
        Uses X-proximity matching so each name pairs with its nearest scale.
        Returns list of {no, name, scale} sorted by (no, y-position), else None.
        """
        if not self.text_coords:
            return None

        # Sort top→bottom (Y descending), then left→right (X ascending)
        items = sorted(self.text_coords, key=lambda t: (-t[2], t[1]))

        # Group into Y-rows: each row's Y anchor is the first item's Y
        Y_TOL = 5.0
        rows: list[list[tuple[str, float, float]]] = []
        cur: list[tuple[str, float, float]] = []
        cur_y: float | None = None
        for item in items:
            if cur_y is None or abs(item[2] - cur_y) <= Y_TOL:
                cur.append(item)
                if cur_y is None:
                    cur_y = item[2]
            else:
                if cur:
                    rows.append(sorted(cur, key=lambda t: t[1]))
                cur = [item]
                cur_y = item[2]
        if cur:
            rows.append(sorted(cur, key=lambda t: t[1]))

        # Count scale-containing rows to decide if this is a drawing list
        scale_rows = sum(
            1 for row in rows
            if any(self._SCALE_RE.fullmatch(t[0].strip()) for t in row)
        )
        if scale_rows < 5:
            return None

        # Estimate X-width of the scale column by sampling all scale objects
        all_scale_xs = [t[1] for t in self.text_coords
                        if self._SCALE_RE.fullmatch(t[0].strip())]
        # Scale is typically a narrow column on the right of each name group.
        # We use a generous proximity: scale "belongs" to the name whose X is
        # closest to (scale_x - half_gap), where half_gap is estimated below.
        # In practice: for each scale, find the name with the largest X < scale_x.

        table: list[dict[str, str]] = []

        for row in rows:
            # Skip pure-header rows
            if all(self._HDR_RE.match(t[0].strip()) or
                   re.fullmatch(r'A\d:\d+/\s*\d+', t[0].strip()) for t in row):
                continue

            scales_in_row = [(idx, it) for idx, it in enumerate(row)
                             if self._SCALE_RE.fullmatch(it[0].strip())]
            nums_in_row   = [(idx, it) for idx, it in enumerate(row)
                             if self._NUM_RE.fullmatch(it[0].strip())]
            names_in_row  = [(idx, it) for idx, it in enumerate(row)
                             if not self._SCALE_RE.fullmatch(it[0].strip())
                             and not self._NUM_RE.fullmatch(it[0].strip())
                             and not re.fullmatch(r'A\d:\d+/\s*\d+', it[0].strip())
                             and not self._HDR_RE.match(it[0].strip())]

            if not names_in_row:
                continue

            # Pair each scale with the closest name to its LEFT (by X)
            paired_name_idxs: set[int] = set()
            scale_for_name: dict[int, str] = {}   # name_list_index → scale_str
            for _, scale_item in scales_in_row:
                sx = scale_item[1]
                # Names whose X < sx
                left_names = [(ni, ni_item) for ni, (_, ni_item) in enumerate(names_in_row)
                              if ni_item[1] < sx]
                if not left_names:
                    continue
                # Closest name to the left of the scale
                best_ni, _ = max(left_names, key=lambda p: names_in_row[p[0]][1][1])
                scale_for_name[best_ni] = scale_item[0]
                paired_name_idxs.add(best_ni)

            # Pair each number with the closest name to its RIGHT (by X)
            no_for_name: dict[int, str] = {}
            for _, no_item in nums_in_row:
                nx = no_item[1]
                right_names = [(ni, ni_item) for ni, (_, ni_item) in enumerate(names_in_row)
                               if ni_item[1] > nx]
                if not right_names:
                    continue
                best_ni, _ = min(right_names, key=lambda p: names_in_row[p[0]][1][1])
                if best_ni not in no_for_name:
                    no_for_name[best_ni] = no_item[0]

            for ni, (_, name_item) in enumerate(names_in_row):
                table.append({
                    "no":    no_for_name.get(ni, ""),
                    "name":  name_item[0],
                    "scale": scale_for_name.get(ni, ""),
                    "_y":    name_item[2],   # used for sorting unnumbered entries
                    "_x":    name_item[1],
                })

        if len(table) < 10:
            return None

        # X boundary of column 1: leftmost panel (approx < 200 DXF units from left)
        if table:
            min_x = min(r["_x"] for r in table)
            max_x = max(r["_x"] for r in table)
            col1_boundary = min_x + (max_x - min_x) * 0.15  # first ~15% of width

        # Title-block filter: collapse spaces before matching
        # Collapse spaces before checking (handles 事 務 所 → 事務所, etc.)
        _TITLEBLOCK_RE = re.compile(
            r'(TEL（|TEL\(|登録第\d|事務所（|一級建築士登録|株式会社|有限会社|'
            r'^\d{4}\.\d{2}$|縮尺.*A|図面名称|新築工事|完成図)'
        )
        named_entries: set[str] = {r["name"] for r in table if r["no"]}

        cleaned: list[dict[str, str]] = []
        for r in table:
            if not r["no"] and not r["scale"]:
                # Collapse spaces for title-block detection
                name_nospace = re.sub(r'\s+', '', r["name"])
                if _TITLEBLOCK_RE.search(name_nospace):
                    continue
                if r["name"] in named_entries:
                    continue
            cleaned.append(r)

        if len(cleaned) < 10:
            return None

        # Sort: numbered entries by number; unnumbered by (X, -Y) so they appear
        # in reading order (top of page first, left column before right column)
        def sort_key(r: dict[str, str]) -> tuple[int, float, float]:
            try:
                return (int(r["no"]), 0.0, 0.0)
            except ValueError:
                # Unnumbered: sort BEFORE numbered (key=0) by X then -Y (top first)
                return (0, r["_x"], -r["_y"])

        result = sorted(cleaned, key=sort_key)
        # Strip internal _y/_x from output
        for r in result:
            r.pop("_y", None)
            r.pop("_x", None)
        return result

    # ── Parse vertical LINE dividers ──────────────────────────────────────────

    def _find_vertical_dividers(self, min_span: float = 150.0) -> list[float]:
        """Return sorted X positions of significant vertical LINE entities.
        These are the column-dividing rules drawn in the DXF.
        """
        dividers: list[float] = []
        parts = self._content.split('\nLINE\n')
        for part in parts[1:]:
            seg = part.split('\n  0\n')[0]
            coords: dict[str, str] = {}
            for code, val in re.findall(r'\n\s*(\d+)\n([^\n]+)', seg):
                coords.setdefault(code, val)
            try:
                x1 = float(coords.get('10', 'nan'))
                y1 = float(coords.get('20', 'nan'))
                x2 = float(coords.get('11', 'nan'))
                y2 = float(coords.get('21', 'nan'))
            except ValueError:
                continue
            if abs(x1 - x2) < 3 and abs(y1 - y2) >= min_span:
                dividers.append((x1 + x2) / 2)

        # Cluster nearby X positions
        if not dividers:
            return []
        dividers.sort()
        clusters: list[float] = []
        cur = [dividers[0]]
        for x in dividers[1:]:
            if x - cur[-1] < 5:
                cur.append(x)
            else:
                clusters.append(sum(cur) / len(cur))
                cur = [x]
        clusters.append(sum(cur) / len(cur))
        return sorted(clusters)

    # ── Layout-aware spec document reconstruction ────────────────────────────

    def _reconstruct_spec_document(self) -> list[str] | None:
        """Detect multi-column spec document layout and return Markdown lines.

        Logic (mirrors PDF text extraction):
          1. Ignore outlier X positions (far-right title block area)
          2. Find the major vertical column gap via X clustering
          3. For each column: group text into Y-rows, join same-row items
          4. Detect section headers (short ALL-CAPS or 工事-style labels at left edge)
          5. Detect table header rows (many items at identical Y) → Markdown table
        """
        if not self.text_coords:
            return None

        # ── Step 1: use vertical LINE dividers as column boundaries ─────────────
        dividers = self._find_vertical_dividers(min_span=400.0)

        # Drop title-block outliers: biggest X gap in text positions
        xs_all = sorted(t[1] for t in self.text_coords)
        if len(xs_all) < 2:
            return None
        big_gap_idx = max(range(len(xs_all)-1), key=lambda i: xs_all[i+1]-xs_all[i])
        x_cutoff = xs_all[big_gap_idx]
        main = [(t, x, y) for t, x, y in self.text_coords if x <= x_cutoff]
        if len(main) < 50:
            return None

        x_min = min(x for _, x, _ in main)
        x_max = max(x for _, x, _ in main)

        # Use dividers in the main zone to build column boundaries
        main_dividers = sorted(d for d in dividers if x_min - 5 < d < x_max + 5)

        if main_dividers:
            boundaries = [x_min - 1.0] + main_dividers + [x_max + 1.0]
        else:
            # Fallback to gap-based detection
            all_xs = sorted(set(round(x, 1) for _, x, _ in main))
            x_span = x_max - x_min
            col_boundaries: list[float] = []
            for i in range(len(all_xs)-1):
                if all_xs[i+1] - all_xs[i] > x_span * 0.04:
                    col_boundaries.append((all_xs[i] + all_xs[i+1]) / 2)
            boundaries = [x_min - 1.0] + col_boundaries + [x_max + 1.0]

        columns: list[list[tuple[str, float, float]]] = []
        for i in range(len(boundaries)-1):
            lo, hi = boundaries[i], boundaries[i+1]
            col = [(t, x, y) for t, x, y in main if lo < x <= hi]
            if col:
                columns.append(col)

        Y_ROW_TOL = 4.0

        def _group_rows(col: list[tuple[str, float, float]]
                        ) -> list[list[tuple[str, float, float]]]:
            """Group items into Y-rows."""
            items = sorted(col, key=lambda t: (-t[2], t[1]))
            rows: list[list[tuple[str, float, float]]] = []
            cur: list[tuple[str, float, float]] = []
            cur_y: float | None = None
            for item in items:
                if cur_y is None or abs(item[2] - cur_y) <= Y_ROW_TOL:
                    cur.append(item)
                    if cur_y is None:
                        cur_y = item[2]
                else:
                    if cur:
                        rows.append(sorted(cur, key=lambda t: t[1]))
                    cur = [item]
                    cur_y = item[2]
            if cur:
                rows.append(sorted(cur, key=lambda t: t[1]))
            return rows

        _SECTION_HDR = re.compile(
            r'^(\d+\.\s*.{1,20}|[一-龥ぁ-んァ-ン]{2,12}(概要|仕様|工事|方針|区分|写真|書等|金額)?)\s*$'
        )
        _NUMBERED   = re.compile(r'^\s*(\d+[\.\．]|[①-⑳]|（\d+）|\(\d+\))\s+')

        # ── Split each main column into sub-columns by internal X gaps ───────────
        def _split_subcols(col: list[tuple[str, float, float]],
                           ) -> list[list[tuple[str, float, float]]]:
            """Split a column into sub-columns at *density cliff* boundaries.

            Strategy: find tight X clusters (many items within ±1.5 units of a
            single X) that are preceded by a gap >= 15 units from any previous
            item.  Each such cluster starts a new sub-column.  This handles the
            case where indented spec-text (spread x=263-407) sits next to a
            dense table column (72 items tightly at x=428), where the
            intra-cluster gap is too small (≈21 u) for simple gap detection.
            """
            if not col:
                return []
            col_x_min = min(x for _, x, _ in col)
            col_x_max = max(x for _, x, _ in col)
            if col_x_max - col_x_min < 30:
                return [col]

            CLUSTER_R     = 2.0   # items within ±2 u are in the same cluster
            MIN_CLUSTER_N = 15    # minimum items to call it a dense cluster
            MIN_GAP_BEFORE = 12.0 # gap from prev item before cluster = boundary

            xs_sorted = sorted(x for _, x, _ in col)

            # Build density clusters: find local peaks
            dense_cluster_starts: list[float] = []
            visited: set[int] = set()
            for i, x in enumerate(xs_sorted):
                if i in visited:
                    continue
                cluster = [j for j, xj in enumerate(xs_sorted)
                           if abs(xj - x) <= CLUSTER_R]
                if len(cluster) >= MIN_CLUSTER_N:
                    cx = sum(xs_sorted[j] for j in cluster) / len(cluster)
                    # Find approach gap: largest sorted X strictly before cluster start
                    cx_lo = min(xs_sorted[j] for j in cluster)
                    prev_x = next((xs_sorted[j] for j in reversed(range(len(xs_sorted)))
                                   if xs_sorted[j] < cx_lo - 0.1), None)
                    items_before = sum(1 for _, xj, _ in col if xj < cx_lo - 0.1)
                    if (prev_x is None or (cx_lo - prev_x) >= MIN_GAP_BEFORE) \
                            and items_before >= MIN_CLUSTER_N:
                        dense_cluster_starts.append(cx_lo)
                    visited.update(cluster)

            if not dense_cluster_starts:
                return [col]

            # Use approach midpoints as split boundaries
            splits: list[float] = []
            for cx_lo in sorted(dense_cluster_starts):
                # Midpoint between last item before this cluster and cluster start
                cx_lo_val = cx_lo
                prev_items = [x for _, x, _ in col if x < cx_lo_val - 0.1]
                if prev_items:
                    splits.append((max(prev_items) + cx_lo_val) / 2)

            if not splits:
                return [col]

            boundaries = [col_x_min - 1.0] + sorted(splits) + [col_x_max + 1.0]
            result: list[list[tuple[str, float, float]]] = []
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i+1]
                sub = [(t, x, y) for t, x, y in col if lo < x <= hi]
                if sub:
                    result.append(sub)
            return result if len(result) > 1 else [col]

        # Expand each main column into (main_col_idx, subcol_idx, items)
        expanded: list[tuple[int, int, list[tuple[str, float, float]]]] = []
        for ci, col in enumerate(columns):
            subcols = _split_subcols(col)
            for si, sub in enumerate(subcols):
                expanded.append((ci, si, sub))

        md_lines: list[str] = []
        # ── Emit columns sequentially: left-col top→bottom, then next col ────────
        all_rows_with_col: list[tuple[int, int, float, list[tuple[str, float, float]]]] = []
        for ci, si, sub in expanded:
            for row in _group_rows(sub):
                if row:
                    all_rows_with_col.append((ci, si, -row[0][2], row))
        all_rows_with_col.sort(key=lambda r: (r[0], r[1], r[2]))

        X_PROXIMITY = 30.0
        prev_was_table = False
        prev_key = (-1, -1)

        for ci_val, si_val, _, row in all_rows_with_col:
            cur_key = (ci_val, si_val)
            if cur_key != prev_key:
                # Sub-column header derived from its topmost meaningful text
                _, _, sub_items = next(
                    ((ci2, si2, sub) for ci2, si2, sub in expanded
                     if ci2 == ci_val and si2 == si_val), (ci_val, si_val, [])
                )
                sub_texts = sorted(sub_items, key=lambda t: -t[2])
                col_title = next((t[0].strip() for t in sub_texts
                                  if len(t[0].strip()) > 2), f"列 {ci_val+1}-{si_val+1}")
                md_lines += ["", f"---", f"", f"### {col_title[:30]}", ""]
                prev_key = cur_key
                prev_was_table = False
            items_in_row = [(t[0].strip(), t[1]) for t in row if t[0].strip()]
            if not items_in_row:
                continue

            # Split row into sub-groups by X proximity
            sub_groups: list[list[str]] = []
            cur_grp: list[str] = []
            prev_x: float | None = None
            for txt, x in sorted(items_in_row, key=lambda t: t[1]):
                if prev_x is None or x - prev_x <= X_PROXIMITY:
                    cur_grp.append(txt)
                else:
                    if cur_grp:
                        sub_groups.append(cur_grp)
                    cur_grp = [txt]
                prev_x = x
            if cur_grp:
                sub_groups.append(cur_grp)

            # Detect table row: ≥4 short items across sub-groups
            flat = [t for sg in sub_groups for t in sg]
            if len(flat) >= 4 and all(len(t) <= 12 for t in flat):
                if not prev_was_table:
                    md_lines.append("| " + " | ".join(flat) + " |")
                    md_lines.append("|" + "---|" * len(flat))
                    prev_was_table = True
                else:
                    md_lines.append("| " + " | ".join(flat) + " |")
                continue
            else:
                if prev_was_table:
                    md_lines.append("")
                prev_was_table = False

            # Emit each sub-group as a separate logical item
            for sg in sub_groups:
                if not sg:
                    continue
                joined = "　".join(sg)
                if len(sg) == 1:
                    t = sg[0]
                    if _SECTION_HDR.match(t) and len(t) <= 25:
                        md_lines += ["", f"**{t}**"]
                    elif _NUMBERED.match(t):
                        md_lines.append(f"- {t}")
                    else:
                        md_lines.append(f"  {t}")
                else:
                    first = sg[0]
                    if len(first) <= 15 and _SECTION_HDR.match(first):
                        md_lines += ["", f"**{first}**",
                                     f"  {'　'.join(sg[1:])}"]
                    else:
                        md_lines.append(f"  {joined}")

        return md_lines if len(md_lines) > 10 else None

    # ── Markdown export ───────────────────────────────────────────────────────

    def _try_gen_grid_md(self) -> str | None:
        """Delegate to gen_grid_md (ezdxf-based) for spec-sheet DXF files.

        Returns the generated Markdown string, or None if gen_grid_md is
        unavailable or the file is not a recognised spec-sheet.
        """
        try:
            import sys as _sys
            _tools_dir = str(Path(__file__).parent)
            if _tools_dir not in _sys.path:
                _sys.path.insert(0, _tools_dir)
            import gen_grid_md as _ggm  # type: ignore[import]

            # Only apply gen_grid_md to confirmed spec-sheet files to avoid
            # running heavy ezdxf processing on every DXF.
            _SPEC_MARKERS = {
                # D003 trade-assignment table headers
                "建築関係", "冷暖房換気設備関係", "電気設備関係",
                "昇降機設備関係", "給排水設備関係", "機械駐車設備関係",
                # D004 work-division diagram
                "工事区分",
                # General spec sheet keywords
                "特記仕様書", "設備特記仕様書",
            }
            text_set = set(self.texts)
            if not (text_set & _SPEC_MARKERS):
                return None

            return _ggm.generate(self.path)
        except Exception:
            return None

    def export_markdown(self, out_path: Path | None = None) -> Path:
        """Export analysis result to a Markdown file."""
        out_path = out_path or self.path.with_suffix(".md")

        total_area: float | None = None
        if self.areas:
            try:
                total_area = sum(float(a.replace(',', '')) for a in self.areas)
            except ValueError:
                pass

        lines: list[str] = []
        lines += [
            f"# CAD Drawing Analysis",
            f"",
            f"| Item | Value |",
            f"|------|-------|",
            f"| File | `{self.path.name}` |",
            f"| Encoding | {self.encoding} |",
            f"| Layers | {len(self.layers)} |",
            f"| Blocks/Symbols | {len(self.blocks)} |",
            f"| Text objects | {len(self.texts)} |",
            f"| Area values | {len(self.areas)} |",
            f"| Total entities | {sum(self.entities.values())} |",
        ]
        if total_area is not None:
            lines.append(f"| Total area | {total_area:,.2f} ㎡ |")

        # Layers
        lines += ["", "## Layers (レイヤー)", ""]
        if self.layers:
            for i, layer in enumerate(self.layers, 1):
                lines.append(f"{i}. `{layer}`")
        else:
            lines.append("_No layers found._")

        # Blocks — merged with usage count
        insert_names = re.findall(r'\nINSERT\n.*?\n  2\n([^\n]+)', self._content, re.DOTALL)
        from collections import Counter
        sym_counts: Counter[str] = Counter(n.strip() for n in insert_names)

        lines += ["", "## Blocks / Symbols (ブロック)", ""]
        if self.blocks:
            lines += ["| Block | Used |", "|-------|------|"]
            for block in self.blocks:
                count = sym_counts.get(block, 0)
                used = str(count) if count else "—"
                lines.append(f"| `{block}` | {used} |")
        else:
            lines.append("_No blocks found._")

        # ── Title Block (表題欄) ──────────────────────────────────────────────────
        tb_table = self._get_title_block_table()
        if tb_table:
            lines += ["", "## 表題欄 (Title Block)", "",
                      "| 項目 | 内容 |", "|------|------|"]
            for label, value in tb_table:
                lines.append(f"| {label} | {value.replace('|', '｜')} |")

        # For spec sheets (D003/D004/D005): delegate entirely to gen_grid_md
        # which uses ezdxf for accurate grid/text extraction.
        drawing_table = self._reconstruct_drawing_table()
        if not drawing_table:
            spec_md = self._try_gen_grid_md()
            if spec_md is not None:
                out_path.write_text(spec_md, encoding="utf-8")
                return out_path

        spec_layout = None if drawing_table else self._reconstruct_spec_document()

        lines += ["", "## Text Content (テキスト)", ""]
        if drawing_table:
            lines += [
                "### 図面リスト (Drawing List)",
                "",
                "| No. | 図面名称 | 縮尺(A3) |",
                "|-----|---------|---------|",
            ]
            for row in drawing_table:
                no    = row["no"].replace("|", "｜")
                name  = row["name"].replace("|", "｜")
                scale = row["scale"].replace("|", "｜")
                lines.append(f"| {no} | {name} | {scale} |")

        elif spec_layout:
            lines += spec_layout

        elif self.texts:
            def _is_noise(t: str) -> bool:
                s = t.strip()
                if re.fullmatch(r'\d{1,4}', s):
                    return True
                collapsed = re.sub(r'\s+', '', s)
                if len(collapsed) <= 2:
                    return True
                if re.fullmatch(r'A\d:\d+/\s*\d+', s):
                    return True
                if re.fullmatch(r'\d+/\s*[\d・/\s]+', s):
                    return True
                return False

            meaningful = [t for t in self.texts if not _is_noise(t)]

            kv_pairs: list[tuple[str, str]] = []
            standalone: list[str] = []
            used: set[int] = set()
            for i, t in enumerate(meaningful):
                if i in used:
                    continue
                next_t = meaningful[i + 1] if i + 1 < len(meaningful) else None
                is_label = (
                    next_t is not None
                    and len(re.sub(r'\s+', '', t)) <= 12
                    and not re.search(r'\d{4}', t)
                    and not re.search(r'[（(]', t)
                    and len(next_t) > len(t)
                )
                if is_label:
                    kv_pairs.append((t, next_t))
                    used.add(i); used.add(i + 1)
                else:
                    standalone.append(t)
                    used.add(i)

            if standalone:
                for text in standalone:
                    lines.append(f"- {text}")

            if kv_pairs:
                lines += ["", "### Key-Value Pairs (ラベル→値)",
                          "", "| Label | Value |", "|-------|-------|"]
                for k, v in kv_pairs:
                    lines.append(f"| {k.replace('|','｜')} | {v.replace('|','｜')} |")
        else:
            lines.append("_No text found._")

        # Areas
        if self.areas:
            lines += ["", "## Areas (面積)", ""]
            for i, area in enumerate(self.areas, 1):
                lines.append(f"{i}. {area} ㎡")
            if total_area is not None:
                lines += ["", f"**Total: {total_area:,.2f} ㎡**"]

        # Entity counts
        lines += ["", "## Entity Counts (エンティティ数)", ""]
        if self.entities:
            lines += ["| Entity | Count |", "|--------|-------|"]
            for etype, count in sorted(self.entities.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {etype} | {count} |")
        else:
            lines.append("_No entities found._")

        # (Symbol references already merged into Blocks section above)

        content = "\n".join(lines) + "\n"
        out_path.write_text(content, encoding="utf-8")
        return out_path


    def _parse_circles(self) -> list[tuple[float, float, float]]:
        """Return list of (cx, cy, radius) for all CIRCLE entities."""
        result: list[tuple[float, float, float]] = []
        parts = self._content.split('\nCIRCLE\n')
        for part in parts[1:]:
            seg = part.split('\n  0\n')[0]
            coords: dict[str, str] = {}
            for code, val in re.findall(r'\n\s*(\d+)\n([^\n]+)', seg):
                coords.setdefault(code, val)
            try:
                x = float(coords.get('10', 'nan'))
                y = float(coords.get('20', 'nan'))
                r = float(coords.get('40', '0'))
                result.append((x, y, r))
            except ValueError:
                pass
        return result

    def _get_title_block_table(self) -> list[tuple[str, str]]:
        """Return [(label, value), ...] for title block texts, sorted top-to-bottom.

        Labels are inferred from content patterns so the table self-documents
        without needing label texts to be present in the DXF.
        """
        _SCALE_HDR_RE = re.compile(r'^A\d:\d+/\s*\d+$')

        def _classify(s: str) -> str | None:
            sc = re.sub(r'\s+', '', s)
            if re.search(r'TEL[（(]', sc):
                return '住所・TEL'
            if re.search(r'事務所.{0,10}登録\s*第\s*\d', sc):
                return '事務所登録'
            if re.search(r'[一二]級建築士登録', sc):
                return '設計者'
            if re.fullmatch(r'\d{4}[./]\d{2}[./]\d{2}', s):
                return '確認日'
            if _SCALE_HDR_RE.fullmatch(s):
                return '縮尺'
            if len(sc) <= 20 and re.search(r'(株式会社|有限会社|合同会社)', sc):
                return '設計事務所'
            if re.fullmatch(r'(株式会社|有限会社|合同会社).{1,15}', sc):
                return '設計事務所'
            if re.fullmatch(r'.{1,15}(株式会社|有限会社|合同会社)', sc):
                return '設計事務所'
            return None

        seen: set[str] = set()
        # Collect with Y coordinate for spatial ordering (top → bottom)
        items: list[tuple[float, str, str]] = []   # (-y, label, value)
        for t, x, y in self.text_coords:
            s = t.strip()
            if not s or s in seen:
                continue
            label = _classify(s)
            if label:
                seen.add(s)
                items.append((-y, label, s))

        items.sort(key=lambda it: it[0])
        return [(lbl, val) for _, lbl, val in items]


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse as _ap
    p = _ap.ArgumentParser(description="DXF Parser — extract layers, text, entities")
    p.add_argument("dxf_file", help="Path to .dxf file")
    p.add_argument("--md", nargs="?", const="", metavar="OUTPUT.md",
                   help="Export Markdown report (default: same name as DXF)")
    args = p.parse_args()

    path = Path(args.dxf_file)
    if not path.exists():
        print(f"❌  File not found: {path}")
        sys.exit(1)

    print("=" * 70)
    print("🏗  DXF PARSER — CAD Drawing Analysis")
    print("=" * 70)

    parser = DXFParser(str(path))
    parser.parse()
    parser.print_summary()

    if args.md is not None:
        md_path = Path(args.md) if args.md else None
        out = parser.export_markdown(md_path)
        print(f"\n📄 Markdown saved: {out}")

    print("\n✨ Analysis complete!")


if __name__ == "__main__":
    main()
