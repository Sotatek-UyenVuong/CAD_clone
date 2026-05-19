#!/usr/bin/env python3
"""cad_analyzer.py — Universal DXF architectural drawing analyzer.

Extracts structured information from any DXF file:
  - 図面情報         Drawing metadata (title block)
  - レイヤー分類     Layer catalog with semantic classification
  - 部屋・室名リスト Room names and areas
  - 建具・設備集計   Door / window / equipment counts
  - 閉合境界         Closed polyline boundaries (rooms)
  - 寸法・高さ情報   Dimension values
  - 注記・仕様       Annotations and technical notes
  - ブロック一覧     All block (symbol) types used

Usage:
  python3 cad_analyzer.py <file.dxf>
  python3 cad_analyzer.py <file.dxf> -o output.md
  python3 cad_analyzer.py <file.dxf> --layers-only   # just dump layer catalog
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
from ezdxf.document import Drawing


# ─────────────────────────────────────────────────────────────────────────────
# Layer semantic classification — loaded from layer_categories.json
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORIES_FILE = Path(__file__).with_name("layer_categories.json")


def _load_categories() -> dict[str, tuple[str, list[str]]]:
    """Load layer categories from JSON; fall back to empty dict on failure."""
    try:
        raw: dict[str, dict] = json.loads(_CATEGORIES_FILE.read_text(encoding="utf-8"))
        return {key: (v["label"], v["keywords"]) for key, v in raw.items()}
    except Exception as exc:
        print(f"[cad_analyzer] WARNING: could not load {_CATEGORIES_FILE}: {exc}", file=sys.stderr)
        return {}


LAYER_CATEGORIES: dict[str, tuple[str, list[str]]] = _load_categories()


def classify_layer(layer_name: str) -> str:
    """Return the best-matching semantic category for a layer name (longest keyword match wins)."""
    up = layer_name.upper()
    best_cat = "other"
    best_score = 0
    for cat, (_, keywords) in LAYER_CATEGORIES.items():
        for kw in keywords:
            if kw.upper() in up:
                score = len(kw)
                if score > best_score:
                    best_score = score
                    best_cat = cat
    return best_cat


def layer_label(cat: str) -> str:
    return LAYER_CATEGORIES.get(cat, ("その他", []))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextItem:
    text: str
    x: float
    y: float
    height: float
    layer: str
    category: str = ""


@dataclass
class InsertItem:
    block_name: str
    x: float
    y: float
    layer: str
    category: str = ""


@dataclass
class BoundaryInfo:
    points: list[tuple[float, float]] = field(default_factory=list)
    area_units: float = 0.0   # in drawing units²
    layer: str = ""
    category: str = ""


@dataclass
class HatchInfo:
    area_units: float = 0.0   # area of outermost boundary in drawing units²
    layer: str = ""
    category: str = ""
    pattern: str = ""         # hatch pattern name (e.g. SOLID, ANSI31)


@dataclass
class PerPageStats:
    name: str = ""            # layout name ("Model", "Layout1", ...)
    is_model: bool = False
    insert_count: int = 0
    inserts_by_cat: dict = field(default_factory=dict)   # {cat: count}
    hatch_count: int = 0
    hatch_area_by_cat: dict = field(default_factory=dict) # {cat: area_units}
    entity_totals: dict = field(default_factory=dict)    # {entity_type: count}


# ─────────────────────────────────────────────────────────────────────────────
# Entity extractors
# ─────────────────────────────────────────────────────────────────────────────

def extract_texts(msp) -> list[TextItem]:
    """Extract all TEXT and MTEXT entities from modelspace."""
    items: list[TextItem] = []
    for e in msp:
        t = ""
        x = y = 0.0
        h = 2.5
        layer = getattr(e.dxf, "layer", "0")
        if e.dxftype() == "TEXT":
            t = (e.dxf.text or "").strip()
            ins = e.dxf.insert
            x, y = ins.x, ins.y
            h = getattr(e.dxf, "height", 2.5) or 2.5
        elif e.dxftype() == "MTEXT":
            try:
                t = e.plain_text().strip()
            except Exception:
                t = ""
            ins = e.dxf.insert
            x, y = ins.x, ins.y
            h = getattr(e.dxf, "char_height", 2.5) or 2.5
        else:
            continue
        if t:
            cat = classify_layer(layer)
            items.append(TextItem(text=t, x=x, y=y, height=h, layer=layer, category=cat))
    return items


def extract_inserts(msp) -> list[InsertItem]:
    """Extract all INSERT (block reference) entities from modelspace."""
    items: list[InsertItem] = []
    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        layer = getattr(e.dxf, "layer", "0")
        ins = e.dxf.insert
        cat = classify_layer(layer)
        items.append(InsertItem(
            block_name=e.dxf.name,
            x=ins.x, y=ins.y,
            layer=layer, category=cat,
        ))
    return items


def extract_closed_polylines(msp) -> list[BoundaryInfo]:
    """Extract closed LWPOLYLINE entities as potential room/space boundaries."""
    boundaries: list[BoundaryInfo] = []
    for e in msp:
        if e.dxftype() != "LWPOLYLINE":
            continue
        if not e.closed:
            continue
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) < 3:
            continue
        area = _polygon_area(pts)
        if area < 1.0:   # skip noise
            continue
        layer = getattr(e.dxf, "layer", "0")
        boundaries.append(BoundaryInfo(
            points=pts,
            area_units=area,
            layer=layer,
            category=classify_layer(layer),
        ))
    return boundaries


def extract_layer_entity_stats(msp) -> dict[str, dict[str, int]]:
    """Count entity types per layer → {layer_name: {entity_type: count}}."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in msp:
        layer = getattr(e.dxf, "layer", "0")
        stats[layer][e.dxftype()] += 1
    return {k: dict(v) for k, v in stats.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace formula — returns area in drawing units²."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _point_in_polygon(px: float, py: float, pts: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors
# ─────────────────────────────────────────────────────────────────────────────

AREA_RE    = re.compile(r"(\d[\d,]*\.?\d*)\s*[㎡m²]", re.UNICODE)
NUM_ONLY   = re.compile(r"^[\d.,\s]+$")
CODE_RE    = re.compile(r"^[A-Z]{1,4}[-\d]*$")   # short codes like "GL", "FL", "EV"

# Room-purpose keyword groupings (Japanese)
ROOM_PURPOSES: dict[str, list[str]] = {
    "居室・リビング": ["居室", "居間", "リビング", "LDK", "LD ", "DK", " L ", "ダイニング"],
    "寝室・洋室・和室": ["寝室", "洋室", "和室", "子供室", "主寝室", "ベッドルーム"],
    "水廻り": ["浴室", "洗面", "トイレ", "WC", "脱衣", "洗濯", "洗い場", "ユニットバス"],
    "収納": ["収納", "納戸", "クロゼット", "押入", "物入", "倉庫", "パントリー"],
    "廊下・ホール・玄関": ["廊下", "ホール", "玄関", "エントランス", "ロビー"],
    "共用部・設備室": ["EV ", "エレベーター", "階段", "機械室", "電気室", "PS", "DS", "ゴミ"],
    "共有施設": ["集会室", "管理", "ラウンジ", "サービス", "交流", "キッズ", "フィットネス"],
    "駐車・駐輪": ["駐車", "駐輪", "車庫", "ガレージ", "バイク"],
    "店舗・テナント": ["店舗", "テナント", "事務所", "SHOP", "オフィス"],
    "バルコニー・屋外": ["バルコニー", "テラス", "屋上", "屋外", "外廊下", "ポーチ"],
}


def classify_room_purpose(name: str) -> str:
    for purpose, keywords in ROOM_PURPOSES.items():
        if any(kw in name for kw in keywords):
            return purpose
    return "その他"


def extract_room_candidates(texts: list[TextItem]) -> list[TextItem]:
    """Filter text items that look like room names."""
    rooms: list[TextItem] = []
    for t in texts:
        txt = t.text
        # Skip short, numeric, code-like, or GL/FL/etc labels
        if len(txt) < 2:
            continue
        if NUM_ONLY.match(txt):
            continue
        if CODE_RE.match(txt) and len(txt) <= 3:
            continue
        if re.match(r"^(GL|FL|SL|CH|PH|B\.)[\d+\-]", txt):
            continue
        if re.search(r"[A-Z0-9]FL[+\-]", txt):
            continue
        if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩]", txt):
            continue
        # Skip very tall text (likely titles / section headers)
        if t.height > 15:
            continue
        # Prefer text on room or text layers
        if t.category in ("room", "text", "finish", "other", "floor", "ceiling"):
            rooms.append(t)
    return rooms


def pair_rooms_with_areas(room_texts: list[TextItem], all_texts: list[TextItem]) -> list[dict]:
    """Associate each room candidate with the nearest area text (if within radius)."""
    area_texts = [t for t in all_texts if AREA_RE.search(t.text)]
    result: list[dict] = []
    for r in room_texts:
        best_dist = 80.0   # search radius in drawing units
        best_area_text = ""
        best_area_m2 = 0.0
        for a in area_texts:
            dist = math.hypot(a.x - r.x, a.y - r.y)
            if dist < best_dist:
                best_dist = dist
                m = AREA_RE.search(a.text)
                if m:
                    best_area_text = a.text
                    try:
                        best_area_m2 = float(m.group(1).replace(",", ""))
                    except ValueError:
                        pass
        result.append({
            "name": r.text,
            "area_text": best_area_text,
            "area_m2": best_area_m2,
            "x": r.x, "y": r.y,
            "layer": r.layer,
            "purpose": classify_room_purpose(r.text),
        })
    return result


# Block name → semantic type mapping
BLOCK_SEMANTIC: list[tuple[str, re.Pattern]] = [
    ("ドア",         re.compile(r"(DOOR|扉|SD[\d-]|FD[\d-]|AD[\d-]|WD[\d-]|D[-_]\d)", re.I)),
    ("窓",           re.compile(r"(WIN|WINDOW|MADO|窓|SASH|サッシ|W[-_]\d)", re.I)),
    ("エレベーター", re.compile(r"(ELV|ELEV|エレベーター|EV[-_])", re.I)),
    ("階段",         re.compile(r"(STAIR|KAIDAN|階段)", re.I)),
    ("便器",         re.compile(r"(便器|TOILET|WC[\d]|便所|BENKI)", re.I)),
    ("洗面器",       re.compile(r"(洗面|LAVATORY|LAVAT|LAV-|WASH)", re.I)),
    ("浴槽・ユニットバス", re.compile(r"(浴槽|BATH|バス|UB[-_])", re.I)),
    ("空調機器",     re.compile(r"(AHU|FCU|空調|HVAC|AC[-_]|FAN)", re.I)),
    ("消火設備",     re.compile(r"(消火|FIRE|SPRINK|スプリンク)", re.I)),
    ("照明器具",     re.compile(r"(照明|LIGHT|LAMP|ライト)", re.I)),
    ("駐車スペース", re.compile(r"(CAR|PARKING|駐車|SPACE)", re.I)),
    ("家具",         re.compile(r"(FURN|家具|TABLE|CHAIR|DESK|SOFA)", re.I)),
    ("厨房機器",     re.compile(r"(KITCHEN|キッチン|厨房|SINK|流し|コンロ)", re.I)),
    ("手すり・フェンス", re.compile(r"(HANDRAIL|手すり|FENCE|フェンス|RAILING)", re.I)),
]


def classify_block(block_name: str) -> str:
    for label, pat in BLOCK_SEMANTIC:
        if pat.search(block_name):
            return label
    return ""


def summarize_blocks(inserts: list[InsertItem]) -> dict[str, dict]:
    """Group inserts by semantic type (block-name regex) and block name."""
    by_semantic: dict[str, list[InsertItem]] = defaultdict(list)
    by_name: dict[str, int] = defaultdict(int)
    for ins in inserts:
        sem = classify_block(ins.block_name)
        if sem:
            by_semantic[sem].append(ins)
        by_name[ins.block_name] += 1

    return {
        "by_semantic": {k: len(v) for k, v in by_semantic.items()},
        "by_name": dict(sorted(by_name.items(), key=lambda x: -x[1])),
    }


def summarize_inserts_by_category(inserts: list[InsertItem]) -> dict[str, dict]:
    """Group INSERT entities by their layer's semantic category.

    Returns:
        {
          category_key: {
            "label": str,            # Japanese label from LAYER_CATEGORIES
            "count": int,            # total inserts in this category
            "blocks": {block_name: count},   # top block names used
            "layers": {layer_name: count},   # which layers they appear on
          }
        }
    """
    by_cat: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "blocks": defaultdict(int),
        "layers": defaultdict(int),
    })

    for ins in inserts:
        entry = by_cat[ins.category]
        entry["count"] += 1
        entry["blocks"][ins.block_name] += 1
        entry["layers"][ins.layer] += 1

    # Freeze inner defaultdicts and attach labels
    result: dict[str, dict] = {}
    for cat, entry in by_cat.items():
        result[cat] = {
            "label":  layer_label(cat),
            "count":  entry["count"],
            "blocks": dict(sorted(entry["blocks"].items(), key=lambda x: -x[1])),
            "layers": dict(sorted(entry["layers"].items(), key=lambda x: -x[1])),
        }
    return result


def extract_dimension_texts(texts: list[TextItem]) -> list[str]:
    """Extract unique texts from dimension layers."""
    dim_texts = [t.text for t in texts if t.category == "dimension"]
    if not dim_texts:
        # Fallback: texts matching common dimension patterns
        dim_texts = [
            t.text for t in texts
            if re.search(r"^[\d,]+$|GL[+-]|FL[+-]|H=\d|W=\d|1/\d{2,3}", t.text)
        ]
    return list(dict.fromkeys(dim_texts))[:60]


def extract_notes(texts: list[TextItem], min_len: int = 8) -> list[TextItem]:
    """Extract annotation/note text items (longer text, non-numeric)."""
    seen: set[str] = set()
    notes: list[TextItem] = []
    for t in sorted(texts, key=lambda t: -t.y):
        txt = t.text
        if len(txt) < min_len:
            continue
        if NUM_ONLY.match(txt):
            continue
        if AREA_RE.search(txt) and len(txt) < 12:
            continue
        if txt in seen:
            continue
        seen.add(txt)
        notes.append(t)
    return notes[:60]


# ─────────────────────────────────────────────────────────────────────────────
# Title block extraction (generic — works for any DXF)
# ─────────────────────────────────────────────────────────────────────────────

TITLE_ATTRIB_TAGS: dict[str, str] = {
    "PROJ": "工事名", "PROJECT": "工事名", "PROJNAME": "工事名", "工事名": "工事名",
    "DATE": "日付", "DRAWDATE": "日付", "作成日": "日付",
    "DRAWNO": "図面番号", "DWG_NO": "図面番号", "DWGNO": "図面番号", "図面番号": "図面番号",
    "TITLE": "図面名称", "DRAWTITLE": "図面名称", "図面名": "図面名称",
    "SCALE": "縮尺", "DRAWSCALE": "縮尺", "縮尺": "縮尺",
    "CLIENT": "建築主", "建築主": "建築主",
    "ARCH": "設計事務所", "ARCHITECT": "設計事務所",
    "STAGE": "図面状態",
}


def extract_title_block(doc: Drawing, msp) -> dict[str, str]:
    info: dict[str, str] = {}

    # 1. Known title block names in blocks section
    for blk_name in ("図面枠", "TITLE", "TITLE_BLOCK", "表題", "BORDER"):
        blk = doc.blocks.get(blk_name)
        if not blk:
            continue
        for e in blk:
            tp = e.dxftype()
            if tp not in ("TEXT", "ATTDEF"):
                continue
            t = (getattr(e.dxf, "text", "") or getattr(e.dxf, "default", "")).strip()
            if not t or t.startswith("%"):
                continue
            _classify_title_text(t, info)

    # 2. ATTRIB tags on INSERT entities in modelspace
    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        try:
            for att in e.attribs:
                tag = getattr(att.dxf, "tag", "").upper().strip()
                val = getattr(att.dxf, "text", "").strip()
                if not val or val.startswith("%"):
                    continue
                if tag in TITLE_ATTRIB_TAGS:
                    info.setdefault(TITLE_ATTRIB_TAGS[tag], val)
                else:
                    # Fallback: content-based classification
                    _classify_title_text(val, info)
        except Exception:
            pass

    # 3. Modelspace text near bottom (Y < 60) — common title strip location
    all_texts = extract_texts(msp)
    strip = [t for t in all_texts if t.y < 60]
    for t in sorted(strip, key=lambda t: t.x):
        _classify_title_text(t.text, info)

    return info


def _classify_title_text(t: str, info: dict[str, str]) -> None:
    """Heuristically classify a text string into title block fields."""
    if re.search(r"\d{4}[./年]\d{1,2}", t):
        info.setdefault("日付", t)
    elif re.search(r"S\s*=\s*1\s*[:／/]\s*\d{2,4}", t):
        info.setdefault("縮尺", t.strip())
    elif re.search(r"工事|PROJECT|プロジェクト", t) and len(t) > 4:
        info.setdefault("工事名", t)
    elif re.search(r"事.{0,2}務.{0,2}所", t):
        info.setdefault("設計事務所", re.sub(r"\s+", "", t))
    elif re.search(r"完成|竣工|実施|基本|計画", t) and "図" in t:
        info.setdefault("図面状態", re.sub(r"\s+", "", t))
    elif re.search(r"建築主|施主|発注者", t):
        info.setdefault("建築主", t)


# ─────────────────────────────────────────────────────────────────────────────
# Scale detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_scale(texts: list[TextItem]) -> float:
    """Try to detect drawing scale (returns unit multiplier, e.g. 100 for 1:100)."""
    for t in texts:
        m = re.search(r"1\s*[:／/]\s*(\d{2,4})", t.text)
        if m:
            return float(m.group(1))
    return 100.0   # default assumption: 1:100


def units_to_m2(area_units: float, scale: float) -> float:
    """Convert drawing-units² area to m² using scale factor.
    
    In a 1:100 drawing, 1 mm on paper = 100 mm real = 0.1 m.
    area_real_mm² = area_drawing_mm² × scale²
    area_m²       = area_real_mm² / 1_000_000
    """
    return area_units * (scale ** 2) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report sections
# ─────────────────────────────────────────────────────────────────────────────

def _md_title_block(info: dict[str, str]) -> str:
    if not info:
        return ""
    order = ["工事名", "図面番号", "図面名称", "縮尺", "日付", "建築主", "設計事務所", "図面状態"]
    rows = [(k, info[k]) for k in order if k in info]
    rows += [(k, v) for k, v in info.items() if k not in dict(rows)]
    lines = ["## 図面情報\n", "| 項目 | 内容 |", "|------|------|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def _md_layer_catalog(doc: Drawing, layer_stats: dict[str, dict[str, int]]) -> str:
    """Generate layer catalog table with semantic category and entity counts."""
    # Group layers by semantic category
    by_cat: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    for layer in doc.layers:
        name = layer.dxf.name
        cat = classify_layer(name)
        stats = layer_stats.get(name, {})
        by_cat[cat].append((name, stats))

    lines = ["\n## レイヤー分類\n"]
    lines.append("| カテゴリ | レイヤー名 | エンティティ数 |")
    lines.append("|----------|-----------|---------------|")

    # Print known categories first
    cat_order = list(LAYER_CATEGORIES.keys()) + ["other"]
    for cat in cat_order:
        entries = by_cat.get(cat, [])
        if not entries:
            continue
        label = LAYER_CATEGORIES.get(cat, ("その他", []))[0]
        for layer_name, stats in sorted(entries, key=lambda x: x[0]):
            total = sum(stats.values())
            detail = ", ".join(f"{k}:{v}" for k, v in sorted(stats.items(), key=lambda x: -x[1])[:4])
            lines.append(f"| {label} | `{layer_name}` | {total} ({detail}) |")

    return "\n".join(lines) + "\n"


def _md_rooms(rooms: list[dict]) -> str:
    # Deduplicate room names
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rooms:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    if not unique:
        return ""

    # Purpose-based summary
    by_purpose: dict[str, list[dict]] = defaultdict(list)
    for r in unique:
        by_purpose[r["purpose"]].append(r)

    lines = ["\n## 部屋・室名リスト\n"]

    # Summary table
    lines.append("### 用途別集計\n")
    lines.append("| 用途 | 室数 | 合計面積 |")
    lines.append("|------|------|----------|")
    total_count = 0
    total_area = 0.0
    for purpose in list(ROOM_PURPOSES.keys()) + ["その他"]:
        rlist = by_purpose.get(purpose, [])
        if not rlist:
            continue
        count = len(rlist)
        area = sum(r["area_m2"] for r in rlist)
        area_str = f"{area:.2f} ㎡" if area > 0 else "—"
        lines.append(f"| {purpose} | {count} | {area_str} |")
        total_count += count
        total_area += area
    lines.append(f"| **合計** | **{total_count}** | **{total_area:.2f} ㎡** |" if total_area > 0
                 else f"| **合計** | **{total_count}** | — |")

    # Detailed room list
    lines.append("\n### 室名詳細\n")
    lines.append("| 室名 | 面積 | レイヤー |")
    lines.append("|------|------|----------|")
    for r in sorted(unique, key=lambda r: -r["y"]):
        area_str = f"{r['area_m2']:.2f} ㎡" if r["area_m2"] > 0 else (r["area_text"] or "—")
        lines.append(f"| {r['name']} | {area_str} | `{r['layer']}` |")

    return "\n".join(lines) + "\n"


def _md_symbols_by_category(cat_summary: dict[str, dict]) -> str:
    """Render INSERT counts grouped by layer category."""
    if not cat_summary:
        return ""

    # Sort: known categories first (in LAYER_CATEGORIES order), then other
    cat_order = list(LAYER_CATEGORIES.keys()) + ["other"]
    sorted_cats = sorted(
        cat_summary.items(),
        key=lambda x: (cat_order.index(x[0]) if x[0] in cat_order else len(cat_order), -x[1]["count"]),
    )

    lines = ["\n## カテゴリ別シンボル集計（INSERTエンティティ）\n"]
    lines.append("> 各レイヤーのセマンティックカテゴリ別にブロック挿入数を集計\n")
    lines.append("| カテゴリ | 挿入数 | 主なブロック名 | 主なレイヤー |")
    lines.append("|----------|--------|--------------|------------|")

    total_inserts = 0
    for cat, entry in sorted_cats:
        top_blocks = list(entry["blocks"].items())[:3]
        top_layers = list(entry["layers"].items())[:2]
        blocks_str = ", ".join(f"`{b}`×{c}" for b, c in top_blocks) or "—"
        layers_str = ", ".join(f"`{l}`" for l, _ in top_layers) or "—"
        lines.append(f"| {entry['label']} | {entry['count']:,} | {blocks_str} | {layers_str} |")
        total_inserts += entry["count"]

    lines.append(f"\n**合計: {total_inserts:,} 個のブロック挿入**\n")
    return "\n".join(lines) + "\n"


def _md_equipment(block_summary: dict) -> str:
    semantic = block_summary.get("by_semantic", {})
    by_name  = block_summary.get("by_name", {})

    if not semantic and not by_name:
        return ""

    lines = ["\n## 建具・設備集計\n"]

    if semantic:
        FIXTURE_TYPES = {"ドア", "窓"}
        EQUIP_TYPES = {k for k in semantic if k not in FIXTURE_TYPES}

        fixture_rows = [(k, v) for k, v in semantic.items() if k in FIXTURE_TYPES]
        equip_rows   = [(k, v) for k, v in semantic.items() if k in EQUIP_TYPES]

        if fixture_rows:
            lines.append("### 建具\n")
            lines.append("| 種類 | 数量 |")
            lines.append("|------|------|")
            for k, v in sorted(fixture_rows, key=lambda x: -x[1]):
                lines.append(f"| {k} | {v} |")

        if equip_rows:
            lines.append("\n### 設備機器\n")
            lines.append("| 種類 | 数量 |")
            lines.append("|------|------|")
            for k, v in sorted(equip_rows, key=lambda x: -x[1]):
                lines.append(f"| {k} | {v} |")

    # All block names (top 30 by usage)
    if by_name:
        lines.append("\n### ブロック一覧（上位30種）\n")
        lines.append("| ブロック名 | 数量 | 種別 |")
        lines.append("|-----------|------|------|")
        for bname, cnt in list(by_name.items())[:30]:
            sem = classify_block(bname) or "—"
            lines.append(f"| `{bname}` | {cnt} | {sem} |")

    return "\n".join(lines) + "\n"


def _md_boundaries(boundaries: list[BoundaryInfo], scale: float) -> str:
    if not boundaries:
        return ""

    # Convert to m² and filter noise:
    # - Skip tiny (< 1 m²) and enormous (> 5000 m²) boundaries
    # - 5000 m² threshold removes drawing borders but keeps large floor areas
    MAX_ROOM_M2 = 5000.0
    useful: list[tuple[BoundaryInfo, float]] = []
    for b in boundaries:
        m2 = units_to_m2(b.area_units, scale)
        if 1.0 <= m2 <= MAX_ROOM_M2:
            useful.append((b, m2))

    if not useful:
        return ""

    by_cat: dict[str, list[tuple[BoundaryInfo, float]]] = defaultdict(list)
    for b, m2 in useful:
        by_cat[b.category].append((b, m2))

    lines = ["\n## 閉合ポリライン境界（室境界候補）\n"]
    lines.append(f"> 縮尺 1:{scale:.0f} で換算 (図面単位mm → ㎡)  |  面積範囲: 1〜{MAX_ROOM_M2:.0f} ㎡\n")
    lines.append("| カテゴリ | レイヤー | 境界数 | 最小面積 | 最大面積 | 合計面積 |")
    lines.append("|----------|----------|--------|----------|----------|----------|")

    all_m2 = 0.0
    for cat, entries in sorted(by_cat.items()):
        label = layer_label(cat)
        areas_m2 = [m2 for _, m2 in entries]
        min_a, max_a, sum_a = min(areas_m2), max(areas_m2), sum(areas_m2)
        all_m2 += sum_a
        layer_names = list(dict.fromkeys(b.layer for b, _ in entries))[:3]
        layers_str = ", ".join(f"`{n}`" for n in layer_names)
        lines.append(
            f"| {label} | {layers_str} | {len(entries)} "
            f"| {min_a:.1f} ㎡ | {max_a:.1f} ㎡ | {sum_a:.1f} ㎡ |"
        )

    lines.append(f"\n**有効境界合計: {all_m2:.1f} ㎡**  \n")
    lines.append("> ※ 重複計算の可能性あり。`壁・躯体` レイヤーの閉合ポリラインが実際の室境界です。")

    return "\n".join(lines) + "\n"


def _md_dimensions(dim_texts: list[str]) -> str:
    if not dim_texts:
        return ""
    lines = ["\n## 寸法・高さ情報\n"]
    lines.append(", ".join(f"`{v}`" for v in dim_texts))
    return "\n".join(lines) + "\n"


def _md_notes(notes: list[TextItem]) -> str:
    if not notes:
        return ""
    lines = ["\n## 注記・仕様\n"]
    for n in notes:
        lines.append(f"- {n.text}  *(layer: `{n.layer}`)*")
    return "\n".join(lines) + "\n"


def _md_layer_text_samples(texts: list[TextItem]) -> str:
    """Per-layer text samples — useful for understanding what each layer contains."""
    by_layer: dict[str, list[TextItem]] = defaultdict(list)
    for t in texts:
        by_layer[t.layer].append(t)

    lines = ["\n## レイヤー別テキストサンプル\n"]
    lines.append("> 各レイヤーに含まれる代表的なテキスト（最大10件）\n")

    for layer_name in sorted(by_layer.keys()):
        items = by_layer[layer_name]
        cat = classify_layer(layer_name)
        label = layer_label(cat)
        samples = list(dict.fromkeys(t.text for t in items))[:10]
        lines.append(f"\n**`{layer_name}`** — {label} ({len(items)} テキスト)")
        for s in samples:
            # Escape pipe for Markdown
            lines.append(f"  - {s.replace('|', '｜')}")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze(dxf_path: str | Path, layers_only: bool = False) -> str:
    p = Path(dxf_path)
    print(f"[cad_analyzer] Reading {p.name} …", file=sys.stderr)
    doc = ezdxf.readfile(str(p))
    msp = doc.modelspace()

    print("[cad_analyzer] Extracting entities …", file=sys.stderr)
    texts     = extract_texts(msp)
    inserts   = extract_inserts(msp)
    bounds    = extract_closed_polylines(msp)
    lyr_stats = extract_layer_entity_stats(msp)

    print(f"[cad_analyzer]   texts={len(texts)}, inserts={len(inserts)}, boundaries={len(bounds)}", file=sys.stderr)

    title = extract_title_block(doc, msp)
    scale = detect_scale(texts)
    print(f"[cad_analyzer]   scale=1:{scale:.0f}", file=sys.stderr)

    md = f"# {p.stem}\n\n"
    md += _md_title_block(title)
    md += _md_layer_catalog(doc, lyr_stats)

    if layers_only:
        md += _md_layer_text_samples(texts)
        return md

    # Full analysis
    room_candidates  = extract_room_candidates(texts)
    rooms            = pair_rooms_with_areas(room_candidates, texts)
    block_summary    = summarize_blocks(inserts)
    cat_summary      = summarize_inserts_by_category(inserts)
    dim_texts        = extract_dimension_texts(texts)
    notes            = extract_notes(texts)

    md += _md_rooms(rooms)
    md += _md_symbols_by_category(cat_summary)
    md += _md_equipment(block_summary)
    md += _md_boundaries(bounds, scale)
    md += _md_dimensions(dim_texts)
    md += _md_notes(notes)
    md += _md_layer_text_samples(texts)

    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal DXF architectural drawing analyzer — generates structured Markdown"
    )
    parser.add_argument("dxf", help="Path to .dxf file")
    parser.add_argument("-o", "--output", help="Output .md file (default: <stem>_analysis.md)")
    parser.add_argument(
        "--layers-only", action="store_true",
        help="Only output layer catalog + text samples (fast mode for exploring a new file)"
    )
    args = parser.parse_args()

    p = Path(args.dxf)
    if not p.exists():
        print(f"Error: {p} does not exist", file=sys.stderr)
        sys.exit(1)

    md = analyze(p, layers_only=args.layers_only)

    out_path = Path(args.output) if args.output else p.with_name(p.stem + "_analysis.md")
    out_path.write_text(md, encoding="utf-8")
    print(f"Written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
