"""count_tool.py — Count symbols in CAD drawings.

Logic ưu tiên:
  1a. DXF available + symbol DB match → đếm INSERT entity           (chính xác 100%)
  1b. DXF available + no INSERT match → symbol-dict + TEXT label    (fallback DXF)
       · Scan block definitions: tìm block nào chứa text khớp query
       · Đếm INSERT của những block đó (composite symbol blocks)
       · Nếu vẫn 0 → đếm TEXT entity trong modelspace khớp query
         (áp dụng cho sàn vẽ thủ công như B1F — EV1/EV2 là TEXT riêng lẻ)
  2.  PDF/image only → Gemini Flash đọc context_md                  (fallback cuối)

Symbol DB (symbols_enriched.json) có:
  - label       → tên human-readable
  - block_names → mã block thực tế trong file DXF (e.g. "ドア", "A$C123...")
  - group       → nhóm (door, valve, electrical_lighting, ...)
  - keywords    → từ khóa tìm kiếm
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from cad_pipeline.config import (
    SYMBOLS_JSON, SYMBOL_GROUPS_JSON, GEMINI_API_KEY,
    GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL, OBJECT_DESCRIPTIONS_JSON,
)


def _attach_viz_image(
    result: dict,
    image_path: str | Path | None,
    dxf_path: str | Path | None = None,
) -> dict:
    """Attach a visualization image when result contains drawable positions."""
    positions = result.get("positions")
    if not isinstance(positions, list) or not positions:
        return result
    if not image_path:
        return result

    src = Path(image_path)
    if not src.exists():
        return result

    try:
        from cad_pipeline.tools.viz_tool import draw_count_boxes, get_viewport_bounds_from_dxf
    except Exception:
        return result

    viewport_bounds = result.get("viewport_bounds")
    if viewport_bounds is None and dxf_path:
        dxf_src = Path(dxf_path)
        if dxf_src.exists():
            viewport_bounds = get_viewport_bounds_from_dxf(dxf_src)

    try:
        out_path = draw_count_boxes(
            image_path=src,
            count_result=result,
            viewport_bounds=viewport_bounds,
        )
        result["image_url"] = str(out_path)
        result["viz_image_path"] = str(out_path)
    except Exception:
        # Keep counting response resilient even if visualization fails.
        pass
    return result


# ── Symbol database (lazy) ─────────────────────────────────────────────────

_symbols_cache: dict | None = None
_groups_cache: dict | None = None


def _load_symbols() -> dict:
    global _symbols_cache
    if _symbols_cache is None:
        with open(SYMBOLS_JSON, encoding="utf-8") as f:
            _symbols_cache = json.load(f)
    return _symbols_cache


def _load_groups() -> dict:
    global _groups_cache
    if _groups_cache is None:
        with open(SYMBOL_GROUPS_JSON, encoding="utf-8") as f:
            _groups_cache = json.load(f)
    return _groups_cache


# ── Object descriptions DB (lazy) ─────────────────────────────────────────

_obj_desc_cache: list[dict] | None = None


def _load_obj_descriptions() -> list[dict]:
    global _obj_desc_cache
    if _obj_desc_cache is None:
        if OBJECT_DESCRIPTIONS_JSON.exists():
            with open(OBJECT_DESCRIPTIONS_JSON, encoding="utf-8") as f:
                _obj_desc_cache = json.load(f).get("objects", [])
        else:
            _obj_desc_cache = []
    return _obj_desc_cache


def _build_obj_index() -> str:
    """Build a compact JSON index: {category: [{id, name_vi, name_en, name_ja}]}
    used as context for Gemini Flash to select relevant ids."""
    objects = _load_obj_descriptions()
    index: dict[str, list[dict]] = {}
    for obj in objects:
        cat = obj.get("category", "other")
        index.setdefault(cat, []).append({
            "id": obj["id"],
            "vi": obj.get("name_vi", ""),
            "en": obj.get("name_en", ""),
            "ja": obj.get("name_ja", ""),
        })
    return json.dumps(index, ensure_ascii=False)


_obj_index_cache: str | None = None


def _find_object_descriptions(query: str) -> list[dict]:
    """Ask Gemini Flash which object id(s) from object_descriptions.json match the query.

    Passes a compact category→[{id, vi, en, ja}] index so Gemini can do semantic
    multilingual matching. Falls back to empty list if Gemini fails.
    Returns full description dicts for matched ids (max 3).
    """
    from google import genai  # type: ignore

    global _obj_index_cache
    if _obj_index_cache is None:
        _obj_index_cache = _build_obj_index()

    objects = _load_obj_descriptions()
    id_to_obj = {o["id"]: o for o in objects}

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""You are a CAD symbol classifier for Japanese architectural floor plan drawings.

Available object types (by category):
{_obj_index_cache}

User query: "{query}"

Task: Return the id(s) from the list above that represent what the user is asking about.
- Understand any language (Vietnamese, Japanese, English).
- Be strict: only ids that clearly match the query.
- Return at most 3 ids, ordered by relevance.
- If nothing matches, return [].

Reply ONLY as a JSON array of id strings. Example: ["toilet", "squat_toilet"]"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        ids: list[str] = json.loads(raw)
        if not isinstance(ids, list):
            return []
        return [id_to_obj[i] for i in ids if i in id_to_obj]
    except Exception:
        return []


def _format_obj_desc_for_prompt(objs: list[dict]) -> str:
    """Format matched object descriptions into a concise prompt block."""
    if not objs:
        return ""
    lines = ["\n--- Symbol Visual Reference (from CAD drawing standards) ---"]
    for obj in objs:
        name = obj.get("name_vi") or obj.get("name_en") or obj.get("id")
        name_ja = obj.get("name_ja", "")
        name_en = obj.get("name_en", "")
        desc = obj.get("description", "")
        shape = obj.get("shape_hint", "")
        excludes = obj.get("exclude_hints", [])

        lines.append(f"\n[{name}{' / ' + name_ja if name_ja else ''}{' / ' + name_en if name_en else ''}]")
        if desc:
            lines.append(f"  Description: {desc}")
        if shape:
            lines.append(f"  Shape hint: {shape}")
        if excludes:
            lines.append(f"  Do NOT count: {'; '.join(excludes[:3])}")
    lines.append("---")
    return "\n".join(lines)


# ── Symbol DB lookup ───────────────────────────────────────────────────────
#
# Flow: query → Gemini picks group(s) from 21 groups
#             → group  → labels
#             → labels → block_names  (from symbols.json)
#             → block_names → đếm INSERT trong DXF
#
# Dùng group làm tầng trung gian vì:
#   - Chỉ có 21 nhóm → prompt Gemini rất nhỏ, nhanh, chính xác
#   - Mỗi group đã có keywords đa ngôn ngữ (JP/EN) → tăng khả năng match
#   - Từ group → labels → block_names là tra cứu tĩnh, không cần LLM nữa

_label_to_blocks_cache: dict[str, set[str]] | None = None  # label → set of block_names


def _build_label_to_blocks() -> dict[str, set[str]]:
    """Build label → block_names mapping from symbols.json (lazy, cached)."""
    global _label_to_blocks_cache
    if _label_to_blocks_cache is not None:
        return _label_to_blocks_cache

    symbols = _load_symbols()
    idx: dict[str, set[str]] = {}
    for sym in symbols.values():
        lbl = sym.get("label", "")
        if not lbl:
            continue
        if lbl not in idx:
            idx[lbl] = set()
        idx[lbl].update(sym.get("block_names", []))

    _label_to_blocks_cache = idx
    return idx


def _gemini_match_groups(query: str) -> list[str]:
    """Ask Gemini Flash which group(s) from the 21 groups match the user query.

    Passes a compact summary of all 21 groups (name + keywords).
    Returns matched group name strings.
    """
    from google import genai  # type: ignore

    groups = _load_groups()
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Compact: "group_name: kw1, kw2, kw3 | desc_en" per line
    group_lines = "\n".join(
        f'{g}: {", ".join(d.get("keywords", [])[:6])} | {d.get("description_en", "")}'
        for g, d in groups.items()
    )

    prompt = f"""You are a CAD symbol classifier for Japanese architectural drawings.

Available symbol groups (21 total):
---
{group_lines}
---

User query: "{query}"

Task: Return ONLY the group name(s) from the list above that contain symbols the user is asking about.
- Understand any language (Vietnamese, Japanese, English).
- Be strict: only groups that clearly contain relevant symbols.
- If nothing matches, return [].

Reply ONLY as a JSON array of group name strings. Example:
["stair_ramp", "elevator_escalator"]"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return [g for g in result if g in groups]
        return []
    except Exception:
        return []


def find_symbols_by_query(query: str) -> list[dict]:
    """Tìm symbols khớp query qua flow: query → group → labels → block_names.

    1. Gemini Flash chọn group(s) từ 21 nhóm (prompt nhỏ, nhanh)
    2. Lấy tất cả labels thuộc group đó
    3. Map labels → block_names từ symbols.json
    Fallback: keyword substring match trên group nếu Gemini lỗi.

    Returns: List of {label, group, block_names}
    """
    groups      = _load_groups()
    label_index = _build_label_to_blocks()

    # ── Primary: Gemini picks group(s) ──────────────────────────────────────
    matched_groups = _gemini_match_groups(query)

    # ── Fallback: keyword substring match on group keywords ─────────────────
    if not matched_groups:
        q = query.lower()
        for g, d in groups.items():
            kws = [k.lower() for k in d.get("keywords", [])]
            if q in g.lower() or any(q in k for k in kws):
                matched_groups.append(g)

    # ── Collect labels from matched groups ───────────────────────────────────
    result: list[dict] = []
    seen: set[str] = set()
    for g in matched_groups:
        for lbl in groups[g].get("labels", []):
            lbl = lbl.strip()
            if lbl in seen or lbl not in label_index:
                continue
            seen.add(lbl)
            result.append({
                "label":       lbl,
                "group":       g,
                "block_names": list(label_index[lbl]),
            })

    return result


def get_block_names_for_query(query: str) -> set[str]:
    """Lấy tập hợp tất cả block_names liên quan đến query."""
    matched = find_symbols_by_query(query)
    block_names: set[str] = set()
    for sym in matched:
        block_names.update(sym["block_names"])
    return block_names


def list_symbol_groups() -> list[str]:
    return list(_load_groups().keys())


# ── Gemini label classifier ────────────────────────────────────────────────

def _gemini_filter_unit_labels(query: str, candidate_labels: list[str]) -> list[str]:
    """Ask Gemini Flash to filter TEXT labels: keep only ones that are actual
    countable units (e.g. EV1, EV2) vs. room names, halls, pits, annotations.
    """
    from google import genai  # type: ignore

    if not candidate_labels:
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    labels_str = "\n".join(f"- {lbl}" for lbl in candidate_labels)
    prompt = f"""You are analyzing text labels extracted from a Japanese architectural CAD drawing (DXF file).

The user wants to count: "{query}"

Below are text labels found in the drawing that contain a keyword related to the query:
{labels_str}

Task: Return ONLY the labels that represent actual countable individual units/instances of what the user is asking for.
EXCLUDE:
- Room/space/area names (e.g. halls, lobbies, pits, machine rooms: EVホール, EV機械室, 上部EVピット, 階段室)
- Generic/ambiguous single-letter labels (e.g. "EV" alone without a number)
- Annotation text, notes, or remarks (e.g. "※A棟階段は...")
- Structural/mechanical spaces (e.g. "（上部EVピット）", "EV昇降路")

INCLUDE:
- Labels that directly name individual units: EV1, EV2, 階段（1）, 避難階段（2）, etc.

Reply ONLY as a JSON array of the labels to keep. Example:
["EV1", "EV2", "EV3"]"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        if isinstance(result, list):
            valid = set(candidate_labels)
            return [lbl for lbl in result if lbl in valid]
        return candidate_labels
    except Exception:
        return candidate_labels


def _collect_text_positions(
    msp,
    query: str,
    matched_groups: list[str],
    viewport_bounds: dict | None = None,
) -> list[dict]:
    """Scan modelspace TEXT/MTEXT entities matching query + group keywords."""
    groups_db = _load_groups()
    search_terms = list(dict.fromkeys(
        [query.upper()] +
        [kw.upper() for g in matched_groups for kw in groups_db.get(g, {}).get("keywords", [])]
    ))
    if viewport_bounds:
        x_min = float(viewport_bounds.get("x_min", 0.0))
        x_max = float(viewport_bounds.get("x_max", 0.0))
        y_min = float(viewport_bounds.get("y_min", 0.0))
        y_max = float(viewport_bounds.get("y_max", 0.0))
    else:
        x_min = x_max = y_min = y_max = 0.0

    raw_hits: list[tuple[str, float, float]] = []
    for entity in msp:
        if entity.dxftype() not in ("TEXT", "MTEXT"):
            continue
        try:
            t = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
            t_clean = re.sub(r"\\[A-Za-z0-9;]+", "", t.strip()).replace("\\P", " ").strip()
            if not any(st in t_clean.upper() for st in search_terms):
                continue
            ex = float(entity.dxf.insert.x)
            ey = float(entity.dxf.insert.y)
            if viewport_bounds and not (x_min <= ex <= x_max and y_min <= ey <= y_max):
                continue
            raw_hits.append((t_clean, ex, ey))
        except Exception:
            pass

    if not raw_hits:
        return []

    seen: set[str] = set()
    unique: list[tuple[str, float, float]] = []
    for t, x, y in raw_hits:
        if t not in seen:
            seen.add(t)
            unique.append((t, x, y))

    candidate_labels = [t for t, _, _ in unique]
    filtered_labels = set(_gemini_filter_unit_labels(query, candidate_labels))

    return [
        {"label": t, "block_name": "TEXT_ENTITY", "wcs_x": x, "wcs_y": y}
        for t, x, y in unique
        if t in filtered_labels
    ]


def count_in_dxf(
    dxf_path: str | Path,
    query: str,
    block_names: set[str] | None = None,
    viewport_bounds: dict | None = None,
) -> dict:
    """Đếm INSERT entity trong file DXF có block_name khớp symbol DB."""
    try:
        import ezdxf  # type: ignore
    except ImportError as exc:
        raise ImportError("ezdxf is required: pip install ezdxf") from exc

    if block_names is None:
        block_names = get_block_names_for_query(query)

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    if viewport_bounds:
        x_min = float(viewport_bounds.get("x_min", 0.0))
        x_max = float(viewport_bounds.get("x_max", 0.0))
        y_min = float(viewport_bounds.get("y_min", 0.0))
        y_max = float(viewport_bounds.get("y_max", 0.0))

        def _in_scope(entity) -> bool:
            try:
                ex = float(entity.dxf.insert.x)
                ey = float(entity.dxf.insert.y)
            except Exception:
                return False
            return x_min <= ex <= x_max and y_min <= ey <= y_max
    else:
        def _in_scope(entity) -> bool:
            return True

    counts_per_block: dict[str, int] = {}
    if block_names:
        for entity in msp.query("INSERT"):
            if not _in_scope(entity):
                continue
            name = entity.dxf.name
            if name in block_names:
                counts_per_block[name] = counts_per_block.get(name, 0) + 1

    sym_map: dict[str, str] = {}
    for sym in find_symbols_by_query(query):
        for bn in sym["block_names"]:
            sym_map[bn] = sym["label"]

    total = sum(counts_per_block.values())

    if total > 0:
        matched_symbols = [
            {"label": sym_map.get(bn, bn), "block_name": bn, "occurrences": cnt}
            for bn, cnt in sorted(counts_per_block.items(), key=lambda x: -x[1])
        ]
        insert_positions = []
        for entity in msp.query("INSERT"):
            if not _in_scope(entity):
                continue
            if entity.dxf.name in counts_per_block:
                try:
                    insert_positions.append({
                        "label": sym_map.get(entity.dxf.name, entity.dxf.name),
                        "block_name": entity.dxf.name,
                        "wcs_x": entity.dxf.insert.x,
                        "wcs_y": entity.dxf.insert.y,
                    })
                except Exception:
                    pass

        matched_groups = [s["group"] for s in find_symbols_by_query(query)]
        text_positions = _collect_text_positions(
            msp,
            query,
            list(dict.fromkeys(matched_groups)),
            viewport_bounds=viewport_bounds,
        )
        all_positions = insert_positions + text_positions
        final_count = max(total, len(text_positions)) if text_positions else total

        return {
            "count": final_count,
            "count_insert": total,
            "count_text": len(text_positions),
            "query": query,
            "matched_symbols": matched_symbols,
            "positions": all_positions,
            "dxf_file": Path(dxf_path).name,
            "mode": "dxf_exact",
            "viewport_bounds": viewport_bounds,
            "details": (
                f"INSERT: {total}, TEXT labels: {len(text_positions)} → total: {final_count} "
                f"in {Path(dxf_path).name}"
            ),
        }

    q_upper = query.upper()
    ev_block_dict: dict[str, list[str]] = {}

    def _get_texts_in_block(bname: str, depth: int = 0) -> list[str]:
        if depth > 2:
            return []
        blk = doc.blocks.get(bname)
        if not blk:
            return []
        texts: list[str] = []
        for e in blk:
            if e.dxftype() in ("TEXT", "MTEXT"):
                try:
                    t = (e.dxf.text if e.dxftype() == "TEXT" else e.text).strip()
                    if q_upper in t.upper():
                        texts.append(t)
                except Exception:
                    pass
            if e.dxftype() == "INSERT":
                texts += _get_texts_in_block(e.dxf.name, depth + 1)
        return texts

    for blk in doc.blocks:
        bname = blk.name
        if bname.startswith("*"):
            continue
        txts = _get_texts_in_block(bname)
        if txts:
            ev_block_dict[bname] = txts

    composite_counts: dict[str, int] = {}
    for entity in msp.query("INSERT"):
        if not _in_scope(entity):
            continue
        bname = entity.dxf.name
        if bname in ev_block_dict:
            composite_counts[bname] = composite_counts.get(bname, 0) + 1

    composite_total = sum(composite_counts.values())

    from collections import Counter as _Counter
    text_hits: list[str] = []
    for entity in msp:
        if entity.dxftype() not in ("TEXT", "MTEXT"):
            continue
        if not _in_scope(entity):
            continue
        try:
            t = (entity.dxf.text if entity.dxftype() == "TEXT" else entity.text).strip()
            t_clean = re.sub(r"\\[A-Za-z0-9;]+", "", t).replace("\\P", " ").strip()
            if q_upper in t_clean.upper():
                text_hits.append(t_clean)
        except Exception:
            pass

    label_counts = _Counter(text_hits)
    unique_text_labels: list[str] = list(dict.fromkeys(text_hits))
    if unique_text_labels:
        unique_text_labels = _gemini_filter_unit_labels(query, unique_text_labels)
    text_total = len(unique_text_labels)

    if composite_total > 0:
        matched_symbols = [
            {"label": k, "block_name": k, "occurrences": v}
            for k, v in sorted(composite_counts.items(), key=lambda x: -x[1])
        ]
        positions = []
        for entity in msp.query("INSERT"):
            if not _in_scope(entity):
                continue
            if entity.dxf.name in composite_counts:
                try:
                    positions.append({
                        "label": entity.dxf.name,
                        "block_name": entity.dxf.name,
                        "wcs_x": entity.dxf.insert.x,
                        "wcs_y": entity.dxf.insert.y,
                    })
                except Exception:
                    pass
        return {
            "count": composite_total,
            "query": query,
            "matched_symbols": matched_symbols,
            "positions": positions,
            "dxf_file": Path(dxf_path).name,
            "mode": "dxf_symbol_dict",
            "viewport_bounds": viewport_bounds,
            "details": (
                f"Found {composite_total} INSERT(s) of composite symbol blocks "
                f"containing '{query}' text in {Path(dxf_path).name}"
            ),
        }

    if text_total > 0:
        seen_pos: set[str] = set()
        positions_text = []
        for entity in msp:
            if entity.dxftype() not in ("TEXT", "MTEXT"):
                continue
            if not _in_scope(entity):
                continue
            try:
                t = (entity.dxf.text if entity.dxftype() == "TEXT" else entity.text).strip()
                t_clean = re.sub(r"\\[A-Za-z0-9;]+", "", t).replace("\\P", " ").strip()
                if t_clean in unique_text_labels and t_clean not in seen_pos:
                    positions_text.append({
                        "label": t_clean,
                        "block_name": "TEXT_ENTITY",
                        "wcs_x": entity.dxf.insert.x,
                        "wcs_y": entity.dxf.insert.y,
                    })
                    seen_pos.add(t_clean)
            except Exception:
                pass
        return {
            "count": text_total,
            "query": query,
            "matched_symbols": [
                {
                    "label": t,
                    "block_name": "TEXT_ENTITY",
                    "occurrences": 1,
                    "floor_count": label_counts[t],
                }
                for t in unique_text_labels
            ],
            "positions": positions_text,
            "dxf_file": Path(dxf_path).name,
            "mode": "dxf_text_label",
            "viewport_bounds": viewport_bounds,
            "details": (
                f"Found {text_total} unique '{query}' TEXT label(s) "
                f"in {Path(dxf_path).name}: {unique_text_labels[:10]}"
            ),
        }

    return {
        "count": 0,
        "query": query,
        "matched_symbols": [],
        "dxf_file": Path(dxf_path).name,
        "mode": "dxf_exact",
        "viewport_bounds": viewport_bounds,
        "details": f"No INSERT or TEXT matching '{query}' found in {Path(dxf_path).name}",
    }


def count_in_image(
    image_path: str | Path,
    query: str,
) -> dict:
    """Gemini Pro đọc trực tiếp ảnh bản vẽ và đếm symbols bằng vision."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)
    image_path = Path(image_path)

    matched = find_symbols_by_query(query)
    related_labels = list({s["label"] for s in matched})[:12]
    label_hint = ""
    if related_labels:
        label_hint = f"\nRelated symbol types in CAD drawings: {', '.join(related_labels)}"

    obj_descs = _find_object_descriptions(query)
    obj_desc_block = _format_obj_desc_for_prompt(obj_descs)

    prompt = f"""You are analyzing a Japanese architectural CAD floor plan drawing.

Task: Count the exact number of "{query}" visible in this drawing, and return their positions.{label_hint}{obj_desc_block}

Rules:
- Count each individual unit/instance you can see
- Do NOT count room labels, halls, corridors, or space names — only actual fixtures/symbols
- Do NOT count legend/key items in the title block
- If a symbol appears in a table/schedule showing quantities, use that number
- Be precise; if uncertain about some items, still count what you can clearly identify

For each found item, provide its bounding box as PERCENTAGE of image size (0–100):
  x_min, y_min = top-left corner as % of image width/height
  x_max, y_max = bottom-right corner as % of image width/height
  origin (0, 0) = top-left of image; (100, 100) = bottom-right
  Example: an item at roughly the center occupying 5% of width: x_min=47, y_min=48, x_max=53, y_max=52

Reply ONLY as JSON:
{{
  "count": <integer>,
  "details": "<brief description of what and where>",
  "confidence": "high" | "medium" | "low",
  "positions": [
    {{"label": "<name e.g. EV1>", "x_min": 47, "y_min": 23, "x_max": 53, "y_max": 27}},
    ...
  ]
}}"""

    try:
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/png")

        response = client.models.generate_content(
            model=GEMINI_PRO_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        result["mode"] = "vision_pro"
        result["query"] = query
        result["image"] = image_path.name
        return result
    except Exception as exc:
        return {
            "count": 0,
            "query": query,
            "details": f"Error: {exc}",
            "confidence": "low",
            "mode": "vision_pro",
            "image": str(image_path),
        }


def count_in_context_md(
    query: str,
    context_md: str,
) -> dict:
    """Fallback count from page markdown context using Gemini Flash."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""You are counting symbols/entities from CAD page markdown context.

User query: "{query}"

Page context:
---
{context_md}
---

Task:
- Determine the best count answer for the query using ONLY the context text above.
- Match by meaning, not exact token.
  Example: if query is Vietnamese "cầu thang", terms like Japanese "階段" / "避難階段"
  or English "stair / staircase / emergency stair" are equivalent and should be counted.
- If an entity appears as a labeled instance (e.g. "避難階段(1)"), treat it as one instance.
- If the context implies at least one clear instance, count must be >= 1.
- If the context does not contain enough information, return count=0 and explain briefly.
- Do not hallucinate values.

Reply ONLY as JSON:
{{
  "count": <integer>,
  "details": "<brief explanation>",
  "confidence": "high" | "medium" | "low"
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)
        count = parsed.get("count", 0)
        if not isinstance(count, int):
            try:
                count = int(count)
            except Exception:
                count = 0
        details = str(parsed.get("details", ""))
        confidence = str(parsed.get("confidence", "low"))
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        # Guard against strict-token false negatives:
        # if details says equivalent term exists once, do not keep count=0.
        if count == 0:
            dnorm = details.lower()
            if (
                ("once" in dnorm or "1" in dnorm or "một" in dnorm)
                and ("階段" in details or "避難階段" in details or "stair" in dnorm)
            ):
                count = 1
        return {
            "count": max(count, 0),
            "query": query,
            "details": details or "Counted from page context.",
            "confidence": confidence,
            "mode": "context_md",
        }
    except Exception as exc:
        return {
            "count": 0,
            "query": query,
            "details": f"Error: {exc}",
            "confidence": "low",
            "mode": "context_md",
        }


def run_count_tool(
    query: str,
    dxf_path: str | Path | None = None,
    image_path: str | Path | None = None,
    context_md: str | None = None,
    use_symbol_db: bool = True,
    layout_hint: str | None = None,
) -> dict:
    """Main entry point cho count tool."""
    _ = use_symbol_db  # Backward-compatible argument; current DXF flow always uses symbol DB first.

    if dxf_path and Path(dxf_path).exists():
        viewport_bounds = None
        if layout_hint:
            try:
                from cad_pipeline.tools.viz_tool import get_viewport_bounds_from_dxf
                viewport_bounds = get_viewport_bounds_from_dxf(dxf_path, layout_name=layout_hint)
            except Exception:
                viewport_bounds = None
        # Strict page/layout-scoped DXF counting:
        # do NOT fallback to global modelspace counting when viewport cannot be resolved.
        if viewport_bounds is None:
            if context_md and context_md.strip():
                fallback = count_in_context_md(query, context_md)
                fallback["mode"] = "context_md_scope_fallback"
                fallback["details"] = (
                    f"DXF scoped count skipped: cannot resolve layout viewport"
                    f"{f' for hint {layout_hint!r}' if layout_hint else ''}. "
                    f"Falling back to page context."
                )
                if layout_hint:
                    fallback["layout_hint"] = layout_hint
                return fallback
            return {
                "count": 0,
                "query": query,
                "matched_symbols": [],
                "dxf_file": Path(dxf_path).name,
                "mode": "dxf_scope_missing",
                "layout_hint": layout_hint or "",
                "details": (
                    "DXF scoped count skipped because viewport/layout scope could not be resolved."
                ),
            }

        result = count_in_dxf(dxf_path, query, viewport_bounds=viewport_bounds)
        if layout_hint:
            result["layout_hint"] = layout_hint
        return _attach_viz_image(result, image_path=image_path, dxf_path=dxf_path)

    if image_path and Path(image_path).exists():
        result = count_in_image(image_path, query)
        return _attach_viz_image(result, image_path=image_path, dxf_path=None)

    if context_md and context_md.strip():
        return count_in_context_md(query, context_md)

    return {
        "count": 0,
        "query": query,
        "details": "Cần cung cấp dxf_path, image_path hoặc context_md để đếm.",
        "mode": "none",
    }
