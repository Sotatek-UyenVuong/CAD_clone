"""area_tool.py — Calculate floor areas from CAD context or room catalog.

Two modes:
  1. LLM-based: Extract and sum dimensions/areas mentioned in context_md.
  2. Room catalog: Look up area from the unit_room_catalog.json.

Tatami → m²: 1 tatami (畳) ≈ 1.62 m² (standard Japanese)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cad_pipeline.config import (
    GEMINI_API_KEY,
    GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL,
)

_CAD_PIPELINE_DIR = Path(__file__).resolve().parents[1]
UNIT_ROOM_CATALOG_PATH = _CAD_PIPELINE_DIR / "unit_room_catalog.json"
TATAMI_TO_M2 = 1.62  # standard conversion


def _attach_viz_image_if_positions(
    result: dict,
    image_path: str | Path | None,
) -> dict:
    """Attach visualization image if area result includes drawable positions."""
    positions = result.get("positions")
    if not isinstance(positions, list) or not positions:
        return result
    if not image_path:
        return result

    src = Path(image_path)
    if not src.exists():
        return result

    viz_payload = dict(result)
    viz_payload.setdefault("mode", "vision_pro")
    viz_payload.setdefault("count", len(positions))
    viz_payload.setdefault("query", str(result.get("query") or "area"))

    try:
        from cad_pipeline.tools.viz_tool import draw_count_boxes
        out_path = draw_count_boxes(image_path=src, count_result=viz_payload)
        result["image_url"] = str(out_path)
        result["viz_image_path"] = str(out_path)
    except Exception:
        # Do not fail area extraction when visualization fails.
        pass
    return result


# ── Room catalog (lazy) ────────────────────────────────────────────────────

_catalog_cache: dict | None = None


def _load_catalog() -> dict:
    global _catalog_cache
    if _catalog_cache is None:
        with open(UNIT_ROOM_CATALOG_PATH, encoding="utf-8") as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


# ── Mode 1: LLM-based area extraction ─────────────────────────────────────

def extract_area_from_context(
    context_md: str,
    query: str,
) -> dict:
    """Ask Gemini Flash to extract and sum areas from page context.

    Args:
        context_md: Full markdown context of relevant pages.
        query: e.g. "total floor area", "居室面積", "parking area"

    Returns:
        {"area": str, "unit": str, "details": str, "breakdown": [...]}
    """
    from google import genai  # type: ignore
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""You are analyzing CAD architectural drawing content.

Content from drawing pages:

---
{context_md[:8000]}
---

Task: Calculate or extract the total area for: "{query}"

Rules:
- Look for tables, dimension annotations, or area schedules
- Convert all values to m² (1 tatami = 1.62 m²)
- Show formula/breakdown if multiple rooms
- If area is not found, say so clearly

Reply ONLY as JSON:
{{
  "area": "<total value e.g. '120.5'>",
  "unit": "m²",
  "details": "<explanation of how you calculated it>",
  "breakdown": [
    {{"name": "<room/space name>", "area": "<value>", "unit": "m²"}}
  ],
  "confidence": "high" | "low"
}}"""

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        return result
    except Exception as exc:
        return {
            "area": "unknown",
            "unit": "m²",
            "details": f"Error: {exc}",
            "breakdown": [],
            "confidence": "low",
        }


def extract_area_from_image(
    image_path: str | Path,
    query: str,
) -> dict:
    """Ask Gemini Pro to extract area values directly from the uploaded image."""
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)
    image_path = Path(image_path)

    prompt = f"""You are analyzing a Japanese architectural CAD drawing image.

Task: Calculate or extract the total area for: "{query}" based ONLY on this uploaded image.

Rules:
- Read area annotations, schedules, and labels visible in the image.
- Convert all values to m² when needed (1 tatami = 1.62 m²).
- If multiple spaces are included, provide a clear breakdown and total.
- If area data is not visible or not reliable, return area="unknown" and explain why.
- Do not use any external context.

Reply ONLY as JSON:
{{
  "area": "<total value e.g. '120.5' or 'unknown'>",
  "unit": "m²",
  "details": "<brief explanation>",
  "breakdown": [
    {{"name": "<space name>", "area": "<value>", "unit": "m²"}}
  ],
  "confidence": "high" | "medium" | "low"
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
            "area": "unknown",
            "unit": "m²",
            "details": f"Error: {exc}",
            "breakdown": [],
            "confidence": "low",
            "mode": "vision_pro",
            "query": query,
            "image": str(image_path),
        }


# ── Mode 2: Room catalog lookup ────────────────────────────────────────────

def get_unit_area(unit_label: str) -> dict:
    """Look up area info for an apartment unit type from the catalog.

    Args:
        unit_label: e.g. "apt_unit_100A", "apt_unit_30A"

    Returns:
        {"unit": str, "total_tatami": float, "total_m2": float, "rooms": [...]}
    """
    catalog = _load_catalog()
    unit = catalog.get(unit_label)
    if not unit:
        return {"error": f"Unit '{unit_label}' not found in catalog"}

    rooms_with_m2 = []
    for room in unit.get("rooms", []):
        tatami = room.get("tatami")
        rooms_with_m2.append(
            {
                "name": room.get("name", ""),
                "tatami": tatami,
                "m2": round(tatami * TATAMI_TO_M2, 2) if tatami else None,
            }
        )

    total_tatami = unit.get("total_tatami", 0)
    return {
        "unit": unit_label,
        "block_name": unit.get("block_name", ""),
        "total_tatami": total_tatami,
        "total_m2": round(total_tatami * TATAMI_TO_M2, 2),
        "room_count": unit.get("room_count", 0),
        "rooms": rooms_with_m2,
    }


def list_unit_types() -> list[str]:
    """Return all available unit type labels."""
    return list(_load_catalog().keys())


def get_all_units_summary() -> list[dict]:
    """Return a compact summary of all unit types: label, total_m2."""
    catalog = _load_catalog()
    return [
        {
            "unit": k,
            "block_name": v.get("block_name", ""),
            "total_tatami": v.get("total_tatami", 0),
            "total_m2": round(v.get("total_tatami", 0) * TATAMI_TO_M2, 2),
            "room_count": v.get("room_count", 0),
        }
        for k, v in catalog.items()
    ]


# ── Unified entry point ────────────────────────────────────────────────────

def run_area_tool(
    query: str,
    context_md: str | None = None,
    unit_label: str | None = None,
    image_path: str | Path | None = None,
) -> dict:
    """Main entry point for the area tool.

    Args:
        query: What area to calculate.
        context_md: Page context for LLM-based extraction.
        unit_label: If set, look up directly in unit catalog.
        image_path: Optional image input for vision-based extraction.

    Returns:
        Area result dict.
    """
    if unit_label:
        return get_unit_area(unit_label)

    if image_path and Path(image_path).exists():
        result = extract_area_from_image(image_path, query)
        return _attach_viz_image_if_positions(result, image_path)

    if context_md:
        return extract_area_from_context(context_md, query)

    return {
        "area": "unknown",
        "unit": "m²",
        "details": "No context or unit label provided",
        "confidence": "low",
    }
