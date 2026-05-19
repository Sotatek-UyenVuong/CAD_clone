"""search_tool.py — Semantic search tool with optional image input.

Flow:
  text query (optional)
  + image bytes (optional)
    → Gemini Flash Vision: describe image in detail
    → enrich query: "<query> <description>"
  → title-block deterministic lookup
  → fallback Mongo lexical page retrieval (no embeddings)
  → Return top-N results with rank, file_name, page_number, image_url,
    short_summary, vector_score (compat key)
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_pipeline.config import (
    GEMINI_API_KEY,
    GEMINI_FLASH_MODEL,
    TOP_K,
    TOP_N,
)
from cad_pipeline.storage import mongo


# ── Image description via Gemini Flash Vision ──────────────────────────────

def _describe_image(
    image_bytes: bytes,
    hint: str = "",
    model: str | None = None,
) -> str:
    """Ask Gemini Flash to describe an image for use as a search query.

    Returns a concise description string (1–3 sentences).
    Falls back to empty string on error.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types as _gt  # type: ignore

        _model = model or GEMINI_FLASH_MODEL
        client = genai.Client(api_key=GEMINI_API_KEY)

        hint_text = f'\nUser context: "{hint}"' if hint else ""
        prompt = (
            "You are analysing a CAD architectural drawing image for semantic search.\n"
            "Describe the image concisely: type of drawing (floor plan / elevation / "
            "section / detail / table / diagram), floor/level if visible, main elements "
            "(rooms, equipment, structures, symbols), and any notable labels or numbers."
            f"{hint_text}\n"
            "Reply in 1–3 sentences. Be specific. Output only the description, no preamble."
        )

        b64 = base64.standard_b64encode(image_bytes).decode()
        response = client.models.generate_content(
            model=_model,
            contents=[
                _gt.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
        )
        return response.text.strip()
    except Exception:
        return ""


def _extract_title_block_query(
    image_bytes: bytes,
    hint: str = "",
    model: str | None = None,
) -> dict[str, str]:
    """Extract title-block metadata with layout-first strategy.

    Strategy:
    1) Try layout detection and crop the best title_block region.
    2) Run Gemini extraction on cropped title_block.
    3) If no title_block detected or extraction fails, fallback to full image.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types as _gt  # type: ignore
        from cad_pipeline.core.layout_detect import LayoutDetector  # lazy import to avoid heavy load at module import time

        _model = model or GEMINI_FLASH_MODEL
        client = genai.Client(api_key=GEMINI_API_KEY)

        hint_text = f'\nUser context: "{hint}"' if hint else ""
        prompt = (
            "You are reading title-block metadata from an architectural CAD drawing image.\n"
            "Extract likely drawing number/title/project fields if visible."
            f"{hint_text}\n"
            "Return ONLY JSON:\n"
            '{ "drawing_no": "<string or empty>", "drawing_title": "<string or empty>", "project": "<string or empty>" }'
        )
        def _extract_from_bytes(img_bytes: bytes) -> dict[str, str]:
            response = client.models.generate_content(
                model=_model,
                contents=[
                    _gt.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            raw = response.text.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            parsed = json.loads(raw)
            return {
                "drawing_no": str(parsed.get("drawing_no", "")).strip(),
                "drawing_title": str(parsed.get("drawing_title", "")).strip(),
                "project": str(parsed.get("project", "")).strip(),
            }

        # 1) Try detectron layout and crop title_block
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is not None:
                detector = LayoutDetector.get()
                blocks = detector.predict_image(image)
                title_blocks = [b for b in blocks if b.label == "title_block" and b.width > 0 and b.height > 0]
                if title_blocks:
                    # Pick most reliable title block region.
                    best = max(title_blocks, key=lambda b: float(b.score) * max(1, b.width * b.height))
                    crop = best.crop(image)
                    if hasattr(cv2, "imencode"):
                        ok, buf = cv2.imencode(".png", crop)
                        if ok:
                            return _extract_from_bytes(buf.tobytes())
                    rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                        Image.fromarray(rgb).save(tmp.name, format="PNG")
                        return _extract_from_bytes(Path(tmp.name).read_bytes())
        except Exception:
            # fall through to full-image extraction
            pass

        # 2) Fallback: full image
        return _extract_from_bytes(image_bytes)
    except Exception:
        return {"drawing_no": "", "drawing_title": "", "project": ""}


def _extract_title_block_query_from_text(
    query: str,
    model: str | None = None,
) -> dict[str, str]:
    """Extract potential title-block metadata directly from text query."""
    try:
        from google import genai  # type: ignore

        text = (query or "").strip()
        if not text:
            return {"drawing_no": "", "drawing_title": "", "project": ""}

        _model = model or GEMINI_FLASH_MODEL
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""Extract potential title-block metadata from this user query.
If a field is not explicitly present, return empty string for that field.
User query: "{text}"
Return ONLY JSON:
{{ "drawing_no": "<string or empty>", "drawing_title": "<string or empty>", "project": "<string or empty>" }}"""
        response = client.models.generate_content(
            model=_model,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)
        return {
            "drawing_no": str(parsed.get("drawing_no", "")).strip(),
            "drawing_title": str(parsed.get("drawing_title", "")).strip(),
            "project": str(parsed.get("project", "")).strip(),
        }
    except Exception:
        return {"drawing_no": "", "drawing_title": "", "project": ""}


def _norm(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[-_‐‑‒–—―]", "", text)
    return text


def _title_block_index_lookup(
    *,
    title_query: dict[str, str],
    folder_id: str | None = None,
    file_id: str | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Deterministic lookup by file.title_block_index."""
    q_no = _norm(title_query.get("drawing_no"))
    q_title = _norm(title_query.get("drawing_title"))
    q_project = _norm(title_query.get("project"))
    if not (q_no or q_title or q_project):
        return []

    db = mongo.get_db()
    query: dict = {}
    if file_id:
        query["_id"] = file_id
    elif folder_id:
        query["folder_id"] = folder_id
    query["title_block_index.0"] = {"$exists": True}

    file_docs = list(
        db["files"].find(
            query,
            {"_id": 1, "folder_id": 1, "file_name": 1, "title_block_index": 1},
        )
    )
    if not file_docs:
        return []

    folder_ids = {str(f.get("folder_id", "")) for f in file_docs if f.get("folder_id")}
    folders_map: dict[str, str] = {}
    if folder_ids:
        for fol in db["folders"].find({"_id": {"$in": list(folder_ids)}}, {"name": 1}):
            folders_map[str(fol["_id"])] = fol.get("name", str(fol["_id"]))

    page_cache: dict[str, dict[int, dict]] = {}
    scored: list[dict] = []
    for file_doc in file_docs:
        fid = str(file_doc.get("_id", ""))
        rows = file_doc.get("title_block_index") or []
        if not isinstance(rows, list):
            continue
        if fid not in page_cache:
            pages = mongo.get_pages_by_file(
                fid,
                projection={"page_number": 1, "image_url": 1, "short_summary": 1},
            )
            page_cache[fid] = {int(p.get("page_number", 0)): p for p in pages}

        for row in rows:
            if not isinstance(row, dict):
                continue
            score = 0.0
            no = _norm(row.get("drawing_no"))
            title = _norm(row.get("drawing_title"))
            project = _norm(row.get("project"))

            if q_no and no:
                if q_no == no:
                    score += 0.8
                elif q_no in no or no in q_no:
                    score += 0.5
            if q_title and title:
                if q_title == title:
                    score += 0.5
                elif q_title in title or title in q_title:
                    score += 0.3
            if q_project and project:
                if q_project == project:
                    score += 0.2
                elif q_project in project or project in q_project:
                    score += 0.1

            if score < 0.45:
                continue

            pno = int(row.get("page_number") or 0)
            page_doc = page_cache.get(fid, {}).get(pno, {})
            folder_key = str(file_doc.get("folder_id", ""))
            scored.append(
                {
                    "file_id": fid,
                    "file_name": file_doc.get("file_name", fid),
                    "folder_id": folder_key,
                    "folder_name": folders_map.get(folder_key, folder_key),
                    "page_number": pno,
                    "image_url": page_doc.get("image_url", ""),
                    "short_summary": str(page_doc.get("short_summary", ""))[:200],
                    "vector_score": round(score, 4),
                    "_score": score,
                }
            )

    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    dedup: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for item in scored:
        key = (str(item["file_id"]), int(item["page_number"]))
        if key in seen:
            continue
        seen.add(key)
        item["rank"] = len(dedup) + 1
        item.pop("_score", None)
        dedup.append(item)
        if len(dedup) >= top_n:
            break
    return dedup


# ── Core search logic ───────────────────────────────────────────────────────

def run_search_tool(
    query: str | None = None,
    image_bytes: bytes | None = None,
    image_path: str | Path | None = None,
    folder_id: str | None = None,
    file_id: str | None = None,
    top_n: int = 10,
    top_k: int | None = None,
    gemini_model: str | None = None,
) -> dict:
    """Semantic search across indexed pages.  Accepts text, image, or both.

    Args:
        query:        Natural language query (any language). Optional if image
                      is provided.
        image_bytes:  Raw image bytes (PNG / JPG / WebP). Takes precedence over
                      image_path.
        image_path:   Path to an image file. Used if image_bytes is None.
        folder_id:    Restrict search to a specific folder.
        file_id:      Restrict search to a specific file.
        top_n:        Max results to return (default 10).
        top_k:        Mongo lexical candidates (default TOP_K from config).
        gemini_model: Override Gemini model for image description.

    Returns:
        {
          "query_used":          str,   # final query sent to embedder
          "image_description":   str,   # Gemini's description (empty if no image)
          "total":               int,
          "results": [
            {
              "rank":           int,
              "file_id":        str,
              "file_name":      str,
              "page_number":    int,
              "image_url":      str,
              "short_summary":  str,    # first 200 chars of page short_summary
              "vector_score":   float,
            },
            ...
          ]
        }
    """
    # ── 1. Resolve image bytes ──────────────────────────────────────────────
    _img: bytes | None = image_bytes
    if _img is None and image_path:
        try:
            _img = Path(image_path).read_bytes()
        except Exception:
            _img = None

    # ── 2. Build optional image/title metadata ──────────────────────────────
    description = ""
    title_query = {"drawing_no": "", "drawing_title": "", "project": ""}
    if _img:
        description = _describe_image(_img, hint=query or "", model=gemini_model)
        title_query = _extract_title_block_query(_img, hint=query or "", model=gemini_model)
    elif query:
        # Text-only queries can still target drawing number/title/project.
        title_query = _extract_title_block_query_from_text(query, model=gemini_model)

    # ── 2.5 Title-block deterministic lookup (image or text) ────────────────
    title_hits = _title_block_index_lookup(
        title_query=title_query,
        folder_id=folder_id,
        file_id=file_id,
        top_n=top_n,
    )
    if title_hits:
        return {
            "query_used": query or "",
            "image_description": description,
            "title_block_query": title_query,
            "retrieval_mode": "title_block_index",
            "total": len(title_hits),
            "results": title_hits,
        }

    # ── 3. Build final query ────────────────────────────────────────────────
    parts = [p for p in [query, description] if p]
    if not parts:
        return {
            "query_used": "",
            "image_description": "",
            "total": 0,
            "results": [],
            "error": "No query text or image provided.",
        }
    final_query = " ".join(parts)

    # ── 4. Lexical fallback retrieval (no embeddings) ──────────────────────
    _top_k = top_k or TOP_K
    candidates = mongo.search_pages_lexical(
        q=final_query,
        limit=_top_k,
        folder_id=folder_id,
        file_id=file_id,
    )

    if not candidates:
        return {
            "query_used": final_query,
            "image_description": description,
            "title_block_query": title_query,
            "retrieval_mode": "lexical_search",
            "total": 0,
            "results": [],
        }

    # ── 5. Fetch MongoDB metadata ───────────────────────────────────────────

    # Deduplicate by (file_id, page_number) first so "top_n pages" is truly page-unique.
    unique_candidates: list[dict] = []
    seen_page_keys: set[tuple[str, int]] = set()
    for c in candidates:
        fid = str(c.get("file_id", "")).strip()
        try:
            pno = int(c.get("page_number", 0) or 0)
        except Exception:
            pno = 0
        if not fid or pno <= 0:
            continue
        key = (fid, pno)
        if key in seen_page_keys:
            continue
        seen_page_keys.add(key)
        unique_candidates.append(c)
        if len(unique_candidates) >= top_n:
            break

    # Batch-fetch folder names to avoid N+1 lookups
    folder_ids = {c["folder_id"] for c in unique_candidates}
    folders_map: dict[str, str] = {}
    for fid in folder_ids:
        fol = mongo.get_folder(fid)
        folders_map[fid] = fol.get("name", fid) if fol else fid

    results: list[dict] = []
    for rank, c in enumerate(unique_candidates, start=1):
        file_doc  = mongo.get_file(c["file_id"]) or {}
        summary   = c.get("short_summary", "")
        fol_id    = c.get("folder_id", file_doc.get("folder_id", ""))
        results.append({
            "rank":          rank,
            "file_id":       c["file_id"],
            "file_name":     file_doc.get("file_name") or file_doc.get("original_name", c["file_id"]),
            "folder_id":     fol_id,
            "folder_name":   folders_map.get(fol_id, fol_id),
            "page_number":   c["page_number"],
            "image_url":     c.get("image_url", ""),
            "short_summary": summary[:200],
            "vector_score":  round(c["score"], 4),
        })

    return {
        "query_used":        final_query,
        "image_description": description,
        "title_block_query": title_query,
        "retrieval_mode": "lexical_search",
        "total":             len(results),
        "results":           results,
    }
