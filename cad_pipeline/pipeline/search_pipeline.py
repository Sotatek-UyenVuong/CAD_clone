"""search_pipeline.py — Global lexical search pipeline.

Flow:
  User Query
  → Mongo lexical page retrieval (summary/context regex)
  → Return top-N results sorted by lexical score
"""

from __future__ import annotations

from cad_pipeline.config import TOP_K, TOP_N
from cad_pipeline.storage import mongo


def run_search(
    query: str,
    top_k: int = TOP_K,
    top_n: int = TOP_N,
    folder_id: str | None = None,
    file_id: str | None = None,
) -> list[dict]:
    """Lexical search across indexed pages (no embeddings).

    Args:
        query: User's search query.
        top_k: Candidates to retrieve from Mongo lexical search.
        top_n: Final results to return.
        folder_id: Optional filter by folder.
        file_id: Optional filter by file.

    Returns:
        List of {file_name, page_number, image_url, vector_score}
        sorted by score descending, limited to top_n.
    """
    candidates = mongo.search_pages_lexical(
        q=query,
        limit=top_k,
        folder_id=folder_id,
        file_id=file_id,
    )

    if not candidates:
        return []

    results: list[dict] = []
    for c in candidates:
        file_doc = mongo.get_file(c["file_id"]) or {}
        results.append({
            "page_id": c["page_id"],
            "file_id": c["file_id"],
            "file_name": file_doc.get("file_name", c["file_id"]),
            "page_number": c["page_number"],
            "image_url": c.get("image_url", ""),
            "short_summary": c.get("short_summary", ""),
            # Keep key name for backward compatibility with existing frontend.
            "vector_score": round(float(c["score"]), 4),
        })

    return results[:top_n]
