"""file_agent.py — Level 2 agent: decides if file summary is sufficient or escalates to pages."""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
from cad_pipeline.prompts.agent_prompts import build_file_agent_prompt


def run_file_agent(
    query: str,
    file_id: str,
    file_name: str,
    file_short_summary: str,
    file_summary: str,
    context_summary: str | None = None,
    recent_citations: list[dict] | None = None,
    explicit_pages_requested: list[int] | None = None,
    language_context: str | None = None,
) -> dict:
    """Run the file-level agent.

    Args:
        query: User's question.
        file_id: MongoDB file ID.
        file_name: Human-readable file name.
        file_short_summary: Short file-level overview.
        file_summary: Detailed file-level summary (aggregated from pages).

    Returns:
        {
          "action": "answer"|"go_to_page",
          "answer": str,
          "reason": str,
          "candidate_pages": list[int]
        }
    """
    from google import genai  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = build_file_agent_prompt(
        file_name=file_name,
        file_id=file_id,
        file_short_summary=file_short_summary,
        file_summary=file_summary,
        context_summary=str(context_summary or "").strip(),
        recent_citations_text=json.dumps(recent_citations or [], ensure_ascii=False),
        explicit_pages_text=json.dumps(explicit_pages_requested or [], ensure_ascii=False),
        language_context=str(language_context or "same as user query"),
        query=query,
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_FLASH_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Invalid file agent result format")

        action = str(result.get("action", "go_to_page"))
        answer = str(result.get("answer", ""))
        reason = str(result.get("reason", ""))

        candidate_pages_raw = result.get("candidate_pages", [])
        candidate_pages: list[int] = []
        if isinstance(candidate_pages_raw, list):
            seen: set[int] = set()
            for item in candidate_pages_raw:
                try:
                    page_no = int(item)
                except (TypeError, ValueError):
                    continue
                if page_no <= 0 or page_no in seen:
                    continue
                seen.add(page_no)
                candidate_pages.append(page_no)
        candidate_pages.sort()

        if action == "answer":
            candidate_pages = []

        return {
            "action": action,
            "answer": answer,
            "reason": reason,
            "candidate_pages": candidate_pages,
        }
    except Exception as exc:
        return {
            "action": "go_to_page",
            "answer": "",
            "reason": f"Agent error: {exc}",
            "candidate_pages": [],
        }
