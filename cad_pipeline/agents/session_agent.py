"""session_agent.py — Session-level agent for file-summary routing.

Input:
  - list of session-scoped file short summaries + file_ids

Output:
  - {"action": "answer", "answer": "...", "file_ids": []}
  - {"action": "go_to_file", "answer": "", "file_ids": ["id1", "id2"]}
"""

from __future__ import annotations

import json
import re

from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL
from cad_pipeline.prompts.agent_prompts import build_session_agent_prompt


def run_session_agent(
    query: str,
    files: list[dict],
    context_summary: str | None = None,
    recent_citations: list[dict] | None = None,
    explicit_pages_requested: list[int] | None = None,
    language_context: str | None = None,
) -> dict:
    """Run the session-level summary router agent."""
    from google import genai  # type: ignore

    client = genai.Client(api_key=GEMINI_API_KEY)

    files_text = "\n".join(
        f"- file_id={f['file_id']} | name={f.get('file_name','?')} | summary={f.get('summary','')[:300]}"
        for f in files
    )
    summary_text = (context_summary or "").strip()
    citations_text = json.dumps(recent_citations or [], ensure_ascii=False)
    explicit_pages_text = json.dumps(explicit_pages_requested or [], ensure_ascii=False)

    prompt = build_session_agent_prompt(
        files_text=files_text,
        summary_text=summary_text,
        citations_text=citations_text,
        explicit_pages_text=explicit_pages_text,
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
        return result
    except Exception as exc:
        return {
            "action": "go_to_file",
            "answer": "",
            "file_ids": [f["file_id"] for f in files[:2]],
            "reason": f"Router error: {exc}",
        }

