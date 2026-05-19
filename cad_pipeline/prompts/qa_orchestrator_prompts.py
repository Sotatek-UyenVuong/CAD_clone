from __future__ import annotations


def build_language_context_prompt(query_text: str, history_ctx: str) -> str:
    return f"""Choose a single output language for this CAD QA turn.

User query:
"{query_text}"

Recent chat context:
{history_ctx}

Rules:
- Prefer the language used by the user in current query.
- If mixed language, choose the dominant language used for intent-bearing words.
- Keep continuity with recent turns unless user clearly switched language.
- Return a BCP-47 language tag when possible (examples: "vi", "ja", "en", "fr", "de", "es", "zh-CN").

Reply ONLY JSON:
{{"language_context":"<language tag>"}}"""


def build_extract_explicit_pages_prompt(query_text: str) -> str:
    return f"""Extract explicitly mentioned page numbers from this CAD QA query.

Query:
"{query_text}"

Rules:
- Return only page numbers explicitly mentioned in the query text.
- If no explicit page number is mentioned, return empty list.
- Do not infer implicit pages.

Reply ONLY JSON:
{{
  "pages": [<int>, ...]
}}"""


def build_extract_explicit_pages_retry_prompt(query_text: str) -> str:
    return f"""Return ONLY strict JSON with explicit page numbers from query.

Query:
"{query_text}"

Output format:
{{"pages":[<int>, ...]}}

If there is no explicit page number, return:
{{"pages":[]}}"""


def build_page_context_summary_prompt(
    query_text: str,
    history_ctx: str,
    files_payload: str,
    citations_payload: str,
) -> str:
    return f"""You are a context summarizer for a CAD QA pipeline.

User query:
"{query_text}"

Recent conversation:
{history_ctx}

Working file candidates:
{files_payload}

Recent grounded citations:
{citations_payload}

Task:
- Extract only information necessary for answering the current query.
- Keep only relevant entities: target object, floor/layout, page clues, file clues, constraints.
- Prioritize cited file/page continuity when relevant.
- Do not invent facts.
- Keep summary concise and high-signal.

Reply ONLY strict JSON:
{{
  "summary_text": "<concise context for downstream answer model>",
  "relevant_file_ids": ["<file_id>", "..."],
  "relevant_pages": [<int>, ...]
}}"""


def build_direct_non_doc_prompt(query_text: str) -> str:
    return f"""You are a chat assistant in a CAD-document Q&A app.

Task:
Decide whether the user query can be answered directly WITHOUT using any document/image/page context.

Use direct answer only for:
- greetings, casual small talk
- general chit-chat
- broad non-document questions unrelated to CAD files, pages, uploaded images, architectural plan analysis

Do NOT use direct answer for:
- questions that depend on project documents/pages/files/citations
- uploaded image analysis
- CAD/layout count/area/search/report tasks

User query:
"{query_text}"

Reply ONLY JSON:
{{
  "is_direct": true | false,
  "answer": "<direct helpful answer in user's language, empty if is_direct=false>",
  "reason": "<short reason>"
}}"""


def build_orchestrator_action_plan_prompt(
    query_text: str,
    language_ctx: str,
    history_ctx: str,
    context_summary: str,
    files_payload: str,
    recent_citations_payload: str,
    explicit_pages_payload: str,
    action_history_payload: str,
    tool_hint: str,
    has_image: bool,
) -> str:
    return f"""You are a single Orchestrator for a CAD QA pipeline.

Your job:
- Decide exactly ONE next action for this step.
- You control global flow; executors are dumb tools/functions.
- Prefer evidence-grounded actions before finalizing.

Available actions:
- "file_agent": reason over one file summary to answer high-level or suggest candidate pages
- "page_reason": reason over page contexts, produce grounded answer + citations
- "search": retrieve relevant pages/files
- "count": run counting on scoped page context
- "area": run area extraction on scoped page context
- "viz": prepare visual-oriented response from scoped pages
- "report_pdf": generate PDF report
- "report_docx": generate DOCX report
- "report_excel": generate Excel report
- "direct_answer": answer directly only for non-document small talk
- "finalize": stop loop and return final response

Inputs:
- Query: "{query_text}"
- Target language: "{language_ctx}"
- Has uploaded image: {str(has_image).lower()}
- Tool hint from lightweight classifier: "{tool_hint}"
- Recent chat history: {history_ctx}
- High-signal context summary: {context_summary}
- Session files in scope: {files_payload}
- Recent grounded citations: {recent_citations_payload}
- Explicit pages requested: {explicit_pages_payload}
- Actions already executed: {action_history_payload}

Rules:
- If query is document-dependent, avoid "direct_answer".
- Use "file_agent" when query likely depends on one file's overview/structure before diving into page-level context.
- If explicit pages are requested, prioritize actions that can ground those pages.
- Use "finalize" only when an answer is already likely sufficient and grounded.
- Avoid repeating same ineffective action.
- Output ONE action only.

Reply ONLY JSON:
{{
  "action": "file_agent" | "page_reason" | "search" | "count" | "area" | "viz" | "report_pdf" | "report_docx" | "report_excel" | "direct_answer" | "finalize",
  "reason": "<short reason>"
}}"""
