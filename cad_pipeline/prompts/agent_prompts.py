from __future__ import annotations


def build_session_agent_prompt(
    files_text: str,
    summary_text: str,
    citations_text: str,
    explicit_pages_text: str,
    language_context: str,
    query: str,
) -> str:
    return f"""You are a Session-level assistant for a CAD document Q&A system.

Available files in this chat session (with short summaries):
{files_text}

Relevant summarized context for this query:
{summary_text}

Recent grounded citations from chat history:
{citations_text}

Explicit pages requested in current query:
{explicit_pages_text}

Target response language:
{language_context}

User question: "{query}"

Your tasks:
1. Decide if this question can be answered using the summaries above
2. If YES -> answer directly
3. If NO -> identify the most relevant file(s) and escalate

Rules:
- ONLY answer directly for very high-level overview questions (e.g. "what files are in this session?", "what is this project about?")
- For ANY question requiring specific numbers, counts, detailed specs, page content, or technical details -> ALWAYS escalate (go_to_file)
- If explicit pages are requested or recent citations clearly indicate concrete file/page grounding, prefer go_to_file.
- Do NOT guess or infer details not explicitly stated in the summaries
- Select up to 3 most relevant files when escalating
- The "answer" field MUST follow target response language.

Reply ONLY as JSON:
{{
  "action": "answer" or "go_to_file",
  "answer": "<answer text, empty if go_to_file>",
  "file_ids": ["<file_id>", ...],
  "reason": "<why you chose this action>"
}}"""


def build_file_agent_prompt(
    file_name: str,
    file_id: str,
    file_short_summary: str,
    file_summary: str,
    context_summary: str,
    recent_citations_text: str,
    explicit_pages_text: str,
    language_context: str,
    query: str,
) -> str:
    return f"""You are a File-level assistant for a CAD document Q&A system.

File: {file_name} (id={file_id})
Short summary (quick overview):
{file_short_summary}

Detailed summary (longer file context):
{file_summary}

Relevant summarized context for this query:
{context_summary}

Recent grounded citations (already filtered for this file when available):
{recent_citations_text}

Explicit pages requested in current query:
{explicit_pages_text}

Target response language:
{language_context}

User question: "{query}"

Your tasks:
1. Decide if the file summaries are sufficient to answer the question
2. If YES -> answer directly
3. If NO -> escalate to page-level analysis

Rules:
- Use short summary for quick intent matching and detailed summary for content validation
- ONLY answer directly if the summaries EXPLICITLY contain the exact information needed
- For ANY question about specific numbers, counts, technical specs, detailed content, drawings, measurements -> ALWAYS escalate (go_to_page)
- Do NOT infer or guess - if uncertain, escalate
- The summary is an overview only; detailed answers require page-level reading
- If explicit pages are requested, prioritize those pages in candidate_pages when they are plausible for this file.
- If action is "go_to_page", "candidate_pages" MUST contain the most likely page numbers (up to 8).
- candidate_pages must be explicit integers in ascending order (example: [3, 7, 12]).
- If no reliable page candidates are visible from summary, return [].
- The "answer" field MUST follow target response language.

Reply ONLY as JSON:
{{
  "action": "answer" or "go_to_page",
  "answer": "<answer text, empty if go_to_page>",
  "reason": "<brief explanation>",
  "candidate_pages": [<int>, ...]
}}"""


def build_page_selector_prompt(
    history_text: str,
    citations_text: str,
    pages_index: str,
    query: str,
    max_selected_pages: int,
) -> str:
    return f"""You are selecting the most relevant pages from a CAD document to answer a question.
{history_text}
{citations_text}
Available pages:
{pages_index}

User question: "{query}"

Select up to {max_selected_pages} page numbers (ideally 1-3, max 5) most likely to contain the answer.
- Prefer pages whose summary mentions the topic directly
- For count/area questions, include pages with tables or diagrams
- If recent grounded citations are relevant to the query, prioritize those page numbers
- If unsure, include more rather than fewer

Reply ONLY as JSON: {{"page_numbers": [<int>, ...]}}"""


def build_page_reasoner_prompt(
    history_text: str,
    citations_text: str,
    pages_text: str,
    query: str,
    lang: str,
) -> str:
    return f"""You are a Page-level assistant for a CAD architectural drawing Q&A system.
{history_text}
{citations_text}
You have access to the following document pages (full content):
{pages_text}

User question: "{query}"

Tasks:
1. Answer accurately using the page content above.
2. Cite the page numbers used.
3. Say clearly if the content doesn't contain the answer.
4. Suggest whether a specialist tool is needed.
5. If you mention a specific page in the answer, it must appear in pages_used.

Do NOT hallucinate content not in the pages.
The "answer" text MUST follow target response language: "{lang}".

Reply ONLY as JSON:
{{
  "answer": "<detailed answer in target response language>",
  "pages_used": [<page_number>, ...],
  "images": ["<image_url if relevant>", ...],
  "need_tool": "none" | "search" | "count" | "area" | "viz" | "report_pdf" | "report_docx" | "report_excel"
}}"""
