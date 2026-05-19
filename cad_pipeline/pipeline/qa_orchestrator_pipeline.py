"""qa_orchestrator_pipeline.py — single orchestrator + pure executors."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from cad_pipeline.agents.file_agent import run_file_agent
from cad_pipeline.agents.page_agent import run_page_agent
from cad_pipeline.agents.tool_router import classify_tool
from cad_pipeline.observability.phoenix_tracing import traced_span
from cad_pipeline.prompts.qa_orchestrator_prompts import (
    build_direct_non_doc_prompt,
    build_extract_explicit_pages_prompt,
    build_extract_explicit_pages_retry_prompt,
    build_language_context_prompt,
    build_orchestrator_action_plan_prompt,
    build_page_context_summary_prompt,
)
from cad_pipeline.storage import mongo
from cad_pipeline.tools.area_tool import run_area_tool
from cad_pipeline.tools.count_tool import run_count_tool
from cad_pipeline.tools.report_tool import run_report_docx, run_report_excel, run_report_pdf
from cad_pipeline.tools.search_tool import run_search_tool


def run_qa(
    query: str,
    session_id: str,
    file_id: str | None = None,
    session_file_ids: list[str] | None = None,
    user_email: str | None = None,
    save_history: bool = True,
    image_bytes: bytes | None = None,
    user_image_url: str | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    answer_stream_callback: Callable[[str], None] | None = None,
) -> dict:
    _ = user_image_url
    query = (query or "").strip()
    session_scope = (session_id or "").strip()
    if not session_scope:
        raise ValueError("session_id is required")

    def _emit(phase: str, message: str, *, tool: str = "", status: str = "running", step_id: str = "") -> None:
        if progress_callback is None:
            return
        progress_callback({"phase": phase, "message": message, "step_id": step_id, "tool": tool, "status": status})

    def _parse_json_text(raw: str) -> dict:
        txt = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not txt:
            return {}
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            s = txt.find("{")
            e = txt.rfind("}")
            if s >= 0 and e > s:
                try:
                    obj = json.loads(txt[s : e + 1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
        return {}

    def _history_context(turns: list[dict], limit: int = 5) -> str:
        lines: list[str] = []
        for t in turns[-limit:]:
            u = str(t.get("role_user", "") or "")
            a = str(t.get("role_assistant", "") or "")
            if u:
                lines.append(f"User: {u}")
            if a:
                lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def _norm_doc_name(value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[‐‑‒–—―]", "-", text)
        return text

    def _collect_recent_citations(turns: list[dict], limit: int = 5) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for turn in reversed(turns[-limit:]):
            meta = turn.get("assistant_meta", {}) if isinstance(turn.get("assistant_meta", {}), dict) else {}
            raw = meta.get("citations", []) if isinstance(meta.get("citations", []), list) else []
            for c in raw:
                fid = str(c.get("file_id", "")).strip()
                pno = int(c.get("page_number", 0) or 0)
                if not fid or pno <= 0:
                    continue
                key = (fid, pno)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"file_id": fid, "file_name": str(c.get("file_name", "") or fid), "page_number": pno})
        out.reverse()
        return out

    def _extract_explicit_pages(query_text: str) -> list[int]:
        if not query_text:
            return []
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=build_extract_explicit_pages_prompt(query_text=query_text),
            )
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            pages = parsed.get("pages", [])
            if not isinstance(pages, list) or not pages:
                retry = client.models.generate_content(
                    model=GEMINI_FLASH_MODEL,
                    contents=build_extract_explicit_pages_retry_prompt(query_text=query_text),
                )
                parsed = _parse_json_text(str(getattr(retry, "text", "") or ""))
                pages = parsed.get("pages", [])
            out: list[int] = []
            seen: set[int] = set()
            for p in pages if isinstance(pages, list) else []:
                try:
                    pno = int(p)
                except Exception:
                    continue
                if pno <= 0 or pno in seen:
                    continue
                seen.add(pno)
                out.append(pno)
            return out
        except Exception:
            return []

    def _decide_language(query_text: str, turns: list[dict]) -> str:
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=GEMINI_FLASH_MODEL,
                contents=build_language_context_prompt(query_text=query_text, history_ctx=_history_context(turns, limit=5)),
            )
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            lang = str(parsed.get("language_context", "") or parsed.get("reply_language", "")).strip()
            if lang:
                return lang
        except Exception:
            pass
        return "auto"

    def _msg(lang: str, vi: str, ja: str, en: str) -> str:
        lang_norm = str(lang or "").strip().lower()
        if lang_norm.startswith("vi"):
            return vi
        if lang_norm.startswith("ja"):
            return ja
        if lang_norm in {"", "auto"} or lang_norm.startswith("en"):
            return en
        # For other languages, translate fallback system text on the fly.
        return _translate_text(en, lang_norm)

    _translation_cache: dict[tuple[str, str], str] = {}

    def _translate_text(text: str, target_language: str) -> str:
        key = (text, target_language)
        cached = _translation_cache.get(key)
        if cached:
            return cached
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = (
                "Translate the following system message to the target language.\n"
                "Keep meaning, tone, and brevity.\n"
                f"Target language: {target_language}\n"
                "Return ONLY translated text.\n\n"
                f"Message:\n{text}"
            )
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=prompt)
            translated = str(getattr(resp, "text", "") or "").strip()
            if translated:
                _translation_cache[key] = translated
                return translated
        except Exception:
            pass
        return text

    def _resolve_working_files() -> list[dict]:
        if session_file_ids is not None:
            files = mongo.get_files_by_ids([str(fid) for fid in session_file_ids])
            allowed = {str(fid) for fid in session_file_ids}
            files = [f for f in files if str(f.get("_id", "")) in allowed]
        else:
            sess = mongo.get_chat_session(session_scope, user_email=user_email)
            if sess and isinstance(sess.get("file_ids"), list):
                files = mongo.get_files_by_ids([str(fid) for fid in sess.get("file_ids", [])])
            else:
                files = mongo.list_files(session_scope)
        if file_id:
            files = [f for f in files if str(f.get("_id", "")) == str(file_id)]
        return files

    def _build_working_file_context(files: list[dict]) -> list[dict]:
        out: list[dict] = []
        for f in files:
            fid = str(f.get("_id", "")).strip()
            if not fid:
                continue
            out.append(
                {
                    "file_id": fid,
                    "file_name": str(f.get("file_name", "") or fid),
                    "short_summary": str(f.get("short_summary") or f.get("summary") or ""),
                    "summary": str(f.get("summary") or ""),
                    "dxf_path": str(f.get("dxf_path", "") or ""),
                }
            )
        return out

    def _load_pages_for_files(files_ctx: list[dict], explicit_pages: list[int] | None = None) -> list[dict]:
        pages: list[dict] = []
        for f in files_ctx:
            fid = str(f.get("file_id", "")).strip()
            if not fid:
                continue
            rows = mongo.get_pages_by_file(
                fid,
                projection={
                    "_id": 1,
                    "file_id": 1,
                    "page_number": 1,
                    "image_url": 1,
                    "short_summary": 1,
                    "context_md": 1,
                    "dxf_path": 1,
                },
                page_numbers=explicit_pages or None,
            )
            for r in rows:
                pages.append(
                    {
                        "page_id": str(r.get("_id", "")),
                        "file_id": fid,
                        "file_name": str(f.get("file_name", fid)),
                        "page_number": int(r.get("page_number", 0) or 0),
                        "image_url": str(r.get("image_url", "") or ""),
                        "short_summary": str(r.get("short_summary", "") or ""),
                        "context_md": str(r.get("context_md", "") or ""),
                        "dxf_path": str(r.get("dxf_path", "") or ""),
                    }
                )
        pages.sort(key=lambda x: (str(x.get("file_name", "")), int(x.get("page_number", 0) or 0)))
        return pages

    def _build_citations_from_pages(scope_pages: list[dict], target_pages: list[int] | None = None) -> list[dict]:
        target = set(target_pages or [])
        out: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for p in scope_pages:
            fid = str(p.get("file_id", "")).strip()
            pno = int(p.get("page_number", 0) or 0)
            if not fid or pno <= 0:
                continue
            if target and pno not in target:
                continue
            key = (fid, pno)
            if key in seen:
                continue
            seen.add(key)
            out.append({"file_id": fid, "file_name": str(p.get("file_name", "") or fid), "page_number": pno})
        return out

    def _build_citations_from_hits(hits: list[dict], files_ctx: list[dict]) -> list[dict]:
        file_id_by_name: dict[str, str] = {}
        for f in files_ctx:
            fid = str(f.get("file_id", "")).strip()
            fname_key = _norm_doc_name(f.get("file_name", ""))
            if fid and fname_key and fname_key not in file_id_by_name:
                file_id_by_name[fname_key] = fid
        citations: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for h in hits:
            page_number = int(h.get("page_number", 0) or 0)
            file_name = str(h.get("file_name", h.get("file_id", ""))).strip()
            fid = str(h.get("file_id", "")).strip()
            if not fid:
                fid = file_id_by_name.get(_norm_doc_name(file_name), "")
            if not fid or page_number <= 0:
                continue
            key = (fid, page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append({"file_id": fid, "file_name": file_name or fid, "page_number": page_number})
        return citations

    def _pick_tool_scope_pages(all_pages: list[dict], explicit_pages: list[int], citations_hint: list[dict]) -> list[dict]:
        if explicit_pages:
            page_set = set(explicit_pages)
            scoped = [p for p in all_pages if int(p.get("page_number", 0) or 0) in page_set]
            if scoped:
                return scoped
        if citations_hint:
            wanted = {(str(c.get("file_id", "")), int(c.get("page_number", 0) or 0)) for c in citations_hint}
            scoped = [p for p in all_pages if (str(p.get("file_id", "")), int(p.get("page_number", 0) or 0)) in wanted]
            if scoped:
                return scoped
        # No implicit page fallback: avoid running tools on guessed scope.
        return []

    def _build_context_summary(query_text: str, turns: list[dict], files_ctx: list[dict], citations: list[dict]) -> str:
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = build_page_context_summary_prompt(
                query_text=query_text,
                history_ctx=_history_context(turns, limit=5),
                files_payload=json.dumps(files_ctx, ensure_ascii=False),
                citations_payload=json.dumps(citations, ensure_ascii=False),
            )
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=prompt)
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            return str(parsed.get("summary_text", "") or "").strip()
        except Exception:
            return ""

    def _try_direct_non_doc_answer(query_text: str) -> str | None:
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=build_direct_non_doc_prompt(query_text))
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            if not bool(parsed.get("is_direct", False)):
                return None
            answer = str(parsed.get("answer", "") or "").strip()
            return answer or None
        except Exception:
            return None

    def _plan_next_action(
        query_text: str,
        language_ctx: str,
        history_ctx: str,
        context_summary: str,
        files_ctx: list[dict],
        recent_citations: list[dict],
        explicit_pages: list[int],
        action_history: list[dict],
        tool_hint: str,
        has_image: bool,
    ) -> tuple[str, str]:
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = build_orchestrator_action_plan_prompt(
                query_text=query_text,
                language_ctx=language_ctx,
                history_ctx=history_ctx,
                context_summary=context_summary,
                files_payload=json.dumps(files_ctx, ensure_ascii=False),
                recent_citations_payload=json.dumps(recent_citations, ensure_ascii=False),
                explicit_pages_payload=json.dumps(explicit_pages, ensure_ascii=False),
                action_history_payload=json.dumps(action_history, ensure_ascii=False),
                tool_hint=tool_hint,
                has_image=has_image,
            )
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=prompt)
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            action = str(parsed.get("action", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip() or "plan"
            valid = {"page_reason", "search", "count", "area", "viz", "report_pdf", "report_docx", "report_excel", "direct_answer", "finalize"}
            valid = valid | {"file_agent"}
            if action in valid:
                return action, reason
        except Exception:
            pass
        if explicit_pages:
            return "page_reason", "fallback_explicit_pages"
        if tool_hint in {"search", "count", "area", "viz", "report_pdf", "report_docx", "report_excel"}:
            return tool_hint, "fallback_tool_hint"
        return "page_reason", "fallback_default"

    def _review_step(
        query_text: str,
        language_ctx: str,
        action_name: str,
        answer_text: str,
        tool_result: dict | None,
        pages_used: list[int],
        citations: list[dict],
        explicit_pages: list[int],
    ) -> tuple[bool, str, str]:
        try:
            from google import genai  # type: ignore
            from cad_pipeline.config import GEMINI_API_KEY, GEMINI_FLASH_MODEL

            client = genai.Client(api_key=GEMINI_API_KEY)
            cited_pages = sorted(
                {
                    int(c.get("page_number", 0) or 0)
                    for c in citations
                    if int(c.get("page_number", 0) or 0) > 0
                }
            )
            prompt = f"""You are reviewing a CAD QA step result.
Decide whether current answer is sufficient for the user query.

Query: "{query_text}"
Target language: "{language_ctx}"
Executed action: "{action_name}"
Answer:
---
{answer_text}
---
Lite evidence summary:
- citations_count: {len(citations)}
- pages_used_count: {len(pages_used)}
- explicit_pages_requested: {json.dumps(explicit_pages, ensure_ascii=False)}
- cited_pages: {json.dumps(cited_pages, ensure_ascii=False)}

Rules:
- If query is document-grounded and citations are missing/weak -> do not finalize.
- If explicit pages are requested and not covered by cited_pages -> do not finalize.
- If answer is sufficient and grounded -> finalize.

Reply ONLY JSON:
{{
  "finalize_now": true | false,
  "reason": "<short reason>",
  "next_action": "file_agent" | "page_reason" | "search" | "count" | "area" | "viz" | "report_pdf" | "report_docx" | "report_excel" | "direct_answer" | "finalize"
}}"""
            resp = client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=prompt)
            parsed = _parse_json_text(str(getattr(resp, "text", "") or ""))
            return (
                bool(parsed.get("finalize_now", False)),
                str(parsed.get("reason", "") or "review"),
                str(parsed.get("next_action", "page_reason")).strip().lower(),
            )
        except Exception:
            pass
        if explicit_pages:
            cited_pages = {int(c.get("page_number", 0) or 0) for c in citations}
            if not set(explicit_pages).issubset(cited_pages):
                return False, "explicit_pages_not_covered", "page_reason"
        if answer_text.strip() and (citations or action_name in {"direct_answer", "search"}):
            return True, "fallback_finalize", "finalize"
        return False, "fallback_continue", "page_reason"

    def _save_turn(answer: str, citations: list[dict], tool_result: dict | None) -> None:
        if not save_history:
            return
        try:
            mongo.append_chat_turn(
                session_id=session_scope,
                user_message=query,
                assistant_message=answer,
                folder_id=session_scope,
                user_email=user_email,
                user_meta={},
                assistant_meta={"citations": citations, "tool_result": tool_result or {}},
            )
        except Exception:
            pass

    with traced_span("qa.run", query=query, session_id=session_scope, file_id=file_id or ""):
        _emit("step_started", "Load session context")
        turns = mongo.get_chat_history(session_scope, user_email=user_email)
        recent_citations = _collect_recent_citations(turns, limit=5)
        language_context = _decide_language(query, turns)
        explicit_pages_requested = _extract_explicit_pages(query)
        files_ctx = _build_working_file_context(_resolve_working_files())
        context_summary = _build_context_summary(query, turns, files_ctx, recent_citations)
        history_ctx = _history_context(turns, limit=5)

        if not files_ctx and not image_bytes:
            answer = _msg(language_context, "Phiên này chưa có tài liệu nguồn để trả lời.", "このセッションには回答に使える資料がまだありません。", "This session has no source documents available for answering yet.")
            _save_turn(answer, [], {"mode": "no_files"})
            return {"answer": answer, "pages_used": [], "citations": [], "images": [], "tool_result": {"mode": "no_files"}, "source_file": "session", "query_type": "qa"}

        all_pages = _load_pages_for_files(files_ctx, explicit_pages_requested or None)
        if explicit_pages_requested and not all_pages:
            answer = _msg(language_context, "Không tìm thấy trang được chỉ định trong phạm vi session hiện tại.", "現在のセッション範囲で指定ページが見つかりませんでした。", "Requested explicit pages were not found in current session scope.")
            _save_turn(answer, [], {"mode": "explicit_page_not_found", "explicit_pages": explicit_pages_requested})
            return {"answer": answer, "pages_used": [], "citations": [], "images": [], "tool_result": {"mode": "explicit_page_not_found", "explicit_pages": explicit_pages_requested}, "source_file": "session", "query_type": "qa"}

        tool_hint = classify_tool(query=query, image_bytes=image_bytes).get("tool", "none")
        trace: list[dict] = []
        loop_guard: set[str] = set()

        final_answer = ""
        final_pages_used: list[int] = []
        final_citations: list[dict] = []
        final_images: list[str] = []
        final_tool_result: dict | None = None
        final_source_file = "session"

        for idx in range(4):
            action, plan_reason = _plan_next_action(
                query_text=query,
                language_ctx=language_context,
                history_ctx=history_ctx,
                context_summary=context_summary,
                files_ctx=files_ctx,
                recent_citations=recent_citations,
                explicit_pages=explicit_pages_requested,
                action_history=trace,
                tool_hint=tool_hint,
                has_image=bool(image_bytes),
            )
            step_id = f"s{idx + 1}"
            _emit("step_started", f"Orchestrator action: {action}", step_id=step_id)

            key = f"{action}:{len(final_citations)}"
            if key in loop_guard and action != "finalize":
                action = "page_reason"
            loop_guard.add(key)

            answer = ""
            pages_used: list[int] = []
            citations: list[dict] = []
            images: list[str] = []
            tool_result: dict | None = None
            source_file = "session"

            if action == "direct_answer":
                answer = _try_direct_non_doc_answer(query) or ""
                if answer:
                    tool_result = {"mode": "direct_non_doc"}
                else:
                    action = "page_reason"

            if action == "page_reason":
                _emit("tool_called", "Run page_reason executor", step_id=step_id, tool="page_reason")
                res = run_page_agent(
                    query=query,
                    pages=all_pages,
                    use_tools=False,
                    chat_history=turns,
                    recent_citations=recent_citations,
                    context_summary=context_summary,
                    language_context=language_context,
                    folder_id=session_scope,
                    file_id=file_id,
                    image_bytes=image_bytes,
                    answer_stream_callback=answer_stream_callback,
                    strict_page_scope=bool(explicit_pages_requested),
                )
                answer = str(res.get("answer", "") or "")
                pages_used = [int(p) for p in res.get("pages_used", []) if int(p) > 0]
                citations = res.get("citations", []) if isinstance(res.get("citations", []), list) else []
                if not citations:
                    citations = _build_citations_from_pages(all_pages, pages_used if pages_used else None)
                images = [str(v) for v in res.get("images", []) if str(v)]
                tool_result = {"mode": "page_reason", "need_tool": res.get("need_tool", "none")}
                source_file = "page_context"

            elif action == "search":
                _emit("tool_called", "Run search executor", step_id=step_id, tool="search")
                search_result = run_search_tool(
                    query=query,
                    image_bytes=image_bytes,
                    folder_id=None,
                    file_id=file_id,
                    top_n=10,
                )
                hits = search_result.get("results", []) if isinstance(search_result.get("results", []), list) else []
                allowed = {str(f.get("file_id", "")) for f in files_ctx}
                hits = [h for h in hits if str(h.get("file_id", "")).strip() in allowed] or hits
                citations = _build_citations_from_hits(hits, files_ctx)
                pages_used = [int(c.get("page_number", 0) or 0) for c in citations if int(c.get("page_number", 0) or 0) > 0]
                answer = _msg(
                    language_context,
                    ("Tìm thấy các trang liên quan:\n" + "\n".join([f"- {c['file_name']} (page {c['page_number']})" for c in citations])) if citations else "Không tìm thấy trang liên quan rõ ràng trong phạm vi session.",
                    ("関連ページが見つかりました:\n" + "\n".join([f"- {c['file_name']} (page {c['page_number']})" for c in citations])) if citations else "セッション範囲で明確な関連ページは見つかりませんでした。",
                    ("Found relevant pages:\n" + "\n".join([f"- {c['file_name']} (page {c['page_number']})" for c in citations])) if citations else "No clear relevant pages found in current session scope.",
                )
                tool_result = search_result
                source_file = "search_tool"

            elif action == "file_agent":
                target_file = None
                if final_citations or recent_citations:
                    ordered_citations = final_citations or recent_citations
                    fid_hint = str(ordered_citations[-1].get("file_id", "")).strip()
                    target_file = next((f for f in files_ctx if str(f.get("file_id", "")).strip() == fid_hint), None)
                if target_file is None and explicit_pages_requested:
                    explicit_set = set(explicit_pages_requested)
                    hinted_pages = [p for p in all_pages if int(p.get("page_number", 0) or 0) in explicit_set]
                    if hinted_pages:
                        fid_hint = str(hinted_pages[0].get("file_id", "")).strip()
                        target_file = next((f for f in files_ctx if str(f.get("file_id", "")).strip() == fid_hint), None)
                if target_file is None and files_ctx:
                    target_file = files_ctx[0]
                if target_file is None:
                    answer = _msg(
                        language_context,
                        "Không có file phù hợp để chạy file_agent.",
                        "file_agent を実行する対象ファイルがありません。",
                        "No suitable file is available for file_agent.",
                    )
                    tool_result = {"mode": "file_agent_scope_missing"}
                else:
                    file_result = run_file_agent(
                        query=query,
                        file_id=str(target_file.get("file_id", "")),
                        file_name=str(target_file.get("file_name", "")),
                        file_short_summary=str(target_file.get("short_summary", "")),
                        file_summary=str(target_file.get("summary", "")),
                        context_summary=context_summary,
                        recent_citations=[c for c in (final_citations or recent_citations) if str(c.get("file_id", "")) == str(target_file.get("file_id", ""))],
                        explicit_pages_requested=explicit_pages_requested,
                        language_context=language_context,
                    )
                    tool_result = {"mode": "file_agent", **file_result}
                    source_file = str(target_file.get("file_name", "session"))
                    if str(file_result.get("action", "")).strip().lower() == "answer":
                        answer = str(file_result.get("answer", "") or "").strip()
                    else:
                        candidate_pages = [
                            int(p)
                            for p in (file_result.get("candidate_pages") or [])
                            if isinstance(p, int) and p > 0
                        ]
                        file_pages = [
                            p
                            for p in all_pages
                            if str(p.get("file_id", "")).strip() == str(target_file.get("file_id", "")).strip()
                        ]
                        citations = _build_citations_from_pages(file_pages, candidate_pages or None)
                        pages_used = [int(c.get("page_number", 0) or 0) for c in citations if int(c.get("page_number", 0) or 0) > 0]
                        if citations:
                            lines = [f"- {c['file_name']} (page {c['page_number']})" for c in citations]
                            answer = _msg(
                                language_context,
                                "File agent gợi ý các trang liên quan:\n" + "\n".join(lines),
                                "file_agent が関連ページを提案しました:\n" + "\n".join(lines),
                                "File agent suggests relevant pages:\n" + "\n".join(lines),
                            )
                        else:
                            answer = _msg(
                                language_context,
                                "File agent chưa xác định được trang cụ thể, chuyển tiếp page-level.",
                                "file_agent は具体的なページを特定できず、page-level へ進みます。",
                                "File agent could not determine specific pages; continue to page-level.",
                            )

            elif action in {"count", "area", "viz", "report_pdf", "report_docx", "report_excel"}:
                scope_pages = _pick_tool_scope_pages(all_pages, explicit_pages_requested, final_citations or recent_citations)
                if not scope_pages:
                    if action == "count":
                        file_with_dxf = next((f for f in files_ctx if str(f.get("dxf_path", "")).strip()), None)
                        if file_with_dxf:
                            dxf_path = str(file_with_dxf.get("dxf_path", "")).strip()
                            tool_result = run_count_tool(query=query, dxf_path=dxf_path, context_md="")
                            answer = str(tool_result.get("details", "") or tool_result.get("count", ""))
                            source_file = str(file_with_dxf.get("file_name", "session"))
                        else:
                            answer = _msg(language_context, "Không có trang phù hợp để chạy tác vụ chuyên biệt.", "専用タスクを実行するための対象ページがありません。", "No suitable scoped pages available for specialist task.")
                            tool_result = {"mode": "scope_missing", "action": action}
                    else:
                        answer = _msg(language_context, "Không có trang phù hợp để chạy tác vụ chuyên biệt.", "専用タスクを実行するための対象ページがありません。", "No suitable scoped pages available for specialist task.")
                        tool_result = {"mode": "scope_missing", "action": action}
                else:
                    citations = _build_citations_from_pages(scope_pages)
                    pages_used = [int(c["page_number"]) for c in citations]
                    source_file = str(scope_pages[0].get("file_name", "session"))
                    if action == "count":
                        p = scope_pages[0]
                        tool_result = run_count_tool(query=query, dxf_path=p.get("dxf_path") or None, context_md=str(p.get("context_md", "") or ""))
                        answer = str(tool_result.get("details", "") or tool_result.get("count", ""))
                    elif action == "area":
                        p = scope_pages[0]
                        tool_result = run_area_tool(query=query, context_md=str(p.get("context_md", "") or ""))
                        answer = str(tool_result.get("details", "") or tool_result.get("area", ""))
                    elif action == "viz":
                        images = [str(p.get("image_url", "") or "") for p in scope_pages if str(p.get("image_url", "")).strip()]
                        tool_result = {"mode": "viz_scope", "image_count": len(images)}
                        answer = _msg(language_context, "Đã chuẩn bị các trang hình ảnh liên quan để bạn quan sát trực quan.", "視覚確認用に関連ページ画像を準備しました。", "Prepared relevant page images for visual inspection.")
                    elif action == "report_pdf":
                        tool_result = run_report_pdf(query=query, pages=scope_pages, tool_result=final_tool_result)
                        answer = _msg(language_context, f"Đã tạo báo cáo PDF: {tool_result.get('file_name', '')}", f"PDFレポートを生成しました: {tool_result.get('file_name', '')}", f"Generated PDF report: {tool_result.get('file_name', '')}")
                    elif action == "report_docx":
                        tool_result = run_report_docx(query=query, pages=scope_pages, tool_result=final_tool_result)
                        answer = _msg(language_context, f"Đã tạo báo cáo DOCX: {tool_result.get('file_name', '')}", f"DOCXレポートを生成しました: {tool_result.get('file_name', '')}", f"Generated DOCX report: {tool_result.get('file_name', '')}")
                    else:
                        tool_result = run_report_excel(query=query, pages=scope_pages, tool_result=final_tool_result, answer_text=final_answer, chat_history=turns)
                        answer = _msg(language_context, f"Đã tạo file Excel: {tool_result.get('file_name', '')}", f"Excelファイルを生成しました: {tool_result.get('file_name', '')}", f"Generated Excel file: {tool_result.get('file_name', '')}")

            elif action == "finalize":
                if final_answer:
                    break
                action = "page_reason"
                continue

            if answer:
                final_answer = answer
                final_pages_used = pages_used
                final_citations = citations
                final_images = images
                final_tool_result = tool_result
                final_source_file = source_file

            finalize_now, review_reason, next_action = _review_step(
                query_text=query,
                language_ctx=language_context,
                action_name=action,
                answer_text=answer or final_answer,
                tool_result=tool_result,
                pages_used=pages_used or final_pages_used,
                citations=citations or final_citations,
                explicit_pages=explicit_pages_requested,
            )
            trace.append({"step": idx + 1, "action": action, "plan_reason": plan_reason, "review_reason": review_reason, "next_action": next_action, "citations_count": len(citations or final_citations)})
            _emit("tool_result", f"Action {action}: {review_reason}", step_id=step_id, tool=action, status="done")
            if finalize_now and final_answer:
                break

        if not final_answer:
            final_answer = _msg(language_context, "Mình chưa đủ dữ liệu để trả lời chắc chắn trong phạm vi session hiện tại.", "現在のセッション範囲では確実に回答するための情報が不足しています。", "I do not yet have enough grounded evidence in this session to answer confidently.")

        if explicit_pages_requested:
            cited_pages = {int(c.get("page_number", 0) or 0) for c in final_citations}
            if not set(explicit_pages_requested).issubset(cited_pages):
                missing = [p for p in explicit_pages_requested if p not in cited_pages]
                final_tool_result = dict(final_tool_result or {})
                final_tool_result["explicit_pages_missing"] = missing
                if not final_citations:
                    fallback = _build_citations_from_pages(all_pages, explicit_pages_requested)
                    if fallback:
                        final_citations = fallback
                        final_pages_used = [int(c["page_number"]) for c in final_citations]

        final_tool_result = dict(final_tool_result or {})
        final_tool_result.setdefault("mode", "orchestrator_single")
        final_tool_result["orchestrator_trace"] = trace
        final_tool_result["language_context"] = language_context
        final_tool_result["explicit_pages_requested"] = explicit_pages_requested
        final_tool_result["context_summary"] = context_summary

        _save_turn(final_answer, final_citations, final_tool_result)
        _emit("finalizing", "Finalize response", status="done")
        return {
            "answer": final_answer,
            "pages_used": final_pages_used,
            "citations": final_citations,
            "images": final_images,
            "tool_result": final_tool_result,
            "source_file": final_source_file,
            "query_type": "qa",
        }
