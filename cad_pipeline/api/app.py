"""api/app.py — FastAPI server exposing the CAD pipeline as REST endpoints.

Endpoints:
  POST /upload                   → Upload & index a file
  POST /qa                       → Q&A query (text; optional image for search)
  POST /search                   → Semantic text search (JSON)
  GET  /tools/search/suggest     → Instant file/folder name lookup (no AI, for live typeahead)
  POST /tools/search             → Semantic search: text + optional image (multipart, on Enter)
  GET  /folders                  → List all folders
  GET  /folders/{id}/files       → List files in folder
  GET  /files/{id}/pages         → List pages in file
  GET  /tools/count              → Count symbols by keyword
  GET  /tools/area/units         → List unit types with areas
  GET  /health                   → Health check
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import queue
import shutil
import tempfile
import threading
import uuid
import hashlib
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest, urlopen

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cad_pipeline.api.auth import get_current_user, require_role, router as auth_router, decode_token
from cad_pipeline.agents.router import classify_query
from cad_pipeline.config import API_BASE_URL, LOCAL_CHAT_UPLOADS_DIR, LOCAL_IMAGES_DIR, USE_S3, REPORTS_DIR
from cad_pipeline.pipeline.upload_pipeline import run_upload_pipeline
from cad_pipeline.pipeline.qa_orchestrator_pipeline import run_qa
from cad_pipeline.pipeline.search_pipeline import run_search
from cad_pipeline.pipeline.delete_pipeline import delete_file as _delete_file, delete_folder as _delete_folder
from cad_pipeline.storage import mongo
from cad_pipeline.tools.count_tool import (
    run_count_tool,
    list_symbol_groups,
)
from cad_pipeline.tools.area_tool import (
    run_area_tool,
    get_all_units_summary,
    list_unit_types,
    get_unit_area,
)
from cad_pipeline.tools.search_tool import run_search_tool

app = FastAPI(
    title="CAD Pipeline API",
    description="Upload CAD drawings, index them, and run Q&A or semantic search.",
    version="1.0.0",
)

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve page/block images locally when USE_S3=false
# Accessible at: GET /images/{file_id}/pages/page_N.png
if not USE_S3:
    LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=LOCAL_IMAGES_DIR, follow_symlink=True), name="images")

# Persist uploaded chat images so history restore can render them after reload.
LOCAL_CHAT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/chat-uploads",
    StaticFiles(directory=LOCAL_CHAT_UPLOADS_DIR, follow_symlink=True),
    name="chat-uploads",
)

# Shared directory for generated reports (PDF / Excel)
# Accessible at: GET /reports/{filename}
_REPORTS_DIR = REPORTS_DIR


def _attach_report_download_url(result: dict) -> dict:
    """If tool_result has generated report file, expose stable download_url."""
    tool = result.get("tool_result") or {}
    if tool.get("format") not in ("pdf", "excel", "docx"):
        return result
    if not tool.get("file_path"):
        return result

    src = Path(tool["file_path"])
    if not src.exists():
        return result

    if src.resolve().parent != _REPORTS_DIR.resolve():
        dst = _REPORTS_DIR / src.name
        shutil.move(str(src), str(dst))
        src = dst
    result["download_url"] = f"/reports/{src.name}"
    return result


def _persist_chat_upload_image(image_bytes: bytes, filename: str | None = None) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        ext = ".png"
    digest = hashlib.sha1(image_bytes).hexdigest()[:16]
    safe_name = f"{uuid.uuid4().hex[:10]}_{digest}{ext}"
    out = LOCAL_CHAT_UPLOADS_DIR / safe_name
    out.write_bytes(image_bytes)
    return f"{API_BASE_URL}/chat-uploads/{safe_name}"


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── In-progress upload tracking ────────────────────────────────────────────

_upload_status: dict[str, dict] = {}
_qa_jobs: dict[str, dict] = {}
_ALLOWED_UPLOAD_EXTS = {
    ".pdf",
    ".dwg",
    ".dxf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "cad_pipeline"}


# ── Upload ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id: str
    file_id: str
    status: str


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    folder_id: str = Form("default"),
    folder_name: str = Form("Default Folder"),
    file_id: str = Form(None),
    _user: dict = Depends(require_role("viewer", "user", "editor", "admin")),
):
    """Upload a supported file to index into the pipeline."""
    job_id = str(uuid.uuid4())
    fid = file_id or str(uuid.uuid4())[:8]
    filename = file.filename or "upload.bin"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_EXTS:
        allowed = ", ".join(sorted(_ALLOWED_UPLOAD_EXTS))
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. Supported: {allowed}",
        )

    # Save upload to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    _upload_status[job_id] = {"status": "processing", "progress": [], "file_id": fid}
    user_email = _user["email"]
    notif_id = f"{user_email}::upload::{job_id}"
    mongo.upsert_notification(
        notification_id=notif_id,
        user_email=user_email,
        kind="upload",
        title="Upload is processing",
        message=f"{filename} is being processed in the background.",
        is_read=True,
        status="processing",
        job_id=job_id,
        file_id=fid,
        file_name=filename,
    )

    def _progress(msg: str) -> None:
        _upload_status[job_id]["progress"].append(msg)

    def _run() -> None:
        try:
            result = run_upload_pipeline(
                file_path=tmp_path,
                file_id=fid,
                file_name=file.filename,
                folder_id=folder_id,
                folder_name=folder_name,
                progress_callback=_progress,
            )
            _upload_status[job_id]["status"] = "done"
            _upload_status[job_id]["result"] = result
            mongo.update_notification_by_job(
                user_email=user_email,
                job_id=job_id,
                updates={
                    "title": "Upload completed",
                    "message": f"{filename} finished processing.",
                    "status": "done",
                    "is_read": False,
                },
            )
        except Exception as exc:
            _upload_status[job_id]["status"] = "error"
            _upload_status[job_id]["error"] = str(exc)
            mongo.update_notification_by_job(
                user_email=user_email,
                job_id=job_id,
                updates={
                    "title": "Upload failed",
                    "message": f"{filename} failed: {str(exc)}",
                    "status": "error",
                    "is_read": False,
                },
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    background_tasks.add_task(_run)

    return UploadResponse(job_id=job_id, file_id=fid, status="processing")


@app.get("/upload/{job_id}/status")
def upload_status(job_id: str):
    """Poll upload job status."""
    status = _upload_status.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


class NotificationReadRequest(BaseModel):
    is_read: bool = True


@app.get("/notifications")
def list_notifications(limit: int = 50, _user: dict = Depends(get_current_user)):
    rows = mongo.list_notifications(_user["email"], limit=max(1, min(limit, 200)))
    unread = sum(1 for r in rows if not r.get("is_read"))
    return {"notifications": rows, "unread_count": unread}


@app.patch("/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: str,
    req: NotificationReadRequest,
    _user: dict = Depends(get_current_user),
):
    ok = mongo.mark_notification_read(notification_id, _user["email"], is_read=bool(req.is_read))
    if not ok:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"id": notification_id, "is_read": bool(req.is_read)}


@app.patch("/notifications/read-all")
def mark_all_notifications_read(_user: dict = Depends(get_current_user)):
    modified = mongo.mark_all_notifications_read(_user["email"])
    return {"modified": modified}


# ── Q&A ────────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    query: str
    folder_id: str | None = None
    session_id: str
    file_id: str | None = None


class StartQAJobResponse(BaseModel):
    job_id: str
    status: str


@app.post("/qa")
def qa_endpoint(req: QARequest, _user: dict = Depends(get_current_user)):
    """Ask a text question about documents in a folder.
    For image-based queries use POST /qa/image (multipart).
    """
    scope_id = (req.session_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session_doc = mongo.get_chat_session(scope_id, user_email=_user["email"])
    if not session_doc:
        raise HTTPException(status_code=404, detail=f"Chat session '{scope_id}' not found")
    session_file_ids = session_doc.get("file_ids") or []
    session_folder_id = str(session_doc.get("folder_id") or "")
    allowed_set = {str(fid) for fid in session_file_ids}

    query_type = classify_query(req.query)

    if query_type == "search":
        results = run_search(
            query=req.query,
            folder_id=session_folder_id or None,
            file_id=req.file_id,
        )
        results = [r for r in results if str(r.get("file_id", "")) in allowed_set]
        return {"query_type": "search", "results": results}

    result = run_qa(
        query=req.query,
        session_id=scope_id,
        file_id=req.file_id,
        session_file_ids=session_file_ids,
        user_email=_user["email"],
    )
    return _attach_report_download_url(result)


@app.post("/qa/stream")
def qa_stream_endpoint(req: QARequest, _user: dict = Depends(get_current_user)):
    """Stream Q&A progress + answer chunks (SSE)."""
    scope_id = (req.session_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session_doc = mongo.get_chat_session(scope_id, user_email=_user["email"])
    if not session_doc:
        raise HTTPException(status_code=404, detail=f"Chat session '{scope_id}' not found")
    session_file_ids = session_doc.get("file_ids") or []

    events: queue.Queue[tuple[str, dict]] = queue.Queue()
    streamed_any_delta = False

    def _emit_step(step: dict) -> None:
        if isinstance(step, dict):
            payload = {
                "phase": str(step.get("phase", "step")),
                "message": str(step.get("message", "")),
                "step_id": str(step.get("step_id", "")),
                "tool": str(step.get("tool", "")),
                "status": str(step.get("status", "")),
            }
            events.put(("step", payload))
            return
        events.put(("step", {"phase": "step", "message": str(step)}))

    def _emit_delta(text: str) -> None:
        nonlocal streamed_any_delta
        if not text:
            return
        streamed_any_delta = True
        events.put(("delta", {"text": text}))

    def _run() -> None:
        try:
            result = run_qa(
                query=req.query,
                session_id=scope_id,
                file_id=req.file_id,
                session_file_ids=session_file_ids,
                user_email=_user["email"],
                progress_callback=_emit_step,
                answer_stream_callback=_emit_delta,
            )
            result = _attach_report_download_url(result)
            answer = str(result.get("answer") or "")
            if answer and not streamed_any_delta:
                chunk_size = 120
                for i in range(0, len(answer), chunk_size):
                    events.put(("delta", {"text": answer[i:i + chunk_size]}))
            events.put(("final", result))
        except Exception as exc:
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(("done", {}))

    threading.Thread(target=_run, daemon=True).start()

    def _stream():
        yield _sse_event("open", {"status": "ok"})
        while True:
            try:
                event, payload = events.get(timeout=20)
                yield _sse_event(event, payload)
                if event == "done":
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/qa/jobs", response_model=StartQAJobResponse)
def start_qa_job(
    req: QARequest,
    background_tasks: BackgroundTasks,
    _user: dict = Depends(get_current_user),
):
    """Start an async Q&A job and report pipeline progress by polling."""
    scope_id = (req.session_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session_doc = mongo.get_chat_session(scope_id, user_email=_user["email"])
    if not session_doc:
        raise HTTPException(status_code=404, detail=f"Chat session '{scope_id}' not found")
    session_file_ids = session_doc.get("file_ids") or []

    job_id = str(uuid.uuid4())
    _qa_jobs[job_id] = {
        "status": "processing",
        "user_email": _user["email"],
        "query": req.query,
        "session_id": scope_id,
        "folder_id": session_folder_id or scope_id,
        "file_id": req.file_id,
        "steps": [],
        "current_step": "",
        "result": None,
        "error": None,
    }

    def _append_step(step: dict) -> None:
        payload = _qa_jobs.get(job_id)
        if not payload:
            return
        if isinstance(step, dict):
            message = str(step.get("message", ""))
            payload["steps"].append(step)
            payload["current_step"] = message
        else:
            step_text = str(step)
            payload["steps"].append({"phase": "step", "message": step_text})
            payload["current_step"] = step_text

    def _run() -> None:
        try:
            result = run_qa(
                query=req.query,
                session_id=scope_id,
                file_id=req.file_id,
                session_file_ids=session_file_ids,
                user_email=_user["email"],
                progress_callback=_append_step,
            )
            payload = _qa_jobs.get(job_id)
            if payload is None:
                return
            payload["result"] = _attach_report_download_url(result)
            payload["status"] = "done"
        except Exception as exc:
            payload = _qa_jobs.get(job_id)
            if payload is None:
                return
            payload["status"] = "error"
            payload["error"] = str(exc)

    background_tasks.add_task(_run)
    return StartQAJobResponse(job_id=job_id, status="processing")


@app.get("/qa/jobs/{job_id}")
def get_qa_job_status(job_id: str, _user: dict = Depends(get_current_user)):
    payload = _qa_jobs.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="QA job not found")
    if payload.get("user_email") != _user["email"]:
        raise HTTPException(status_code=404, detail="QA job not found")
    return {k: v for k, v in payload.items() if k != "user_email"}


@app.post("/qa/image")
async def qa_image_endpoint(
    query: Annotated[str, Form()] = "",
    folder_id: Annotated[str, Form()] = "",
    session_id: Annotated[str, Form()] = "",
    file_id: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    _user: dict = Depends(get_current_user),
):
    """Q&A with optional image upload.

    Accepts multipart/form-data:
      - query     : text question (optional if image provided)
      - folder_id : folder to scope search
      - file_id   : file to scope search (optional)
      - image     : image file (PNG/JPG/WebP, optional)

    When an image is provided and the intent is "search", runs image-based
    search (title-block lookup first, then lexical fallback).
    Otherwise runs the standard Q&A pipeline with the image for additional
    context.
    """
    scope_id = (session_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session_doc = mongo.get_chat_session(scope_id, user_email=_user["email"])
    if not session_doc:
        raise HTTPException(status_code=404, detail=f"Chat session '{scope_id}' not found")
    session_file_ids = session_doc.get("file_ids") or []
    allowed_set = {str(fid) for fid in session_file_ids}
    session_folder_id = str(session_doc.get("folder_id") or "")

    img_bytes: bytes | None = None
    user_image_url: str | None = None
    if image and image.filename:
        img_bytes = await image.read()
        if img_bytes:
            user_image_url = _persist_chat_upload_image(img_bytes, image.filename)

    # Image only → force search_tool
    if img_bytes and not query:
        result = run_search_tool(
            query=None,
            image_bytes=img_bytes,
            folder_id=session_folder_id or None,
            file_id=file_id,
            top_n=10,
        )
        hits = result.get("results", []) or []
        result["results"] = [h for h in hits if str(h.get("file_id", "")) in allowed_set]
        result["total"] = len(result["results"])
        return {"query_type": "search", "tool_result": result, **result}

    result = run_qa(
        query=query,
        session_id=scope_id,
        file_id=file_id,
        session_file_ids=session_file_ids,
        user_email=_user["email"],
        image_bytes=img_bytes,
        user_image_url=user_image_url,
    )

    return _attach_report_download_url(result)


@app.post("/qa/image/stream")
async def qa_image_stream_endpoint(
    query: Annotated[str, Form()] = "",
    folder_id: Annotated[str, Form()] = "",
    session_id: Annotated[str, Form()] = "",
    file_id: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    _user: dict = Depends(get_current_user),
):
    """Stream Q&A progress + answer chunks (SSE) with optional uploaded image."""
    scope_id = (session_id or "").strip()
    if not scope_id:
        raise HTTPException(status_code=422, detail="session_id is required")
    session_doc = mongo.get_chat_session(scope_id, user_email=_user["email"])
    if not session_doc:
        raise HTTPException(status_code=404, detail=f"Chat session '{scope_id}' not found")
    session_file_ids = session_doc.get("file_ids") or []
    session_folder_id = str(session_doc.get("folder_id") or "")

    img_bytes: bytes | None = None
    user_image_url: str | None = None
    if image and image.filename:
        img_bytes = await image.read()
        if img_bytes:
            user_image_url = _persist_chat_upload_image(img_bytes, image.filename)

    events: queue.Queue[tuple[str, dict]] = queue.Queue()
    streamed_any_delta = False

    def _emit_step(step: dict) -> None:
        if isinstance(step, dict):
            payload = {
                "phase": str(step.get("phase", "step")),
                "message": str(step.get("message", "")),
                "step_id": str(step.get("step_id", "")),
                "tool": str(step.get("tool", "")),
                "status": str(step.get("status", "")),
            }
            events.put(("step", payload))
            return
        events.put(("step", {"phase": "step", "message": str(step)}))

    def _emit_delta(text: str) -> None:
        nonlocal streamed_any_delta
        if not text:
            return
        streamed_any_delta = True
        events.put(("delta", {"text": text}))

    def _run() -> None:
        try:
            result = run_qa(
                query=query,
                session_id=scope_id,
                file_id=file_id,
                session_file_ids=session_file_ids,
                user_email=_user["email"],
                image_bytes=img_bytes,
                user_image_url=user_image_url,
                progress_callback=_emit_step,
                answer_stream_callback=_emit_delta,
            )
            result = _attach_report_download_url(result)
            answer = str(result.get("answer") or "")
            if answer and not streamed_any_delta:
                chunk_size = 120
                for i in range(0, len(answer), chunk_size):
                    events.put(("delta", {"text": answer[i:i + chunk_size]}))
            events.put(("final", result))
        except Exception as exc:
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(("done", {}))

    threading.Thread(target=_run, daemon=True).start()

    def _stream():
        yield _sse_event("open", {"status": "ok"})
        while True:
            try:
                event, payload = events.get(timeout=20)
                yield _sse_event(event, payload)
                if event == "done":
                    break
            except queue.Empty:
                yield ": keep-alive\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Reports download ───────────────────────────────────────────────────────

@app.get("/reports/{filename}")
def download_report(filename: str):
    """Download a generated PDF or Excel report by filename."""
    filepath = _REPORTS_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")
    # Prevent path traversal
    try:
        filepath.resolve().relative_to(_REPORTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    media = (
        "application/pdf" if filename.endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if filename.endswith(".xlsx")
        else "application/octet-stream"
    )
    return FileResponse(path=str(filepath), media_type=media, filename=filename)


# ── Search ─────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    folder_id: str | None = None
    file_id: str | None = None


@app.post("/search")
def search_endpoint(req: SearchRequest, _user: dict = Depends(get_current_user)):
    """Semantic text search (JSON body)."""
    results = run_search(
        query=req.query,
        top_k=req.top_k,
        folder_id=req.folder_id,
        file_id=req.file_id,
    )
    return {"results": results, "total": len(results)}


@app.post("/tools/search")
async def tools_search_endpoint(
    query: Annotated[str, Form()] = "",
    folder_id: Annotated[str | None, Form()] = None,
    file_id: Annotated[str | None, Form()] = None,
    top_n: Annotated[int, Form()] = 10,
    image: Annotated[UploadFile | None, File()] = None,
    _user: dict = Depends(get_current_user),
):
    """Semantic search tool: text + optional image (multipart/form-data).

    Fields:
      - query     : search text (optional if image provided)
      - folder_id : restrict to folder (optional)
      - file_id   : restrict to file (optional)
      - top_n     : max results (default 10)
      - image     : image file (PNG/JPG/WebP, optional)
                    → Gemini Flash describes the image
                    → description is appended to query before embedding

    Returns top-N pages ranked by retrieval score with:
      rank, file_name, page_number, image_url, short_summary, vector_score,
      plus query_used and image_description.
    """
    img_bytes: bytes | None = None
    if image and image.filename:
        img_bytes = await image.read()

    if not query and not img_bytes:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: query (text) or image file.",
        )

    result = run_search_tool(
        query=query or None,
        image_bytes=img_bytes,
        folder_id=folder_id,
        file_id=file_id,
        top_n=top_n,
    )
    return result


@app.get("/tools/search/suggest")
def search_suggest(
    q: str,
    limit: int = 20,
    folder_id: str | None = None,
    _user: dict = Depends(get_current_user),
):
    """Instant file/folder typeahead — no AI, no embeddings.

    Called on every keystroke; returns matching files and folders
    directly from MongoDB using a case-insensitive regex on file_name,
    original_name, and tags.  When `folder_id` is supplied the search
    is scoped to that folder.

    Query params:
      q         : search text (required, min 1 char)
      limit     : max files to return (default 20)
      folder_id : optional folder scope

    Response shape:
    ```json
    {
      "query": "D318",
      "files": [
        {
          "file_id":     "…",
          "file_name":   "竣工図.pdf",
          "folder_id":   "…",
          "folder_name": "建築意匠図",
          "total_pages": 334,
          "tags":        ["竣工図", "意匠"]
        }
      ],
      "folders": [
        {
          "folder_id":  "…",
          "name":       "建築意匠図",
          "file_count": 12
        }
      ],
      "total_files":   1,
      "total_folders": 0
    }
    ```
    """
    if not q or not q.strip():
        raise HTTPException(status_code=422, detail="q must be a non-empty string.")

    files   = mongo.search_files(q.strip(), limit=limit, folder_id=folder_id)
    folders = mongo.search_folders(q.strip(), limit=10) if not folder_id else []

    return {
        "query":         q.strip(),
        "files":         files,
        "folders":       folders,
        "total_files":   len(files),
        "total_folders": len(folders),
    }


# ── Folders & Files ────────────────────────────────────────────────────────

class CreateFolderRequest(BaseModel):
    folder_id: str
    name: str
    summary: str = ""


class UpdateChatSessionSourcesRequest(BaseModel):
    folder_id: str | None = None
    file_ids: list[str]
    session_name: str | None = None


@app.post("/folders", status_code=201)
def create_folder(
    req: CreateFolderRequest,
    _user: dict = Depends(require_role("viewer", "user", "editor", "admin")),
):
    """Create a new folder (editor or admin only)."""
    existing = mongo.get_folder(req.folder_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Folder '{req.folder_id}' already exists")
    mongo.upsert_folder(req.folder_id, req.name, req.summary)
    return {"folder_id": req.folder_id, "name": req.name}


@app.get("/folders")
def list_folders(_user: dict = Depends(get_current_user)):
    folders = mongo.list_folders()
    for f in folders:
        f["id"] = str(f.pop("_id", ""))
    return {"folders": folders}


@app.get("/folders/{folder_id}/files")
def list_files(folder_id: str, _user: dict = Depends(get_current_user)):
    files = mongo.list_files(folder_id)
    for f in files:
        f["id"] = str(f.pop("_id", ""))
    return {"files": files}


@app.get("/files/{file_id}/pages")
def list_pages(file_id: str, _user: dict = Depends(get_current_user)):
    pages = mongo.get_pages_by_file(
        file_id,
        projection={"page_number": 1, "image_url": 1, "short_summary": 1},
    )
    for p in pages:
        p["id"] = str(p.pop("_id", ""))
    return {"pages": pages}


@app.get("/files/{file_id}")
def get_file_meta(file_id: str, _user: dict = Depends(get_current_user)):
    file_doc = mongo.get_file(file_id)
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "id": str(file_doc.get("_id", file_id)),
        "file_name": file_doc.get("file_name", file_id),
        "folder_id": file_doc.get("folder_id", ""),
        "total_pages": int(file_doc.get("total_pages") or 0),
        "updated_at": file_doc.get("updated_at"),
    }


@app.get("/files/{file_id}/pages/{page_number}")
def get_page_detail(
    file_id: str,
    page_number: int,
    _user: dict = Depends(get_current_user),
):
    """Get full page detail including blocks and context_md for the Blocks tab."""
    pages = mongo.get_pages_by_file(
        file_id,
        projection={
            "page_number": 1,
            "image_url": 1,
            "short_summary": 1,
            "context_md": 1,
            "blocks": 1,
        },
    )
    page = next((p for p in pages if p.get("page_number") == page_number), None)
    if not page:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found in file {file_id}")
    page["id"] = str(page.pop("_id", ""))
    return page


@app.api_route("/files/{file_id}/original", methods=["GET", "HEAD"])
def get_original_file(
    file_id: str,
    request: Request,
    access_token: str | None = None,
):
    """Stream the original uploaded file (PDF/DXF/other) for preview/download."""
    # Support both Authorization header and token query for iframe/img preview,
    # where setting custom headers is not always convenient.
    token: str | None = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif access_token:
        token = access_token
    if not token:
        raise HTTPException(status_code=401, detail="Authorization required")
    decode_token(token)

    file_doc = mongo.get_file(file_id)
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")

    file_url = file_doc.get("file_url")
    dxf_path = str(file_doc.get("dxf_path") or "").strip()
    if not file_url and not dxf_path:
        raise HTTPException(status_code=404, detail="Original file URL not found")

    file_name = str(file_doc.get("file_name") or file_id)
    file_ext = Path(file_name).suffix.lower()
    prefer_dxf = file_ext in {".dwg", ".dxf"} and bool(dxf_path)
    original_ref = dxf_path if prefer_dxf else str(file_url or "")
    if not original_ref and dxf_path:
        original_ref = dxf_path

    parsed = urlparse(str(original_ref))
    is_remote_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    filename = str(file_doc.get("file_name") or Path(parsed.path).name or file_id)
    if prefer_dxf:
        base = Path(file_name).stem or Path(parsed.path).stem or file_id
        filename = f"{base}.dxf"

    if is_remote_url:
        try:
            method = "HEAD" if request.method.upper() == "HEAD" else "GET"
            req = UrlRequest(str(original_ref), method=method)
            with urlopen(req, timeout=20) as upstream:
                content_type = upstream.headers.get("Content-Type", "") or "application/octet-stream"
                headers = {"Content-Disposition": f'inline; filename="{filename}"'}
                if method == "HEAD":
                    return Response(status_code=200, media_type=content_type, headers=headers)
                return Response(content=upstream.read(), media_type=content_type, headers=headers)
        except HTTPError as exc:
            raise HTTPException(status_code=exc.code or 502, detail=f"Cannot fetch original file URL: HTTP {exc.code}") from exc
        except URLError as exc:
            raise HTTPException(status_code=502, detail="Cannot fetch original file URL") from exc

    path = Path(str(original_ref))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type=media_type or "application/octet-stream",
        content_disposition_type="inline",
    )


# ── Delete ─────────────────────────────────────────────────────────────────

@app.delete("/files/{file_id}")
def delete_file_endpoint(
    file_id: str,
    folder_id: str,
    _user: dict = Depends(require_role("admin")),
):
    """Delete a file and all its data; rebuilds folder summary. Admin only."""
    file_doc = mongo.get_file(file_id)
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")
    result = _delete_file(file_id=file_id, folder_id=folder_id)
    return result


@app.delete("/folders/{folder_id}")
def delete_folder_endpoint(
    folder_id: str,
    _user: dict = Depends(require_role("admin")),
):
    """Delete a folder and all its files, pages, and chat history. Admin only."""
    folder = mongo.get_folder(folder_id)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    result = _delete_folder(folder_id=folder_id)
    mongo.delete_chat_sessions_by_folder(folder_id)
    return result


# ── Chat history ────────────────────────────────────────────────────────────

@app.get("/folders/{folder_id}/chat-history")
def get_chat_history(folder_id: str, _user: dict = Depends(get_current_user)):
    """Get the last 5 chat turns for a folder."""
    turns = mongo.get_chat_history(folder_id, user_email=_user["email"])
    return {"folder_id": folder_id, "turns": turns}


@app.delete("/folders/{folder_id}/chat-history")
def clear_chat_history(
    folder_id: str,
    _user: dict = Depends(get_current_user),
):
    """Clear chat history for a folder."""
    mongo.delete_chat_history(folder_id, user_email=_user["email"])
    return {"folder_id": folder_id, "cleared": True}


@app.get("/chat-sessions/{session_id}")
def get_chat_session_state(session_id: str, _user: dict = Depends(get_current_user)):
    doc = mongo.get_chat_session(session_id, user_email=_user["email"])
    if not doc:
        return {"session_id": session_id, "folder_id": session_id, "file_ids": [], "exists": False}
    return {
        "session_id": str(doc.get("session_id", session_id)),
        "folder_id": doc.get("folder_id", session_id),
        "file_ids": doc.get("file_ids", []),
        "session_name": doc.get("session_name"),
        "exists": True,
    }


@app.get("/chat-sessions")
def list_chat_sessions(_user: dict = Depends(get_current_user)):
    """List current user's chat sessions (user-scoped, not shared)."""
    rows = mongo.list_user_chat_sessions(_user["email"])
    result: list[dict] = []
    for row in rows:
        session_id = str(row.get("session_id") or row.get("folder_id") or "")
        if not session_id:
            continue
        folder_id = str(row.get("folder_id") or session_id)
        folder = mongo.get_folder(folder_id) or {}
        file_ids = row.get("file_ids", []) or []
        session_name = str(row.get("session_name") or "").strip()
        if folder:
            folder_name = session_name or folder.get("name", folder_id)
            file_count = len(mongo.list_files(folder_id))
        else:
            folder_name = session_name or session_id
            file_count = len(file_ids)
        result.append(
            {
                "session_id": session_id,
                "folder_id": folder_id,
                "folder_name": folder_name,
                "session_name": session_name or None,
                "file_count": file_count,
                "updated_at": row.get("updated_at"),
            }
        )
    return {"sessions": result}


@app.delete("/chat-sessions/{session_id}")
def delete_chat_session_state(
    session_id: str,
    clear_history: bool = True,
    _user: dict = Depends(get_current_user),
):
    """Delete chat-session state only (does NOT delete folder/files)."""
    doc = mongo.get_chat_session(session_id, user_email=_user["email"])
    folder_id = (doc or {}).get("folder_id", session_id)
    deleted_session = mongo.delete_chat_session(session_id, user_email=_user["email"])
    if clear_history:
        mongo.delete_chat_history(folder_id, user_email=_user["email"])
    return {
        "session_id": session_id,
        "folder_id": folder_id,
        "deleted_session": deleted_session,
        "cleared_history": clear_history,
    }


@app.put("/chat-sessions/{session_id}/sources")
def update_chat_session_sources(
    session_id: str,
    req: UpdateChatSessionSourcesRequest,
    _user: dict = Depends(get_current_user),
):
    file_ids = [fid for fid in req.file_ids if mongo.get_file(fid) is not None]
    folder_id = req.folder_id or (mongo.get_chat_session(session_id, user_email=_user["email"]) or {}).get("folder_id") or session_id
    mongo.upsert_chat_session_sources(
        session_id=session_id,
        folder_id=folder_id,
        file_ids=file_ids,
        session_name=(req.session_name or "").strip() or None,
        user_email=_user["email"],
    )
    return {
        "session_id": session_id,
        "folder_id": folder_id,
        "file_ids": file_ids,
        "session_name": (req.session_name or "").strip() or None,
        "exists": True,
    }


# ── Tools ──────────────────────────────────────────────────────────────────

@app.get("/tools/count/groups")
def get_symbol_groups(_user: dict = Depends(get_current_user)):
    """List all symbol group names available in the symbol database."""
    return {"groups": list_symbol_groups()}


@app.get("/tools/count")
def count_symbols(
    keyword: str,
    use_symbol_db: bool = True,
    _user: dict = Depends(get_current_user),
):
    """Count symbols matching a keyword or group name."""
    result = run_count_tool(query=keyword, use_symbol_db=use_symbol_db)
    return result


@app.get("/tools/area/units")
def get_unit_list(_user: dict = Depends(get_current_user)):
    """List all apartment unit types with their floor areas."""
    return {"units": get_all_units_summary()}


@app.get("/tools/area/units/{unit_label}")
def get_unit_detail(unit_label: str, _user: dict = Depends(get_current_user)):
    """Get detailed room breakdown for a specific unit type."""
    return get_unit_area(unit_label)


@app.post("/tools/count/context")
def count_in_page_context(query: str, page_id: str):
    """Count objects in a specific page's context using LLM."""
    page = mongo.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    context_md = page.get("context_md", "")
    result = run_count_tool(query=query, context_md=context_md, use_symbol_db=False)
    return result


@app.post("/tools/area/context")
def area_in_page_context(query: str, page_id: str):
    """Calculate area from a specific page's context using LLM."""
    page = mongo.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    context_md = page.get("context_md", "")
    result = run_area_tool(query=query, context_md=context_md)
    return result


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cad_pipeline.api.app:app", host="0.0.0.0", port=8001, reload=True)
