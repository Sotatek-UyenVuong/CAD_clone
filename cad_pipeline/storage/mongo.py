"""mongo.py — MongoDB CRUD operations for the CAD pipeline.

Collections:
  - folders   → folder metadata + summary
  - files     → file metadata + summary
  - pages     → page metadata + context_md + blocks
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any
import unicodedata

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from cad_pipeline.config import MONGODB_DB, MONGODB_URI


# ── Connection (lazy singleton) ────────────────────────────────────────────

_client: MongoClient | None = None
_db: Database | None = None


def get_db() -> Database:
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGODB_URI)
        _db = _client[MONGODB_DB]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db: Database) -> None:
    db["pages"].create_index([("file_id", ASCENDING)])
    db["pages"].create_index([("folder_id", ASCENDING)])
    db["pages"].create_index([("file_id", ASCENDING), ("page_number", ASCENDING)])
    db["files"].create_index([("folder_id", ASCENDING)])
    db["chat_sessions"].create_index([("folder_id", ASCENDING)])
    db["chat_sessions"].create_index([("user_email", ASCENDING), ("session_id", ASCENDING)])
    db["chat_history"].create_index([("session_id", ASCENDING)])
    db["chat_history"].create_index([("folder_id", ASCENDING)])
    db["chat_history"].create_index([("user_email", ASCENDING), ("session_id", ASCENDING)])
    db["chat_history"].create_index([("user_email", ASCENDING), ("folder_id", ASCENDING)])
    db["notifications"].create_index([("user_email", ASCENDING), ("created_at", ASCENDING)])
    db["notifications"].create_index([("user_email", ASCENDING), ("is_read", ASCENDING)])
    db["notifications"].create_index([("user_email", ASCENDING), ("job_id", ASCENDING)], unique=True, sparse=True)


# ── Folders ────────────────────────────────────────────────────────────────

def upsert_folder(folder_id: str, name: str, summary: str = "") -> None:
    db = get_db()
    db["folders"].update_one(
        {"_id": folder_id},
        {
            "$set": {
                "name": name,
                "summary": summary,
                "updated_at": _now(),
            },
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )


def get_folder(folder_id: str) -> dict | None:
    return get_db()["folders"].find_one({"_id": folder_id})


def list_folders() -> list[dict]:
    db = get_db()
    folders = list(db["folders"].find())
    for f in folders:
        f["file_count"] = db["files"].count_documents({"folder_id": f["_id"]})
    return folders


# ── Files ──────────────────────────────────────────────────────────────────

def upsert_file(
    file_id: str,
    folder_id: str,
    file_name: str,
    file_url: str,
    total_pages: int,
    summary: str = "",
    tags: list[str] | None = None,
    dxf_path: str | None = None,
) -> None:
    db = get_db()
    db["files"].update_one(
        {"_id": file_id},
        {
            "$set": {
                "folder_id": folder_id,
                "file_name": file_name,
                "file_url": file_url,
                "total_pages": total_pages,
                "summary": summary,
                "tags": tags or [],
                "dxf_path": dxf_path,   # local path đến file DXF gốc nếu có
                "updated_at": _now(),
            },
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )


def get_file(file_id: str) -> dict | None:
    return get_db()["files"].find_one({"_id": file_id})


def get_files_by_ids(file_ids: list[str]) -> list[dict]:
    if not file_ids:
        return []
    return list(get_db()["files"].find({"_id": {"$in": file_ids}}))


def list_files(folder_id: str) -> list[dict]:
    return list(get_db()["files"].find({"folder_id": folder_id}))


def search_files(
    q: str,
    limit: int = 20,
    folder_id: str | None = None,
) -> list[dict]:
    """Instant name-based file search across the whole library (no AI).

    Matches `file_name`, `original_name`, and `tags` using a
    case-insensitive regex.  Optionally scope to a single folder.

    Each result includes folder metadata for grouping in the UI.
    Returns up to `limit` files sorted by file_name.
    """
    db = get_db()
    q_norm = unicodedata.normalize("NFKC", q).strip()

    # Build filter
    tokens = re.findall(r"\w+", q_norm, flags=re.UNICODE)
    connector = r"[\s　_\-./\\()（）\[\]【】・,:：;；]*"
    pattern_str = connector.join(re.escape(t) for t in tokens) if tokens else re.escape(q_norm)
    pattern = re.compile(pattern_str, re.IGNORECASE)
    name_filter: dict = {
        "$or": [
            {"file_name":     {"$regex": pattern}},
            {"original_name": {"$regex": pattern}},
            {"tags":          {"$regex": pattern}},
        ]
    }
    if folder_id:
        name_filter["folder_id"] = folder_id

    files = list(
        db["files"]
        .find(name_filter, {
            "_id": 1, "folder_id": 1, "file_name": 1, "original_name": 1,
            "total_pages": 1, "short_summary": 1, "tags": 1, "created_at": 1,
        })
        .sort("file_name", ASCENDING)
        .limit(limit)
    )

    # Attach folder name (batch fetch to avoid N+1)
    fids = {f["folder_id"] for f in files}
    folders_map = {
        fol["_id"]: fol.get("name", fol["_id"])
        for fol in db["folders"].find({"_id": {"$in": list(fids)}}, {"name": 1})
    }
    for f in files:
        f["file_id"]    = str(f.pop("_id"))
        f["folder_name"] = folders_map.get(f["folder_id"], f["folder_id"])

    return files


def search_folders(q: str, limit: int = 10) -> list[dict]:
    """Instant folder name search across the whole library."""
    db = get_db()
    q_norm = unicodedata.normalize("NFKC", q).strip()
    tokens = re.findall(r"\w+", q_norm, flags=re.UNICODE)
    connector = r"[\s　_\-./\\()（）\[\]【】・,:：;；]*"
    pattern_str = connector.join(re.escape(t) for t in tokens) if tokens else re.escape(q_norm)
    pattern = re.compile(pattern_str, re.IGNORECASE)
    folders = list(
        db["folders"]
        .find({"name": {"$regex": pattern}}, {"_id": 1, "name": 1})
        .sort("name", ASCENDING)
        .limit(limit)
    )
    for fol in folders:
        fol["folder_id"]   = str(fol.pop("_id"))
        fol["file_count"]  = db["files"].count_documents({"folder_id": fol["folder_id"]})
    return folders


def update_file_summary(file_id: str, summary: str, tags: list[str] | None = None) -> None:
    update: dict = {"summary": summary, "updated_at": _now()}
    if tags is not None:
        update["tags"] = tags
    get_db()["files"].update_one({"_id": file_id}, {"$set": update})


def update_file_short_summary(file_id: str, short_summary: str) -> None:
    get_db()["files"].update_one(
        {"_id": file_id},
        {"$set": {"short_summary": short_summary, "updated_at": _now()}},
    )


def update_file_title_block_index(file_id: str, title_block_index: list[dict]) -> None:
    """Store per-page title-block metadata index for fast lookup."""
    get_db()["files"].update_one(
        {"_id": file_id},
        {"$set": {"title_block_index": title_block_index, "updated_at": _now()}},
    )


def update_folder_summary(folder_id: str, summary: str) -> None:
    get_db()["folders"].update_one(
        {"_id": folder_id},
        {"$set": {"summary": summary, "updated_at": _now()}},
    )


def delete_file(file_id: str) -> int:
    """Delete a file and all its pages. Returns number of pages deleted."""
    db = get_db()
    result = db["pages"].delete_many({"file_id": file_id})
    db["files"].delete_one({"_id": file_id})
    return result.deleted_count


def delete_folder(folder_id: str) -> dict:
    """Delete a folder, all its files, and all their pages."""
    db = get_db()
    file_ids = [f["_id"] for f in list_files(folder_id)]
    pages_deleted = db["pages"].delete_many({"folder_id": folder_id}).deleted_count
    files_deleted = db["files"].delete_many({"folder_id": folder_id}).deleted_count
    db["folders"].delete_one({"_id": folder_id})
    return {"file_ids": file_ids, "files_deleted": files_deleted, "pages_deleted": pages_deleted}


# ── Chat history ────────────────────────────────────────────────────────────

CHAT_HISTORY_LIMIT = 5


def append_chat_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    folder_id: str | None = None,
    user_email: str | None = None,
    user_meta: dict | None = None,
    assistant_meta: dict | None = None,
) -> None:
    """Append a Q&A turn to session chat history, keeping only the last 5."""
    db = get_db()
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    resolved_folder_id = str(folder_id or session_id)
    turn = {
        "role_user": user_message,
        "role_assistant": assistant_message,
        "user_meta": user_meta or {},
        "assistant_meta": assistant_meta or {},
        "ts": _now(),
    }
    db["chat_history"].update_one(
        {"_id": scope_id},
        {
            "$push": {
                "turns": {
                    "$each": [turn],
                    "$slice": -CHAT_HISTORY_LIMIT,
                }
            },
            "$set": {
                "session_id": session_id,
                "folder_id": resolved_folder_id,
                "user_email": user_email,
                "updated_at": _now(),
            },
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )
    # Keep chat_sessions in sync so session list/debugging can rely on a
    # concrete session document even when user never updates source file_ids.
    if user_email is not None:
        touch_chat_session(session_id=session_id, folder_id=resolved_folder_id, user_email=user_email)


def get_chat_history(session_id: str, user_email: str | None = None) -> list[dict]:
    """Return the last N chat turns for a session."""
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    doc = get_db()["chat_history"].find_one({"_id": scope_id})
    return doc.get("turns", []) if doc else []


def delete_chat_history(session_id: str, user_email: str | None = None) -> None:
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    get_db()["chat_history"].delete_one({"_id": scope_id})


def delete_chat_histories_by_folder(folder_id: str) -> int:
    result = get_db()["chat_history"].delete_many({"folder_id": folder_id})
    return result.deleted_count


# ── Chat sessions (source file selection per chatbot session) ──────────────

def upsert_chat_session_sources(
    session_id: str,
    folder_id: str,
    file_ids: list[str],
    session_name: str | None = None,
    user_email: str | None = None,
) -> None:
    """Persist selected source file ids for one chatbot session."""
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    set_fields: dict[str, Any] = {
        "session_id": session_id,
        "user_email": user_email,
        "folder_id": folder_id,
        "file_ids": file_ids,
        "updated_at": _now(),
    }
    if session_name is not None:
        set_fields["session_name"] = session_name
    get_db()["chat_sessions"].update_one(
        {"_id": scope_id},
        {
            "$set": set_fields,
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )


def touch_chat_session(
    session_id: str,
    folder_id: str,
    user_email: str,
) -> None:
    """Ensure a user-scoped chat session doc exists and update timestamp.

    Does not overwrite existing file_ids.
    """
    scope_id = f"{user_email}::{session_id}"
    get_db()["chat_sessions"].update_one(
        {"_id": scope_id},
        {
            "$set": {
                "session_id": session_id,
                "user_email": user_email,
                "folder_id": folder_id,
                "updated_at": _now(),
            },
            "$setOnInsert": {
                "file_ids": [],
                "created_at": _now(),
            },
        },
        upsert=True,
    )


def get_chat_session(session_id: str, user_email: str | None = None) -> dict | None:
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    return get_db()["chat_sessions"].find_one({"_id": scope_id})


def delete_chat_session(session_id: str, user_email: str | None = None) -> bool:
    scope_id = f"{user_email}::{session_id}" if user_email else session_id
    result = get_db()["chat_sessions"].delete_one({"_id": scope_id})
    return result.deleted_count > 0


def delete_chat_sessions_by_folder(folder_id: str, user_email: str | None = None) -> int:
    query: dict[str, Any] = {"folder_id": folder_id}
    if user_email is not None:
        query["user_email"] = user_email
    result = get_db()["chat_sessions"].delete_many(query)
    return result.deleted_count


def list_user_chat_sessions(user_email: str) -> list[dict]:
    """List user-scoped chat sessions (union of chat_sessions and chat_history)."""
    db = get_db()
    merged: dict[str, dict] = {}

    # Source 1: explicit chat session state (sources/file_ids)
    for doc in db["chat_sessions"].find({"user_email": user_email}):
        folder_id = str(doc.get("folder_id") or doc.get("session_id") or "")
        if not folder_id:
            continue
        merged[folder_id] = {
            "session_id": str(doc.get("session_id") or folder_id),
            "folder_id": folder_id,
            "file_ids": doc.get("file_ids", []),
            "session_name": doc.get("session_name"),
            "updated_at": doc.get("updated_at") or doc.get("created_at"),
        }

    # Source 2: chat history docs (session had messages, even without explicit sources)
    for doc in db["chat_history"].find({"user_email": user_email}):
        session_id = str(doc.get("session_id") or "")
        folder_id = str(doc.get("folder_id") or "")
        if not session_id:
            # Backward-compat: derive from scoped _id format "email::session_id"
            doc_id = str(doc.get("_id", ""))
            if "::" in doc_id:
                session_id = doc_id.split("::", 1)[1]
        if not session_id:
            continue
        if not folder_id:
            folder_id = session_id
        rec = merged.get(session_id)
        cand_ts = doc.get("updated_at") or doc.get("created_at")
        rec_ts = rec.get("updated_at") if rec else None
        if rec is None or (cand_ts is not None and (rec_ts is None or cand_ts > rec_ts)):
            merged[session_id] = {
                "session_id": session_id,
                "folder_id": folder_id,
                "file_ids": [],
                "session_name": None,
                "updated_at": cand_ts,
            }

    sessions = list(merged.values())
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    sessions.sort(key=lambda s: s.get("updated_at") or min_dt, reverse=True)
    return sessions


# ── Pages ──────────────────────────────────────────────────────────────────

def upsert_page(
    page_id: str,
    file_id: str,
    folder_id: str,
    page_number: int,
    image_url: str,
    short_summary: str,
    context_md: str,
    blocks: list[dict],
    dxf_path: str | None = None,
) -> None:
    db = get_db()
    doc: dict = {
        "file_id": file_id,
        "folder_id": folder_id,
        "page_number": page_number,
        "image_url": image_url,
        "short_summary": short_summary,
        "context_md": context_md,
        "blocks": blocks,
        "updated_at": _now(),
    }
    if dxf_path is not None:
        doc["dxf_path"] = dxf_path
    db["pages"].update_one(
        {"_id": page_id},
        {
            "$set": doc,
            "$setOnInsert": {"created_at": _now()},
        },
        upsert=True,
    )


def get_page(page_id: str) -> dict | None:
    return get_db()["pages"].find_one({"_id": page_id})


def get_pages_by_file(
    file_id: str,
    projection: dict | None = None,
    page_numbers: list[int] | None = None,
) -> list[dict]:
    proj = projection or {"context_md": 1, "page_number": 1, "image_url": 1, "short_summary": 1, "dxf_path": 1}
    query: dict = {"file_id": file_id}
    if page_numbers:
        normalized_pages: list[int] = []
        for p in page_numbers:
            if isinstance(p, int) and p > 0:
                normalized_pages.append(p)
        if normalized_pages:
            query["page_number"] = {"$in": sorted(set(normalized_pages))}
    return list(
        get_db()["pages"]
        .find(query, proj)
        .sort("page_number", ASCENDING)
    )


def get_pages_by_ids(page_ids: list[str]) -> list[dict]:
    return list(get_db()["pages"].find({"_id": {"$in": page_ids}}))


def search_pages_lexical(
    q: str,
    limit: int = 50,
    folder_id: str | None = None,
    file_id: str | None = None,
) -> list[dict]:
    """Search pages by lexical regex over summary/context (no embeddings)."""
    db = get_db()
    q_norm = unicodedata.normalize("NFKC", q).strip()
    if not q_norm:
        return []
    # Guard query explosion: oversized regex payload can trigger Mongo BSON parser errors.
    q_norm = q_norm[:1200]

    tokens = re.findall(r"\w+", q_norm, flags=re.UNICODE)
    # Keep lexical pattern bounded and meaningful.
    tokens = [t[:48] for t in tokens if t][:32]
    connector = r"[\s　_\-./\\()（）\[\]【】・,:：;；]*"
    pattern_str = connector.join(re.escape(t) for t in tokens) if tokens else re.escape(q_norm[:256])
    try:
        pattern = re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        # Fallback to plain escaped prefix if pattern still invalid.
        pattern = re.compile(re.escape(q_norm[:256]), re.IGNORECASE)

    text_filter: dict = {
        "$or": [
            {"short_summary": {"$regex": pattern}},
            {"context_md": {"$regex": pattern}},
        ]
    }
    scope_filter: dict[str, str] = {}
    if folder_id:
        scope_filter["folder_id"] = folder_id
    if file_id:
        scope_filter["file_id"] = file_id

    mongo_filter: dict = text_filter if not scope_filter else {"$and": [scope_filter, text_filter]}
    candidate_limit = max(int(limit) * 8, 80)
    docs = list(
        db["pages"].find(
            mongo_filter,
            {
                "_id": 1,
                "file_id": 1,
                "folder_id": 1,
                "page_number": 1,
                "image_url": 1,
                "short_summary": 1,
                "context_md": 1,
            },
        ).limit(candidate_limit)
    )

    ranked: list[dict] = []
    for doc in docs:
        summary = str(doc.get("short_summary", ""))
        context = str(doc.get("context_md", ""))
        summary_hits = len(pattern.findall(summary))
        context_hits = len(pattern.findall(context))
        score = float(summary_hits * 3 + context_hits)
        if score <= 0:
            continue
        ranked.append({
            "page_id": str(doc.get("_id", "")),
            "file_id": str(doc.get("file_id", "")),
            "folder_id": str(doc.get("folder_id", "")),
            "page_number": int(doc.get("page_number", 0) or 0),
            "image_url": str(doc.get("image_url", "")),
            "short_summary": summary,
            "score": score,
        })

    ranked.sort(key=lambda item: (-item["score"], item["page_number"]))
    return ranked[: max(1, int(limit))]


# ── Users ──────────────────────────────────────────────────────────────────

def upsert_user(email: str, hashed_password: str, role: str) -> None:
    """Create or update a user account."""
    db = get_db()
    db["users"].update_one(
        {"_id": email},
        {
            "$set": {
                "hashed_password": hashed_password,
                "role": role,
                "updated_at": _now(),
            },
            "$setOnInsert": {"created_at": _now(), "last_login": None},
        },
        upsert=True,
    )


def get_user(email: str) -> dict | None:
    return get_db()["users"].find_one({"_id": email})


def list_users() -> list[dict]:
    users = list(get_db()["users"].find({}, {"hashed_password": 0}))
    for u in users:
        u["email"] = u.pop("_id")
    return users


def update_user(email: str, updates: dict) -> None:
    updates["updated_at"] = _now()
    get_db()["users"].update_one({"_id": email}, {"$set": updates})


def update_user_last_login(email: str) -> None:
    get_db()["users"].update_one(
        {"_id": email},
        {"$set": {"last_login": _now()}},
    )


def delete_user(email: str) -> bool:
    result = get_db()["users"].delete_one({"_id": email})
    return result.deleted_count > 0


# ── Notifications ───────────────────────────────────────────────────────────

def upsert_notification(
    notification_id: str,
    user_email: str,
    kind: str,
    title: str,
    message: str,
    is_read: bool = False,
    status: str | None = None,
    job_id: str | None = None,
    file_id: str | None = None,
    file_name: str | None = None,
) -> None:
    doc: dict[str, Any] = {
        "user_email": user_email,
        "kind": kind,
        "title": title,
        "message": message,
        "is_read": is_read,
        "updated_at": _now(),
    }
    if status is not None:
        doc["status"] = status
    if job_id is not None:
        doc["job_id"] = job_id
    if file_id is not None:
        doc["file_id"] = file_id
    if file_name is not None:
        doc["file_name"] = file_name
    get_db()["notifications"].update_one(
        {"_id": notification_id},
        {"$set": doc, "$setOnInsert": {"created_at": _now()}},
        upsert=True,
    )


def update_notification_by_job(
    user_email: str,
    job_id: str,
    updates: dict[str, Any],
) -> bool:
    if not updates:
        return False
    updates["updated_at"] = _now()
    result = get_db()["notifications"].update_one(
        {"user_email": user_email, "job_id": job_id},
        {"$set": updates},
    )
    return result.matched_count > 0


def list_notifications(user_email: str, limit: int = 50) -> list[dict]:
    rows = list(
        get_db()["notifications"]
        .find({"user_email": user_email})
        .sort("created_at", -1)
        .limit(limit)
    )
    out: list[dict] = []
    for r in rows:
        row = dict(r)
        row["id"] = str(row.pop("_id"))
        out.append(row)
    return out


def mark_notification_read(notification_id: str, user_email: str, is_read: bool = True) -> bool:
    result = get_db()["notifications"].update_one(
        {"_id": notification_id, "user_email": user_email},
        {"$set": {"is_read": is_read, "updated_at": _now()}},
    )
    return result.matched_count > 0


def mark_all_notifications_read(user_email: str) -> int:
    result = get_db()["notifications"].update_many(
        {"user_email": user_email, "is_read": False},
        {"$set": {"is_read": True, "updated_at": _now()}},
    )
    return int(result.modified_count)


# ── Helper ─────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)
