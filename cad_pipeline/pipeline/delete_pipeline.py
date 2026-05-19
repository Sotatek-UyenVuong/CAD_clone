"""delete_pipeline.py — Delete a file or folder from all stores.

File deletion:
  1. Delete MongoDB pages + file doc
  2. Rebuild folder summary from remaining files

Folder deletion:
  1. Delete MongoDB pages + files + folder doc
  2. Delete chat history for folder
"""

from __future__ import annotations

from cad_pipeline.core.context_builder import build_folder_summary
from cad_pipeline.storage import mongo


def delete_file(file_id: str, folder_id: str) -> dict:
    """Remove a file and all its data; rebuild folder summary.

    Returns:
        {"file_id": str, "pages_deleted": int, "folder_summary_updated": bool}
    """
    # 1. MongoDB pages + file doc
    pages_deleted = mongo.delete_file(file_id)

    # 2. Rebuild folder summary from remaining files
    remaining = mongo.list_files(folder_id)
    if remaining:
        short_summaries = [
            f.get("short_summary") or f.get("summary", "")
            for f in remaining
            if f.get("short_summary") or f.get("summary")
        ]
        folder_summary = build_folder_summary(short_summaries)
        mongo.update_folder_summary(folder_id, folder_summary)
        folder_summary_updated = True
    else:
        # No files left — clear folder summary
        mongo.update_folder_summary(folder_id, "")
        folder_summary_updated = True

    return {
        "file_id": file_id,
        "pages_deleted": pages_deleted,
        "folder_summary_updated": folder_summary_updated,
    }


def delete_folder(folder_id: str) -> dict:
    """Remove a folder and everything inside it.

    Returns:
        {"folder_id": str, "files_deleted": int, "pages_deleted": int}
    """
    # 1. MongoDB — pages + files + folder doc
    result = mongo.delete_folder(folder_id)

    # 2. Chat histories (all users)
    mongo.delete_chat_histories_by_folder(folder_id)

    return {
        "folder_id": folder_id,
        "files_deleted": result["files_deleted"],
        "pages_deleted": result["pages_deleted"],
    }
