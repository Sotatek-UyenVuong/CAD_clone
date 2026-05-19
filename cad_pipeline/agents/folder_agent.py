"""Backward-compat wrapper for session-level summary router."""

from __future__ import annotations

from cad_pipeline.agents.session_agent import run_session_agent


def run_folder_agent(
    query: str,
    files: list[dict],
) -> dict:
    """Deprecated name. Use `run_session_agent` instead."""
    return run_session_agent(query=query, files=files)
