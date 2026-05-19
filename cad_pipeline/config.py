"""config.py — Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the sub-project directory first, then fall back to root.
# First file uses override=True so stale DATABASE_* from the shell does not win over cad_pipeline/.env.
_HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_HERE / ".env", override=True)
load_dotenv(dotenv_path=_HERE.parent / ".env", override=False)

# ── MongoDB ────────────────────────────────────────────────────────────────
MONGODB_URI: str = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("DATABASE_NAME", "cad_pipeline")

# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
GEMINI_PRO_MODEL: str = "gemini-3.1-pro-preview"
# ── Search tuning ──────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "100"))
TOP_N: int = int(os.getenv("TOP_N", "15"))

# ── Layout detection model ─────────────────────────────────────────────────
PROJECT_ROOT: Path = _HERE.parent
LAYOUT_WEIGHTS: Path = (
    PROJECT_ROOT
    / "layout_detect"
    / "models"
    / "checkpoints"
    / "cad_layout_v7_swapsplit"
    / "model_final.pth"
)
LAYOUT_CLASSES: list[str] = ["text", "table", "title_block", "diagram", "image"]
LAYOUT_SCORE_THR: float = float(os.getenv("LAYOUT_SCORE_THR", "0.5"))
LAYOUT_MIN_SIZE: int = 1280
LAYOUT_MAX_SIZE: int = 2000

# ── Symbol database ────────────────────────────────────────────────────────
SYMBOL_DB_DIR: Path = PROJECT_ROOT / "symbol_db"
SYMBOLS_JSON: Path = SYMBOL_DB_DIR / "symbols_enriched.json"
SYMBOL_GROUPS_JSON: Path = SYMBOL_DB_DIR / "symbol_groups.json"
OBJECT_DESCRIPTIONS_JSON: Path = _HERE / "object_descriptions.json"

# ── Cloudflare R2 credentials ──────────────────────────────────────────────
R2_ACCOUNT_ID: str = os.getenv("ACCOUNT_ID", "")
S3_ACCESS_KEY: str = os.getenv("CLIENT_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.getenv("CLIENT_SECRET", "")
S3_BUCKET: str = os.getenv("R2_BUCKET_NAME", "")
S3_PUBLIC_BASE_URL: str = os.getenv("R2_PUBLIC_URL", "")

# Auto-derived — không cần điền tay
S3_ENDPOINT_URL: str = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
S3_REGION: str = "auto"
USE_S3: bool = os.getenv("USE_S3", "false").lower() == "true"

# ── Local fallback (used when USE_S3=false) ────────────────────────────────
LOCAL_IMAGES_DIR: Path = Path(os.getenv("IMAGES_DIR", str(_HERE / "data" / "images")))
LOCAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_ORIGINALS_DIR: Path = Path(os.getenv("ORIGINALS_DIR", str(_HERE / "data" / "originals")))
LOCAL_ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CHAT_UPLOADS_DIR: Path = Path(os.getenv("CHAT_UPLOADS_DIR", str(_HERE / "data" / "chat_uploads")))
LOCAL_CHAT_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Base URL cho static file serving (FastAPI mount /images)
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")

# ── PDF rendering ──────────────────────────────────────────────────────────
PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))

# ── Marker API (Datalab.to) ────────────────────────────────────────────────
MARKER_API_KEY: str = os.getenv("MARKER_API_KEY", "")
MARKER_API_URL: str = "https://www.datalab.to/api/v1/marker"
MARKER_POLL_INTERVAL: float = 2.0   # seconds between status polls
MARKER_MAX_POLLS: int = 150         # max ~5 min

# ── Agent settings ─────────────────────────────────────────────────────────
AGENT_MAX_PAGES: int = int(os.getenv("AGENT_MAX_PAGES", "25"))

# ── Reports output directory ───────────────────────────────────────────────
REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", str(_HERE.parent / "reports")))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
