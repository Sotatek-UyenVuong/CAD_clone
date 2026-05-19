"""upload_pipeline.py — Full upload & indexing pipeline.

Flow:
  File → S3 (original)
       → Images → S3 (pages)
       → Layout Detection
       → Per-block processing → S3 (block crops, optional)
       → Build Context (with S3 URLs)
      → Save MongoDB
      → Build file/folder summaries

USE_S3=true  → upload original + pages + blocks to S3, store S3 URLs in Mongo
USE_S3=false → keep images local, store local paths in Mongo
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import re
import shutil
import subprocess
import tempfile
from collections import Counter
import cv2
import numpy as np
from pathlib import Path
from typing import Callable
from PIL import Image

from cad_pipeline.config import API_BASE_URL, LOCAL_IMAGES_DIR, LOCAL_ORIGINALS_DIR, PDF_DPI, USE_S3
from cad_pipeline.core.pdf_to_images import pdf_to_page_images
from cad_pipeline.core.layout_detect import LayoutDetector
from cad_pipeline.core.marker_pdf import marker_ocr_document
from cad_pipeline.core.page_processor import process_page_blocks, generate_page_summary
from cad_pipeline.core.context_builder import (
    build_page_context,
    build_file_summary,
    build_folder_summary,
    generate_file_short_summary,
)
from cad_pipeline.storage import mongo

_MARKER_EXCEL_EXTS = {".xls", ".xlsx"}
_DOC_TO_PDF_EXTS = {".doc", ".docx"}
_CAD_DWG_DXF_EXTS = {".dwg", ".dxf"}
_FATAL_LLM_ERROR_TOKENS = (
    "resource_exhausted",
    "monthly spending cap",
    "billing account has exceeded",
    "api_key_invalid",
    "api key not valid",
    "invalid api key",
    "permission_denied",
)


def _safe_load_image(image_path: str | Path):
    """Load image robustly even if cv2.imread is unavailable."""
    image_path = Path(image_path)
    if hasattr(cv2, "imread"):
        image = cv2.imread(str(image_path))
        if image is not None:
            return image

    # Fallback path for environments where cv2 is a stub module.
    with Image.open(image_path) as pil_image:
        rgb = pil_image.convert("RGB")
    return np.asarray(rgb)[:, :, ::-1].copy()


def _encode_png_bytes(image: np.ndarray) -> bytes:
    """Encode BGR/gray numpy image as PNG bytes with OpenCV/Pillow fallback."""
    if hasattr(cv2, "imencode"):
        success, buf = cv2.imencode(".png", image)
        if success:
            return buf.tobytes()

    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image[:, :, ::-1])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pil_image.save(tmp.name, format="PNG")
        return Path(tmp.name).read_bytes()


def _contains_fatal_llm_error(text: object) -> bool:
    value = str(text or "").lower()
    if not value:
        return False
    return any(token in value for token in _FATAL_LLM_ERROR_TOKENS)


def _iter_text_values(obj: object):
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_text_values(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_text_values(item)


def _raise_on_fatal_page_errors(page_number: int, short_summary: str, processed_blocks: list[dict]) -> None:
    if _contains_fatal_llm_error(short_summary):
        raise RuntimeError(
            "Gemini fatal API error while generating page summary "
            f"(page {page_number})."
        )
    for text in _iter_text_values(processed_blocks):
        if _contains_fatal_llm_error(text):
            raise RuntimeError(
                "Gemini fatal API error while processing page blocks "
                f"(page {page_number})."
            )


def _raise_on_fatal_summary_error(text: str, stage: str) -> None:
    if _contains_fatal_llm_error(text):
        raise RuntimeError(f"Gemini fatal API error while {stage}.")


def run_upload_pipeline(
    file_path: str | Path,
    file_id: str | None = None,
    file_name: str | None = None,
    folder_id: str = "default",
    folder_name: str = "Default Folder",
    dpi: int = PDF_DPI,
    score_thr: float = 0.5,
    upload_blocks: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run the full upload and indexing pipeline for a file.

    Args:
        file_path: Path to the PDF, image, or document to index.
        file_id: Optional explicit file ID (auto-generated if None).
        file_name: Display name for the file (defaults to filename).
        folder_id: Which folder this file belongs to.
        folder_name: Display name for the folder.
        dpi: Resolution for PDF rendering.
        score_thr: Layout detection confidence threshold.
        upload_blocks: Whether to also upload individual block crops to S3.
        progress_callback: Optional callable(message) for progress updates.

    Returns:
        {"file_id", "folder_id", "total_pages", "page_ids", "original_url"}
    """
    file_path = Path(file_path)
    file_id = file_id or _make_id(str(file_path))
    file_name = file_name or file_path.name

    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Upload original file to S3 or persist local original ───────
    original_url = str(file_path)
    if USE_S3:
        log(f"[1/8] Uploading original file to S3...")
        from cad_pipeline.storage.s3_store import upload_original_file
        original_url = upload_original_file(file_path, folder_id, file_id, file_name)
    else:
        log(f"[1/8] S3 disabled — persisting original file locally")
        safe_name = Path(file_name).name or file_path.name
        target_dir = LOCAL_ORIGINALS_DIR / file_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        src_resolved = file_path.resolve()
        dst_resolved = target_path.resolve()
        if src_resolved != dst_resolved:
            shutil.copy2(file_path, target_path)
        original_url = str(target_path)

    suffix = file_path.suffix.lower()
    processing_path = file_path
    conversion_tmp_dir: Path | None = None
    if suffix in _DOC_TO_PDF_EXTS:
        log("[2/8] Converting DOC/DOCX to PDF...")
        conversion_tmp_dir = Path(tempfile.mkdtemp(prefix="cad_doc2pdf_"))
        processing_path = _convert_doc_to_pdf(file_path, conversion_tmp_dir)
        log(f"[2/8] Converted file ready: {processing_path.name}")

    if suffix in _MARKER_EXCEL_EXTS:
        return _run_marker_text_pipeline(
            file_path=file_path,
            file_id=file_id,
            file_name=file_name,
            folder_id=folder_id,
            folder_name=folder_name,
            original_url=original_url,
            log=log,
        )

    if suffix in _CAD_DWG_DXF_EXTS:
        cad_tmp_dir: Path | None = None
        try:
            dxf_path = file_path
            if suffix == ".dwg":
                log("[2/6] Converting DWG to DXF...")
                cad_tmp_dir = Path(tempfile.mkdtemp(prefix="cad_dwg2dxf_"))
                dxf_path = _convert_dwg_to_dxf(file_path, cad_tmp_dir)
                log(f"[2/6] Converted DWG -> DXF: {dxf_path.name}")
            dxf_path = _persist_dxf_file(dxf_path, file_id=file_id, file_name=file_name)
            log(f"[3/6] Persisted DXF: {dxf_path}")
            return _run_dxf_single_page_pipeline(
                source_file=file_path,
                dxf_path=dxf_path,
                file_id=file_id,
                file_name=file_name,
                folder_id=folder_id,
                folder_name=folder_name,
                original_url=original_url,
                log=log,
            )
        finally:
            if cad_tmp_dir is not None:
                shutil.rmtree(cad_tmp_dir, ignore_errors=True)

    # ── Step 2: Render PDF → page images ────────────────────────────────────
    log(f"[2/8] Rendering {file_name} → page images (dpi={dpi})...")
    pages_info = pdf_to_page_images(processing_path, file_id, LOCAL_IMAGES_DIR, dpi=dpi)
    total_pages = len(pages_info)

    # ── Step 3: Ensure folder + file records in MongoDB ─────────────────────
    log(f"[3/8] Creating folder + file records in MongoDB...")
    mongo.upsert_folder(folder_id, folder_name)
    mongo.upsert_file(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        file_url=original_url,
        total_pages=total_pages,
    )

    detector = LayoutDetector.get(score_thr=score_thr)
    layout_detection_enabled = True
    page_summaries: list[str] = []
    page_ids: list[str] = []
    title_block_index: list[dict] = []

    for page_info in pages_info:
        page_number = page_info["page_number"]
        image_path = page_info["image_path"]

        log(f"[4/8] Processing page {page_number}/{total_pages}...")

        # ── Step 4a: Upload page image to S3 or build local HTTP URL ────────
        if USE_S3:
            from cad_pipeline.storage.s3_store import upload_page_image
            image_url = upload_page_image(image_path, folder_id, file_id, page_number)
        else:
            # Serve via FastAPI StaticFiles mount at /images
            rel = Path(image_path).relative_to(LOCAL_IMAGES_DIR)
            image_url = f"{API_BASE_URL}/images/{rel}"

        # ── Step 4b: Load image for processing ──────────────────────────────
        image = _safe_load_image(image_path)

        # ── Step 5: Layout detection ─────────────────────────────────────────
        log(f"[5/8] Layout detection — page {page_number}...")
        if layout_detection_enabled:
            try:
                blocks = detector.predict_file(image_path)
            except ImportError as exc:
                layout_detection_enabled = False
                blocks = []
                log(
                    "[5/8] Layout detection unavailable "
                    f"({exc}) — continuing without layout blocks."
                )
        else:
            blocks = []

        # ── Step 6: Page summary + block processing — run concurrently ─────
        log(f"[6/8] Gemini: page summary + {len(blocks)} blocks in parallel — page {page_number}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _page_executor:
            _summary_future = _page_executor.submit(generate_page_summary, image)
            _blocks_future = _page_executor.submit(
                process_page_blocks, blocks, image
            )
            short_summary = _summary_future.result()
            processed_blocks = _blocks_future.result()
        _raise_on_fatal_page_errors(page_number, short_summary, processed_blocks)
        page_summaries.append(short_summary)
        title_block_index.extend(_extract_title_block_entries(page_number, processed_blocks))

        # ── Step 6c: Upload block crops to S3 (optional) ─────────────────────
        if USE_S3 and upload_blocks:
            _upload_block_crops(
                blocks=blocks,
                image=image,
                processed_blocks=processed_blocks,
                folder_id=folder_id,
                file_id=file_id,
                page_number=page_number,
            )

        # ── Step 7: Build context_md ─────────────────────────────────────────
        context_md = build_page_context(
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            blocks=processed_blocks,
        )

        page_id = f"{file_id}_p{page_number}"
        page_ids.append(page_id)

        # ── Step 7b: Save page to MongoDB ────────────────────────────────────
        mongo.upsert_page(
            page_id=page_id,
            file_id=file_id,
            folder_id=folder_id,
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            context_md=context_md,
            blocks=processed_blocks,
        )

    # ── Step 8: Build summaries ──────────────────────────────────────────────
    log("[8/8] Finalizing summaries...")

    # Build + save file-level summary (full, stored in DB for agent retrieval)
    file_summary = build_file_summary(page_summaries, file_name)
    mongo.update_file_summary(file_id, file_summary)

    # Generate a short summary for this file via Gemini Flash
    file_short_summary = generate_file_short_summary(file_name, page_summaries)
    _raise_on_fatal_summary_error(file_short_summary, "generating file short summary")
    mongo.update_file_short_summary(file_id, file_short_summary)
    mongo.update_file_title_block_index(file_id, title_block_index)

    # Rebuild folder-level summary using short summaries of all files in folder
    all_file_docs = mongo.list_files(folder_id)
    all_short_summaries = [f.get("short_summary") or f.get("summary", "") for f in all_file_docs if f.get("short_summary") or f.get("summary")]
    folder_summary = build_folder_summary(all_short_summaries)
    mongo.update_folder_summary(folder_id, folder_summary)

    log("✓ Done!")
    if conversion_tmp_dir is not None:
        shutil.rmtree(conversion_tmp_dir, ignore_errors=True)
    return {
        "file_id": file_id,
        "folder_id": folder_id,
        "total_pages": total_pages,
        "page_ids": page_ids,
        "original_url": original_url,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _run_marker_text_pipeline(
    *,
    file_path: Path,
    file_id: str,
    file_name: str,
    folder_id: str,
    folder_name: str,
    original_url: str,
    log: Callable[[str], None],
) -> dict:
    """Handle Excel (XLS/XLSX) with Marker text extraction.

    Each extracted sheet/page from Marker is treated as one page in CAD DB.
    """
    ext = file_path.suffix.lower()
    is_excel = ext in {".xls", ".xlsx"}
    log(f"[2/6] Marker extraction for {'Excel sheets' if is_excel else 'document pages'}...")
    marker_pages = marker_ocr_document(file_path)
    if not marker_pages:
        raise RuntimeError("Marker returned no page content for this document.")

    total_pages = len(marker_pages)

    log("[3/6] Creating folder + file records in MongoDB...")
    mongo.upsert_folder(folder_id, folder_name)
    mongo.upsert_file(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        file_url=original_url,
        total_pages=total_pages,
    )

    page_ids: list[str] = []
    page_summaries: list[str] = []
    for page_number in sorted(marker_pages):
        page_md = str(marker_pages.get(page_number, "") or "").strip()
        if not page_md:
            page_md = "(Empty content)"
        page_short = _marker_page_short_summary(page_md)
        page_summaries.append(page_short)
        context_md = (
            f"# {'Sheet' if is_excel else 'Page'} {page_number}\n\n"
            f"{page_md}"
        )
        page_id = f"{file_id}_p{page_number}"
        page_ids.append(page_id)
        mongo.upsert_page(
            page_id=page_id,
            file_id=file_id,
            folder_id=folder_id,
            page_number=page_number,
            image_url="",
            short_summary=page_short,
            context_md=context_md,
            blocks=[],
        )
    log("[4/6] Saving extracted pages to MongoDB...")

    log("[5/6] Building file/folder summaries...")
    file_summary = build_file_summary(page_summaries, file_name)
    mongo.update_file_summary(file_id, file_summary)
    file_short_summary = generate_file_short_summary(file_name, page_summaries)
    _raise_on_fatal_summary_error(file_short_summary, "generating file short summary")
    mongo.update_file_short_summary(file_id, file_short_summary)
    mongo.update_file_title_block_index(file_id, [])

    all_file_docs = mongo.list_files(folder_id)
    all_short_summaries = [
        f.get("short_summary") or f.get("summary", "")
        for f in all_file_docs
        if f.get("short_summary") or f.get("summary")
    ]
    folder_summary = build_folder_summary(all_short_summaries)
    mongo.update_folder_summary(folder_id, folder_summary)

    log("✓ Done!")
    return {
        "file_id": file_id,
        "folder_id": folder_id,
        "total_pages": total_pages,
        "page_ids": page_ids,
        "original_url": original_url,
    }


def _marker_page_short_summary(markdown: str, max_len: int = 900) -> str:
    text = re.sub(r"\s+", " ", markdown or "").strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _convert_doc_to_pdf(source_path: Path, output_dir: Path) -> Path:
    """Convert DOC/DOCX to PDF via LibreOffice headless."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "DOC/DOCX conversion to PDF failed. Ensure LibreOffice is installed "
            f"and callable as `soffice`. Details: {details[:300]}"
        )
    pdf_files = sorted(output_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdf_files:
        raise RuntimeError("DOC/DOCX conversion completed but produced no PDF output.")
    return pdf_files[0]


def _convert_dwg_to_dxf(source_path: Path, output_dir: Path) -> Path:
    """Convert DWG to DXF by reusing tools/_archive/dwg_to_dxf_converter.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parents[2] / "tools" / "_archive" / "dwg_to_dxf_converter.py"
    if not script_path.exists():
        raise RuntimeError(f"DWG converter script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("cad_archive_dwg_to_dxf", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter module spec from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    convert_fn = getattr(module, "convert", None)
    if not callable(convert_fn):
        raise RuntimeError("Converter script does not expose callable convert(dwg, out_dir).")

    converted = convert_fn(source_path, output_dir)
    if converted is None:
        raise RuntimeError(
            "DWG conversion failed via tools/_archive/dwg_to_dxf_converter.py. "
            "Ensure ODAFileConverter and xvfb-run are installed."
        )
    dxf_path = Path(converted)
    if not dxf_path.exists():
        raise RuntimeError(f"Converter returned missing DXF path: {dxf_path}")
    return dxf_path


def _persist_dxf_file(dxf_src: Path, *, file_id: str, file_name: str) -> Path:
    """Persist DXF artifact to stable storage directory for later tool use."""
    target_dir = LOCAL_ORIGINALS_DIR / file_id / "cad"
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(file_name).stem.strip() or dxf_src.stem.strip() or "drawing"
    target_path = target_dir / f"{base_name}.dxf"
    if dxf_src.resolve() != target_path.resolve():
        shutil.copy2(dxf_src, target_path)
    return target_path


def _clean_dxf_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\\[A-Za-z0-9;]+", "", text).replace("\\P", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_unit_label(text: str) -> bool:
    t = _clean_dxf_text(text)
    if not t:
        return False
    if len(t) > 40:
        return False
    # Keep labels that usually represent countable units.
    return bool(
        re.search(r"\d", t)
        or re.search(r"[A-Za-z]{1,6}\d{1,4}$", t)
        or re.search(r"[（(]\d{1,3}[)）]", t)
    )


def _extract_dxf_catalog(dxf_path: Path) -> dict:
    """Extract deterministic symbol/unit catalog from DXF for Q&A grounding."""
    try:
        import ezdxf  # type: ignore
    except Exception:
        return {"available": False, "reason": "ezdxf_unavailable", "top_symbols": [], "unit_labels": []}

    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
    except Exception as exc:
        return {
            "available": False,
            "reason": f"dxf_parse_error:{type(exc).__name__}",
            "top_symbols": [],
            "unit_labels": [],
        }

    insert_counts: Counter[str] = Counter()
    for entity in msp.query("INSERT"):
        try:
            name = str(entity.dxf.name or "").strip()
        except Exception:
            continue
        if not name or name.startswith("*"):
            continue
        insert_counts[name] += 1

    unit_label_counts: Counter[str] = Counter()
    for entity in msp:
        if entity.dxftype() not in ("TEXT", "MTEXT"):
            continue
        try:
            raw = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
        except Exception:
            continue
        text = _clean_dxf_text(str(raw or ""))
        if not _looks_like_unit_label(text):
            continue
        unit_label_counts[text] += 1

    def _collect_block_texts(block_name: str, depth: int = 0, seen: set[str] | None = None) -> list[str]:
        if depth > 2:
            return []
        if seen is None:
            seen = set()
        key = f"{block_name}@{depth}"
        if key in seen:
            return []
        seen.add(key)

        block = doc.blocks.get(block_name)
        if block is None:
            return []

        texts: list[str] = []
        for e in block:
            if e.dxftype() in ("TEXT", "MTEXT"):
                try:
                    raw = e.dxf.text if e.dxftype() == "TEXT" else e.text
                except Exception:
                    continue
                cleaned = _clean_dxf_text(str(raw or ""))
                if cleaned:
                    texts.append(cleaned)
            elif e.dxftype() == "INSERT":
                try:
                    nested_name = str(e.dxf.name or "").strip()
                except Exception:
                    nested_name = ""
                if nested_name and not nested_name.startswith("*"):
                    texts.extend(_collect_block_texts(nested_name, depth + 1, seen))
        # Keep order, unique.
        deduped: list[str] = []
        seen_text: set[str] = set()
        for t in texts:
            if t in seen_text:
                continue
            seen_text.add(t)
            deduped.append(t)
        return deduped

    def _pick_block_alias(texts: list[str]) -> str:
        if not texts:
            return ""
        priority_patterns = [
            r"[A-Za-z]{1,8}\d{1,4}",     # EV1, M2, etc.
            r"[（(]\d{1,3}[)）]",         # (1), （2）
            r"[一-龯ぁ-ゔァ-ヴー]{1,20}",  # Japanese words
        ]
        for pat in priority_patterns:
            for t in texts:
                if len(t) > 40:
                    continue
                if re.search(pat, t):
                    return t
        for t in texts:
            if 1 <= len(t) <= 24:
                return t
        return ""

    block_meta: dict[str, dict] = {}
    for name in insert_counts:
        texts = _collect_block_texts(name)
        block_meta[name] = {
            "alias": _pick_block_alias(texts),
            "samples": texts[:3],
        }

    top_symbols = [
        {
            "name": name,
            "count": int(cnt),
            "alias": str((block_meta.get(name) or {}).get("alias", "") or ""),
            "samples": (block_meta.get(name) or {}).get("samples", []),
        }
        for name, cnt in insert_counts.most_common(120)
    ]
    top_units = [{"label": label, "count": int(cnt)} for label, cnt in unit_label_counts.most_common(120)]
    return {
        "available": True,
        "insert_total": int(sum(insert_counts.values())),
        "distinct_symbol_count": int(len(insert_counts)),
        "distinct_unit_label_count": int(len(unit_label_counts)),
        "top_symbols": top_symbols,
        "unit_labels": top_units,
    }


def _build_dxf_catalog_markdown(catalog: dict) -> str:
    if not bool(catalog.get("available")):
        reason = str(catalog.get("reason", "unknown"))
        return f"## DXF symbol catalog\n- Catalog unavailable ({reason})."

    symbols = catalog.get("top_symbols", []) if isinstance(catalog.get("top_symbols", []), list) else []
    units = catalog.get("unit_labels", []) if isinstance(catalog.get("unit_labels", []), list) else []
    lines = [
        "## DXF symbol catalog",
        f"- INSERT total: {int(catalog.get('insert_total', 0) or 0)}",
        f"- Distinct symbol blocks: {int(catalog.get('distinct_symbol_count', 0) or 0)}",
        f"- Distinct unit/text labels: {int(catalog.get('distinct_unit_label_count', 0) or 0)}",
    ]

    if symbols:
        lines.append("- Top symbol blocks:")
        for item in symbols[:30]:
            name = str(item.get("name", "")).strip()
            cnt = int(item.get("count", 0) or 0)
            alias = str(item.get("alias", "")).strip()
            samples = item.get("samples", []) if isinstance(item.get("samples", []), list) else []
            if name:
                if alias:
                    lines.append(f"  - {name} ({alias}): {cnt}")
                else:
                    lines.append(f"  - {name}: {cnt}")
                if samples:
                    sample_text = "; ".join(str(v).strip() for v in samples[:2] if str(v).strip())
                    if sample_text:
                        lines.append(f"    - sample: {sample_text}")
    else:
        lines.append("- Top symbol blocks: (none detected)")

    if units:
        lines.append("- Unit/text labels:")
        for item in units[:30]:
            label = str(item.get("label", "")).strip()
            cnt = int(item.get("count", 0) or 0)
            if label:
                lines.append(f"  - {label}: {cnt}")
    else:
        lines.append("- Unit/text labels: (none detected)")

    return "\n".join(lines)


def _run_dxf_single_page_pipeline(
    *,
    source_file: Path,
    dxf_path: Path,
    file_id: str,
    file_name: str,
    folder_id: str,
    folder_name: str,
    original_url: str,
    log: Callable[[str], None],
) -> dict:
    """Index a DXF drawing as a single-page CAD file for count tool support."""
    dxf_path_abs = str(dxf_path.resolve())
    catalog = _extract_dxf_catalog(dxf_path)
    catalog_md = _build_dxf_catalog_markdown(catalog)
    distinct_symbols = int(catalog.get("distinct_symbol_count", 0) or 0)
    distinct_units = int(catalog.get("distinct_unit_label_count", 0) or 0)
    top_unit_names = [str(v.get("label", "")).strip() for v in (catalog.get("unit_labels", []) or []) if str(v.get("label", "")).strip()]
    top_unit_preview = ", ".join(top_unit_names[:8]) if top_unit_names else "none"

    log("[3/6] Creating folder + file records in MongoDB...")
    mongo.upsert_folder(folder_id, folder_name)
    mongo.upsert_file(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        file_url=original_url,
        total_pages=1,
        dxf_path=dxf_path_abs,
    )

    page_id = f"{file_id}_p1"
    page_short_summary = (
        f"DXF catalog ready ({distinct_symbols} symbol blocks, {distinct_units} unit/text labels). "
        f"Top unit labels: {top_unit_preview}. Source: {Path(dxf_path_abs).name}."
    )
    context_md = (
        "# Page 1\n\n"
        f"- Source file: {source_file.name}\n"
        f"- DXF path: {dxf_path_abs}\n"
        "- This file is indexed as CAD geometry for count workflows.\n\n"
        f"{catalog_md}"
    )
    log("[4/6] Creating DXF-backed page context...")
    mongo.upsert_page(
        page_id=page_id,
        file_id=file_id,
        folder_id=folder_id,
        page_number=1,
        image_url="",
        short_summary=page_short_summary,
        context_md=context_md,
        blocks=[],
        dxf_path=dxf_path_abs,
    )

    log("[5/6] Building file/folder summaries...")
    file_summary = (
        f"File: {file_name}\n"
        "Type: CAD drawing (DWG/DXF)\n"
        f"DXF source: {Path(dxf_path_abs).name}\n"
        "This file is prepared for geometry-based counting.\n\n"
        f"{catalog_md}"
    )
    mongo.update_file_summary(file_id, file_summary)
    mongo.update_file_short_summary(file_id, page_short_summary)
    mongo.update_file_title_block_index(file_id, [])

    all_file_docs = mongo.list_files(folder_id)
    all_short_summaries = [
        f.get("short_summary") or f.get("summary", "")
        for f in all_file_docs
        if f.get("short_summary") or f.get("summary")
    ]
    folder_summary = build_folder_summary(all_short_summaries)
    mongo.update_folder_summary(folder_id, folder_summary)
    log("✓ Done!")
    return {
        "file_id": file_id,
        "folder_id": folder_id,
        "total_pages": 1,
        "page_ids": [page_id],
        "original_url": original_url,
    }

def _upload_block_crops(
    blocks,
    image,
    processed_blocks: list[dict],
    folder_id: str,
    file_id: str,
    page_number: int,
) -> None:
    """Upload each block crop to S3 and attach the crop_url to processed_blocks."""
    from cad_pipeline.storage.s3_store import upload_block_crop

    for i, (block, proc) in enumerate(zip(blocks, processed_blocks)):
        crop = block.crop(image)
        try:
            png_bytes = _encode_png_bytes(crop)
        except Exception:
            continue
        crop_url = upload_block_crop(
            image_bytes=png_bytes,
            folder_id=folder_id,
            file_id=file_id,
            page_number=page_number,
            block_type=block.label,
            block_index=i,
        )
        proc["crop_url"] = crop_url


def _make_id(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:12]


def _normalize_title_block_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_title_block_entries(page_number: int, processed_blocks: list[dict]) -> list[dict]:
    """Extract normalized title-block metadata from processed blocks for indexing."""
    entries: list[dict] = []
    for block in processed_blocks:
        if str(block.get("type", "")) != "title_block":
            continue
        content = block.get("content")
        if not isinstance(content, dict):
            continue
        drawing_no = _normalize_title_block_text(
            content.get("drawing_no")
            or content.get("drawing_number")
            or content.get("sheet_no")
            or content.get("sheet_number")
            or content.get("図面番号")
            or content.get("図番")
        )
        drawing_title = _normalize_title_block_text(
            content.get("drawing_title")
            or content.get("title")
            or content.get("sheet_title")
            or content.get("図面名称")
            or content.get("図面名")
        )
        project = _normalize_title_block_text(
            content.get("project")
            or content.get("project_name")
            or content.get("工事名")
            or content.get("物件名")
        )
        if not (drawing_no or drawing_title or project):
            continue
        entries.append(
            {
                "page_number": int(page_number),
                "drawing_no": drawing_no,
                "drawing_title": drawing_title,
                "project": project,
                "source": "title_block",
                "raw": content,
            }
        )
    return entries
