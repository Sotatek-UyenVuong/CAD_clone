"""upload_rendered.py — Upload pipeline using pre-rendered images & pre-detected layout.

Designed for the TAE dataset where:
  - Page PNGs already exist in a rendered_*_dpi300 directory
  - Layout detection .txt files (YOLO format) already exist alongside each PNG
  - DXF files exist in a dxf_output directory, mapped by page index

This skips:
  - Step 2: PDF → PNG rendering (images already exist)
  - Step 5: Layout detection (results already in .txt files)

Adds:
  - Per-page DXF path stored in MongoDB
  - Parallel page processing (N_PARALLEL pages at a time)

Checkpoint/resume:
  - A JSON checkpoint file tracks done_pages
  - On resume: done pages are already in MongoDB, only interrupted pages are re-processed
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import shutil
import sys
import threading
from itertools import islice
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cad_pipeline.config import API_BASE_URL, LOCAL_IMAGES_DIR, USE_S3
from cad_pipeline.core.layout_detect import LayoutBlock, LAYOUT_CLASSES
from cad_pipeline.core.page_processor import process_page_blocks, generate_page_summary
from cad_pipeline.core.context_builder import (
    build_page_context,
    build_file_summary,
    build_folder_summary,
    generate_file_short_summary,
)
from cad_pipeline.storage import mongo

# Number of pages to process concurrently
N_PARALLEL = 10


def _safe_load_image(image_path: str | Path):
    """Load image robustly even if cv2.imread is unavailable."""
    image_path = Path(image_path)
    if hasattr(cv2, "imread"):
        image = cv2.imread(str(image_path))
        if image is not None:
            return image

    with Image.open(image_path) as pil_image:
        rgb = pil_image.convert("RGB")
    return np.asarray(rgb)[:, :, ::-1].copy()


# ── DXF mapping helpers ─────────────────────────────────────────────────────

def build_dxf_page_map(dxf_root: Path) -> dict[int, str]:
    """Recursively scan dxf_root for .dxf files and build {0-based-page-index: abs_path}.

    Patterns handled:
      D000_表紙.dxf   → page 0    E0000　図面.dxf → page 0
      S09_ PCa.dxf   → page 9    M-001_機器.dxf  → page 1
      S00-08_*.dxf   → page 0   (first number in a range)
      S10A_*.dxf     → page 10  (trailing letter ignored)
    """
    page_map: dict[int, str] = {}
    if not dxf_root.exists():
        return page_map
    for dxf_path in sorted(dxf_root.rglob("*.dxf")):
        page_idx = _extract_page_index(dxf_path.stem)
        if page_idx is not None and page_idx not in page_map:
            page_map[page_idx] = str(dxf_path)
    return page_map


def _extract_page_index(stem: str) -> int | None:
    stem = stem.strip()
    m = re.match(r"^[A-Za-z]*[-_]?(\d+)", stem)
    if m:
        return int(m.group(1))
    m2 = re.match(r"^(\d+)", stem)
    return int(m2.group(1)) if m2 else None


# ── Class names ──────────────────────────────────────────────────────────────

def load_class_names(rendered_dir: Path) -> list[str]:
    """Read class order from classes.txt. Falls back to LAYOUT_CLASSES."""
    classes_txt = rendered_dir / "classes.txt"
    if classes_txt.exists():
        names = [l.strip() for l in classes_txt.read_text().splitlines() if l.strip()]
        if names:
            return names
    return list(LAYOUT_CLASSES)


# ── YOLO → LayoutBlock conversion ───────────────────────────────────────────

def load_blocks_from_yolo_txt(
    txt_path: Path,
    img_width: int,
    img_height: int,
    class_names: list[str],
) -> list[LayoutBlock]:
    """Parse YOLO-format .txt into LayoutBlock objects.

    Format: class_id x_center y_center width height  (normalized 0-1)
    """
    blocks: list[LayoutBlock] = []
    if not txt_path.exists():
        return blocks
    with txt_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            x1 = max(0, int((xc - w / 2) * img_width))
            y1 = max(0, int((yc - h / 2) * img_height))
            x2 = min(img_width, int((xc + w / 2) * img_width))
            y2 = min(img_height, int((yc + h / 2) * img_height))
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            label = class_names[cls_id] if cls_id < len(class_names) else "text"
            blocks.append(
                LayoutBlock(label=label, score=1.0, x1=x1, y1=y1, x2=x2, y2=y2, class_id=cls_id)
            )
    return blocks


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def save_checkpoint(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ── Image URL helpers ─────────────────────────────────────────────────────────

def _ensure_image_url(png_path: Path, file_id: str, page_number: int, folder_id: str) -> str:
    dst_dir = Path(LOCAL_IMAGES_DIR) / file_id / "pages"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"page_{page_number}.png"
    if not dst_path.exists():
        try:
            dst_path.symlink_to(png_path.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(str(png_path), str(dst_path))

    if USE_S3:
        from cad_pipeline.storage.s3_store import upload_page_image

        return upload_page_image(dst_path, folder_id, file_id, page_number)
    rel = dst_path.relative_to(LOCAL_IMAGES_DIR)
    return f"{API_BASE_URL}/images/{rel}"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_upload_rendered(
    pdf_path: Path | str,
    rendered_dir: Path | str,
    dxf_dir: Path | str | None,
    folder_id: str,
    folder_name: str,
    file_id: str | None = None,
    file_name: str | None = None,
    checkpoint_path: Path | str | None = None,
    n_parallel: int = N_PARALLEL,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Upload pipeline using pre-rendered images and pre-detected YOLO layout.

    Processes n_parallel pages concurrently and checkpoints completed pages.
    """
    pdf_path = Path(pdf_path)
    rendered_dir = Path(rendered_dir)
    dxf_dir = Path(dxf_dir) if dxf_dir else None
    file_id = file_id or _make_id(str(pdf_path))
    file_name = file_name or pdf_path.name
    if checkpoint_path is None:
        checkpoint_path = rendered_dir.parent / f".checkpoint_{file_id}.json"
    checkpoint_path = Path(checkpoint_path)

    def log(msg: str) -> None:
        print(msg, flush=True)
        if progress_callback:
            progress_callback(msg)

    dxf_map: dict[int, str] = {}
    if dxf_dir:
        dxf_map = build_dxf_page_map(dxf_dir)
        log(f"[DXF] Mapped {len(dxf_map)} DXF files from {dxf_dir}")

    png_files = sorted(rendered_dir.glob("page_*.png"), key=_page_sort_key)
    if not png_files:
        raise FileNotFoundError(f"No page_*.png files found in {rendered_dir}")
    total_pages = len(png_files)
    class_names = load_class_names(rendered_dir)
    log(f"[RENDERED] {total_pages} pages in {rendered_dir.name}")
    log(f"[CLASSES]  {class_names}")

    ckpt = load_checkpoint(checkpoint_path)
    done_pages: set[int] = set(ckpt.get("done_pages", []))
    if done_pages:
        log(f"[RESUME] {len(done_pages)}/{total_pages} pages already done")
    else:
        ckpt = {
            "file_id": file_id,
            "folder_id": folder_id,
            "file_name": file_name,
            "total_pages": total_pages,
            "done_pages": [],
        }

    log("[3] Upserting folder + file in MongoDB...")
    mongo.upsert_folder(folder_id, folder_name)
    mongo.upsert_file(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        file_url=str(pdf_path),
        total_pages=total_pages,
    )

    _lock = threading.Lock()

    def _process_page(png_path: Path) -> tuple[int, dict]:
        page_idx = _page_sort_key(png_path)
        page_number = page_idx + 1

        if page_number in done_pages:
            stored = mongo.get_page(f"{file_id}_p{page_number}")
            return page_number, {
                "page_id": f"{file_id}_p{page_number}",
                "short_summary": (stored or {}).get("short_summary", ""),
                "indexed": True,
            }

        image = _safe_load_image(png_path)
        if image is None:
            log(f"  ⚠ page {page_number}: could not load image — skipping")
            return page_number, {
                "page_id": f"{file_id}_p{page_number}",
                "short_summary": "",
                "indexed": False,
            }

        img_h, img_w = image.shape[:2]
        image_url = _ensure_image_url(png_path, file_id, page_number, folder_id)

        txt_path = png_path.with_suffix(".txt")
        blocks = load_blocks_from_yolo_txt(txt_path, img_w, img_h, class_names)
        dxf_path_str: str | None = dxf_map.get(page_idx)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_summary = pool.submit(generate_page_summary, image)
            f_blocks = pool.submit(process_page_blocks, blocks, image)
            short_summary = f_summary.result()
            processed_blocks = f_blocks.result()

        context_md = build_page_context(
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            blocks=processed_blocks,
        )
        page_id = f"{file_id}_p{page_number}"

        mongo.upsert_page(
            page_id=page_id,
            file_id=file_id,
            folder_id=folder_id,
            page_number=page_number,
            image_url=image_url,
            short_summary=short_summary,
            context_md=context_md,
            blocks=processed_blocks,
            dxf_path=dxf_path_str,
        )

        with _lock:
            log(
                f"  ✓ page {page_number}/{total_pages}"
                + (f" [DXF: {Path(dxf_path_str).name}]" if dxf_path_str else "")
            )

        return page_number, {
            "page_id": page_id,
            "short_summary": short_summary,
            "indexed": True,
        }

    all_results: dict[int, dict] = {}

    for batch in _batched(png_files, n_parallel):
        batch_nums = [_page_sort_key(p) + 1 for p in batch]
        pending_in_batch = [p for p in batch if _page_sort_key(p) + 1 not in done_pages]
        already_in_batch = [p for p in batch if _page_sort_key(p) + 1 in done_pages]

        if pending_in_batch:
            log(
                f"[BATCH] Processing pages {batch_nums[0]}–{batch_nums[-1]} "
                f"({len(pending_in_batch)} new, {len(already_in_batch)} cached)..."
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = {pool.submit(_process_page, png): png for png in batch}
            for fut in concurrent.futures.as_completed(futures):
                page_number, result = fut.result()
                all_results[page_number] = result

        new_done = {n for n in batch_nums if bool(all_results.get(n, {}).get("indexed"))}
        done_pages.update(new_done)
        ckpt["done_pages"] = sorted(done_pages)
        save_checkpoint(checkpoint_path, ckpt)

    log("[9] Building summaries...")
    sorted_nums = sorted(all_results.keys())
    page_ids = [all_results[n]["page_id"] for n in sorted_nums]
    page_summaries = [all_results[n]["short_summary"] for n in sorted_nums]

    file_summary = build_file_summary(page_summaries, file_name)
    mongo.update_file_summary(file_id, file_summary)

    file_short_summary = generate_file_short_summary(file_name, page_summaries)
    mongo.update_file_short_summary(file_id, file_short_summary)

    all_file_docs = mongo.list_files(folder_id)
    all_short = [
        f.get("short_summary") or f.get("summary", "")
        for f in all_file_docs
        if f.get("short_summary") or f.get("summary")
    ]
    folder_summary = build_folder_summary(all_short)
    mongo.update_folder_summary(folder_id, folder_summary)

    ckpt["status"] = "done"
    save_checkpoint(checkpoint_path, ckpt)
    log("✓ Done!")

    return {
        "file_id": file_id,
        "folder_id": folder_id,
        "total_pages": total_pages,
        "page_ids": page_ids,
        "original_url": str(pdf_path),
    }


def _batched(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def _page_sort_key(png_path: Path) -> int:
    m = re.search(r"page_(\d+)", png_path.stem)
    return int(m.group(1)) if m else 0


def _make_id(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:12]
