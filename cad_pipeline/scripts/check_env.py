"""check_env.py - Quick runtime environment validation for CAD pipeline.

Usage:
  python -m cad_pipeline.scripts.check_env
  python -m cad_pipeline.scripts.check_env --strict

Exit code:
  0: checks passed (or warnings only without --strict)
  1: at least one required check failed
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv


def _load_dotenv_files() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    cad_pipeline_dir = project_root / "cad_pipeline"
    load_dotenv(dotenv_path=cad_pipeline_dir / ".env", override=True)
    load_dotenv(dotenv_path=project_root / ".env", override=False)


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _check_module(name: str, required_attrs: tuple[str, ...] = ()) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - defensive runtime check
        return False, f"{name}: import error ({exc})"

    missing = [attr for attr in required_attrs if not hasattr(module, attr)]
    if missing:
        return False, f"{name}: missing attributes {missing}"

    mod_ver = getattr(module, "__version__", None)
    if mod_ver:
        return True, f"{name}: ok (version {mod_ver})"
    return True, f"{name}: ok"


def _check_paths() -> list[tuple[bool, str, bool]]:
    from cad_pipeline.config import LAYOUT_WEIGHTS, OBJECT_DESCRIPTIONS_JSON, SYMBOLS_JSON, SYMBOL_GROUPS_JSON

    checks: list[tuple[bool, str, bool]] = []
    for label, path, required in (
        ("layout weights", LAYOUT_WEIGHTS, True),
        ("symbols json", SYMBOLS_JSON, True),
        ("symbol groups json", SYMBOL_GROUPS_JSON, True),
        ("object descriptions json", OBJECT_DESCRIPTIONS_JSON, False),
    ):
        checks.append((Path(path).exists(), f"{label}: {path}", required))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CAD pipeline runtime environment.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    args = parser.parse_args()

    _load_dotenv_files()

    failures: list[str] = []
    warnings: list[str] = []

    required_module_checks = (
        ("numpy", ()),
        ("PIL", ()),
        ("cv2", ("imread", "imdecode", "imencode")),
        ("fitz", ()),
        ("pymongo", ()),
    )
    for mod_name, attrs in required_module_checks:
        ok, message = _check_module(mod_name, attrs)
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            failures.append(message)

    optional_module_checks = (
        ("detectron2", ()),
        ("google.genai", ()),
    )
    for mod_name, attrs in optional_module_checks:
        ok, message = _check_module(mod_name, attrs)
        if ok:
            print(f"[OK] {message}")
        else:
            print(f"[WARN] {message}")
            warnings.append(message)

    for ok, message, required in _check_paths():
        if ok:
            print(f"[OK] {message}")
            continue
        if required:
            print(f"[FAIL] {message}")
            failures.append(message)
        else:
            print(f"[WARN] {message}")
            warnings.append(message)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    marker_key = os.getenv("MARKER_API_KEY", "").strip()
    database_url = os.getenv("DATABASE_URL", "").strip()

    env_checks = (
        ("GEMINI_API_KEY", bool(gemini_key), True),
        ("DATABASE_URL", bool(database_url), True),
        ("MARKER_API_KEY", bool(marker_key), False),
    )
    for key, present, required in env_checks:
        if present:
            print(f"[OK] env {key}: set")
        else:
            level = "FAIL" if required else "WARN"
            print(f"[{level}] env {key}: missing")
            if required:
                failures.append(f"env {key}: missing")
            else:
                warnings.append(f"env {key}: missing")

    opencv = _pkg_version("opencv-python")
    opencv_headless = _pkg_version("opencv-python-headless")
    if opencv and opencv_headless:
        msg = (
            "Both opencv-python and opencv-python-headless are installed "
            f"({opencv} / {opencv_headless}). Keep only one to avoid conflicts."
        )
        print(f"[WARN] {msg}")
        warnings.append(msg)
    elif opencv_headless:
        print(f"[OK] opencv-python-headless pinned: {opencv_headless}")
    elif opencv:
        print(f"[WARN] opencv-python installed ({opencv}); prefer headless on servers.")
        warnings.append(f"opencv-python installed ({opencv})")
    else:
        print("[FAIL] Neither opencv-python nor opencv-python-headless is installed.")
        failures.append("opencv package missing")

    print("")
    print("---- check summary ----")
    print(f"failures: {len(failures)}")
    print(f"warnings: {len(warnings)}")

    if failures:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
