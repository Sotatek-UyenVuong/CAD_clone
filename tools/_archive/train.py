"""
train.py
--------
Train a YOLO model on the prepared CAD drawing dataset.

Prerequisites:
  pip install ultralytics
  python tools/prepare_dataset.py          # builds dataset/ directory first

Usage:
  # Quick training (YOLOv8n, 50 epochs) – for smoke-testing
  python tools/train.py

  # Full training (YOLOv8m, 200 epochs, custom img size)
  python tools/train.py --model yolov8m.pt --epochs 200 --imgsz 1280

  # Resume an interrupted run
  python tools/train.py --resume runs/detect/cad_yolo/weights/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent   # /mnt/sda/uyen/CAD
DATASET_YAML = ROOT / "dataset" / "cad_dataset.yaml"
RUNS_DIR = ROOT / "runs"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on CAD drawing dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="yolov8m.pt",
        help=(
            "YOLO backbone to use.  Use a pretrained checkpoint path to fine-tune, "
            "or a .yaml config to train from scratch.  "
            "Choices: yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt | "
            "yolo11n.pt | yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt"
        ),
    )
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--imgsz",    type=int,   default=1280,
                        help="Training image size (square, pixels)")
    parser.add_argument("--batch",    type=int,   default=-1,
                        help="Batch size (-1 = auto)")
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--device",   default="0",
                        help="CUDA device id(s), e.g. '0', '0,1', or 'cpu'")
    parser.add_argument("--project",  type=Path,  default=RUNS_DIR / "detect",
                        help="Directory where run folders are saved")
    parser.add_argument("--name",     default="cad_yolo",
                        help="Run subdirectory name")
    parser.add_argument("--data",     type=Path,  default=DATASET_YAML,
                        help="Path to dataset YAML")
    parser.add_argument("--resume",   type=str,   default=None,
                        help="Path to last.pt to resume training")
    parser.add_argument("--patience", type=int,   default=30,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--lr0",      type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--lrf",      type=float, default=0.001,
                        help="Final learning rate (lr0 * lrf)")
    parser.add_argument("--optimizer",default="auto",
                        choices=["SGD", "Adam", "AdamW", "auto"],
                        help="Optimizer choice")
    parser.add_argument("--augment",  action="store_true",
                        help="Enable heavy augmentation (mosaic, mixup, etc.)")
    parser.add_argument("--amp",      action="store_true", default=True,
                        help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--val",      action="store_true", default=True,
                        help="Run validation after every epoch")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained ImageNet weights")
    return parser.parse_args(argv)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: ultralytics not installed.\n"
            "  pip install ultralytics"
        )

    # Validate dataset YAML
    if not args.data.exists():
        sys.exit(
            f"ERROR: Dataset YAML not found: {args.data}\n"
            f"  Run: python tools/prepare_dataset.py"
        )

    # Load or resume model
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"ERROR: Checkpoint not found: {resume_path}")
        print(f"▶  Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        print(f"▶  Loading base model: {args.model}")
        model = YOLO(args.model)

    print(f"▶  Dataset YAML:    {args.data}")
    print(f"▶  Image size:      {args.imgsz}px")
    print(f"▶  Epochs:          {args.epochs}")
    print(f"▶  Batch size:      {'auto' if args.batch == -1 else args.batch}")
    print(f"▶  Device:          {args.device}")
    print()

    # Build augmentation kwargs when --augment is requested
    aug_kwargs: dict = {}
    if args.augment:
        aug_kwargs = dict(
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        resume=bool(args.resume),
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf / args.lr0,          # ultralytics uses lrf as a multiplier
        optimizer=args.optimizer,
        amp=args.amp,
        val=args.val,
        pretrained=args.pretrained,
        plots=True,
        save=True,
        save_period=10,                    # save checkpoint every N epochs
        verbose=True,
        **aug_kwargs,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✓ Training complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Results dir  : {results.save_dir}")
    print(f"\n  Validate on test set:")
    print(f"    python tools/evaluate.py --weights {best_weights}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
