"""Download PubLayNet pretrained weights via LayoutParser and copy to pretrained/."""
from pathlib import Path
import shutil

import layoutparser as lp  # type: ignore

PRETRAINED = Path(__file__).resolve().parent.parent / "pretrained"
PRETRAINED.mkdir(exist_ok=True)

print("▶  Loading PubLayNet Faster R-CNN via LayoutParser (will download if needed)…")
model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")

src = Path(model.cfg.MODEL.WEIGHTS)
dst = PRETRAINED / "publaynet_faster_rcnn_R50_FPN.pth"

if src.exists() and not dst.exists():
    shutil.copy2(src, dst)
    print(f"✓ Copied weights to: {dst}")
else:
    print(f"✓ Weights at: {src}")
    print(f"  Config:     {model.cfg.MODEL.WEIGHTS}")

print(f"\n  Use this path in training script:")
print(f"  cfg.MODEL.WEIGHTS = '{src}'")
