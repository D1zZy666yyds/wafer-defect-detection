"""从断点恢复训练：python scripts/resume_train.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
from ultralytics import YOLO

last = ROOT / "runs" / "train" / "wafer_defect_seg" / "weights" / "last.pt"
assert last.exists(), f"断点不存在: {last}"

model = YOLO(str(last))
model.train(resume=True)
