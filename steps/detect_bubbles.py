"""Stage 1, step: 03_normalized -> 04_detections. Speech bubble polygons, one JSON per page."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from steps.common import DETECTIONS_DIR, NORMALIZED_DIR, Context, pick_device, resolve_chapters, select_pages

STEP = "bubbles"


class YoloBubbleDetector:
    """YOLOv8 segmentation model with one class, speech bubble."""

    def __init__(self, cfg: dict):
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        self.cfg = cfg
        self.name = cfg["model_repo"]
        self.device = pick_device()
        self.model = YOLO(hf_hub_download(cfg["model_repo"], cfg["model_file"]))

    def __call__(self, page: Path) -> list[dict]:
        import cv2
        import numpy as np
        result = self.model.predict(
            str(page), conf=self.cfg["conf"], imgsz=self.cfg["imgsz"], device=self.device,
            retina_masks=True, verbose=False,
        )[0]
        if result.masks is None:
            return []
        detections = []
        for polygon, box, conf in zip(result.masks.xy, result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            if len(polygon) < 3:
                continue
            detections.append({
                "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in polygon],
                "box": [round(v, 1) for v in box],
                "confidence": round(conf, 4),
                "area": round(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32)))),
            })
        return detections


def run(ctx: Context, chapters: list[str] | None = None, detector=None) -> None:
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    done = skipped = found = 0
    seconds = 0.0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"bubbles {chapter}", unit="page"):
            dst = ctx.data_dir / DETECTIONS_DIR / chapter / f"{src.stem}.json"
            if dst.exists() and not ctx.force:
                skipped += 1
                continue
            detector = detector or YoloBubbleDetector(ctx.cfg["detect_bubbles"])
            start = time.perf_counter()
            detections = detector(src)
            seconds += time.perf_counter() - start
            with Image.open(src) as im:
                width, height = im.size
            for i, det in enumerate(detections, start=1):
                det["id"] = i
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps({
                "page": src.name, "width": width, "height": height,
                "model": getattr(detector, "name", ""), "detections": detections,
            }, indent=1), encoding="utf-8")
            done += 1
            found += len(detections)
    per_page = f", {seconds / done:.1f}s per page" if done else ""
    print(f"Bubbles: {done} pages, {found} bubbles found{per_page}, {skipped} already done")
