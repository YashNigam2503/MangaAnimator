"""Rough panel boxes, used to give the diffusion model a whole panel as context.

A panel's frame and the artwork touching it form one connected body of ink, so
each large connected ink component is taken as a panel and its bounding box
returned. Panels joined by a bubble or a character that crosses the gutter come
out as one large box; callers fall back to their usual window when a box is too
big. This is not the panel splitting of the later stage, only a context hint.
"""
from __future__ import annotations

import cv2
import numpy as np


def find_panels(grey: np.ndarray, cfg: dict) -> list[list[int]]:
    """Bounding boxes [x0, y0, x1, y1] of the panels on a greyscale page, largest first."""
    ink = (grey < cfg["ink_threshold"]).astype(np.uint8)
    if cfg["close_px"] > 0:
        size = 2 * cfg["close_px"] + 1
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    page_area = grey.shape[0] * grey.shape[1]
    boxes = []
    for x, y, w, h, _ in stats[1:]:
        if w * h >= cfg["min_area_fraction"] * page_area:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def panel_for(box: list[int], panels: list[list[int]]) -> list[int] | None:
    """The smallest panel holding the centre of a region box, widened to hold the whole region."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    holding = [p for p in panels if p[0] <= cx < p[2] and p[1] <= cy < p[3]]
    if not holding:
        return None
    panel = min(holding, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
    return [min(panel[0], box[0]), min(panel[1], box[1]), max(panel[2], box[2]), max(panel[3], box[3])]
