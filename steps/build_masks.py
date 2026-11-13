"""Stage 1, step: 04_detections + 04_text_masks (+ the page itself) -> 04_masks, 04_regions.

The final mask is white where something has to be removed. It is split into
regions (connected components), each with a square window of surrounding art
that the inpainting step works on.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from steps.common import (
    DETECTIONS_DIR, MASKS_DIR, NORMALIZED_DIR, REGIONS_DIR, TEXT_MASKS_DIR, Context, write_json,
    resolve_chapters, save_png, select_pages,
)

STEP = "masks"


def rasterise_polygons(polygons: list, size: tuple[int, int]) -> np.ndarray:
    """Filled polygons as a 0/255 mask. size is (width, height)."""
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.round(np.asarray(polygon)).astype(np.int32)], 255)
    return mask


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components of a mask: (label image, stats). Label 0 is the background.

    build_masks and inpaint both call this on the saved final mask, so region ids agree.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return labels, stats


def remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, stats = label_regions(mask)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Close small unmasked islands enclosed by the mask, which would survive as leftover specks."""
    labels, stats = label_regions(255 - mask)
    x, y, w, h, area = (stats[:, i] for i in range(5))
    enclosed = (x > 0) & (y > 0) & (x + w < mask.shape[1]) & (y + h < mask.shape[0]) & (area < max_area)
    enclosed[0] = False
    return mask | (enclosed[labels] * 255).astype(np.uint8)


def grow_into_tail(polygon_mask: np.ndarray, grey: np.ndarray, cfg: dict) -> np.ndarray:
    """Extend one bubble's mask over the white interior it sits on, which reaches into the tail.

    The detector's polygon follows the bubble body and usually stops short of a pointed
    tail. The tail's inside is white and connected to the bubble's inside, so the white
    area under the polygon is followed outwards. If it spreads too far (an open outline
    leaking into a white background), the polygon is kept as it is.
    """
    ys, xs = np.nonzero(polygon_mask)
    if len(xs) == 0:
        return polygon_mask
    reach = cfg["reach_px"]
    y0, y1 = max(ys.min() - reach, 0), min(ys.max() + reach + 1, grey.shape[0])
    x0, x1 = max(xs.min() - reach, 0), min(xs.max() + reach + 1, grey.shape[1])
    inside = polygon_mask[y0:y1, x0:x1] > 0
    white = (grey[y0:y1, x0:x1] >= cfg["white_threshold"]).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    overlap = np.bincount(labels[inside], minlength=len(stats))
    overlap[0] = 0
    best = int(overlap.argmax())
    if overlap[best] < cfg["min_white_share"] * inside.sum():
        return polygon_mask                       # not a white bubble
    interior = labels == best
    x, y, w, h = stats[best, :4]
    cut_off = ((x == 0 and x0 > 0) or (y == 0 and y0 > 0)
               or (x + w == x1 - x0 and x1 < grey.shape[1]) or (y + h == y1 - y0 and y1 < grey.shape[0]))
    grown = inside | interior
    if cut_off or grown.sum() > cfg["max_growth"] * inside.sum():
        return polygon_mask                       # leaked out of the bubble
    result = polygon_mask.copy()
    result[y0:y1, x0:x1][grown] = 255
    return result


def build_mask(polygons: list, text_mask: np.ndarray, cfg: dict,
               grey: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final mask plus the dilated bubble and text masks it is the union of.

    grey is the page in greyscale; with it, each bubble is grown over its tail first.
    """
    size = (text_mask.shape[1], text_mask.shape[0])
    bubbles = np.zeros((size[1], size[0]), dtype=np.uint8)
    for polygon in polygons:
        one = rasterise_polygons([polygon], size)
        if grey is not None and cfg["tail_growth"]["enabled"]:
            one = grow_into_tail(one, grey, cfg["tail_growth"])
        bubbles |= one
    bubbles = dilate(bubbles, cfg["bubble_dilate_px"])
    text = dilate((text_mask > 0).astype(np.uint8) * 255, cfg["text_dilate_px"])
    final = fill_holes(remove_specks(bubbles | text, cfg["min_area_px"]), cfg["min_area_px"])
    return final, bubbles, text


def window_for(box: list[int], page: tuple[int, int], cfg: dict) -> list[int]:
    """Square window around a box: grown by context_factor, clamped to the window limits and the page."""
    x0, y0, x1, y1 = box
    longest = max(x1 - x0, y1 - y0)
    side = min(max(round(longest * cfg["context_factor"]), cfg["min_window"]), cfg["max_window"])
    side = max(side, longest)  # never smaller than the region itself
    window = []
    for lo, hi, limit in ((x0, x1, page[0]), (y0, y1, page[1])):
        length = min(side, limit)
        start = (lo + hi) // 2 - length // 2
        start = min(max(start, 0), limit - length)
        window.append((start, start + length))
    return [window[0][0], window[1][0], window[0][1], window[1][1]]


def merge_windows(windows: list[list[int]], overlap: float, max_side: int) -> list[int]:
    """Group windows that overlap heavily. Returns a group number per window, numbered from 1.

    A merge is only made while the combined window stays within max_side, so
    chains of neighbours cannot grow into one window covering the whole page.
    """
    rects = [list(w) for w in windows]
    groups = [[i] for i in range(len(windows))]

    def area(w):
        return (w[2] - w[0]) * (w[3] - w[1])

    def try_merge() -> bool:
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                shared = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
                union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                fits = union[2] - union[0] <= max_side and union[3] - union[1] <= max_side
                if fits and shared > overlap * min(area(a), area(b)):
                    rects[i] = union
                    groups[i] += groups.pop(j)
                    rects.pop(j)
                    return True
        return False

    while try_merge():
        pass
    numbers = [0] * len(windows)
    for number, members in enumerate(sorted(groups, key=min), start=1):
        for i in members:
            numbers[i] = number
    return numbers


def find_regions(final: np.ndarray, bubbles: np.ndarray, text: np.ndarray, cfg: dict) -> list[dict]:
    """One entry per connected component of the final mask."""
    height, width = final.shape
    labels, stats = label_regions(final)
    regions = []
    for rid in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[rid])
        box = [x, y, x + w, y + h]
        part = labels[y:y + h, x:x + w] == rid
        from_bubble = bool((bubbles[y:y + h, x:x + w][part] > 0).any())
        from_text = bool((text[y:y + h, x:x + w][part] > 0).any())
        share = area / (width * height)
        regions.append({
            "id": rid,
            "box": box,
            "area": area,
            "area_share": round(share, 5),
            "source": "both" if from_bubble and from_text else "bubble" if from_bubble else "text",
            "label": "hard" if share > cfg["hard_area_fraction"] else "easy",
            "window": window_for(box, (width, height), cfg),
        })
    # Neighbouring regions whose windows overlap heavily are filled together in one window.
    groups = merge_windows([r["window"] for r in regions], cfg["merge_overlap"], cfg["max_window"])
    for region, group in zip(regions, groups):
        region["group"] = group
    for group in set(groups):
        members = [r for r in regions if r["group"] == group]
        merged = [min(r["window"][0] for r in members), min(r["window"][1] for r in members),
                  max(r["window"][2] for r in members), max(r["window"][3] for r in members)]
        for r in members:
            r["window"] = merged
    return regions


def load_masks(ctx: Context, chapter: str, stem: str) -> tuple[list, np.ndarray, np.ndarray]:
    """Bubble polygons, raw text mask and greyscale pixels of a page."""
    detections = json.loads((ctx.data_dir / DETECTIONS_DIR / chapter / f"{stem}.json").read_text(encoding="utf-8"))
    with Image.open(ctx.data_dir / TEXT_MASKS_DIR / chapter / f"{stem}.png") as im:
        text_mask = np.asarray(im.convert("L"))
    with Image.open(ctx.data_dir / NORMALIZED_DIR / chapter / f"{stem}.png") as im:
        grey = np.asarray(im.convert("L"))
    return [d["polygon"] for d in detections["detections"]], text_mask, grey


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    done = skipped = total = hard = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"masks {chapter}", unit="page"):
            mask_dst = ctx.data_dir / MASKS_DIR / chapter / src.name
            regions_dst = ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json"
            if mask_dst.exists() and regions_dst.exists() and not ctx.force:
                skipped += 1
                continue
            for needed in (ctx.data_dir / DETECTIONS_DIR / chapter / f"{src.stem}.json",
                           ctx.data_dir / TEXT_MASKS_DIR / chapter / src.name):
                if not needed.is_file():
                    raise SystemExit(f"Missing {needed}. Run the bubbles and text steps first.")
            polygons, text_mask, grey = load_masks(ctx, chapter, src.stem)
            final, bubbles, text = build_mask(polygons, text_mask, ctx.cfg["mask"], grey)
            regions = find_regions(final, bubbles, text, ctx.cfg["regions"])
            save_png(Image.fromarray(final, mode="L"), mask_dst)
            write_json(regions_dst, {
                "page": src.name, "width": final.shape[1], "height": final.shape[0], "regions": regions,
            })
            done += 1
            total += len(regions)
            hard += sum(r["label"] == "hard" for r in regions)
    print(f"Masks: {done} pages, {total} regions ({hard} hard), {skipped} already done")
