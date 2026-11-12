"""Stage 1, step: debug images in 05_debug, for judging detection and inpainting by eye.

Per page: <page>_overlay.png (bubble regions red, text regions blue) and
<page>_compare.png (original, LaMa result, final result). Per chapter: contact
sheets of every hard region at 100% (original, LaMa, final).
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from steps.build_masks import build_mask, load_masks
from steps.common import (
    CLEAN_DIR, CLEAN_LAMA_DIR, DEBUG_DIR, MASKS_DIR, NORMALIZED_DIR, REGIONS_DIR, Context,
    resolve_chapters, save_png, select_pages,
)

STEP = "debug"
GAP = 12
LABEL_HEIGHT = 18
RED, BLUE = (230, 30, 30), (30, 80, 230)


def tint(pixels: np.ndarray, where: np.ndarray, colour: tuple[int, int, int], alpha: int) -> None:
    pixels[where] = (pixels[where].astype(np.int32) * (255 - alpha) + np.array(colour) * alpha) // 255


def side_by_side(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width = sum(im.width for im in images) + GAP * (len(images) - 1)
    sheet = Image.new("RGB", (width, max(im.height for im in images) + LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for im, label in zip(images, labels):
        draw.text((x + 4, 3), label, fill="black")
        sheet.paste(im, (x, LABEL_HEIGHT))
        x += im.width + GAP
    return sheet


def stack(rows: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + GAP * (len(rows) - 1)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + GAP
    return sheet


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    cfg = ctx.cfg["debug"]
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    made = skipped = sheets = 0
    for chapter in chapters:
        out_dir = ctx.data_dir / DEBUG_DIR / chapter
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        hard_rows = []
        made_before = made
        for src in tqdm(pages, desc=f"debug {chapter}", unit="page"):
            lama_path = ctx.data_dir / CLEAN_LAMA_DIR / chapter / src.name
            final_path = ctx.data_dir / CLEAN_DIR / chapter / src.name
            regions_path = ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json"
            for needed in (lama_path, final_path, regions_path):
                if not needed.is_file():
                    raise SystemExit(f"Missing {needed}. Run the inpaint step first.")
            with Image.open(src) as im:
                original = im.convert("RGB")
            with Image.open(lama_path) as im:
                lama = im.convert("RGB")
            with Image.open(final_path) as im:
                final = im.convert("RGB")

            # Hard regions are collected for every selected page, so the sheets cover the whole selection.
            m = cfg["crop_margin"]
            for region in json.loads(regions_path.read_text(encoding="utf-8"))["regions"]:
                if region["label"] != "hard":
                    continue
                x0, y0, x1, y1 = region["box"]
                box = (max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, original.width), min(y1 + m, original.height))
                hard_rows.append(side_by_side(
                    [im.crop(box) for im in (original, lama, final)],
                    [f"{src.stem} region {region['id']}: original", "LaMa", "final"],
                ))

            overlay_dst = out_dir / f"{src.stem}_overlay.png"
            compare_dst = out_dir / f"{src.stem}_compare.png"
            if overlay_dst.exists() and compare_dst.exists() and not ctx.force:
                skipped += 1
                continue
            polygons, text_mask, grey = load_masks(ctx, chapter, src.stem)
            final_mask, bubbles, text = build_mask(polygons, text_mask, ctx.cfg["mask"], grey)
            pixels = np.array(original)
            tint(pixels, (text > 0) & (final_mask > 0), BLUE, cfg["overlay_alpha"])
            tint(pixels, (bubbles > 0) & (final_mask > 0), RED, cfg["overlay_alpha"])
            save_png(Image.fromarray(pixels), overlay_dst)
            save_png(side_by_side([original, lama, final], ["original", "LaMa", "final"]), compare_dst)
            made += 1

        # Contact sheets are rebuilt whenever a page was redone, since they span pages.
        existing = sorted(out_dir.glob("hard_regions_*.png")) if out_dir.is_dir() else []
        if hard_rows and (made > made_before or not existing or ctx.force):
            for old in existing:
                old.unlink()
            for i in range(0, len(hard_rows), cfg["sheet_rows"]):
                save_png(stack(hard_rows[i:i + cfg["sheet_rows"]]), out_dir / f"hard_regions_{i // cfg['sheet_rows'] + 1:02d}.png")
                sheets += 1
    print(f"Debug: {made} pages, {sheets} hard-region sheets written, {skipped} pages already done")
