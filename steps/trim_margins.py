"""Optional step: 01_png -> 01b_trimmed. Crops blank white borders around the page content."""
from __future__ import annotations

from PIL import Image
from tqdm import tqdm

from steps.common import (
    PNG_DIR, TRIMMED_DIR, Context, copy_file, select_pages, resolve_chapters, save_png, set_notes,
)

STEP = "trim"


def content_box(im: Image.Image, threshold: int) -> tuple[int, int, int, int] | None:
    """Bounding box of everything darker than the white threshold, or None for a blank page."""
    mask = im.convert("L").point(lambda v: 255 if v < threshold else 0)
    return mask.getbbox()


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    cfg = ctx.cfg["trim_margins"]
    src_root = ctx.data_dir / PNG_DIR
    chapters = resolve_chapters(src_root, chapters, "Run the convert step first.")
    trimmed = unchanged = skipped = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"trim {chapter}", unit="page"):
            dst = ctx.data_dir / TRIMMED_DIR / chapter / src.name
            if dst.exists() and not ctx.force:
                skipped += 1
                continue
            row = ctx.manifest.row(chapter, src.stem)
            with Image.open(src) as im:
                box = content_box(im, cfg["threshold"])
                page_area = im.width * im.height
                box_area = (box[2] - box[0]) * (box[3] - box[1]) if box else 0
                if box_area < cfg["min_content_fraction"] * page_area:
                    # Mostly white page rather than a white border: leave it alone.
                    copy_file(src, dst)
                    unchanged += 1
                    set_notes(row, STEP, [f"not trimmed, content covers {box_area / page_area:.0%} of the page"])
                else:
                    pad = cfg["padding"]
                    crop = (
                        max(box[0] - pad, 0), max(box[1] - pad, 0),
                        min(box[2] + pad, im.width), min(box[3] + pad, im.height),
                    )
                    save_png(im.crop(crop), dst)
                    trimmed += 1
                    set_notes(row, STEP, [])
            row["status"] = "trimmed"
    print(f"Trim: {trimmed} trimmed, {unchanged} left unchanged, {skipped} already done")
