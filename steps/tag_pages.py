"""Stage 1, step: tags each page `story` or `cover` in the manifest.

A cover is a colour page where text regions cover a large share of the page.
Nothing is skipped because of the tag; it is for reporting and review.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from tqdm import tqdm

from steps.build_masks import dilate
from steps.common import NORMALIZED_DIR, TEXT_MASKS_DIR, Context, flag, resolve_chapters, select_pages
from steps.inspect_pages import has_colour

STEP = "tag"


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    counts = {"story": 0, "cover": 0}
    skipped = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"tag {chapter}", unit="page"):
            row = ctx.manifest.row(chapter, src.stem)
            if row["page_type"] and not ctx.force:
                skipped += 1
                continue
            mask_path = ctx.data_dir / TEXT_MASKS_DIR / chapter / src.name
            if not mask_path.is_file():
                raise SystemExit(f"Missing {mask_path}. Run the text step first.")
            if not row["is_colour"]:
                with Image.open(src) as im:
                    row["is_colour"] = flag(has_colour(im, ctx.cfg["colour"]))
            with Image.open(mask_path) as im:
                text = dilate(np.asarray(im.convert("L")), ctx.cfg["mask"]["text_dilate_px"])
            text_fraction = float((text > 0).mean())
            is_cover = row["is_colour"] == "true" and text_fraction > ctx.cfg["page_type"]["cover_text_fraction"]
            row["page_type"] = "cover" if is_cover else "story"
            counts[row["page_type"]] += 1
    print(f"Tag: {counts['story']} story, {counts['cover']} cover, {skipped} already done")
