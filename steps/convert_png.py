"""Step 2: 00_raw -> 01_png. Lossless 8-bit RGB PNG, no resizing."""
from __future__ import annotations

from PIL import Image, ImageOps
from tqdm import tqdm

from steps.common import PNG_DIR, RAW_DIR, Context, select_pages, resolve_chapters, save_png, set_notes

STEP = "convert"


def to_rgb(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im)
    if im.mode == "I" or im.mode.startswith("I;16"):
        # 16-bit greyscale: scale down to 8 bits instead of clipping.
        im = im.convert("I").point(lambda v: v * (1 / 256)).convert("L")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, rgba)
    return im.convert("RGB")


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    raw_root = ctx.data_dir / RAW_DIR
    chapters = resolve_chapters(raw_root, chapters, "Put raw pages in data/00_raw/<chapter>/.")
    converted = skipped = failed = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, raw_root / chapter)
        for src in tqdm(pages, desc=f"convert {chapter}", unit="page"):
            dst = ctx.data_dir / PNG_DIR / chapter / f"{src.stem}.png"
            if dst.exists() and not ctx.force:
                skipped += 1
                continue
            row = ctx.manifest.row(chapter, src.stem, file=src.name)
            try:
                with Image.open(src) as im:
                    save_png(to_rgb(im), dst)
            except Exception as e:
                failed += 1
                row["status"] = "error"
                set_notes(row, STEP, [f"failed ({type(e).__name__}: {e})"])
                continue
            converted += 1
            row["status"] = "converted"
            set_notes(row, STEP, [])
    print(f"Convert: {converted} converted, {skipped} already done, {failed} failed")
