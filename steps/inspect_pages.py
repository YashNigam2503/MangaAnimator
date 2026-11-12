"""Step 1: read every raw page and record what it is in the manifest."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageOps
from tqdm import tqdm

from steps.common import (
    RAW_DIR, Context, choose_upscale_factor, flag, select_pages, resolve_chapters, set_notes,
)

STEP = "inspect"
INSPECT_COLUMNS = [
    "format", "width", "height", "mode", "is_spread", "is_colour", "needs_upscale", "upscale_factor",
]


def has_colour(im: Image.Image, colour_cfg: dict) -> bool:
    """True when the page has real colour, not just grey stored in three channels."""
    if im.mode in ("1", "L", "LA", "I", "F") or im.mode.startswith("I;16"):
        return False
    rgb = im.convert("RGB")
    # Shrinking first averages out JPEG chroma noise and keeps this fast.
    rgb.thumbnail((colour_cfg["sample_size"], colour_cfg["sample_size"]))
    r, g, b = rgb.split()
    spread = ImageChops.lighter(ImageChops.difference(r, g), ImageChops.difference(g, b))
    coloured = sum(spread.histogram()[colour_cfg["tolerance"] + 1:])
    return coloured / (rgb.width * rgb.height) >= colour_cfg["min_fraction"]


def inspect_page(src: Path, row: dict, cfg: dict) -> None:
    try:
        with Image.open(src) as im:
            fmt, mode = im.format, im.mode
            im.load()  # decode fully so truncated files are caught here
            # Sizes are recorded as the page will look after convert applies EXIF orientation.
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            colour = has_colour(im, cfg["colour"])
    except Exception as e:
        for col in INSPECT_COLUMNS:
            row[col] = ""
        row["status"] = "error"
        set_notes(row, STEP, [f"unreadable ({type(e).__name__}: {e})"])
        return

    factor = choose_upscale_factor(height, cfg["min_height"], cfg["max_upscale_factor"])
    row.update({
        "format": fmt,
        "width": str(width),
        "height": str(height),
        "mode": mode,
        "is_spread": flag(width > height),
        "is_colour": flag(colour),
        "needs_upscale": flag(height < cfg["min_height"]),
        "upscale_factor": str(factor),
    })
    set_notes(row, STEP, [])
    if row["status"] in ("", "error"):
        row["status"] = "inspected"


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["height"]]
    by_height = sorted(ok, key=lambda r: int(r["height"]))
    return {
        "pages": len(rows),
        "unreadable": len(rows) - len(ok),
        "smallest": by_height[0] if ok else None,
        "largest": by_height[-1] if ok else None,
        "needs_upscale": sum(r["needs_upscale"] == "true" for r in ok),
        "spreads": sum(r["is_spread"] == "true" for r in ok),
        "colour": sum(r["is_colour"] == "true" for r in ok),
    }


def print_summary(summary: dict, duplicates: int) -> None:
    def describe(row):
        return f"{row['chapter']}/{row['file']} ({row['width']}x{row['height']})" if row else "-"

    print("Inspect summary")
    print(f"  pages:          {summary['pages']}")
    print(f"  smallest page:  {describe(summary['smallest'])}")
    print(f"  largest page:   {describe(summary['largest'])}")
    print(f"  need upscaling: {summary['needs_upscale']}")
    print(f"  spreads:        {summary['spreads']}")
    print(f"  colour pages:   {summary['colour']}")
    if summary["unreadable"]:
        print(f"  unreadable:     {summary['unreadable']} (see manifest notes)")
    if duplicates:
        print(f"  duplicate names ignored: {duplicates} (see manifest notes)")


def run(ctx: Context, chapters: list[str] | None = None) -> dict:
    raw_root = ctx.data_dir / RAW_DIR
    chapters = resolve_chapters(raw_root, chapters, "Put raw pages in data/00_raw/<chapter>/.")
    rows: list[dict] = []
    duplicates = 0
    for chapter in chapters:
        pages, dupes = select_pages(ctx, chapter, raw_root / chapter)
        for dupe, kept in dupes:
            row = ctx.manifest.duplicate_row(chapter, dupe.name)
            set_notes(row, STEP, [f"same page name as {kept.name}, ignored"])
        duplicates += len(dupes)
        for src in tqdm(pages, desc=f"inspect {chapter}", unit="page"):
            row = ctx.manifest.row(chapter, src.stem, file=src.name)
            if not row["format"] or ctx.force:
                inspect_page(src, row, ctx.cfg)
            rows.append(row)
    summary = summarize(rows)
    print_summary(summary, duplicates)
    return summary
