"""Runs the pipeline steps in order. See README.md."""
from __future__ import annotations

import argparse
from pathlib import Path

from steps import (
    build_masks, convert_png, detect_bubbles, detect_text, inpaint, inspect_pages, make_debug,
    normalize, tag_pages, trim_margins, upscale,
)
from steps.common import MANIFEST_NAME, PROJECT_ROOT, RAW_DIR, Context, Manifest, list_pages, load_config

STEPS = {
    # stage 0: page preparation
    "inspect": inspect_pages.run,
    "convert": convert_png.run,
    "trim": trim_margins.run,
    "upscale": upscale.run,
    "normalize": normalize.run,
    # stage 1: bubble and text removal
    "bubbles": detect_bubbles.run,
    "text": detect_text.run,
    "tag": tag_pages.run,
    "masks": build_masks.run,
    "inpaint": inpaint.run,
    "debug": make_debug.run,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manga pages to cleaned pages: preparation, then bubble and text removal.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--chapter", help="chapter folder name to process")
    which.add_argument("--all", action="store_true", help="process every chapter")
    parser.add_argument("--only", choices=list(STEPS), help="run a single step")
    parser.add_argument("--from", dest="start", choices=list(STEPS), help="run this step and every step after it")
    parser.add_argument("--region", help="<page>:<id>: redo one region with the diffusion backend, e.g. 07:3")
    parser.add_argument("--seed", type=int, help="seed for --region (default: the configured seed plus the region id)")
    parser.add_argument("--force", action="store_true", help="overwrite outputs that already exist")
    parser.add_argument("--limit", type=int, help="process only N pages of each chapter")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N pages of each chapter")
    parser.add_argument("--backend", choices=["ncnn", "torch", "none"], help="override upscale.backend from the config")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg["upscale"]["backend"] = args.backend
    trim_enabled = cfg["trim_margins"]["enabled"]
    if "trim" in (args.only, args.start) and not trim_enabled:
        raise SystemExit("The trim step is off. Set trim_margins.enabled to true in config.yaml first.")

    ctx = Context(
        cfg=cfg,
        data_dir=args.data_dir,
        manifest=Manifest(args.data_dir / MANIFEST_NAME),
        force=args.force,
        limit=args.limit,
        skip=args.skip,
        region=args.region,
        seed=args.seed,
    )
    chapters = None if args.all else [args.chapter]
    names = [args.only] if args.only else [n for n in STEPS if n != "trim" or trim_enabled]
    if args.start:
        names = names[names.index(args.start):]
    if args.region:
        # Refill that one region, then redraw the debug images of its page.
        inpaint.run(ctx, chapters)
        ctx.manifest.save()
        folder = args.data_dir / RAW_DIR / args.chapter
        stems = [p.stem for p in list_pages(folder)[0]] if folder.is_dir() else []
        page = Path(args.region.partition(":")[0]).stem
        ctx.region, ctx.force = None, True
        if page in stems:
            ctx.skip, ctx.limit = stems.index(page), 1
        names = ["debug"]
    for name in names:
        try:
            STEPS[name](ctx, chapters)
        finally:
            ctx.manifest.save()


if __name__ == "__main__":
    main()
