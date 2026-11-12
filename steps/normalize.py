"""Step 4: 02_upscaled -> 03_normalized. Every page exactly target_height tall."""
from __future__ import annotations

from PIL import Image
from tqdm import tqdm

from steps import upscale
from steps.common import (
    NORMALIZED_DIR, UPSCALED_DIR, Context, copy_file, select_pages, resolve_chapters, save_png, set_notes,
)

STEP = "normalize"


def run(ctx: Context, chapters: list[str] | None = None) -> None:
    target = ctx.cfg["target_height"]
    max_enlarge = ctx.cfg["max_plain_enlarge"]
    src_root = ctx.data_dir / UPSCALED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run the upscale step first.")
    done = skipped = flagged = 0
    for chapter in chapters:
        # Pick up results of an upscale step that ran on another machine.
        upscale.apply_results(ctx, chapter, upscale.read_results(src_root / chapter), set_status=False)
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"normalize {chapter}", unit="page"):
            dst = ctx.data_dir / NORMALIZED_DIR / chapter / src.name
            if dst.exists() and not ctx.force:
                skipped += 1
                continue
            row = ctx.manifest.row(chapter, src.stem)
            with Image.open(src) as im:
                scale = target / im.height
                if im.height == target and im.mode == "RGB":
                    copy_file(src, dst)
                    size = im.size
                else:
                    size = (max(round(im.width * scale), 1), target)
                    save_png(im.convert("RGB").resize(size, Image.LANCZOS), dst)
            notes = []
            if scale > max_enlarge:
                notes.append(f"enlarged {scale:.2f}x by plain resize, should have gone through AI upscaling")
                flagged += 1
            row.update({
                "final_width": str(size[0]),
                "final_height": str(size[1]),
                "normalize_scale": f"{scale:.4f}",
                "status": "normalized",
            })
            set_notes(row, STEP, notes)
            done += 1
    print(f"Normalize: {done} resized to {target}px tall, {skipped} already done, {flagged} flagged")
