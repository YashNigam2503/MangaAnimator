"""Stage 1, step: 03_normalized + 04_masks + 04_regions -> 05_clean_lama, 05_clean.

Every region is filled per window, never on the whole page, and only masked
pixels are pasted back: outside the mask the output is identical to the input.

05_clean_lama has every region filled with LaMa. 05_clean takes easy regions
from LaMa and hard regions from the diffusion backend; when that backend is not
available the LaMa result is kept and the region is reported as diffusion_pending.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from steps.build_masks import label_regions
from steps.common import (
    CLEAN_DIR, CLEAN_LAMA_DIR, MASKS_DIR, NORMALIZED_DIR, REGIONS_DIR, REGIONS_REPORT_NAME, Context,
    pick_device, resolve_chapters, save_png, select_pages,
)

STEP = "inpaint"
REPORT_COLUMNS = [
    "chapter", "page", "page_type", "region_id", "source", "box", "area_share",
    "label", "backend", "seed", "seconds", "status",
]


class LamaBackend:
    """LaMa through IOPaint's batch command. Same result on CPU and GPU, only the speed differs."""
    name = "lama"

    def __init__(self, cfg: dict):
        self.model = cfg["lama_model"]

    def fill(self, crops: list[tuple[Image.Image, Image.Image]]) -> list[Image.Image]:
        """Fill the white area of each (image, mask) pair. One call, so the model loads once."""
        with tempfile.TemporaryDirectory() as tmp:
            images, masks, out = (Path(tmp) / name for name in ("images", "masks", "out"))
            for folder in (images, masks, out):
                folder.mkdir()
            for i, (image, mask) in enumerate(crops):
                image.save(images / f"{i:05d}.png")
                mask.save(masks / f"{i:05d}.png")
            # IOPaint's default cuts images over 800 px into one crop per mask blob, which leaves
            # rectangular seams where crops overlap. The windows are already crops: fill them whole.
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"hd_strategy": "Original"}), encoding="utf-8")
            cmd = [
                sys.executable, "-m", "iopaint", "run", f"--model={self.model}", f"--device={pick_device()}",
                f"--image={images}", f"--mask={masks}", f"--output={out}", f"--config={config}",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            results = []
            for i, (image, _) in enumerate(crops):
                path = out / f"{i:05d}.png"
                if proc.returncode != 0 or not path.is_file():
                    raise SystemExit(
                        f"iopaint run failed (exit code {proc.returncode}). Check that IOPaint is installed "
                        f"(pip install iopaint). Its output:\n{(proc.stderr or proc.stdout)[-2000:]}"
                    )
                with Image.open(path) as result:
                    result = result.convert("RGB")
                if result.size != image.size:
                    raise SystemExit(f"iopaint returned a {result.size} result for a {image.size} window.")
                results.append(result)
            return results


def read_report(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{c: r.get(c) or "" for c in REPORT_COLUMNS} for r in csv.DictReader(f)]


def write_report(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["chapter"], r["page"], int(r["region_id"])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class PageJob:
    """One page to fill: its pixels, its mask split into regions, and the windows to work on."""

    def __init__(self, ctx: Context, chapter: str, src: Path):
        self.chapter, self.src = chapter, src
        with Image.open(src) as im:
            self.original = np.array(im.convert("RGB"))
        with Image.open(ctx.data_dir / MASKS_DIR / chapter / src.name) as im:
            self.mask = np.array(im.convert("L"))
        if self.mask.shape != self.original.shape[:2]:
            raise SystemExit(f"Mask and page differ in size for {chapter}/{src.name}. Rerun the masks step with --force.")
        info = json.loads((ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json").read_text(encoding="utf-8"))
        self.regions = info["regions"]
        self.labels, _ = label_regions(self.mask)
        self.groups: dict[int, list[dict]] = {}
        for region in self.regions:
            self.groups.setdefault(region["group"], []).append(region)

    def window(self, group: int) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.groups[group][0]["window"]
        return slice(y0, y1), slice(x0, x1)

    def crop(self, pixels: np.ndarray, group: int, only: list[dict] | None = None) -> tuple[Image.Image, Image.Image]:
        """Window of a group as (image, mask). The mask covers everything to remove in the window,
        or only the given regions."""
        win = self.window(group)
        if only is None:
            mask = self.mask[win] > 0
        else:
            mask = np.isin(self.labels[win], [r["id"] for r in only])
        return Image.fromarray(pixels[win]), Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    def paste(self, target: np.ndarray, group: int, result: Image.Image, regions: list[dict]) -> None:
        """Copy the result into the target, but only on the masked pixels of these regions."""
        win = self.window(group)
        own = np.isin(self.labels[win], [r["id"] for r in regions])
        target[win][own] = np.asarray(result)[own]


def run(ctx: Context, chapters: list[str] | None = None, lama=None, diffusion=None) -> None:
    cfg = ctx.cfg["inpaint"]
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")

    jobs: list[PageJob] = []
    skipped = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in pages:
            outputs = [ctx.data_dir / folder / chapter / src.name for folder in (CLEAN_LAMA_DIR, CLEAN_DIR)]
            if all(p.exists() for p in outputs) and not ctx.force:
                skipped += 1
                continue
            for needed in (ctx.data_dir / MASKS_DIR / chapter / src.name,
                           ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json"):
                if not needed.is_file():
                    raise SystemExit(f"Missing {needed}. Run the masks step first.")
            jobs.append(PageJob(ctx, chapter, src))

    # LaMa on every window of every page, in one batch.
    windows = [(job, group) for job in jobs for group in sorted(job.groups)]
    lama_results: dict[tuple[int, int], Image.Image] = {}
    lama_seconds: dict[tuple[int, int], float] = {}
    if cfg["easy_backend"] == "lama" and windows:
        lama = lama or LamaBackend(cfg)
        crops = [job.crop(job.original, group) for job, group in windows]
        print(f"Inpaint: filling {len(crops)} windows on {len(jobs)} pages with LaMa ({pick_device()})...")
        start = time.perf_counter()
        results = lama.fill(crops)
        elapsed = time.perf_counter() - start
        # The batch reports one total; share it out by window size.
        pixels = [crop[0].width * crop[0].height for crop in crops]
        for (job, group), result, px in zip(windows, results, pixels):
            lama_results[(id(job), group)] = result
            lama_seconds[(id(job), group)] = elapsed * px / sum(pixels)

    report_path = ctx.data_dir / REGIONS_REPORT_NAME
    report = read_report(report_path)
    counts = {"done": 0, "diffusion_pending": 0, "not_filled": 0}
    for job in tqdm(jobs, desc="inpaint", unit="page"):
        lama_page = job.original.copy()
        rows = {}
        for group, regions in job.groups.items():
            key = (id(job), group)
            filled = key in lama_results
            if filled:
                job.paste(lama_page, group, lama_results[key], regions)
            for region in regions:
                rows[region["id"]] = {
                    "chapter": job.chapter, "page": job.src.name,
                    "page_type": (ctx.manifest.find(job.chapter, job.src.stem) or {}).get("page_type", ""),
                    "region_id": str(region["id"]), "source": region["source"],
                    "box": " ".join(str(v) for v in region["box"]),
                    "area_share": f"{region['area_share']:.5f}", "label": region["label"],
                    "backend": "lama" if filled else "none", "seed": "",
                    "seconds": f"{lama_seconds[key] / len(regions):.2f}" if filled else "",
                    "status": "done" if filled else "not_filled",
                }

        # Hard regions: diffusion over the LaMa-cleaned page, so no lettering is left in the context.
        final_page = lama_page.copy()
        for group, regions in job.groups.items():
            hard = [r for r in regions if r["label"] == "hard"]
            if not hard or cfg["hard_backend"] == "lama":
                continue
            if cfg["hard_backend"] == "none":
                for region in hard:
                    win = job.window(group)
                    own = job.labels[win] == region["id"]
                    final_page[win][own] = job.original[win][own]
                    rows[region["id"]].update(backend="none", seconds="", status="not_filled")
            elif diffusion is None:
                for region in hard:
                    rows[region["id"]]["status"] = "diffusion_pending"
            else:
                for region in hard:
                    seed = cfg["seed"] + region["id"]
                    start = time.perf_counter()
                    image, mask = job.crop(lama_page, group, only=[region])
                    job.paste(final_page, group, diffusion.fill(image, mask, cfg["prompt"], seed), [region])
                    rows[region["id"]].update(
                        backend=diffusion.name, seed=str(seed), seconds=f"{time.perf_counter() - start:.2f}",
                    )

        save_png(Image.fromarray(lama_page), ctx.data_dir / CLEAN_LAMA_DIR / job.chapter / job.src.name)
        save_png(Image.fromarray(final_page), ctx.data_dir / CLEAN_DIR / job.chapter / job.src.name)
        report = [r for r in report if (r["chapter"], r["page"]) != (job.chapter, job.src.name)] + list(rows.values())
        for row in rows.values():
            counts[row["status"]] += 1
    if jobs:
        write_report(report_path, report)
    print(f"Inpaint: {len(jobs)} pages, {counts['done']} regions filled, "
          f"{counts['diffusion_pending']} hard regions pending diffusion, "
          f"{counts['not_filled']} not filled, {skipped} pages already done")
