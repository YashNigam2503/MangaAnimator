"""Stage 1, step: 03_normalized + 04_masks + 04_regions -> 05_clean_lama, 05_clean.

Every region is filled per window, never on the whole page, and only masked
pixels are pasted back: outside the mask the output is identical to the input.

05_clean_lama has every region filled with LaMa. 05_clean takes easy regions
from LaMa and hard regions from the diffusion backend; when that backend is not
available (no GPU) the LaMa result is kept and the region is reported as
diffusion_pending.

Each hard region is diffused with its own prompt (from the prompts step), on
its whole panel when the panel fits the window limit. A fill in which a face
appears inside the mask is rejected and retried with the next seed; when every
attempt fails the LaMa fill is kept and the region is reported as needs_review.

With --region <page>:<id> a single region of an already cleaned page is redone
with the diffusion backend, optionally with another --seed. With --hard-only
every hard region of the selected pages is redone and nothing else is touched.
"""
from __future__ import annotations

import csv
import json
import os
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
    atomic_path, pick_device, resolve_chapters, save_png, select_pages,
)
from steps.describe_regions import load_prompts
from steps.detect_faces import AnimeFaceDetector, face_inside_mask
from steps.diffusion import make_diffusion_backend
from steps.inspect_pages import has_colour
from steps.panels import find_panels, panel_for

STEP = "inpaint"
REPORT_COLUMNS = [
    "chapter", "page", "page_type", "region_id", "source", "box", "area_share",
    "label", "backend", "seed", "attempts", "seconds", "status", "prompt",
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
            # IOPaint pins an old diffusers, so on the GPU machine it lives in its own environment.
            python = os.environ.get("IOPAINT_PYTHON") or sys.executable
            cmd = [
                python, "-m", "iopaint", "run", f"--model={self.model}", f"--device={pick_device()}",
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
    with atomic_path(path) as tmp, open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class PageJob:
    """One page to fill: its pixels, its mask split into regions, and the windows to work on."""

    def __init__(self, ctx: Context, chapter: str, src: Path):
        self.chapter, self.src, self.cfg = chapter, src, ctx.cfg
        with Image.open(src) as im:
            self.original = np.array(im.convert("RGB"))
        with Image.open(ctx.data_dir / MASKS_DIR / chapter / src.name) as im:
            self.mask = np.array(im.convert("L"))
        if self.mask.shape != self.original.shape[:2]:
            raise SystemExit(f"Mask and page differ in size for {chapter}/{src.name}. Rerun the masks step with --force.")
        info = json.loads((ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json").read_text(encoding="utf-8"))
        self.regions = info["regions"]
        # Fills on a black-and-white page are kept grey: the models sometimes add a colour tint.
        row = ctx.manifest.find(chapter, src.stem) or {}
        self.page_type = row.get("page_type", "")
        if row.get("is_colour"):
            self.grey = row["is_colour"] == "false"
        else:
            self.grey = not has_colour(Image.fromarray(self.original), ctx.cfg["colour"])
        self.labels, _ = label_regions(self.mask)
        self.groups: dict[int, list[dict]] = {}
        for region in self.regions:
            self.groups.setdefault(region["group"], []).append(region)
        self._panels = None

    @staticmethod
    def slices(box: list[int]) -> tuple[slice, slice]:
        x0, y0, x1, y1 = box
        return slice(y0, y1), slice(x0, x1)

    def window(self, group: int) -> tuple[slice, slice]:
        return self.slices(self.groups[group][0]["window"])

    def diffusion_window(self, region: dict) -> tuple[list[int], bool]:
        """Box the diffusion model works on: the region's whole panel when that fits within
        max_window, otherwise the region's usual window. The flag says whether it is the panel."""
        if self._panels is None:
            grey = np.asarray(Image.fromarray(self.original).convert("L"))
            self._panels = find_panels(grey, self.cfg["panels"])
        panel = panel_for(region["box"], self._panels)
        limit = self.cfg["regions"]["max_window"]
        if panel and panel[2] - panel[0] <= limit and panel[3] - panel[1] <= limit:
            return panel, True
        return region["window"], False

    def crop(self, pixels: np.ndarray, win: tuple[slice, slice],
             only: list[dict] | None = None) -> tuple[Image.Image, Image.Image]:
        """A window as (image, mask). The mask covers everything to remove in the window,
        or only the given regions."""
        if only is None:
            mask = self.mask[win] > 0
        else:
            mask = np.isin(self.labels[win], [r["id"] for r in only])
        return Image.fromarray(pixels[win]), Image.fromarray((mask * 255).astype(np.uint8), mode="L")

    def paste(self, target: np.ndarray, win: tuple[slice, slice], result: Image.Image, regions: list[dict]) -> None:
        """Copy the result into the target, but only on the masked pixels of these regions."""
        own = np.isin(self.labels[win], [r["id"] for r in regions])
        if self.grey:
            result = result.convert("L")
        target[win][own] = np.asarray(result.convert("RGB"))[own]

    def row(self, region: dict) -> dict:
        """A report row for a region, before any backend has run."""
        row = {c: "" for c in REPORT_COLUMNS}
        row.update({
            "chapter": self.chapter, "page": self.src.name, "page_type": self.page_type,
            "region_id": str(region["id"]), "source": region["source"],
            "box": " ".join(str(v) for v in region["box"]),
            "area_share": f"{region['area_share']:.5f}", "label": region["label"],
            "backend": "none", "status": "not_filled",
        })
        return row


class HardFiller:
    """Fills hard regions with the diffusion backend, checking each fill for invented faces."""

    def __init__(self, ctx: Context, diffusion, faces=None):
        self.ctx, self.cfg = ctx, ctx.cfg["inpaint"]
        self.diffusion = diffusion
        self.faces = faces or AnimeFaceDetector(ctx.cfg["faces"])
        self.prompts: dict[tuple[str, str], dict[int, str]] = {}

    def prepare(self, pages: list[tuple[str, Path]]) -> None:
        """Hand the diffusion backend every prompt these pages will use, so it can encode them
        all before it loads the image model. Missing prompts are reported later, by prompt_for."""
        if not hasattr(self.diffusion, "prepare"):
            return
        prompts = []
        for chapter, src in pages:
            path = self.ctx.data_dir / REGIONS_DIR / chapter / f"{src.stem}.json"
            if not path.is_file():
                continue
            regions = json.loads(path.read_text(encoding="utf-8"))["regions"]
            hard = [r for r in regions if r["label"] == "hard"]
            prompts += load_prompts(self.ctx, chapter, src.stem, hard).values()
        self.diffusion.prepare(prompts)

    def prompt_for(self, job: PageJob, region: dict) -> str:
        key = (job.chapter, job.src.stem)
        if key not in self.prompts:
            self.prompts[key] = load_prompts(self.ctx, job.chapter, job.src.stem, job.regions)
        if region["id"] not in self.prompts[key]:
            raise SystemExit(
                f"No prompt for region {region['id']} of {job.chapter}/{job.src.name}. Run the prompts step first: "
                f"python run.py --chapter {job.chapter} --only prompts (needs ANTHROPIC_API_KEY).")
        return self.prompts[key][region["id"]]

    def fill(self, job: PageJob, region: dict, lama_page: np.ndarray, final_page: np.ndarray,
             row: dict, seed: int | None = None) -> None:
        """Diffuse one region into final_page and record the outcome in its report row.

        The context is the LaMa-cleaned page, so no lettering is left around the region.
        """
        prompt = self.prompt_for(job, region)
        first_seed = seed if seed is not None else self.cfg["seed"] + region["id"]
        box, _ = job.diffusion_window(region)
        win = job.slices(box)
        image, mask = job.crop(lama_page, win, only=[region])
        start = time.perf_counter()
        accepted = None
        for attempt in range(1, self.cfg["diffusion_attempts"] + 1):
            used_seed = first_seed + attempt - 1
            result = self.diffusion.fill(image, mask, prompt, used_seed)
            if face_inside_mask(self.faces(result), mask) is None:
                accepted = result
                break
        row.update(seed=str(used_seed), attempts=str(attempt),
                   seconds=f"{time.perf_counter() - start:.2f}", prompt=prompt)
        if accepted is not None:
            job.paste(final_page, win, accepted, [region])
            row.update(backend=self.diffusion.name, status="done")
        else:
            # Every attempt drew a face where there should be none: keep the LaMa fill.
            group_win = job.window(region["group"])
            own = job.labels[group_win] == region["id"]
            final_page[group_win][own] = lama_page[group_win][own]
            row.update(backend="lama", status="needs_review")


def load_page(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def hard_filler(ctx: Context, diffusion, faces, what: str) -> HardFiller:
    diffusion = diffusion or make_diffusion_backend(ctx.cfg["inpaint"])
    if diffusion is None:
        raise SystemExit(f"{what} uses the diffusion backend, which needs a GPU.")
    return HardFiller(ctx, diffusion, faces)


def update_report(ctx: Context, rows: list[dict]) -> None:
    """Replace the report rows of these regions, keeping every other row."""
    path = ctx.data_dir / REGIONS_REPORT_NAME
    keys = {(r["chapter"], r["page"], r["region_id"]) for r in rows}
    kept = [r for r in read_report(path) if (r["chapter"], r["page"], r["region_id"]) not in keys]
    write_report(path, kept + rows)


def existing_row(ctx: Context, job: PageJob, region: dict) -> dict:
    for row in read_report(ctx.data_dir / REGIONS_REPORT_NAME):
        if (row["chapter"], row["page"], row["region_id"]) == (job.chapter, job.src.name, str(region["id"])):
            return row
    return job.row(region)


def redo_region(ctx: Context, chapters: list[str] | None, diffusion, faces) -> None:
    """--region <page>:<id>: refill one region of an already cleaned page with the diffusion backend."""
    page, _, rid = ctx.region.partition(":")
    if chapters is None or len(chapters) != 1 or not rid.isdigit():
        raise SystemExit("--region needs --chapter and the form <page>:<id>, for example --region 07:3")
    chapter, stem = chapters[0], Path(page).stem
    src = ctx.data_dir / NORMALIZED_DIR / chapter / f"{stem}.png"
    lama_path = ctx.data_dir / CLEAN_LAMA_DIR / chapter / src.name
    final_path = ctx.data_dir / CLEAN_DIR / chapter / src.name
    for needed in (src, lama_path, final_path):
        if not needed.is_file():
            raise SystemExit(f"Missing {needed}. Run the inpaint step on this page first.")
    job = PageJob(ctx, chapter, src)
    region = next((r for r in job.regions if r["id"] == int(rid)), None)
    if region is None:
        raise SystemExit(f"Page {stem} has no region {rid}. It has regions 1 to {len(job.regions)}.")
    filler = hard_filler(ctx, diffusion, faces, "--region")
    filler.prepare([(chapter, src)])
    final_page = load_page(final_path)
    row = existing_row(ctx, job, region)
    filler.fill(job, region, load_page(lama_path), final_page, row, seed=ctx.seed)
    save_png(Image.fromarray(final_page), final_path)
    update_report(ctx, [row])
    print(f"Inpaint: region {rid} of {chapter}/{src.name}: {row['status']} ({row['backend']}), "
          f"seed {row['seed']}, {row['attempts']} attempt(s), {float(row['seconds']):.0f}s")


def redo_hard(ctx: Context, chapters: list[str], diffusion, faces) -> None:
    """--hard-only: refill every hard region of the selected, already cleaned pages. Nothing else changes."""
    filler = None
    counts = {"done": 0, "needs_review": 0}
    pages_done = 0
    selected = {chapter: select_pages(ctx, chapter, ctx.data_dir / NORMALIZED_DIR / chapter)[0] for chapter in chapters}
    announced = False
    for chapter in chapters:
        for src in tqdm(selected[chapter], desc=f"hard regions {chapter}", unit="page"):
            lama_path = ctx.data_dir / CLEAN_LAMA_DIR / chapter / src.name
            if not lama_path.is_file():
                raise SystemExit(f"Missing {lama_path}. Run the inpaint step on this page first.")
            job = PageJob(ctx, chapter, src)
            hard = [r for r in job.regions if r["label"] == "hard"]
            if not hard:
                continue
            filler = filler or hard_filler(ctx, diffusion, faces, "--hard-only")
            if not announced:
                filler.prepare([(c, page) for c in chapters for page in selected[c]])
                announced = True
            lama_page = load_page(lama_path)
            final_page = lama_page.copy()      # easy regions from LaMa, hard regions refilled below
            rows = []
            for region in hard:
                row = existing_row(ctx, job, region)
                filler.fill(job, region, lama_page, final_page, row)
                rows.append(row)
                counts[row["status"]] += 1
            save_png(Image.fromarray(final_page), ctx.data_dir / CLEAN_DIR / chapter / src.name)
            update_report(ctx, rows)
            pages_done += 1
    print(f"Inpaint: hard regions on {pages_done} pages: {counts['done']} filled, "
          f"{counts['needs_review']} kept as LaMa and marked needs_review")
    if filler and hasattr(filler.diffusion, "summary"):
        print(f"Diffusion: {filler.diffusion.summary()}")


def run(ctx: Context, chapters: list[str] | None = None, lama=None, diffusion=None, faces=None) -> None:
    cfg = ctx.cfg["inpaint"]
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    if ctx.region:
        return redo_region(ctx, chapters, diffusion, faces)
    if ctx.hard_only:
        return redo_hard(ctx, chapters, diffusion, faces)

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
        crops = [job.crop(job.original, job.window(group)) for job, group in windows]
        print(f"Inpaint: filling {len(crops)} windows on {len(jobs)} pages with LaMa ({pick_device()})...")
        start = time.perf_counter()
        results = lama.fill(crops)
        elapsed = time.perf_counter() - start
        # The batch reports one total; share it out by window size.
        pixels = [crop[0].width * crop[0].height for crop in crops]
        for (job, group), result, px in zip(windows, results, pixels):
            lama_results[(id(job), group)] = result
            lama_seconds[(id(job), group)] = elapsed * px / sum(pixels)

    needs_diffusion = cfg["hard_backend"] == "diffusion" and any(
        r["label"] == "hard" for job in jobs for r in job.regions)
    if needs_diffusion and diffusion is None:
        diffusion = make_diffusion_backend(cfg)
    filler = HardFiller(ctx, diffusion, faces) if needs_diffusion and diffusion is not None else None
    if filler is not None:
        filler.prepare([(job.chapter, job.src) for job in jobs])

    report_path = ctx.data_dir / REGIONS_REPORT_NAME
    report = read_report(report_path)
    counts = {"done": 0, "diffusion_pending": 0, "needs_review": 0, "not_filled": 0}
    for job in tqdm(jobs, desc="inpaint", unit="page"):
        lama_page = job.original.copy()
        rows = {}
        for group, regions in job.groups.items():
            key = (id(job), group)
            filled = key in lama_results
            if filled:
                job.paste(lama_page, job.window(group), lama_results[key], regions)
            for region in regions:
                rows[region["id"]] = job.row(region)
                if filled:
                    rows[region["id"]].update(
                        backend="lama", seconds=f"{lama_seconds[key] / len(regions):.2f}", status="done")

        final_page = lama_page.copy()
        for region in job.regions:
            if region["label"] != "hard" or cfg["hard_backend"] == "lama":
                continue
            row = rows[region["id"]]
            if cfg["hard_backend"] == "none":
                win = job.window(region["group"])
                own = job.labels[win] == region["id"]
                final_page[win][own] = job.original[win][own]
                row.update(backend="none", seconds="", status="not_filled")
            elif filler is None:
                row["status"] = "diffusion_pending"
            else:
                filler.fill(job, region, lama_page, final_page, row)

        save_png(Image.fromarray(lama_page), ctx.data_dir / CLEAN_LAMA_DIR / job.chapter / job.src.name)
        save_png(Image.fromarray(final_page), ctx.data_dir / CLEAN_DIR / job.chapter / job.src.name)
        report = [r for r in report if (r["chapter"], r["page"]) != (job.chapter, job.src.name)] + list(rows.values())
        # Written after every page: if the run is killed, each finished page still has its report rows.
        write_report(report_path, report)
        for row in rows.values():
            counts[row["status"]] += 1
    print(f"Inpaint: {len(jobs)} pages, {counts['done']} regions filled, "
          f"{counts['diffusion_pending']} hard regions pending diffusion, "
          f"{counts['needs_review']} need review, {counts['not_filled']} not filled, {skipped} pages already done")
    if filler is not None and hasattr(diffusion, "summary"):
        print(f"Diffusion: {diffusion.summary()}")
