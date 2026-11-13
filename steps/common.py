"""Shared helpers: config, folder layout, page listing and the manifest."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = "00_raw"
PNG_DIR = "01_png"
TRIMMED_DIR = "01b_trimmed"
UPSCALED_DIR = "02_upscaled"
NORMALIZED_DIR = "03_normalized"
DETECTIONS_DIR = "04_detections"
TEXT_MASKS_DIR = "04_text_masks"
MASKS_DIR = "04_masks"
REGIONS_DIR = "04_regions"
PROMPTS_DIR = "04_prompts"
CLEAN_LAMA_DIR = "05_clean_lama"
CLEAN_DIR = "05_clean"
DEBUG_DIR = "05_debug"
MANIFEST_NAME = "manifest.csv"
REGIONS_REPORT_NAME = "regions_report.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
UPSCALE_FACTORS = (2, 3, 4)  # the scales Real-ESRGAN offers
NOTE_ORDER = ["inspect", "convert", "trim", "upscale", "normalize"]

MANIFEST_COLUMNS = [
    "chapter", "file", "format", "width", "height", "mode",
    "is_spread", "is_colour", "needs_upscale", "upscale_factor",
    "upscale_applied", "final_width", "final_height", "normalize_scale",
    "page_type", "status", "notes",
]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flag(value: bool) -> str:
    return "true" if value else "false"


def choose_upscale_factor(height: int, min_height: int, max_factor: int) -> int:
    """Smallest factor that brings height to min_height, capped at max_factor. 1 = no upscale."""
    if height >= min_height:
        return 1
    allowed = [f for f in UPSCALE_FACTORS if f <= max_factor]
    if not allowed:
        return 1
    for factor in allowed:
        if height * factor >= min_height:
            return factor
    return allowed[-1]


def list_pages(folder: Path, limit: int | None = None) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Image files in a folder, sorted by name.

    Two files that differ only by extension would collide once converted to PNG,
    so the first one is kept and the rest are returned as (duplicate, kept) pairs.
    """
    files = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )
    pages: list[Path] = []
    dupes: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}
    for p in files:
        stem = p.stem.lower()
        if stem in seen:
            dupes.append((p, seen[stem]))
        else:
            seen[stem] = p
            pages.append(p)
    if limit is not None:
        pages = pages[:limit]
    return pages, dupes


def select_pages(ctx: "Context", chapter: str, folder: Path) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """Pages of a step's input folder, narrowed by --skip and --limit.

    --skip and --limit count pages of the raw chapter, so every step picks the same
    pages whatever its own input folder happens to contain.
    """
    pages, dupes = list_pages(folder)
    if ctx.limit is None and not ctx.skip:
        return pages, dupes
    raw = ctx.data_dir / RAW_DIR / chapter
    reference = list_pages(raw)[0] if raw.is_dir() else pages
    end = None if ctx.limit is None else ctx.skip + ctx.limit
    wanted = {p.stem.lower() for p in reference[ctx.skip:end]}
    return [p for p in pages if p.stem.lower() in wanted], dupes


def resolve_chapters(input_root: Path, chapters: list[str] | None, hint: str) -> list[str]:
    """Chapters to process: the ones asked for, or every subfolder of the step's input folder."""
    if not input_root.is_dir():
        raise SystemExit(f"Input folder not found: {input_root}. {hint}")
    if chapters is None:
        return sorted(p.name for p in input_root.iterdir() if p.is_dir())
    for chapter in chapters:
        if not (input_root / chapter).is_dir():
            raise SystemExit(f"Chapter folder not found: {input_root / chapter}. {hint}")
    return chapters


def _partial(dst: Path) -> Path:
    # Dot-prefixed so an interrupted write is never picked up as a page.
    return dst.with_name(f".{dst.name}.partial")


@contextmanager
def atomic_path(dst: Path):
    """Yields a temporary path next to dst. It is renamed to dst only when the block finishes
    without an error, so a file under its final name is always complete. A run that is killed
    outright leaves at most a dot-prefixed .partial file, which no step reads."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _partial(dst)
    try:
        yield tmp
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def save_png(im, dst: Path) -> None:
    with atomic_path(dst) as tmp:
        im.save(tmp, format="PNG")


def copy_file(src: Path, dst: Path) -> None:
    with atomic_path(dst) as tmp:
        shutil.copyfile(src, tmp)


def write_text(dst: Path, text: str) -> None:
    with atomic_path(dst) as tmp:
        tmp.write_text(text, encoding="utf-8", newline="")


def write_json(dst: Path, data, **kwargs) -> None:
    write_text(dst, json.dumps(data, indent=1, **kwargs))


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_notes(row: dict, step: str, notes: list[str]) -> None:
    """Replace the notes a step left on a row, keeping the notes of other steps."""
    kept = [n for n in row["notes"].split("; ") if n and not n.startswith(f"{step}:")]
    # Pipeline order, so a rerun that rewrites the same notes leaves the row unchanged.
    order = lambda n: NOTE_ORDER.index(n.split(":")[0]) if n.split(":")[0] in NOTE_ORDER else len(NOTE_ORDER)
    row["notes"] = "; ".join(sorted(kept + [f"{step}: {n}" for n in notes], key=order))


class Manifest:
    """data/manifest.csv: one row per input page, updated by every step."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                for raw in csv.DictReader(f):
                    self.rows.append({c: raw.get(c) or "" for c in MANIFEST_COLUMNS})
        self._by_stem = {
            (r["chapter"], Path(r["file"]).stem.lower()): r
            for r in self.rows if r["status"] != "duplicate"
        }

    def _new_row(self, chapter: str, file: str) -> dict:
        row = {c: "" for c in MANIFEST_COLUMNS}
        row["chapter"] = chapter
        row["file"] = file
        self.rows.append(row)
        return row

    def find(self, chapter: str, stem: str) -> dict | None:
        return self._by_stem.get((chapter, stem.lower()))

    def row(self, chapter: str, stem: str, file: str | None = None) -> dict:
        """The row of a page, looked up by name without extension; created if missing."""
        row = self.find(chapter, stem)
        if row is None:
            row = self._new_row(chapter, file or f"{stem}.png")
            self._by_stem[(chapter, stem.lower())] = row
        return row

    def duplicate_row(self, chapter: str, file: str) -> dict:
        for row in self.rows:
            if row["chapter"] == chapter and row["file"] == file and row["status"] == "duplicate":
                return row
        row = self._new_row(chapter, file)
        row["status"] = "duplicate"
        return row

    def save(self) -> None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(self.rows, key=lambda r: (r["chapter"], r["file"].lower())))
        text = buf.getvalue()
        # Leave the file alone when nothing changed, so a rerun does no work.
        if self.path.is_file() and self.path.read_text(encoding="utf-8") == text:
            return
        write_text(self.path, text)


@dataclass
class Context:
    cfg: dict
    data_dir: Path
    manifest: Manifest
    force: bool = False
    limit: int | None = None
    skip: int = 0
    region: str | None = None   # --region <page>:<id>: redo one region with the diffusion backend
    seed: int | None = None     # --seed for that region
    hard_only: bool = False     # --hard-only: redo the hard regions of already cleaned pages
    root: Path = PROJECT_ROOT  # relative paths in the config resolve against this
