"""Step 3: 01_png (or 01b_trimmed) -> 02_upscaled. AI upscaling of pages that are too small.

The decision is made from the image this step actually reads, not from the manifest,
so the step can run on another machine with nothing but its input folder and config.yaml.
What it did is written to _upscale_results.csv next to the output pages; that file
travels with the folder and is merged into the manifest back home.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
import types
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from steps.common import (
    PNG_DIR, TRIMMED_DIR, UPSCALE_FACTORS, UPSCALED_DIR, Context, choose_upscale_factor,
    copy_file, select_pages, resolve_chapters, save_png, set_notes,
)

STEP = "upscale"
RESULTS_NAME = "_upscale_results.csv"
RESULT_COLUMNS = [
    "file", "input_width", "input_height", "factor", "result",
    "output_width", "output_height", "backend", "model", "seconds", "note",
]
RELEASES_URL = "https://github.com/xinntao/Real-ESRGAN/releases"


class BackendUnavailable(Exception):
    """The configured backend cannot run here. The message names the fix."""


class NoneBackend:
    """Upscales nothing: flagged pages are copied unchanged and listed."""
    name = "none"
    model = ""

    def check(self) -> None:
        pass


class NcnnBackend:
    """Calls the realesrgan-ncnn-vulkan executable. Runs on any Vulkan GPU, integrated included."""
    name = "ncnn"

    def __init__(self, cfg: dict, root: Path):
        up = cfg["upscale"]
        self.binary = root / up["ncnn_binary"] if up["ncnn_binary"] else None
        self.model = up["model"]
        self.gpu_id = up["gpu_id"]
        self.tile_size = up["tile_size"]
        self.scales: set[int] = set()

    def check(self) -> None:
        if self.binary is None:
            raise BackendUnavailable(
                "upscale.ncnn_binary is empty in config.yaml. Download realesrgan-ncnn-vulkan from "
                f"{RELEASES_URL}, unzip it into tools/realesrgan/ and set the path to the executable."
            )
        if not self.binary.is_file():
            raise BackendUnavailable(
                f"realesrgan-ncnn-vulkan not found at {self.binary}. Download it from {RELEASES_URL}, "
                "unzip it into tools/realesrgan/ and fix upscale.ncnn_binary in config.yaml."
            )
        # The binary loads models from the folder next to it. animevideov3 ships one
        # file per scale; the x4plus models are a single 4x file.
        models = self.binary.parent / "models"
        self.scales = {f for f in UPSCALE_FACTORS if (models / f"{self.model}-x{f}.param").is_file()}
        if not self.scales and (models / f"{self.model}.param").is_file():
            self.scales = {4}
        if not self.scales:
            found = sorted(p.stem for p in models.glob("*.param")) if models.is_dir() else []
            raise BackendUnavailable(
                f"Model '{self.model}' not found in {models}. Available: {', '.join(found) or 'none'}. "
                "Set upscale.model in config.yaml to one of these (without the -x2/-x3/-x4 suffix)."
            )

    def upscale(self, src: Path, dst: Path, factor: int) -> None:
        native = factor if factor in self.scales else max(self.scales)
        if native < factor:
            raise RuntimeError(f"model '{self.model}' has no {factor}x or larger variant")
        tmp = dst.with_name(f".{dst.stem}.partial.png")
        cmd = [
            str(self.binary), "-i", str(src), "-o", str(tmp),
            "-n", self.model, "-s", str(native), "-f", "png",
            "-m", str(self.binary.parent / "models"), "-t", str(self.tile_size),
        ]
        if self.gpu_id != "auto":
            cmd += ["-g", str(self.gpu_id)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not tmp.is_file():
            # stderr is mostly progress percentages; keep the lines that say something.
            lines = [l for l in proc.stderr.splitlines() if l.strip() and not l.strip().endswith("%")]
            raise RuntimeError(
                f"realesrgan-ncnn-vulkan failed on {src.name} (exit code {proc.returncode}):\n" + "\n".join(lines)
            )
        with Image.open(src) as im:
            expected = (im.width * factor, im.height * factor)
        with Image.open(tmp) as out:
            as_wanted = out.mode == "RGB" and out.size == expected
            if not as_wanted:
                # The model only had a larger scale: bring it down to the factor asked for.
                save_png(out.convert("RGB").resize(expected, Image.LANCZOS), dst)
        if as_wanted:
            os.replace(tmp, dst)
            return
        tmp.unlink()


# name -> (architecture, blocks, weights), as in Real-ESRGAN's inference_realesrgan.py
TORCH_MODELS = {
    "realesr-animevideov3": ("srvgg", 16, f"{RELEASES_URL}/download/v0.2.5.0/realesr-animevideov3.pth"),
    "RealESRGAN_x4plus_anime_6B": ("rrdb", 6, f"{RELEASES_URL}/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "RealESRGAN_x4plus": ("rrdb", 23, f"{RELEASES_URL}/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
# The ncnn build names the same models differently; accept either spelling.
TORCH_ALIASES = {
    "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B",
    "realesrgan-x4plus": "RealESRGAN_x4plus",
}


def _patch_torchvision() -> None:
    """basicsr imports torchvision.transforms.functional_tensor, which torchvision 0.17 removed."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        import torchvision.transforms.functional as F
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules[shim.__name__] = shim


class TorchBackend:
    """Uses the Python Real-ESRGAN package on CUDA. The path for Colab or Kaggle."""
    name = "torch"

    def __init__(self, cfg: dict, root: Path):
        up = cfg["upscale"]
        self.model = TORCH_ALIASES.get(up["model"], up["model"])
        self.tile_size = up["tile_size"]
        self.upsampler = None

    def check(self) -> None:
        if self.model not in TORCH_MODELS:
            raise BackendUnavailable(
                f"Model '{self.model}' is not known to the torch backend. "
                f"Set upscale.model in config.yaml to one of: {', '.join(TORCH_MODELS)}."
            )
        try:
            import torch
        except ImportError:
            raise BackendUnavailable("PyTorch is not installed. Run: pip install torch torchvision realesrgan")
        if not torch.cuda.is_available():
            raise BackendUnavailable(
                "The torch backend needs a CUDA GPU and none was found. On Colab or Kaggle, switch the "
                "runtime to GPU. On this machine, set upscale.backend to ncnn or none in config.yaml."
            )
        self.load("cuda")

    def load(self, device: str) -> None:
        try:
            _patch_torchvision()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except ImportError as e:
            raise BackendUnavailable(f"The Real-ESRGAN package could not be imported ({e}). Run: pip install realesrgan")
        arch, blocks, url = TORCH_MODELS[self.model]
        if arch == "srvgg":
            net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=blocks, upscale=4, act_type="prelu")
        else:
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=blocks, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4, model_path=url, model=net, tile=self.tile_size, tile_pad=10, pre_pad=0,
            half=device == "cuda", device=device,
        )

    def upscale(self, src: Path, dst: Path, factor: int) -> None:
        import numpy as np
        with Image.open(src) as im:
            bgr = np.asarray(im.convert("RGB"))[:, :, ::-1]
        # All these models are 4x networks; outscale resizes the result to the factor asked for.
        out, _ = self.upsampler.enhance(bgr, outscale=factor)
        save_png(Image.fromarray(np.ascontiguousarray(out[:, :, ::-1])), dst)


BACKENDS = {"ncnn": NcnnBackend, "torch": TorchBackend}


def make_backend(ctx: Context):
    name = ctx.cfg["upscale"]["backend"]
    if name == "none":
        return NoneBackend()
    if name not in BACKENDS:
        raise SystemExit(f"Unknown upscale backend '{name}' in config.yaml. Use ncnn, torch or none.")
    return BACKENDS[name](ctx.cfg, ctx.root)


def read_results(chapter_dir: Path) -> dict[str, dict]:
    path = chapter_dir / RESULTS_NAME
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["file"]: {c: r.get(c) or "" for c in RESULT_COLUMNS} for r in csv.DictReader(f)}


def write_results(chapter_dir: Path, results: dict[str, dict]) -> None:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    with open(chapter_dir / RESULTS_NAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results[k] for k in sorted(results))


def apply_results(ctx: Context, chapter: str, results: dict[str, dict], set_status: bool) -> None:
    """Copy what the upscale step recorded into the manifest."""
    for file, res in results.items():
        row = ctx.manifest.row(chapter, Path(file).stem)
        row["upscale_applied"] = res["factor"] if res["result"] == "upscaled" else "1"
        set_notes(row, STEP, [res["note"]] if res["note"] else [])
        if set_status:
            row["status"] = {"upscaled": "upscaled", "copied": "not_upscaled", "skipped": "upscale_skipped"}[res["result"]]


def input_root(ctx: Context) -> Path:
    return ctx.data_dir / (TRIMMED_DIR if ctx.cfg["trim_margins"]["enabled"] else PNG_DIR)


def run(ctx: Context, chapters: list[str] | None = None, backend=None) -> None:
    cfg = ctx.cfg
    backend = backend or make_backend(ctx)
    src_root = input_root(ctx)
    chapters = resolve_chapters(src_root, chapters, "Run the earlier steps first, or upload this folder.")

    jobs = []  # (chapter, src, dst, width, height, factor)
    already_done = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in pages:
            dst = ctx.data_dir / UPSCALED_DIR / chapter / src.name
            if dst.exists() and not ctx.force:
                already_done += 1
                continue
            with Image.open(src) as im:
                width, height = im.size
            factor = choose_upscale_factor(height, cfg["min_height"], cfg["max_upscale_factor"])
            jobs.append((chapter, src, dst, width, height, factor))

    if any(job[5] > 1 for job in jobs):
        try:
            backend.check()
        except BackendUnavailable as e:
            raise SystemExit(f"Upscale backend '{backend.name}' is unavailable: {e}")

    counts = {"upscaled": 0, "copied": 0, "skipped": 0}
    skipped_pages = []
    results = {chapter: read_results(ctx.data_dir / UPSCALED_DIR / chapter) for chapter in chapters}
    touched: dict[str, dict] = {chapter: {} for chapter in chapters}
    try:
        for chapter, src, dst, width, height, factor in tqdm(jobs, desc="upscale", unit="page"):
            res = {c: "" for c in RESULT_COLUMNS}
            res.update({"file": src.name, "input_width": str(width), "input_height": str(height),
                        "factor": str(factor), "backend": backend.name})
            start = time.perf_counter()
            if factor == 1:
                copy_file(src, dst)
                res["result"] = "copied"
            elif backend.name == "none":
                copy_file(src, dst)
                res["result"] = "skipped"
                res["note"] = f"skipped (backend none), needs {factor}x"
                skipped_pages.append(f"{chapter}/{src.name} ({width}x{height}, needs {factor}x)")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    backend.upscale(src, dst, factor)
                except Exception as e:
                    raise SystemExit(f"Upscale failed, stopping. {e}")
                res["result"] = "upscaled"
                res["model"] = backend.model
                res["seconds"] = f"{time.perf_counter() - start:.1f}"
                if height * factor < cfg["min_height"]:
                    res["note"] = f"still under min_height after {factor}x ({height * factor}px)"
            with Image.open(dst) as out:
                res["output_width"], res["output_height"] = str(out.width), str(out.height)
            counts[res["result"]] += 1
            results[chapter][src.name] = res
            touched[chapter][src.name] = res
    finally:
        for chapter in chapters:
            if touched[chapter]:
                write_results(ctx.data_dir / UPSCALED_DIR / chapter, results[chapter])
                apply_results(ctx, chapter, touched[chapter], set_status=True)

    print(f"Upscale ({backend.name}): {counts['upscaled']} upscaled, {counts['copied']} copied unchanged, "
          f"{counts['skipped']} skipped, {already_done} already done")
    if skipped_pages:
        print(f"{len(skipped_pages)} pages need upscaling and were copied unchanged. Upscale them elsewhere "
              f"and drop the results into {ctx.data_dir / UPSCALED_DIR}:")
        for page in skipped_pages:
            print(f"  {page}")
