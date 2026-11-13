"""Stage 1, step: 03_normalized -> 04_text_masks. Binary mask of lettering pixels, white = text."""
from __future__ import annotations

import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from steps.common import (
    NORMALIZED_DIR, TEXT_MASKS_DIR, Context, pick_device, resolve_chapters, save_png, select_pages,
)

STEP = "text"
REPO_URL = "https://github.com/dmMaze/comic-text-detector"
WEIGHTS_URL = "https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1"


@contextmanager
def placeholder_module(name: str):
    """Makes `import <name>` succeed with an empty module for the length of the block.

    The placeholder is always taken out again. Left in sys.modules it has no __spec__, and
    `import diffusers` later in the same run then fails with "wandb.__spec__ is None".
    A module of that name that was already imported is put back afterwards.
    """
    previous = sys.modules.get(name)
    sys.modules[name] = types.ModuleType(name)
    try:
        yield
    finally:
        if previous is not None:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)


class ComicTextDetector:
    """comic-text-detector: text segmentation for manga and comics, any orientation."""

    def __init__(self, cfg: dict, root: Path):
        device = pick_device()
        # The PyTorch weights take minutes per page on CPU; the ONNX build of the same model takes seconds.
        repo, weights = root / cfg["repo_dir"], root / cfg["weights" if device == "cuda" else "weights_cpu"]
        if not (repo / "inference.py").is_file():
            raise SystemExit(f"comic-text-detector not found at {repo}. Run: git clone {REPO_URL} {cfg['repo_dir']}")
        if not weights.is_file():
            raise SystemExit(f"Text detector weights not found at {weights}. Download {weights.name} from {WEIGHTS_URL}.")
        # The detector is a plain script folder, not a package: import it from where it sits.
        sys.path.insert(0, str(repo))
        # It imports wandb for training logs only; a placeholder saves installing it.
        with placeholder_module("wandb"):
            from inference import TextDetector
            from utils.textmask import REFINEMASK_INPAINT
        self.refine_mode = REFINEMASK_INPAINT
        self.model = TextDetector(
            model_path=str(weights), input_size=cfg["input_size"], device=device, act="leaky",
        )

    def __call__(self, page: Image.Image) -> Image.Image:
        import numpy as np
        bgr = np.ascontiguousarray(np.asarray(page.convert("RGB"))[:, :, ::-1])
        _, refined, _ = self.model(bgr, refine_mode=self.refine_mode, keep_undetected_mask=True)
        return Image.fromarray(((refined > 0) * 255).astype(np.uint8), mode="L")


def run(ctx: Context, chapters: list[str] | None = None, detector=None) -> None:
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    done = skipped = 0
    seconds = 0.0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"text {chapter}", unit="page"):
            dst = ctx.data_dir / TEXT_MASKS_DIR / chapter / src.name
            if dst.exists() and not ctx.force:
                skipped += 1
                continue
            detector = detector or ComicTextDetector(ctx.cfg["detect_text"], ctx.root)
            start = time.perf_counter()
            with Image.open(src) as im:
                mask = detector(im)
                if mask.size != im.size:
                    raise SystemExit(f"Text detector returned a {mask.size} mask for a {im.size} page ({src.name}).")
            seconds += time.perf_counter() - start
            save_png(mask.convert("L").point(lambda v: 255 if v else 0), dst)
            done += 1
    per_page = f", {seconds / done:.1f}s per page" if done else ""
    print(f"Text: {done} pages{per_page}, {skipped} already done")
