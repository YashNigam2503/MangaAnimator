"""Stage 1, step: 03_normalized + 04_masks + 04_regions -> 04_prompts.

Writes one diffusion prompt per hard region. The region's panel, with the region
outlined, is sent to a vision-language model (Claude), which answers with one
sentence describing what the artwork should plausibly continue with behind it.

Needs Anthropic credentials (ANTHROPIC_API_KEY). Without them the step prints a
notice and writes nothing; hard regions then cannot be diffused until it has run.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from steps.common import NORMALIZED_DIR, PROMPTS_DIR, Context, resolve_chapters, select_pages, write_json

STEP = "prompts"
ENDING = "no additional people"
# The prompt goes to an image model: naming lettering in it tends to make it draw some.
BANNED = re.compile(r"\b(text|letters?|lettering|speech|bubbles?|balloons?|captions?|words?|writing)\b", re.I)

SYSTEM = f"""You help restore manga artwork. You are shown one manga panel in which a region is outlined in red. \
Everything inside the outline will be erased and repainted by an image model, and your sentence is the instruction that model receives.

Write exactly one sentence describing only what the artwork should plausibly show inside the outlined region once it is repainted: \
the background, objects, and the cut-off parts of characters who are already visible in the panel, such as the rest of a shoulder, arm or piece of clothing. \
Describe it as a continuation of the drawing around it, in the same drawing style as the panel.

Do not introduce any new person or face. Never mention text, letters, speech bubbles, the outline, or what currently covers the region. \
End the sentence with the exact words: {ENDING}"""


class NoCredentials(Exception):
    pass


class ClaudeDescriber:
    def __init__(self, cfg: dict):
        import anthropic
        self.anthropic = anthropic
        self.cfg = cfg
        self.name = cfg["model"]
        self.client = anthropic.Anthropic()

    def __call__(self, panel: Image.Image, reminder: str = "") -> str:
        anthropic = self.anthropic
        buffer = io.BytesIO()
        panel.save(buffer, format="PNG")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}},
            {"type": "text", "text": "Describe what should continue inside the outlined region." + reminder},
        ]
        try:
            response = self.client.beta.messages.create(
                model=self.cfg["model"],
                max_tokens=self.cfg["max_tokens"],
                output_config={"effort": self.cfg["effort"]},
                # If the model declines a request, the API reruns it on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
        except (anthropic.AuthenticationError, TypeError) as e:
            raise NoCredentials(str(e))
        except anthropic.RateLimitError as e:
            raise SystemExit(f"Prompts: the Claude API rate limit was hit and retries ran out ({e.message}). Run the step again later.")
        except anthropic.APIStatusError as e:
            raise SystemExit(f"Prompts: the Claude API returned an error ({e.status_code}: {e.message}).")
        except anthropic.APIConnectionError as e:
            raise SystemExit(f"Prompts: could not reach the Claude API ({e}). Check the network connection.")
        if response.stop_reason == "refusal":
            raise SystemExit("Prompts: the model declined to describe a panel. Write that region's prompt by hand in 04_prompts.")
        return "".join(block.text for block in response.content if block.type == "text")


def outlined_panel(original: np.ndarray, labels: np.ndarray, region: dict, box: list[int], cfg: dict) -> Image.Image:
    """The panel with the region outlined in red, shrunk to the size sent to the model."""
    x0, y0, x1, y1 = box
    crop = original[y0:y1, x0:x1].copy()
    own = (labels[y0:y1, x0:x1] == region["id"]).astype(np.uint8)
    contours, _ = cv2.findContours(own, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, (255, 0, 0), cfg["outline_px"])
    image = Image.fromarray(crop)
    image.thumbnail((cfg["image_max_side"], cfg["image_max_side"]), Image.LANCZOS)
    return image


def clean_prompt(text: str) -> str:
    """One line, ending with the required words."""
    text = " ".join(text.split()).strip(" .")
    if not text.lower().endswith(ENDING):
        text = f"{text}, {ENDING}"
    return text


def describe(describer, panel: Image.Image, retries: int) -> str:
    reminder = ""
    for _ in range(retries + 1):
        prompt = clean_prompt(describer(panel, reminder))
        found = BANNED.search(prompt)
        if not found:
            return prompt
        reminder = f" Your last answer used the word '{found.group(0)}', which is not allowed. Describe only the artwork."
    raise SystemExit(f"Prompts: the model kept using a forbidden word ('{found.group(0)}'): {prompt}")


def prompts_path(ctx: Context, chapter: str, stem: str) -> Path:
    return ctx.data_dir / PROMPTS_DIR / chapter / f"{stem}.json"


def load_prompts(ctx: Context, chapter: str, stem: str, regions: list[dict]) -> dict[int, str]:
    """Stored prompts by region id. A prompt written for a region that has since moved is ignored."""
    path = prompts_path(ctx, chapter, stem)
    if not path.is_file():
        return {}
    stored = {p["id"]: p for p in json.loads(path.read_text(encoding="utf-8"))["prompts"]}
    return {r["id"]: stored[r["id"]]["prompt"] for r in regions
            if r["id"] in stored and stored[r["id"]]["box"] == r["box"]}


def run(ctx: Context, chapters: list[str] | None = None, describer=None) -> None:
    from steps.inpaint import PageJob  # shares the page, mask and panel handling
    cfg = ctx.cfg["prompts"]
    if cfg.get("mode", "claude") == "fixed":
        print("Prompts: mode is 'fixed' in config.yaml, so no prompts are written; every hard region uses prompts.fixed_prompt.")
        return
    src_root = ctx.data_dir / NORMALIZED_DIR
    chapters = resolve_chapters(src_root, chapters, "Run stage 0 first.")
    written = skipped = 0
    for chapter in chapters:
        pages, _ = select_pages(ctx, chapter, src_root / chapter)
        for src in tqdm(pages, desc=f"prompts {chapter}", unit="page"):
            job = PageJob(ctx, chapter, src)
            hard = [r for r in job.regions if r["label"] == "hard"]
            known = {} if ctx.force else load_prompts(ctx, chapter, src.stem, hard)
            missing = [r for r in hard if r["id"] not in known]
            if not missing:
                skipped += 1
                continue
            describer = describer or ClaudeDescriber(cfg)
            try:
                for region in missing:
                    box, _ = job.diffusion_window(region)
                    known[region["id"]] = describe(describer, outlined_panel(job.original, job.labels, region, box, cfg), cfg["retries"])
                    written += 1
            except NoCredentials as e:
                print(f"Prompts: no Anthropic credentials ({e}). Set ANTHROPIC_API_KEY and run this step again. "
                      "Until then hard regions cannot be filled by the diffusion backend.")
                return
            write_json(prompts_path(ctx, chapter, src.stem), {
                "page": src.name, "model": getattr(describer, "name", ""),
                "prompts": [{"id": r["id"], "box": r["box"], "prompt": known[r["id"]]} for r in hard],
            }, ensure_ascii=False)
    print(f"Prompts: {written} region prompts written, {skipped} pages already done")
