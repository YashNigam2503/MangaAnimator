import copy
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from steps.common import MANIFEST_NAME, RAW_DIR, Context, Manifest, load_config  # noqa: E402

CHAPTER = "ch"


@pytest.fixture
def cfg():
    cfg = copy.deepcopy(load_config(ROOT / "config.yaml"))
    cfg["upscale"]["backend"] = "none"
    cfg["trim_margins"]["enabled"] = False
    return cfg


@pytest.fixture
def ctx(tmp_path, cfg):
    data = tmp_path / "data"
    (data / RAW_DIR / CHAPTER).mkdir(parents=True)
    return Context(cfg=cfg, data_dir=data, manifest=Manifest(data / MANIFEST_NAME))


@pytest.fixture
def raw(ctx):
    return ctx.data_dir / RAW_DIR / CHAPTER


def make_page(path: Path, size, colour=(200, 200, 200), mode="RGB", **save_args) -> Path:
    """A synthetic page: flat background with a dark block, so it is not blank."""
    im = Image.new(mode, size, colour)
    w, h = size
    dark = 0 if mode in ("L", "I;16") else (0, 0, 0) + ((255,) if mode == "RGBA" else ())
    im.paste(dark, (w // 4, h // 4, w // 2, h // 2))
    im.save(path, **save_args)
    return path


class StubBackend:
    """Stands in for Real-ESRGAN: a plain nearest-neighbour resize."""
    name = "stub"
    model = "stub-model"

    def __init__(self):
        self.calls = []

    def check(self):
        pass

    def upscale(self, src, dst, factor):
        self.calls.append((src.name, factor))
        with Image.open(src) as im:
            im.resize((im.width * factor, im.height * factor), Image.NEAREST).save(dst)


# --- stage 1 stand-ins: no model downloads, no GPU ---

import math  # noqa: E402

from PIL import ImageDraw  # noqa: E402

LAMA_COLOUR = (255, 0, 0)
DIFFUSION_COLOUR = (0, 255, 0)


def layout(size):
    """Where the stub detectors "find" things on a page of this size."""
    w, h = size
    return {
        "big_bubble": (0.30 * w, 0.30 * h, 0.15 * w, 0.10 * h),      # centre x, centre y, radius x, radius y
        "small_bubble": (0.75 * w, 0.70 * h, 0.06 * w, 0.04 * h),
        "bubble_text": (int(0.22 * w), int(0.27 * h), int(0.38 * w), int(0.33 * h)),
        "free_text": (int(0.10 * w), int(0.85 * h), int(0.30 * w), int(0.89 * h)),
    }


def ellipse_polygon(cx, cy, rx, ry, points=72):
    return [[cx + rx * math.cos(2 * math.pi * i / points), cy + ry * math.sin(2 * math.pi * i / points)]
            for i in range(points)]


class StubBubbleDetector:
    name = "stub-bubbles"

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, page):
        with Image.open(page) as im:
            spots = layout(im.size)
        return [{"polygon": ellipse_polygon(*spots[k]), "box": [0, 0, 0, 0], "confidence": 0.9, "area": 0}
                for k in ("big_bubble", "small_bubble")]


class StubTextDetector:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, page):
        mask = Image.new("L", page.size, 0)
        draw = ImageDraw.Draw(mask)
        spots = layout(page.size)
        for k in ("bubble_text", "free_text"):
            draw.rectangle(spots[k], fill=255)
        draw.rectangle((5, 5, 9, 9), fill=255)     # a speck, too small to keep
        return mask


class StubLama:
    name = "lama"
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def fill(self, crops):
        StubLama.calls += 1
        return [Image.new("RGB", image.size, LAMA_COLOUR) for image, _ in crops]


STUB_PROMPT = "a quiet street continues behind the figure, no additional people"


class StubDiffusion:
    name = "stub-diffusion"

    def __init__(self):
        self.seeds, self.prompts, self.sizes = [], [], []

    def fill(self, image, mask, prompt, seed):
        self.seeds.append(seed)
        self.prompts.append(prompt)
        self.sizes.append(image.size)
        return Image.new("RGB", image.size, DIFFUSION_COLOUR)


class StubDescriber:
    """Stands in for the vision-language model."""
    name = "stub-describer"
    answers = None          # a list of answers to hand out in turn; None = always STUB_PROMPT
    images = []

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, panel, reminder=""):
        StubDescriber.images.append(panel)
        return StubDescriber.answers.pop(0) if StubDescriber.answers else STUB_PROMPT


class StubFaces:
    """Stands in for the face detector. `verdicts` says, per call, whether a face sits in the mask."""
    verdicts = None         # e.g. [True, False]; None = never a face

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        found = StubFaces.verdicts.pop(0) if StubFaces.verdicts else False
        # A face covering the whole image has its centre in the middle of the window, inside the big bubble.
        return [(0.0, 0.0, float(image.width), float(image.height), 0.9)] if found else []


@pytest.fixture(autouse=True)
def stub_models(monkeypatch):
    """Whatever a test runs, the real detectors and LaMa are never loaded."""
    from steps import describe_regions, detect_bubbles, detect_text, inpaint
    monkeypatch.setattr(detect_bubbles, "YoloBubbleDetector", StubBubbleDetector)
    monkeypatch.setattr(detect_text, "ComicTextDetector", StubTextDetector)
    monkeypatch.setattr(inpaint, "LamaBackend", StubLama)
    monkeypatch.setattr(inpaint, "make_diffusion_backend", lambda cfg: None)   # as on a machine without a GPU
    monkeypatch.setattr(inpaint, "AnimeFaceDetector", StubFaces)
    monkeypatch.setattr(describe_regions, "ClaudeDescriber", StubDescriber)
    StubDescriber.answers, StubDescriber.images, StubFaces.verdicts = None, [], None
    StubLama.calls = 0
