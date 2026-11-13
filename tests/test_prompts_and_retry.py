import csv
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import CHAPTER, DIFFUSION_COLOUR, LAMA_COLOUR, STUB_PROMPT, StubDescriber, StubDiffusion, StubFaces
from steps import build_masks, describe_regions, detect_bubbles, detect_text, inpaint, make_debug, tag_pages
from steps.common import CLEAN_DIR, CLEAN_LAMA_DIR, DEBUG_DIR, MASKS_DIR, PROMPTS_DIR, REGIONS_REPORT_NAME
from steps.describe_regions import NoCredentials, clean_prompt, describe
from steps.detect_faces import face_inside_mask, non_max_suppression
from steps.panels import find_panels, panel_for
from test_stage1 import PAGE_SIZE, load, pages, regions_of, run_stage1  # noqa: F401  (pages is a fixture)


def prepare(ctx):
    """Everything up to the masks, so a test can run the prompts and inpaint steps its own way."""
    detect_bubbles.run(ctx)
    detect_text.run(ctx)
    tag_pages.run(ctx)
    build_masks.run(ctx)


def hard_region(ctx, stem="p1"):
    return next(r for r in regions_of(ctx, stem) if r["label"] == "hard")


def report_row(ctx, page, region_id):
    with open(ctx.data_dir / REGIONS_REPORT_NAME, newline="", encoding="utf-8") as f:
        return next(r for r in csv.DictReader(f) if r["page"] == page and r["region_id"] == str(region_id))


def own_pixels(ctx, region, name="p1.png"):
    labels, _ = build_masks.label_regions(load(ctx, MASKS_DIR, name))
    return labels == region["id"]


# --- prompts ---

def test_one_prompt_per_hard_region_is_stored(ctx, pages):
    prepare(ctx)
    describe_regions.run(ctx)
    stored = json.loads((ctx.data_dir / PROMPTS_DIR / CHAPTER / "p1.json").read_text(encoding="utf-8"))
    hard = hard_region(ctx)
    assert stored["prompts"] == [{"id": hard["id"], "box": hard["box"], "prompt": STUB_PROMPT}]
    assert len(StubDescriber.images) == 2                     # one hard region on each of the two pages
    panel = np.asarray(StubDescriber.images[0])
    assert max(panel.shape[:2]) <= ctx.cfg["prompts"]["image_max_side"]
    assert ((panel[:, :, 0] > 200) & (panel[:, :, 1] < 80) & (panel[:, :, 2] < 80)).sum() > 500   # the red outline

    describe_regions.run(ctx)                                  # second run: nothing to ask
    assert len(StubDescriber.images) == 2


def test_prompt_is_one_line_ending_with_the_required_words():
    assert clean_prompt("A wall continues.\n") == "A wall continues, no additional people"
    assert clean_prompt("The sky continues, No Additional People.") == "The sky continues, No Additional People"


def test_forbidden_words_are_retried_then_refused():
    answers = ["The speech bubble area shows sky, no additional people", "Cloudy sky continues, no additional people"]
    reminders = []

    def describer(panel, reminder=""):
        reminders.append(reminder)
        return answers.pop(0)

    assert describe(describer, None, retries=2) == "Cloudy sky continues, no additional people"
    assert reminders[0] == "" and "'speech'" in reminders[1]
    assert describe(lambda panel, reminder="": "Context of a textured wall, no additional people", None, 0)   # not whole words
    with pytest.raises(SystemExit, match="forbidden word"):
        describe(lambda panel, reminder="": "Some letters on a sign, no additional people", None, retries=1)


def test_without_credentials_the_cpu_run_still_completes(ctx, pages, capsys):
    def no_key(panel, reminder=""):
        raise NoCredentials("no key")

    prepare(ctx)
    describe_regions.run(ctx, describer=no_key)
    assert "no Anthropic credentials" in capsys.readouterr().out
    assert not (ctx.data_dir / PROMPTS_DIR).exists()
    inpaint.run(ctx)                                           # no GPU either: hard regions stay pending
    assert report_row(ctx, "p1.png", hard_region(ctx)["id"])["status"] == "diffusion_pending"


def test_diffusion_without_a_prompt_stops_and_names_the_fix(ctx, pages):
    prepare(ctx)
    with pytest.raises(SystemExit, match="Run the prompts step first"):
        inpaint.run(ctx, diffusion=StubDiffusion())


def test_a_prompt_for_a_region_that_moved_is_ignored(ctx, pages):
    prepare(ctx)
    describe_regions.run(ctx)
    path = ctx.data_dir / PROMPTS_DIR / CHAPTER / "p1.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["prompts"][0]["box"][0] += 5
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert describe_regions.load_prompts(ctx, CHAPTER, "p1", regions_of(ctx)) == {}
    describe_regions.run(ctx)                                  # asked again for that region only
    assert len(StubDescriber.images) == 3


# --- diffusion window, prompt in the report, face check ---

def test_diffusion_gets_the_region_prompt_and_the_whole_panel(ctx, pages):
    diffusion = StubDiffusion()
    run_stage1(ctx, diffusion=diffusion)
    assert diffusion.prompts == [STUB_PROMPT, STUB_PROMPT]
    assert diffusion.sizes == [PAGE_SIZE, PAGE_SIZE]          # the noise page is one panel, and it fits
    hard = hard_region(ctx)
    row = report_row(ctx, "p1.png", hard["id"])
    assert (row["prompt"], row["attempts"], row["status"], row["backend"]) == (STUB_PROMPT, "1", "done", "stub-diffusion")
    assert report_row(ctx, "p1.png", next(r["id"] for r in regions_of(ctx) if r["label"] == "easy"))["prompt"] == ""


def test_a_panel_over_the_size_limit_falls_back_to_the_region_window(ctx, pages):
    prepare(ctx)
    describe_regions.run(ctx)
    ctx.cfg["regions"]["max_window"] = 900                    # the 1000x1400 panel no longer fits
    describe_regions.run(ctx)
    diffusion = StubDiffusion()
    inpaint.run(ctx, diffusion=diffusion)
    x0, y0, x1, y1 = hard_region(ctx)["window"]
    assert diffusion.sizes[0] == (x1 - x0, y1 - y0)


def test_a_fill_with_a_face_in_the_mask_is_retried_with_the_next_seed(ctx, pages):
    prepare(ctx)
    describe_regions.run(ctx)
    StubFaces.verdicts = [True, True, False]                  # p1: rejected twice, accepted on the third seed
    diffusion = StubDiffusion()
    inpaint.run(ctx, diffusion=diffusion)
    hard = hard_region(ctx)
    base = ctx.cfg["inpaint"]["seed"] + hard["id"]
    assert diffusion.seeds[:3] == [base, base + 1, base + 2]
    row = report_row(ctx, "p1.png", hard["id"])
    assert (row["status"], row["backend"], row["seed"], row["attempts"]) == ("done", "stub-diffusion", str(base + 2), "3")
    assert (load(ctx, CLEAN_DIR)[own_pixels(ctx, hard)] == DIFFUSION_COLOUR).all()


def test_when_every_attempt_fails_the_lama_fill_is_kept_for_review(ctx, pages, capsys):
    prepare(ctx)
    describe_regions.run(ctx)
    StubFaces.verdicts = [True, True, True]                   # p1 fails all three; p2 then passes
    diffusion = StubDiffusion()
    inpaint.run(ctx, diffusion=diffusion)
    hard = hard_region(ctx)
    assert len(diffusion.seeds) == 3 + 1
    row = report_row(ctx, "p1.png", hard["id"])
    assert (row["status"], row["backend"], row["attempts"], row["prompt"]) == ("needs_review", "lama", "3", STUB_PROMPT)
    assert (load(ctx, CLEAN_DIR)[own_pixels(ctx, hard)] == LAMA_COLOUR).all()
    assert report_row(ctx, "p2.png", hard_region(ctx, "p2")["id"])["status"] == "done"
    assert "1 need review" in capsys.readouterr().out


# --- redoing only the hard regions ---

def test_hard_only_redoes_hard_regions_and_nothing_else(ctx, pages):
    run_stage1(ctx)                                            # CPU run: hard regions pending
    lama_before = {n: (ctx.data_dir / CLEAN_LAMA_DIR / CHAPTER / n).stat().st_mtime_ns for n in ("p1.png", "p2.png")}
    easy = next(r for r in regions_of(ctx) if r["label"] == "easy")
    easy_row_before = report_row(ctx, "p1.png", easy["id"])

    diffusion = StubDiffusion()
    ctx.hard_only = True
    inpaint.run(ctx, diffusion=diffusion)
    ctx.force = True
    make_debug.run(ctx)

    hard = hard_region(ctx)
    final, own = load(ctx, CLEAN_DIR), own_pixels(ctx, hard)
    mask = load(ctx, MASKS_DIR) > 0
    assert (final[own] == DIFFUSION_COLOUR).all()
    assert (final[mask & ~own] == LAMA_COLOUR).all()
    assert len(diffusion.seeds) == 2                           # the two hard regions, one per page
    for name, mtime in lama_before.items():                    # the LaMa baseline was not rerun
        assert (ctx.data_dir / CLEAN_LAMA_DIR / CHAPTER / name).stat().st_mtime_ns == mtime
    assert report_row(ctx, "p1.png", easy["id"]) == easy_row_before
    assert report_row(ctx, "p1.png", hard["id"])["status"] == "done"
    assert (ctx.data_dir / DEBUG_DIR / CHAPTER / "hard_regions_01.png").exists()


def test_hard_only_needs_a_gpu_and_cleaned_pages(ctx, pages):
    prepare(ctx)
    describe_regions.run(ctx)
    ctx.hard_only = True
    with pytest.raises(SystemExit, match="Run the inpaint step on this page first"):
        inpaint.run(ctx, diffusion=StubDiffusion())
    ctx.hard_only = False
    inpaint.run(ctx)
    ctx.hard_only = True
    with pytest.raises(SystemExit, match="needs a GPU"):
        inpaint.run(ctx)


# --- panels and faces ---

def test_panels_are_found_and_matched_to_regions(cfg):
    page = Image.new("L", (1000, 1400), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle((50, 50, 950, 600), outline=0, width=5)
    draw.rectangle((50, 650, 480, 1350), outline=0, width=5)
    draw.rectangle((520, 650, 950, 1350), outline=0, width=5)
    draw.rectangle((700, 300, 720, 320), fill=0)              # a small mark: not a panel
    panels = find_panels(np.asarray(page), cfg["panels"])
    assert panels == [[50, 50, 951, 601], [50, 650, 481, 1351], [520, 650, 951, 1351]]
    assert panel_for([600, 700, 800, 900], panels) == [520, 650, 951, 1351]
    assert panel_for([400, 900, 560, 1000], panels) == [50, 650, 560, 1351]     # widened to hold the region
    assert panel_for([10, 10, 30, 30], panels) is None                           # in the margin


def test_face_centre_decides(cfg):
    mask = Image.new("L", (200, 200), 0)
    mask.paste(255, (100, 100, 200, 200))
    inside, outside = (120, 120, 180, 180, 0.9), (0, 0, 80, 80, 0.9)
    straddling = (60, 60, 130, 130, 0.9)                      # overlaps the mask, centre (95, 95) is outside it
    assert face_inside_mask([outside, straddling], mask) is None
    assert face_inside_mask([outside, inside], mask) == inside


def test_non_max_suppression_keeps_the_best_of_each_cluster():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5], dtype=np.float32)
    assert sorted(non_max_suppression(boxes, scores, 0.5)) == [1, 2]
