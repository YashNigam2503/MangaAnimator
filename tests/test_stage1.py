import csv
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import CHAPTER, DIFFUSION_COLOUR, LAMA_COLOUR, StubDiffusion, StubLama, layout
from steps import build_masks, describe_regions, detect_bubbles, detect_text, inpaint, make_debug, tag_pages
from steps.build_masks import (
    build_mask, dilate, find_regions, merge_windows, rasterise_polygons, remove_specks, window_for,
)
from steps.common import (
    CLEAN_DIR, CLEAN_LAMA_DIR, DEBUG_DIR, DETECTIONS_DIR, MASKS_DIR, NORMALIZED_DIR, REGIONS_DIR,
    REGIONS_REPORT_NAME, TEXT_MASKS_DIR,
)

PAGE_SIZE = (1000, 1400)


@pytest.fixture
def mask_cfg(cfg):
    return cfg["mask"]


@pytest.fixture
def region_cfg(cfg):
    return cfg["regions"]


@pytest.fixture
def pages(ctx):
    """Two textured pages in 03_normalized."""
    folder = ctx.data_dir / NORMALIZED_DIR / CHAPTER
    folder.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for name in ("p1.png", "p2.png"):
        noise = rng.integers(0, 256, (PAGE_SIZE[1], PAGE_SIZE[0], 3), dtype=np.uint8)
        Image.fromarray(noise).save(folder / name, compress_level=1)
    return folder


def run_stage1(ctx, **inpaint_args):
    detect_bubbles.run(ctx)
    detect_text.run(ctx)
    tag_pages.run(ctx)
    build_masks.run(ctx)
    describe_regions.run(ctx)
    inpaint.run(ctx, **inpaint_args)
    make_debug.run(ctx)


def load(ctx, folder, name="p1.png"):
    with Image.open(ctx.data_dir / folder / CHAPTER / name) as im:
        return np.array(im)


def regions_of(ctx, stem="p1"):
    return json.loads((ctx.data_dir / REGIONS_DIR / CHAPTER / f"{stem}.json").read_text())["regions"]


def report_rows(ctx):
    with open(ctx.data_dir / REGIONS_REPORT_NAME, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- mask building ---

def test_polygon_rasterisation():
    mask = rasterise_polygons([[[10, 10], [50, 10], [50, 30], [10, 30]]], (100, 60))
    assert mask.shape == (60, 100) and set(np.unique(mask)) == {0, 255}
    assert mask[20, 30] == 255 and mask[5, 30] == 0 and mask[20, 60] == 0
    assert (mask > 0).sum() == 41 * 21      # fillPoly includes the boundary


def test_dilate_grows_by_the_given_pixels():
    mask = np.zeros((100, 100), np.uint8)
    mask[50, 50] = 255
    grown = dilate(mask, 6)
    ys, xs = np.nonzero(grown)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (44, 56, 44, 56)
    assert (dilate(mask, 0) == mask).all()


def test_bubble_and_text_use_their_own_dilation(mask_cfg):
    cfg = dict(mask_cfg, bubble_dilate_px=12, text_dilate_px=8, min_area_px=1)
    text = np.zeros((400, 600), np.uint8)
    text[300:340, 400:500] = 255
    final, bubbles, grown_text = build_mask([[[100, 100], [200, 100], [200, 180], [100, 180]]], text, cfg)
    ys, xs = np.nonzero(bubbles)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (88, 212, 88, 192)
    ys, xs = np.nonzero(grown_text)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (392, 507, 292, 347)
    assert (final == (bubbles | grown_text)).all()


def test_union_and_speck_removal(mask_cfg):
    cfg = dict(mask_cfg, bubble_dilate_px=0, text_dilate_px=0, min_area_px=400)
    text = np.zeros((300, 300), np.uint8)
    text[120:140, 120:160] = 255       # inside the bubble: already covered
    text[250:260, 250:260] = 255       # 100 px speck
    text[200:230, 20:60] = 255         # 1200 px, kept
    final, _, _ = build_mask([[[100, 100], [200, 100], [200, 200], [100, 200]]], text, cfg)
    assert final[130, 140] == 255 and final[150, 150] == 255
    assert final[255, 255] == 0
    assert final[215, 40] == 255
    assert (remove_specks(text, 400)[250:260, 250:260] == 0).all()


def test_small_holes_in_the_mask_are_closed(mask_cfg):
    cfg = dict(mask_cfg, bubble_dilate_px=0, text_dilate_px=0, min_area_px=400)
    text = np.zeros((300, 300), np.uint8)
    text[50:250, 50:250] = 255
    text[100:110, 100:110] = 0         # small hole: closed
    text[150:200, 150:200] = 0         # 2500 px hole: left open
    final, _, _ = build_mask([], text, cfg)
    assert final[105, 105] == 255 and final[175, 175] == 0 and final[10, 10] == 0


def bubble_page(outline=True, background=120):
    """A white bubble with a pointed tail. Returns (grey page, polygon of the bubble body only)."""
    page = Image.new("L", (600, 600), background)
    draw = ImageDraw.Draw(page)
    edge = 0 if outline else 255
    draw.polygon([(280, 380), (340, 380), (420, 520)], fill=255, outline=edge)      # tail
    draw.ellipse((150, 150, 450, 400), fill=255, outline=edge, width=4)
    draw.rectangle((290, 385, 335, 405), fill=255)                                            # open the tail into the body
    body = [[300 + 150 * np.cos(a), 275 + 125 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 90)]
    return np.asarray(page), body


def test_bubble_mask_grows_over_the_tail(mask_cfg):
    cfg = dict(mask_cfg, bubble_dilate_px=12, min_area_px=1)
    grey, body = bubble_page()
    empty = np.zeros_like(grey)
    plain, _, _ = build_mask([body], empty, cfg)
    grown, _, _ = build_mask([body], empty, cfg, grey)
    assert plain[505, 410] == 0                     # the polygon alone stops short of the tail tip
    assert grown[505, 410] == 255 and grown[520, 420] == 255
    assert grown[100, 100] == 0 and grown[560, 300] == 0
    assert (grown >= plain).all()


def test_tail_growth_gives_up_when_the_white_leaks_or_the_bubble_is_dark(mask_cfg):
    cfg = dict(mask_cfg, bubble_dilate_px=12, min_area_px=1)
    empty = np.zeros((600, 600), np.uint8)
    grey, body = bubble_page(outline=False, background=255)       # white bubble on a white page
    assert (build_mask([body], empty, cfg, grey)[0] == build_mask([body], empty, cfg)[0]).all()
    dark = np.full((600, 600), 30, np.uint8)                      # dark bubble
    assert (build_mask([body], empty, cfg, dark)[0] == build_mask([body], empty, cfg)[0]).all()
    off = dict(cfg, tail_growth=dict(cfg["tail_growth"], enabled=False))
    grey, body = bubble_page()
    assert (build_mask([body], empty, off, grey)[0] == build_mask([body], empty, off)[0]).all()


def test_region_splitting_sources_and_labels(region_cfg):
    bubbles = np.zeros((1000, 1000), np.uint8)
    text = np.zeros((1000, 1000), np.uint8)
    bubbles[100:400, 100:400] = 255     # 9% of the page, with text inside
    text[200:250, 150:350] = 255
    bubbles[700:750, 100:160] = 255     # bubble only
    text[700:740, 700:800] = 255        # text only
    regions = find_regions(bubbles | text, bubbles, text, dict(region_cfg, hard_area_fraction=0.04))
    assert [r["id"] for r in regions] == [1, 2, 3]
    by_box = {tuple(r["box"]): r for r in regions}
    big = by_box[(100, 100, 400, 400)]
    assert (big["source"], big["label"], big["area"], big["area_share"]) == ("both", "hard", 90000, 0.09)
    assert (by_box[(100, 700, 160, 750)]["source"], by_box[(100, 700, 160, 750)]["label"]) == ("bubble", "easy")
    assert by_box[(700, 700, 800, 740)]["source"] == "text"


def test_window_is_square_grown_and_clamped(region_cfg):
    cfg = dict(region_cfg, context_factor=2.0, min_window=512, max_window=1536)
    page = (1400, 2000)
    assert window_for([600, 900, 1000, 1200], page, cfg) == [400, 650, 1200, 1450]       # 400 * 2 = 800 square
    assert window_for([700, 1000, 750, 1040], page, cfg) == [469, 764, 981, 1276]        # min_window
    x0, y0, x1, y1 = window_for([100, 100, 1100, 1900], page, cfg)                       # max_window, then the page
    assert (x1 - x0, y1 - y0) == (1400, 1800) and x0 == 0


def test_window_at_page_edges_stays_inside_the_page(region_cfg):
    cfg = dict(region_cfg, context_factor=2.0, min_window=512, max_window=1536)
    assert window_for([0, 0, 60, 40], (1400, 2000), cfg) == [0, 0, 512, 512]
    assert window_for([1350, 1960, 1400, 2000], (1400, 2000), cfg) == [888, 1488, 1400, 2000]
    assert window_for([10, 10, 50, 50], (300, 400), cfg) == [0, 0, 300, 400]               # page smaller than a window


def test_window_merging(region_cfg):
    a, b, far = [0, 0, 512, 512], [100, 50, 612, 562], [1000, 1000, 1512, 1512]
    assert merge_windows([a, b, far], 0.5, 1536) == [1, 1, 2]
    assert merge_windows([a, [400, 400, 912, 912]], 0.5, 1536) == [1, 2]      # slight overlap: kept apart
    assert merge_windows([a, b, far], 0.5, 600) == [1, 2, 3]           # merged window would exceed the limit
    chain = [[i * 200, 0, i * 200 + 512, 512] for i in range(8)]       # each overlaps the next
    groups = merge_windows(chain, 0.5, 1536)
    assert len(set(groups)) > 1                                         # the chain is not swallowed whole
    bubbles = np.zeros((2000, 1400), np.uint8)
    bubbles[300:360, 300:360] = 255
    bubbles[300:360, 420:480] = 255
    bubbles[1700:1760, 1000:1060] = 255
    regions = find_regions(bubbles, bubbles, np.zeros_like(bubbles), region_cfg)
    assert [r["group"] for r in regions] == [1, 1, 2]
    assert regions[0]["window"] == regions[1]["window"]
    x0, y0, x1, y1 = regions[0]["window"]
    assert x0 <= 300 and x1 >= 480 and y0 <= 300 and y1 >= 360       # the merged window holds both regions


# --- the steps, with stub detectors and backends ---

def test_detection_outputs(ctx, pages):
    detect_bubbles.run(ctx)
    detect_text.run(ctx)
    detections = json.loads((ctx.data_dir / DETECTIONS_DIR / CHAPTER / "p1.json").read_text())
    assert (detections["width"], detections["height"]) == PAGE_SIZE
    assert [d["id"] for d in detections["detections"]] == [1, 2]
    assert len(detections["detections"][0]["polygon"]) == 72
    text = load(ctx, TEXT_MASKS_DIR)
    assert text.shape == (PAGE_SIZE[1], PAGE_SIZE[0]) and set(np.unique(text)) == {0, 255}


def test_masks_and_regions_from_the_stub_layout(ctx, pages):
    detect_bubbles.run(ctx)
    detect_text.run(ctx)
    build_masks.run(ctx)
    mask = load(ctx, MASKS_DIR)
    assert set(np.unique(mask)) == {0, 255}
    assert mask[7, 7] == 0                                            # the speck is gone
    regions = regions_of(ctx)
    assert sorted((r["source"], r["label"]) for r in regions) == [("both", "hard"), ("bubble", "easy"), ("text", "easy")]


def test_paste_back_leaves_unmasked_pixels_identical(ctx, pages):
    run_stage1(ctx)
    for name in ("p1.png", "p2.png"):
        original, mask = load(ctx, NORMALIZED_DIR, name), load(ctx, MASKS_DIR, name) > 0
        for folder in (CLEAN_LAMA_DIR, CLEAN_DIR):
            result = load(ctx, folder, name)
            assert result.shape == original.shape
            assert (result[~mask] == original[~mask]).all()          # untouched outside the mask
            assert (result[mask] == LAMA_COLOUR).all()               # filled everywhere inside it
    assert StubLama.calls == 1                                       # one batch for all pages


def test_cpu_fallback_marks_hard_regions_pending(ctx, pages, capsys):
    run_stage1(ctx)
    assert (load(ctx, CLEAN_DIR) == load(ctx, CLEAN_LAMA_DIR)).all()
    rows = [r for r in report_rows(ctx) if r["page"] == "p1.png"]
    assert len(rows) == 3
    hard = [r for r in rows if r["label"] == "hard"]
    assert len(hard) == 1 and hard[0]["status"] == "diffusion_pending" and hard[0]["backend"] == "lama"
    assert all(r["status"] == "done" and r["backend"] == "lama" for r in rows if r["label"] == "easy")
    assert all(r["page_type"] == "story" and r["seconds"] for r in rows)
    assert "2 hard regions pending diffusion" in capsys.readouterr().out


def test_diffusion_backend_fills_only_hard_regions(ctx, pages):
    diffusion = StubDiffusion()
    run_stage1(ctx, diffusion=diffusion)
    original, final, lama = load(ctx, NORMALIZED_DIR), load(ctx, CLEAN_DIR), load(ctx, CLEAN_LAMA_DIR)
    mask = load(ctx, MASKS_DIR) > 0
    labels, _ = build_masks.label_regions(load(ctx, MASKS_DIR))
    hard = next(r for r in regions_of(ctx) if r["label"] == "hard")
    hard_pixels = labels == hard["id"]
    assert (final[hard_pixels] == DIFFUSION_COLOUR).all()
    assert (final[mask & ~hard_pixels] == LAMA_COLOUR).all()
    assert (final[~mask] == original[~mask]).all()
    assert (lama[mask] == LAMA_COLOUR).all()                          # the LaMa baseline is still complete
    row = next(r for r in report_rows(ctx) if r["page"] == "p1.png" and r["label"] == "hard")
    assert (row["backend"], row["status"], row["seed"]) == ("stub-diffusion", "done", str(ctx.cfg["inpaint"]["seed"] + hard["id"]))


def test_hard_backend_lama_and_none(ctx, pages):
    ctx.cfg["inpaint"]["hard_backend"] = "lama"
    run_stage1(ctx)
    assert all(r["status"] == "done" for r in report_rows(ctx))

    ctx.cfg["inpaint"]["hard_backend"] = "none"
    ctx.force = True
    inpaint.run(ctx)
    original, final = load(ctx, NORMALIZED_DIR), load(ctx, CLEAN_DIR)
    labels, _ = build_masks.label_regions(load(ctx, MASKS_DIR))
    hard = next(r for r in regions_of(ctx) if r["label"] == "hard")
    assert (final[labels == hard["id"]] == original[labels == hard["id"]]).all()
    assert next(r for r in report_rows(ctx) if r["label"] == "hard")["status"] == "not_filled"


def test_tagging_cover_needs_colour_and_lots_of_text(ctx, pages):
    detect_text.run(ctx)
    ctx.cfg["page_type"]["cover_text_fraction"] = 0.01
    ctx.manifest.row(CHAPTER, "p1")["is_colour"] = "true"
    ctx.manifest.row(CHAPTER, "p2")["is_colour"] = "false"
    tag_pages.run(ctx)
    assert ctx.manifest.find(CHAPTER, "p1")["page_type"] == "cover"
    assert ctx.manifest.find(CHAPTER, "p2")["page_type"] == "story"


def test_debug_images_exist(ctx, pages):
    run_stage1(ctx)
    debug = ctx.data_dir / DEBUG_DIR / CHAPTER
    names = sorted(p.name for p in debug.iterdir())
    assert names == ["hard_regions_01.png", "p1_compare.png", "p1_overlay.png", "p2_compare.png", "p2_overlay.png"]
    with Image.open(debug / "p1_overlay.png") as overlay:
        assert overlay.size == PAGE_SIZE
        spots = layout(PAGE_SIZE)
        r, g, b = overlay.getpixel((int(spots["small_bubble"][0]), int(spots["small_bubble"][1])))
        assert r > b                                                  # bubble region is tinted red
        x0, y0, x1, y1 = spots["free_text"]
        r, g, b = overlay.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
        assert b > r                                                  # text region is tinted blue
    with Image.open(debug / "p1_compare.png") as compare:
        assert compare.width > 3 * PAGE_SIZE[0]


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_second_run_does_nothing_and_force_redoes(ctx, pages):
    run_stage1(ctx)
    before = snapshot(ctx.data_dir)
    run_stage1(ctx)
    assert snapshot(ctx.data_dir) == before
    assert StubLama.calls == 1

    ctx.force = True
    run_stage1(ctx)
    after = snapshot(ctx.data_dir)
    assert StubLama.calls == 2
    for name, (content, mtime) in before.items():
        if name.startswith(NORMALIZED_DIR):
            assert after[name] == (content, mtime)                    # the input folder is never touched
        else:
            assert after[name][0] == content and after[name][1] > mtime, name


def test_missing_inputs_give_clear_errors(ctx, pages):
    with pytest.raises(SystemExit, match="Run the bubbles and text steps first"):
        build_masks.run(ctx)
    with pytest.raises(SystemExit, match="Run the masks step first"):
        inpaint.run(ctx)
