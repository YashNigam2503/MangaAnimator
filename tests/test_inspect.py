import pytest
from PIL import Image

from conftest import CHAPTER, make_page
from steps import inspect_pages
from steps.common import choose_upscale_factor


def test_manifest_values_small_large_spread(ctx, raw):
    make_page(raw / "small.jpg", (600, 800))
    make_page(raw / "large.png", (1400, 2000))
    make_page(raw / "spread.webp", (2400, 1700))
    summary = inspect_pages.run(ctx)

    small = ctx.manifest.find(CHAPTER, "small")
    assert (small["file"], small["format"], small["width"], small["height"]) == ("small.jpg", "JPEG", "600", "800")
    assert (small["is_spread"], small["needs_upscale"], small["upscale_factor"]) == ("false", "true", "2")
    assert small["status"] == "inspected"

    large = ctx.manifest.find(CHAPTER, "large")
    assert (large["format"], large["is_spread"], large["needs_upscale"], large["upscale_factor"]) == \
        ("PNG", "false", "false", "1")

    spread = ctx.manifest.find(CHAPTER, "spread")
    assert (spread["format"], spread["is_spread"], spread["needs_upscale"]) == ("WEBP", "true", "false")

    assert summary["pages"] == 3
    assert summary["needs_upscale"] == 1
    assert summary["smallest"]["file"] == "small.jpg"
    assert summary["largest"]["file"] == "large.png"


def test_colour_detection(ctx, raw):
    make_page(raw / "grey_as_rgb.jpg", (300, 400), colour=(180, 180, 180))
    make_page(raw / "grey.png", (300, 400), colour=180, mode="L")
    make_page(raw / "colour.jpg", (300, 400), colour=(200, 60, 40))
    inspect_pages.run(ctx)
    assert ctx.manifest.find(CHAPTER, "grey_as_rgb")["is_colour"] == "false"
    assert ctx.manifest.find(CHAPTER, "grey")["is_colour"] == "false"
    assert ctx.manifest.find(CHAPTER, "colour")["is_colour"] == "true"


def test_size_is_recorded_after_exif_orientation(ctx, raw):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees to display
    make_page(raw / "rotated.jpg", (400, 300), exif=exif)
    inspect_pages.run(ctx)
    row = ctx.manifest.find(CHAPTER, "rotated")
    assert (row["width"], row["height"], row["is_spread"]) == ("300", "400", "false")


def test_corrupt_file_is_reported_not_fatal(ctx, raw):
    make_page(raw / "good.png", (300, 400))
    (raw / "broken.jpg").write_bytes(b"this is not an image")
    good = (raw / "good.png").read_bytes()
    (raw / "truncated.png").write_bytes(good[: len(good) // 2])
    summary = inspect_pages.run(ctx)
    assert summary["pages"] == 3 and summary["unreadable"] == 2
    for name in ("broken", "truncated"):
        row = ctx.manifest.find(CHAPTER, name)
        assert row["status"] == "error"
        assert "inspect: unreadable" in row["notes"]
    assert ctx.manifest.find(CHAPTER, "good")["status"] == "inspected"


def test_duplicate_page_names_are_flagged(ctx, raw):
    make_page(raw / "01.jpg", (300, 400))
    make_page(raw / "01.png", (300, 400))
    summary = inspect_pages.run(ctx)
    assert summary["pages"] == 1
    assert ctx.manifest.find(CHAPTER, "01")["file"] == "01.jpg"
    dupes = [r for r in ctx.manifest.rows if r["status"] == "duplicate"]
    assert [r["file"] for r in dupes] == ["01.png"]
    assert "same page name as 01.jpg" in dupes[0]["notes"]


@pytest.mark.parametrize("height, max_factor, expected", [
    (1600, 4, 1), (2400, 4, 1),
    (1599, 4, 2), (800, 4, 2),
    (799, 4, 3), (534, 4, 3),
    (533, 4, 4), (400, 4, 4),
    (300, 4, 4),   # still too small at the largest factor
    (500, 2, 2),   # would need 4, capped at 2
    (700, 3, 3),
    (500, 1, 1),   # cap below the smallest factor: no upscaling
])
def test_upscale_factor_selection(height, max_factor, expected):
    assert choose_upscale_factor(height, 1600, max_factor) == expected
