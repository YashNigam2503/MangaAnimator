import pytest
from PIL import Image

from conftest import CHAPTER, StubBackend, make_page
from steps import normalize, trim_margins, upscale
from steps.common import NORMALIZED_DIR, PNG_DIR, TRIMMED_DIR, UPSCALED_DIR


@pytest.fixture
def png(ctx):
    folder = ctx.data_dir / PNG_DIR / CHAPTER
    folder.mkdir(parents=True)
    return folder


def upscaled(ctx, name):
    return ctx.data_dir / UPSCALED_DIR / CHAPTER / name


def test_none_backend_copies_and_lists_flagged_pages(ctx, png, capsys):
    make_page(png / "small.png", (300, 400))
    make_page(png / "large.png", (1200, 1700))
    upscale.run(ctx)
    assert upscaled(ctx, "small.png").read_bytes() == (png / "small.png").read_bytes()
    assert upscaled(ctx, "large.png").read_bytes() == (png / "large.png").read_bytes()
    small = ctx.manifest.find(CHAPTER, "small")
    assert small["status"] == "upscale_skipped" and small["upscale_applied"] == "1"
    assert "upscale: skipped (backend none), needs 4x" in small["notes"]
    assert ctx.manifest.find(CHAPTER, "large")["status"] == "not_upscaled"
    printed = capsys.readouterr().out
    assert "small.png" in printed.split("copied unchanged. Upscale them elsewhere")[1]


def test_stub_backend_upscales_only_flagged_pages(ctx, png):
    make_page(png / "small.png", (600, 800))      # needs 2x
    make_page(png / "tiny.png", (100, 300))       # 4x is still not enough
    make_page(png / "large.png", (1200, 1700))
    stub = StubBackend()
    upscale.run(ctx, backend=stub)
    assert sorted(stub.calls) == [("small.png", 2), ("tiny.png", 4)]
    with Image.open(upscaled(ctx, "small.png")) as im:
        assert im.size == (1200, 1600)
    assert upscaled(ctx, "large.png").read_bytes() == (png / "large.png").read_bytes()
    small = ctx.manifest.find(CHAPTER, "small")
    assert (small["status"], small["upscale_applied"], small["notes"]) == ("upscaled", "2", "")
    tiny = ctx.manifest.find(CHAPTER, "tiny")
    assert tiny["upscale_applied"] == "4"
    assert "upscale: still under min_height after 4x (1200px)" in tiny["notes"]
    assert ctx.manifest.find(CHAPTER, "large")["upscale_applied"] == "1"


def test_missing_ncnn_binary_stops_with_a_fix_and_no_fallback(ctx, png):
    make_page(png / "small.png", (300, 400))
    ctx.cfg["upscale"]["backend"] = "ncnn"
    ctx.cfg["upscale"]["ncnn_binary"] = "tools/nowhere/realesrgan-ncnn-vulkan.exe"
    with pytest.raises(SystemExit) as stop:
        upscale.run(ctx)
    message = str(stop.value)
    assert "unavailable" in message and "github.com/xinntao/Real-ESRGAN/releases" in message
    assert not upscaled(ctx, "small.png").exists()


def test_missing_model_lists_the_models_that_exist(ctx, png, tmp_path):
    make_page(png / "small.png", (300, 400))
    models = tmp_path / "bin" / "models"
    models.mkdir(parents=True)
    (tmp_path / "bin" / "realesrgan-ncnn-vulkan.exe").write_bytes(b"")
    (models / "some-model-x2.param").write_text("")
    ctx.cfg["upscale"].update(backend="ncnn", ncnn_binary=str(tmp_path / "bin" / "realesrgan-ncnn-vulkan.exe"))
    with pytest.raises(SystemExit) as stop:
        upscale.run(ctx)
    assert "some-model-x2" in str(stop.value)


def test_unavailable_backend_is_not_needed_when_nothing_is_flagged(ctx, png):
    make_page(png / "large.png", (1200, 1700))
    ctx.cfg["upscale"].update(backend="ncnn", ncnn_binary="")
    upscale.run(ctx)
    assert upscaled(ctx, "large.png").exists()


def test_results_file_travels_with_the_folder(ctx, png):
    """An upscale run elsewhere (no manifest) is merged into the manifest by normalize."""
    make_page(png / "small.png", (600, 800))
    upscale.run(ctx, backend=StubBackend())
    ctx.manifest.rows.clear()
    ctx.manifest._by_stem.clear()
    normalize.run(ctx)
    row = ctx.manifest.find(CHAPTER, "small")
    assert row["upscale_applied"] == "2" and row["status"] == "normalized"


def test_normalize_gives_exact_target_height_and_keeps_aspect(ctx):
    src = ctx.data_dir / UPSCALED_DIR / CHAPTER
    src.mkdir(parents=True)
    sizes = {"page": (1504, 2160), "spread": (3400, 2400), "odd": (1333, 1777), "exact": (1400, 2000)}
    for name, size in sizes.items():
        make_page(src / f"{name}.png", size)
    make_page(src / "small.png", (700, 1000))     # needs 2x plain enlargement
    make_page(src / "grey.png", (1400, 2000), colour=180, mode="L")
    normalize.run(ctx)
    target = ctx.cfg["target_height"]
    for name, (w, h) in sizes.items():
        with Image.open(ctx.data_dir / NORMALIZED_DIR / CHAPTER / f"{name}.png") as im:
            assert im.height == target and im.mode == "RGB" and im.format == "PNG"
            assert abs(im.width - w * target / h) <= 1
        row = ctx.manifest.find(CHAPTER, name)
        assert (row["final_width"], row["final_height"]) == (str(im.width), str(target))
        assert float(row["normalize_scale"]) == pytest.approx(target / h, abs=1e-4)
        assert row["status"] == "normalized" and row["notes"] == ""
    assert "normalize: enlarged 2.00x by plain resize" in ctx.manifest.find(CHAPTER, "small")["notes"]
    with Image.open(ctx.data_dir / NORMALIZED_DIR / CHAPTER / "grey.png") as im:
        assert im.mode == "RGB"


def test_enlarging_by_exactly_the_limit_is_not_flagged(ctx):
    src = ctx.data_dir / UPSCALED_DIR / CHAPTER
    src.mkdir(parents=True)
    make_page(src / "edge.png", (1200, 1600))     # 2000 / 1600 = 1.25
    normalize.run(ctx)
    assert ctx.manifest.find(CHAPTER, "edge")["notes"] == ""


def test_trim_crops_white_border_keeping_padding(ctx, png):
    im = Image.new("RGB", (400, 600), (255, 255, 255))
    im.paste((0, 0, 0), (40, 40, 360, 560))
    im.save(png / "bordered.png")
    Image.new("RGB", (400, 600), (255, 255, 255)).save(png / "blank.png")
    mostly_white = Image.new("RGB", (400, 600), (255, 255, 255))
    mostly_white.paste((0, 0, 0), (180, 280, 220, 320))
    mostly_white.save(png / "mostly_white.png")

    trim_margins.run(ctx)
    pad = ctx.cfg["trim_margins"]["padding"]
    out = ctx.data_dir / TRIMMED_DIR / CHAPTER
    with Image.open(out / "bordered.png") as result:
        assert result.size == (320 + 2 * pad, 520 + 2 * pad)
        assert result.getpixel((pad, pad)) == (0, 0, 0)
        assert result.getpixel((pad - 1, pad - 1)) == (255, 255, 255)
    for name in ("blank", "mostly_white"):
        assert (out / f"{name}.png").read_bytes() == (png / f"{name}.png").read_bytes()
        assert "trim: not trimmed" in ctx.manifest.find(CHAPTER, name)["notes"]


def test_upscale_reads_trimmed_pages_when_trim_is_enabled(ctx, png):
    im = Image.new("RGB", (1300, 1700), (255, 255, 255))
    im.paste((0, 0, 0), (100, 100, 1200, 1500))   # content is 1400 tall: under min_height once trimmed
    im.save(png / "page.png")
    ctx.cfg["trim_margins"]["enabled"] = True
    trim_margins.run(ctx)
    stub = StubBackend()
    upscale.run(ctx, backend=stub)
    assert stub.calls == [("page.png", 2)]
