from PIL import Image

from conftest import CHAPTER, make_page
from steps import convert_png
from steps.common import PNG_DIR


def out(ctx, name):
    return ctx.data_dir / PNG_DIR / CHAPTER / name


def test_exif_rotation_is_applied(ctx, raw):
    # Stored 40x20 with a red left half; orientation 6 means "rotate 90 degrees clockwise to display".
    im = Image.new("RGB", (40, 20), (255, 255, 255))
    im.paste((255, 0, 0), (0, 0, 20, 20))
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(raw / "rotated.png", exif=exif)
    convert_png.run(ctx)
    with Image.open(out(ctx, "rotated.png")) as result:
        assert result.size == (20, 40)
        assert result.getpixel((10, 5)) == (255, 0, 0)       # red half is now on top
        assert result.getpixel((10, 35)) == (255, 255, 255)
        assert result.getexif().get(0x0112) in (None, 1)


def test_transparency_is_flattened_onto_white(ctx, raw):
    im = Image.new("RGBA", (20, 20), (0, 0, 0, 0))            # fully transparent
    im.paste((0, 0, 255, 255), (0, 0, 10, 20))                # opaque blue left half
    im.putpixel((15, 15), (0, 0, 0, 128))                     # half-transparent black
    im.save(raw / "alpha.png")
    convert_png.run(ctx)
    with Image.open(out(ctx, "alpha.png")) as result:
        assert result.mode == "RGB"
        assert result.getpixel((15, 5)) == (255, 255, 255)
        assert result.getpixel((5, 5)) == (0, 0, 255)
        assert result.getpixel((15, 15)) == (127, 127, 127)


def test_palette_transparency_is_flattened(ctx, raw):
    im = Image.new("P", (10, 10), 0)
    im.putpalette([0, 0, 0, 255, 0, 0])
    im.putpixel((5, 5), 1)
    im.save(raw / "palette.png", transparency=0)
    convert_png.run(ctx)
    with Image.open(out(ctx, "palette.png")) as result:
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)
        assert result.getpixel((5, 5)) == (255, 0, 0)


def test_all_formats_become_8bit_rgb_png_without_resizing(ctx, raw):
    make_page(raw / "a.jpg", (60, 80))
    make_page(raw / "b.webp", (70, 90))
    make_page(raw / "grey.png", (50, 70), colour=180, mode="L")
    Image.new("I;16", (30, 40), 65535).save(raw / "deep.png")
    convert_png.run(ctx)
    sizes = {"a": (60, 80), "b": (70, 90), "grey": (50, 70), "deep": (30, 40)}
    for name, size in sizes.items():
        with Image.open(out(ctx, f"{name}.png")) as result:
            assert (result.format, result.mode, result.size) == ("PNG", "RGB", size)
        assert ctx.manifest.find(CHAPTER, name)["status"] == "converted"
    with Image.open(out(ctx, "deep.png")) as result:
        assert result.getpixel((0, 0)) == (255, 255, 255)     # 16-bit white stays white
    with Image.open(out(ctx, "grey.png")) as result:
        assert result.getpixel((0, 0)) == (180, 180, 180)


def test_corrupt_file_is_skipped_and_reported(ctx, raw):
    make_page(raw / "good.png", (30, 40))
    (raw / "broken.jpg").write_bytes(b"nope")
    convert_png.run(ctx)
    assert out(ctx, "good.png").exists()
    assert not out(ctx, "broken.png").exists()
    row = ctx.manifest.find(CHAPTER, "broken")
    assert row["status"] == "error" and "convert: failed" in row["notes"]
