import csv
import hashlib

import pytest
import yaml
from PIL import Image

import run
from conftest import CHAPTER, StubBackend, make_page
from steps import upscale
from steps.common import MANIFEST_NAME, NORMALIZED_DIR, PNG_DIR, RAW_DIR, UPSCALED_DIR


@pytest.fixture
def project(tmp_path, cfg, raw, ctx):
    """A data folder with mixed raw pages and a config file, driven through run.main."""
    make_page(raw / "01.jpg", (600, 800))
    make_page(raw / "02.png", (1400, 2000))
    make_page(raw / "03.webp", (2400, 1700))
    make_page(raw / "04.png", (500, 700), colour=(0, 0, 0, 0), mode="RGBA")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def go(*extra):
        run.main(["--config", str(config), "--data-dir", str(ctx.data_dir), *extra])

    go.data = ctx.data_dir
    return go


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def manifest_rows(data):
    with open(data / MANIFEST_NAME, newline="", encoding="utf-8") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def test_full_run_meets_the_acceptance_criteria(project, capsys):
    raw_before = snapshot(project.data / RAW_DIR)
    project("--chapter", CHAPTER)

    assert snapshot(project.data / RAW_DIR) == raw_before          # raw is untouched
    outputs = sorted(p.name for p in (project.data / NORMALIZED_DIR / CHAPTER).iterdir())
    assert outputs == ["01.png", "02.png", "03.png", "04.png"]     # names preserved
    originals = {"01.png": (600, 800), "02.png": (1400, 2000), "03.png": (2400, 1700), "04.png": (500, 700)}
    for name, (w, h) in originals.items():
        with Image.open(project.data / NORMALIZED_DIR / CHAPTER / name) as im:
            assert (im.format, im.mode, im.height) == ("PNG", "RGB", 2000)
            assert abs(im.width - w * 2000 / h) <= 1

    rows = manifest_rows(project.data)
    assert sorted(rows) == ["01.jpg", "02.png", "03.webp", "04.png"]   # one row per input page
    assert all(r["status"] == "normalized" for r in rows.values())
    assert (rows["01.jpg"]["width"], rows["01.jpg"]["height"], rows["01.jpg"]["final_height"]) == ("600", "800", "2000")
    assert rows["03.webp"]["is_spread"] == "true"
    # backend none: the run completes, flagged pages are listed and noted
    assert "upscale: skipped (backend none), needs 2x" in rows["01.jpg"]["notes"]
    assert "normalize: enlarged 2.50x" in rows["01.jpg"]["notes"]
    assert rows["02.png"]["notes"] == ""
    printed = capsys.readouterr().out
    assert "2 pages need upscaling" in printed and "01.png" in printed and "04.png" in printed


def test_second_run_does_nothing(project):
    project("--chapter", CHAPTER)
    before = snapshot(project.data)
    project("--chapter", CHAPTER)
    assert snapshot(project.data) == before


def test_force_overwrites_outputs_but_not_raw(project):
    project("--chapter", CHAPTER)
    before = snapshot(project.data)
    project("--chapter", CHAPTER, "--force")
    after = snapshot(project.data)
    for name, (digest, mtime) in before.items():
        if name.startswith(RAW_DIR):
            assert after[name] == (digest, mtime)
        elif name.endswith(".png"):
            assert after[name][0] == digest and after[name][1] > mtime, name


def test_limit_and_only(project):
    project("--chapter", CHAPTER, "--limit", "2", "--only", "inspect")
    assert sorted(manifest_rows(project.data)) == ["01.jpg", "02.png"]
    assert not (project.data / PNG_DIR).exists()
    project("--all", "--limit", "2")
    assert len(list((project.data / NORMALIZED_DIR / CHAPTER).iterdir())) == 2


def test_upscaled_pages_are_recorded_through_the_runner(project, monkeypatch):
    monkeypatch.setattr(upscale, "make_backend", lambda ctx: StubBackend())
    project("--chapter", CHAPTER)
    rows = manifest_rows(project.data)
    assert rows["01.jpg"]["upscale_applied"] == "2" and rows["01.jpg"]["notes"] == ""
    assert rows["04.png"]["upscale_applied"] == "3"
    assert rows["02.png"]["upscale_applied"] == "1"
    with Image.open(project.data / UPSCALED_DIR / CHAPTER / "01.png") as im:
        assert im.size == (1200, 1600)


def test_missing_chapter_and_disabled_trim_give_clear_errors(project):
    with pytest.raises(SystemExit, match="Chapter folder not found"):
        project("--chapter", "nope")
    with pytest.raises(SystemExit, match="trim step is off"):
        project("--chapter", CHAPTER, "--only", "trim")
