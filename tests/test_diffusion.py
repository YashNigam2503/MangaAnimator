import csv

import numpy as np
import pytest
from PIL import Image

import run
from conftest import CHAPTER, DIFFUSION_COLOUR, LAMA_COLOUR, StubDiffusion
from steps import build_masks, inpaint
from steps.common import CLEAN_DIR, CLEAN_LAMA_DIR, MASKS_DIR, NORMALIZED_DIR, REGIONS_REPORT_NAME
from steps.diffusion import working_size
from test_stage1 import load, pages, regions_of, run_stage1  # noqa: F401  (pages is a fixture)


def report_rows(ctx):
    with open(ctx.data_dir / REGIONS_REPORT_NAME, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("size, max_side, expected", [
    ((512, 512), 1024, (512, 512)),          # small windows are left alone
    ((1393, 1536), 1024, (928, 1024)),       # shrunk to the limit, sides divisible by 16
    ((1000, 700), 1024, (992, 704)),         # rounded to a multiple of 16
    ((2000, 500), 1024, (1024, 256)),
])
def test_working_size(size, max_side, expected):
    assert working_size(size, max_side) == expected
    assert all(v % 16 == 0 for v in working_size(size, max_side))


def test_redo_one_region_with_another_seed(ctx, pages):
    run_stage1(ctx)                                             # no GPU: the hard region is pending
    hard = next(r for r in regions_of(ctx) if r["label"] == "hard")
    before = load(ctx, CLEAN_DIR)
    labels, _ = build_masks.label_regions(load(ctx, MASKS_DIR))
    own = labels == hard["id"]

    diffusion = StubDiffusion()
    ctx.region, ctx.seed = f"p1:{hard['id']}", 99
    inpaint.run(ctx, [CHAPTER], diffusion=diffusion)

    after = load(ctx, CLEAN_DIR)
    assert diffusion.seeds == [99]
    assert (after[own] == DIFFUSION_COLOUR).all()               # that region was refilled
    assert (after[~own] == before[~own]).all()                  # nothing else on the page changed
    assert (load(ctx, CLEAN_LAMA_DIR)[own] == LAMA_COLOUR).all()   # the LaMa baseline is untouched
    assert (load(ctx, CLEAN_DIR, "p2.png") == load(ctx, CLEAN_LAMA_DIR, "p2.png")).all()
    row = next(r for r in report_rows(ctx) if r["page"] == "p1.png" and r["region_id"] == str(hard["id"]))
    assert (row["backend"], row["seed"], row["status"]) == ("stub-diffusion", "99", "done")
    other = next(r for r in report_rows(ctx) if r["page"] == "p2.png" and r["label"] == "hard")
    assert other["status"] == "diffusion_pending"


def test_redo_region_errors(ctx, pages):
    run_stage1(ctx)
    ctx.region = "p1:999"
    with pytest.raises(SystemExit, match="has no region 999"):
        inpaint.run(ctx, [CHAPTER], diffusion=StubDiffusion())
    ctx.region = "p1:1"
    with pytest.raises(SystemExit, match="needs a GPU"):
        inpaint.run(ctx, [CHAPTER])
    ctx.region = "p1"
    with pytest.raises(SystemExit, match="needs --chapter and the form"):
        inpaint.run(ctx, [CHAPTER], diffusion=StubDiffusion())


def test_fills_on_a_black_and_white_page_are_kept_grey(ctx):
    folder = ctx.data_dir / NORMALIZED_DIR / CHAPTER
    folder.mkdir(parents=True)
    rng = np.random.default_rng(3)
    grey = rng.integers(0, 256, (1400, 1000), dtype=np.uint8)
    Image.fromarray(np.stack([grey] * 3, axis=-1)).save(folder / "bw.png", compress_level=1)
    run_stage1(ctx, diffusion=StubDiffusion())
    result, mask = load(ctx, CLEAN_DIR, "bw.png"), load(ctx, MASKS_DIR, "bw.png") > 0
    filled = result[mask]
    assert (filled[:, 0] == filled[:, 1]).all() and (filled[:, 1] == filled[:, 2]).all()
    assert set(map(tuple, np.unique(filled, axis=0))) == {(76, 76, 76), (150, 150, 150)}   # red and green as grey


def test_from_runs_a_step_and_everything_after_it(ctx, pages, tmp_path, cfg):
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = ["--config", str(config), "--data-dir", str(ctx.data_dir), "--chapter", CHAPTER]
    run.main(args + ["--from", "bubbles"])
    assert (ctx.data_dir / CLEAN_DIR / CHAPTER / "p1.png").exists()
    before = (ctx.data_dir / CLEAN_DIR / CHAPTER / "p1.png").stat().st_mtime_ns
    mask_before = (ctx.data_dir / MASKS_DIR / CHAPTER / "p1.png").stat().st_mtime_ns
    run.main(args + ["--from", "inpaint", "--force"])
    assert (ctx.data_dir / CLEAN_DIR / CHAPTER / "p1.png").stat().st_mtime_ns > before
    assert (ctx.data_dir / MASKS_DIR / CHAPTER / "p1.png").stat().st_mtime_ns == mask_before   # earlier steps untouched
    with pytest.raises(SystemExit, match="trim step is off"):
        run.main(args + ["--from", "trim"])
