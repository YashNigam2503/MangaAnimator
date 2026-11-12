# Plan: Stage 0 – page preparation for the manga-to-video pipeline

## Context

This is the first stage of a pipeline that turns manga pages into video. Later stages (not part of this task) will remove speech bubbles by inpainting, split pages into panels, and send panels to a video model. Those stages need every page to arrive in the same condition: lossless format, enough resolution, and one consistent scale.

Your job is to build that preparation stage: inspect the pages, convert them to PNG, upscale the ones that are too small, and bring everything to one working height.

## Goal

Given a folder of raw manga pages (JPG, PNG or WebP, mixed sizes), produce a folder of PNG pages that are all exactly `target_height` pixels tall, plus a manifest recording what was done to each page.

## Out of scope

Do not build any of these now:

- Speech bubble or text detection, inpainting
- Panel detection or panel cropping (this happens after bubble removal, in a later stage)
- OCR, video generation

## Constraints

- The development machine has no dedicated GPU. Everything except AI upscaling must run on CPU. Upscaling must work through a pluggable backend so it can run locally or on a rented/online GPU (Colab, Kaggle) with the same script.
- Never modify or delete anything in the raw input folder.
- Every step reads one folder and writes another, so a single step can be rerun or run on a different machine.
- Steps are idempotent: an output that already exists is skipped unless `--force` is passed.
- File names are preserved across steps (only the extension changes to `.png`).
- Python 3.10+. Dependencies: Pillow, PyYAML, tqdm, pytest. Add others only if a step needs them, and say why.

## Folder layout

```
project/
  config.yaml
  requirements.txt
  README.md
  run.py                  # runs all steps in order
  steps/
    inspect_pages.py
    convert_png.py
    upscale.py
    normalize.py
    trim_margins.py       # optional step, off by default
  tests/
  data/
    00_raw/<chapter>/     # input, read-only
    01_png/<chapter>/
    02_upscaled/<chapter>/
    03_normalized/<chapter>/   # final output of this stage
    manifest.csv
```

## Configuration (`config.yaml`)

All tunable values live here; no magic numbers in the scripts.

```yaml
min_height: 1600        # pages shorter than this get AI upscaling
target_height: 2000     # final height of every page
max_upscale_factor: 4
upscale:
  backend: ncnn         # ncnn | torch | none
  ncnn_binary: ""       # path to realesrgan-ncnn-vulkan executable
  model: realesr-animevideov3
trim_margins:
  enabled: false
  threshold: 245        # pixel value treated as "white"
  padding: 16           # pixels of margin to keep
```

## Steps

### 1. Inspect (`inspect_pages.py`)

Read every image in `00_raw` and write `data/manifest.csv` with one row per page:

- chapter, file name, format, width, height, colour mode
- `is_spread`: true when width > height (double-page spread)
- `is_colour`: true when the page has real colour, not just a grayscale image stored as RGB
- `needs_upscale`: true when height < `min_height`
- `upscale_factor`: the smallest of 2, 3, 4 that brings height to at least `min_height`; capped at `max_upscale_factor`
- `status` and `notes` columns that later steps update

Print a summary: page count, smallest and largest page, how many need upscaling. Report unreadable or corrupt files in the manifest and continue; do not crash.

### 2. Convert to PNG (`convert_png.py`)

`00_raw` → `01_png`

- Apply EXIF orientation.
- Flatten any transparency onto white.
- Save as 8-bit RGB PNG. (RGB even for grayscale pages, because the later models expect three channels.)
- Do not resize in this step.

### 3. Upscale (`upscale.py`)

`01_png` → `02_upscaled`

- Pages with `needs_upscale = false` are copied unchanged.
- Pages with `needs_upscale = true` are upscaled by their `upscale_factor` with Real-ESRGAN, using an anime/manga-trained model.
- Backends:
  - `ncnn`: call the `realesrgan-ncnn-vulkan` executable as a subprocess. It runs on integrated graphics through Vulkan, so it works without a dedicated GPU.
  - `torch`: use the Python Real-ESRGAN package on `cuda` when available. This is the path for Colab or Kaggle.
  - `none`: copy flagged pages unchanged, mark them `upscale_skipped` in the manifest, and print the list, so they can be upscaled elsewhere and dropped into `02_upscaled` by hand.
- Before writing the backend code, read the Real-ESRGAN README to confirm the current model names, supported scale factors and command-line flags. Do not rely on the values in this plan being current.
- If the configured backend is unavailable (binary missing, no CUDA), stop with a clear message that names the fix. Do not silently fall back to a plain resize.
- A page that is still under `min_height` after the maximum factor is kept and flagged in `notes`.

### 4. Normalize (`normalize.py`)

`02_upscaled` → `03_normalized`

- Resize every page to exactly `target_height` tall, keeping aspect ratio, with Lanczos resampling. Spreads use the same height and are simply wider.
- Record the final width, height and the scale factor applied in the manifest.
- If this step has to enlarge a page by more than 1.25x, flag it in `notes`: that page should have gone through AI upscaling.

### 5. Trim margins (`trim_margins.py`, optional, off by default)

When enabled, runs between steps 2 and 3. Crops away blank white borders around the page content, keeping `padding` pixels. Skip any page where the detected content box covers less than 50% of the page, and flag it, since that usually means a mostly white page rather than a border.

### Runner (`run.py`)

- `python run.py --chapter <name>` runs all enabled steps in order for one chapter; `--all` runs every chapter.
- `--only <step>` runs a single step, `--force` overwrites existing outputs, `--limit N` processes only the first N pages (for quick tests).

## Testing

- Unit tests use small synthetic images generated in the test itself. Do not commit manga pages to the repository.
- Cover at least: manifest values for a small page, a large page and a spread; EXIF rotation; transparency flattening; upscale factor selection (including the cap); normalize producing the exact target height; idempotency (a second run does nothing); `--force` overwriting.
- Upscaling tests use the `none` backend plus a stubbed backend; do not require Real-ESRGAN to be installed for the test suite to pass.
- Finish with one real run on 3 to 5 pages from `00_raw` using `--limit`, and report the manifest rows.

## Acceptance criteria

1. `python run.py --chapter <name>` completes on a folder of mixed JPG/PNG/WebP pages with no manual steps.
2. Every file in `03_normalized` is an RGB PNG exactly `target_height` pixels tall, with its original aspect ratio (within one pixel of rounding).
3. `00_raw` is byte-for-byte unchanged after a run.
4. `manifest.csv` has one row per input page, showing original size, whether it was upscaled and by how much, final size, and any flags.
5. Running the same command twice does no work the second time.
6. With backend `none`, the run still completes and lists the pages that need upscaling.
7. All tests pass.
8. `README.md` explains setup, how to run locally, and how to run only the upscale step on Colab or Kaggle (upload `01_png`, run the step with the `torch` backend, download `02_upscaled`).

## What to report back

- The summary printed by the inspect step for the real test pages
- Which upscale backend you tested and how long it took per page
- Anything in this plan that turned out to be wrong or outdated, especially Real-ESRGAN model names and flags
- Before/after crops at 100% zoom of one upscaled page, so line and screentone quality can be judged by eye
