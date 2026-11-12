# Manga pipeline, stage 0: page preparation

Takes a folder of raw manga pages (JPG, PNG or WebP, mixed sizes) and produces PNG pages that are all exactly `target_height` pixels tall, plus a manifest recording what was done to each page. Later stages (bubble removal, panel splitting, video) build on this output.

## Setup

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For AI upscaling on this machine, download `realesrgan-ncnn-vulkan` for your operating system from the [Real-ESRGAN releases page](https://github.com/xinntao/Real-ESRGAN/releases) (release v0.2.5.0 has the zips), unzip it into `tools/realesrgan/` and point `upscale.ncnn_binary` in `config.yaml` at the executable. Keep the `models` folder next to the executable. On Linux or macOS, run `chmod +x` on it first. It runs on any GPU with Vulkan support, including integrated graphics.

## Running locally

Put the pages of a chapter in `data/00_raw/<chapter>/`, then:

```
python run.py --chapter ch001
```

| Option | Effect |
| --- | --- |
| `--chapter <name>` | process one chapter |
| `--all` | process every chapter |
| `--only <step>` | run one step: `inspect`, `convert`, `trim`, `upscale`, `normalize`, or a stage 1 step (see below) |
| `--force` | redo outputs that already exist |
| `--limit N` | only N pages of each chapter, for quick tests |
| `--skip N` | skip the first N pages of each chapter |
| `--backend <name>` | override `upscale.backend` for this run: `ncnn`, `torch` or `none` |

The steps, each reading one folder and writing the next:

| Step | Reads | Writes | What it does |
| --- | --- | --- | --- |
| inspect | `00_raw` | `manifest.csv` | records format, size, spread, colour and whether the page needs upscaling |
| convert | `00_raw` | `01_png` | applies EXIF orientation, flattens transparency onto white, saves 8-bit RGB PNG |
| trim (off by default) | `01_png` | `01b_trimmed` | crops blank white borders |
| upscale | `01_png` or `01b_trimmed` | `02_upscaled` | Real-ESRGAN on pages shorter than `min_height`; other pages are copied |
| normalize | `02_upscaled` | `03_normalized` | resizes every page to `target_height` with Lanczos |

`data/03_normalized/<chapter>/` is the output of this stage. `data/manifest.csv` has one row per input page: original size, the upscale factor applied (`upscale_applied`, 1 means none), final size, and any flags in `notes`.

Things to know:

- `00_raw` is never modified.
- A step skips any page whose output already exists. This also applies after a config change: if you change `target_height` or the upscale model, rerun with `--force`.
- Two raw files with the same name and different extensions (`01.jpg`, `01.png`) would collide; the first is used and the other is marked `duplicate` in the manifest.
- Unreadable files are marked `error` in the manifest and the run continues.
- If the upscale backend cannot run (binary missing, no CUDA), the run stops and says what to fix. It never falls back to a plain resize.
- All tunable values are in `config.yaml`.

### Upscale backends

- `ncnn`: calls the `realesrgan-ncnn-vulkan` executable. The model must exist in the `models` folder next to it; the Windows build ships `realesr-animevideov3` (2x, 3x, 4x), `realesrgan-x4plus-anime` and `realesrgan-x4plus` (4x only; the result is resized down when a smaller factor is needed).
- `torch`: uses the `realesrgan` Python package on a CUDA GPU. This is the Colab/Kaggle path.
- `none`: upscales nothing. Flagged pages are copied unchanged, noted in the manifest and listed at the end of the run. To upscale them elsewhere, replace those files in `02_upscaled` with the upscaled versions (same file names), then run `python run.py --chapter <name> --only normalize --force`.

## Running only the upscale step on Colab or Kaggle

Locally, run everything before the upscale step:

```
python run.py --chapter ch001 --only inspect
python run.py --chapter ch001 --only convert
```

Zip `run.py`, `config.yaml`, `steps/` and `data/01_png/` (keeping the folder structure) and upload the zip to a notebook with a GPU runtime. Then in the notebook:

```
!unzip -q project.zip -d project
%cd project
!pip install -q realesrgan pyyaml tqdm
!python run.py --all --only upscale --backend torch
!zip -q -r upscaled.zip data/02_upscaled
```

Download `upscaled.zip`, unzip it so the pages land in `data/02_upscaled/<chapter>/`, and finish locally:

```
python run.py --chapter ch001 --only normalize
```

The upscale step decides which pages to upscale from the images themselves, so it needs no manifest on the remote machine. It writes `_upscale_results.csv` into each chapter folder of `02_upscaled`; keep that file with the pages, because the normalize step reads it to fill in the manifest.

Model weights are downloaded automatically on first use. If trimming is enabled, upload `data/01b_trimmed/` instead of `data/01_png/` and keep `trim_margins.enabled: true` in the uploaded config.

## Stage 1: removing speech bubbles and text

Stage 1 reads `data/03_normalized/` and writes cleaned pages with no bubbles and no lettering. `python run.py --chapter ch001` runs stage 0 and then stage 1.

### Extra setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
git clone https://github.com/dmMaze/comic-text-detector tools/comic-text-detector
```

Download `comictextdetector.pt` and `comictextdetector.pt.onnx` from the [manga-image-translator beta-0.2.1 release](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1) into `tools/models/`. The speech bubble model and LaMa download themselves on first use.

### Steps

| Step (`--only`) | Reads | Writes | What it does |
| --- | --- | --- | --- |
| `bubbles` | `03_normalized` | `04_detections` | speech bubble polygons, one JSON per page |
| `text` | `03_normalized` | `04_text_masks` | mask of lettering pixels, inside and outside bubbles |
| `tag` | `04_text_masks`, manifest | manifest | `page_type`: `cover` for a colour page with a lot of text, otherwise `story` |
| `masks` | `04_detections`, `04_text_masks`, `03_normalized` | `04_masks`, `04_regions` | final mask (white = remove), split into regions with a window each |
| `inpaint` | `03_normalized`, `04_masks`, `04_regions` | `05_clean_lama`, `05_clean`, `regions_report.csv` | fills every region |
| `debug` | all of the above | `05_debug` | images for checking the result by eye |

How the mask is built:

- Each bubble polygon is first grown over the white inside of the bubble, which follows it into the tail. The detector's polygon covers the bubble body and usually stops short of a pointed tail. If the white area spreads too far (an open outline), the polygon is kept as it is. The settings are under `mask.tail_growth`.
- Bubbles are then widened by `bubble_dilate_px` to swallow the outline, and text by `text_dilate_px`.
- Parts smaller than `min_area_px` are dropped and small holes inside the mask are closed.

How regions are filled:

- Each connected part of the mask is a region. A region covering more than `hard_area_fraction` of the page is `hard`, the rest are `easy`.
- Each region gets a square window of surrounding art. Windows that overlap heavily are merged while the result stays within `max_window`, so neighbours are filled together.
- Only masked pixels are pasted back. Outside the mask, the output is identical to the input.
- `05_clean_lama` has every region filled with LaMa. `05_clean` is the final result: easy regions from LaMa, hard regions from the diffusion backend.
- The diffusion backend is not built yet. Until it is, `05_clean` equals `05_clean_lama` and hard regions are listed as `diffusion_pending` in `regions_report.csv`. Set `inpaint.hard_backend: lama` to accept LaMa for hard regions.
- The `seconds` column in the report is the LaMa batch time shared out by window size, not a per-region measurement.

Debug images in `data/05_debug/<chapter>/`:

- `<page>_overlay.png`: bubble regions in red, text regions in blue
- `<page>_compare.png`: original, LaMa result and final result side by side
- `hard_regions_NN.png`: every hard region at 100%: original, LaMa, final

To process pages other than the first ones, combine `--skip` and `--limit`: `python run.py --chapter ch001 --skip 5 --limit 4` runs pages 6 to 9.

### Devices

Every step uses `cuda` when PyTorch finds a GPU and `cpu` otherwise. On CPU the text detector uses the ONNX build of its model (`detect_text.weights_cpu`), because the PyTorch weights take about two minutes per page there. The two builds give slightly different masks.

### Models and licences

| Model or tool | Used for | Licence |
| --- | --- | --- |
| [kitsumed/yolov8m_seg-speech-bubble](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble) | speech bubble detection | GPL-3.0 |
| [ultralytics](https://github.com/ultralytics/ultralytics) | runs the bubble model | AGPL-3.0 |
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (weights from manga-image-translator) | text detection | GPL-3.0 |
| [LaMa](https://github.com/advimman/lama) (big-lama) | inpainting | Apache-2.0 |
| [IOPaint](https://github.com/Sanster/IOPaint) | runs LaMa | Apache-2.0 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | stage 0 upscaling | BSD-3-Clause |

## Tests

```
python -m pytest
```

The tests generate synthetic images and replace the detectors and inpainting backends with stand-ins, so they need no model downloads, no GPU and no Real-ESRGAN.
