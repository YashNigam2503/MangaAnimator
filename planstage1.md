# Plan: Stage 1 – remove speech bubbles and text from manga pages

## Context

This project turns manga pages into video. The pipeline is built in stages, each reading one folder and writing another.

- **Stage 0 (done):** inspect, convert to PNG, upscale small pages with Real-ESRGAN, normalize to one height. Output: `data/03_normalized/<chapter>/` (RGB PNG, all `target_height` px tall) and `data/manifest.csv`.
- **Stage 1 (this plan):** remove all speech bubbles and all lettering from the pages, and fill the holes so the artwork continues behind them.
- **Later (not now):** panel splitting, OCR for the script, video generation.

**Deadline: inpainted results on the 5 test pages today.** Work in the order given under "Order of work". Get one page through every step before polishing anything.

## Goal

For every page in `03_normalized`, produce a cleaned page with no bubbles and no text, at the same size, where everything outside the removed regions is untouched. Quality matters more than speed: a visible smear or leftover outline counts as a failure.

## What has to be removed

1. **Speech bubbles** of any shape, including their outline and tail.
2. **Rectangular caption and narration boxes.**
3. **Text drawn directly on the art with no bubble**: sound effects, shouted lines, vertical strips of repeated characters, titles, slogans, credits.
4. **Glow, outline and shadow effects** attached to that lettering.

Most pages are black and white story pages. Colour cover or title pages exist too, with large decorative lettering; they are handled by the same steps but tagged separately (see step 1).

## Out of scope

- Panel detection and cropping
- OCR or translation of the removed text
- Video generation
- Training or fine-tuning any model

## Constraints

- **Hardware:** development machine has no dedicated GPU. Heavy steps run on a rented GPU (Vast.ai, 24 GB card). Every step picks its device at runtime (`cuda` if available, otherwise `cpu`). Detection, mask building, LaMa and debug images must also work on CPU; only the diffusion backend may require a GPU.
- **Conventions from stage 0 still apply:** one folder in, one folder out per step; idempotent, with `--force` to overwrite; file names preserved; all tunables in `config.yaml`; earlier folders are never modified; steps are added to the existing `run.py` and work with `--chapter`, `--only`, `--limit`.
- **Verify before coding:** model names, file names, install methods, command-line flags and licences in this plan are from memory or a quick search. Check each against its model card or repository first, and report anything that differs.
- **Licences:** record the licence of every model used in `README.md`. Some may not allow commercial use; do not pick replacements on that basis yourself, just report.

## Folder layout (additions)

```
steps/
  tag_pages.py
  detect_bubbles.py
  detect_text.py
  build_masks.py
  inpaint.py
  make_debug.py
data/
  04_detections/<chapter>/<page>.json     # bubble polygons
  04_text_masks/<chapter>/<page>.png      # raw text mask
  04_masks/<chapter>/<page>.png           # final mask, white = remove
  04_regions/<chapter>/<page>.json        # one entry per region to fill
  05_clean_lama/<chapter>/<page>.png      # every region filled with LaMa
  05_clean/<chapter>/<page>.png           # final: easy = LaMa, hard = diffusion
  05_debug/<chapter>/
  regions_report.csv
```

## Configuration (add to `config.yaml`)

```yaml
detect_bubbles:
  model_repo: kitsumed/yolov8m_seg-speech-bubble
  conf: 0.25
  imgsz: 1024
detect_text:
  model: comic-text-detector
mask:
  bubble_dilate_px: 12     # must swallow the bubble outline and tail
  text_dilate_px: 8        # raise for lettering with glow or thick outlines
  min_area_px: 400
regions:
  context_factor: 2.0      # window = region box grown by this factor
  min_window: 512
  max_window: 1536
  hard_area_fraction: 0.04 # region covering more than this share of the page is hard
inpaint:
  easy_backend: lama       # lama | none
  hard_backend: diffusion  # diffusion | lama | none
  diffusion_model: flux-fill   # fall back to an SDXL inpainting model if it does not fit
  prompt: "clean manga artwork, background continues naturally, no text, no speech bubble"
  seed: 1234
```

## Steps

### 1. Tag pages (`tag_pages.py`)

Add two columns to `manifest.csv`: `is_colour` (already computed in stage 0; reuse it) and `page_type` with the value `story` or `cover`. Tag a page `cover` when it is colour and, after detection, text regions cover more than 15% of the page. Nothing is skipped because of the tag; it is used for reporting and so cover pages can be reviewed separately.

### 2. Detect bubbles (`detect_bubbles.py`)

`03_normalized` → `04_detections`

- Load the YOLOv8 segmentation model from the configured Hugging Face repo with `ultralytics`. It has one class, speech bubble.
- Save one JSON per page: polygon, bounding box, confidence and pixel area per detection.
- Check on the test pages whether rectangular caption boxes are detected. Report the result either way; boxes it misses are usually still caught by step 3, because their text is.

### 3. Detect text (`detect_text.py`)

`03_normalized` → `04_text_masks`

- Run a text segmentation model for comics on every page and save a binary mask of lettering pixels. Start with comic-text-detector.
- This step is required, not optional. It is what catches text outside bubbles.
- It must find text at any orientation, including vertical text and large stylised sound effects. Report which kinds it misses on the test pages.

### 4. Build masks and regions (`build_masks.py`)

`04_detections` + `04_text_masks` → `04_masks`, `04_regions`

- Rasterise bubble polygons, dilate by `bubble_dilate_px`.
- Dilate the text mask by `text_dilate_px`. Text that lies inside a bubble is already covered; keep only the union.
- Final mask = union of both, minus components smaller than `min_area_px`.
- Split the mask into connected components. For each, record in `04_regions`: id, bounding box, area, share of page area, source (`bubble`, `text` or `both`), and a square window (bounding box grown by `context_factor`, clamped to `min_window`..`max_window` and to the page edges). Merge windows that overlap heavily, so neighbouring regions are filled together.
- Label each region `easy` or `hard`: `hard` when its area share exceeds `hard_area_fraction`. If a character or face detector is cheap to add, also mark a region `hard` when it overlaps a character; otherwise leave this out and say so.

### 5. Inpaint (`inpaint.py`)

`03_normalized` + `04_masks` + `04_regions` → `05_clean_lama`, `05_clean`

Rules for both backends:

- Work per window, never on the whole page.
- Paste back **only the masked pixels**. Outside the mask the output must be pixel-identical to the input.
- Log seconds per region.

**LaMa backend.** Use IOPaint's batch command (`iopaint run`) with the LaMa model: write window crops and mask crops to a temporary folder, run the command, read the results back. Run `iopaint run --help` first and use the flags it lists. LaMa gives the same result on CPU and GPU; only the speed differs.

**Diffusion backend (GPU).** For `hard` regions. Start with FLUX.1 Fill through the `diffusers` library; if it does not fit the GPU or cannot be loaded, use an SDXL inpainting model and report the switch. The weights may be gated on Hugging Face: read the access token from an environment variable, never from a file in the repository. Use the configured prompt and a fixed seed per region (base seed plus region id), and store the seed in the report. Support `--region <page>:<id> --seed <n>` to redo a single region with a different seed.

**Outputs.**

- `05_clean_lama/`: every region filled with LaMa. This always exists and is the baseline.
- `05_clean/`: easy regions from LaMa, hard regions from the diffusion backend. If the diffusion backend is unavailable (no GPU), copy the LaMa result and mark those regions `diffusion_pending` in the report, so the run still completes on CPU.

### 6. Debug images (`make_debug.py`)

`05_debug` gets, per page:

- `<page>_overlay.png`: page with bubble regions in translucent red and text regions in translucent blue
- `<page>_compare.png`: original, LaMa result and final result side by side
- per chapter, a contact sheet of every hard region cropped at 100%: original, LaMa, diffusion

### Report (`regions_report.csv`)

One row per region: chapter, page, page type, region id, source, box, area share, easy/hard, backend used, seed, seconds, status.

## Order of work

1. Bubble detection → masks → LaMa → debug images, on one page, on CPU.
2. Same on all 5 test pages. Tune `bubble_dilate_px` until no outline fragments remain.
3. Add text detection and merge it into the masks. Rerun the 5 pages.
4. Add the diffusion backend. Test it on the GPU instance on the hard regions only.
5. Tests, README, report.

If time runs out, steps 1 to 3 with LaMa everywhere are the minimum result for today; say clearly which steps are unfinished.

## Running on Vast.ai

Add a `README.md` section covering:

- which folders to upload (`03_normalized`, plus the code) and which to download afterwards (`05_clean`, `05_clean_lama`, `05_debug`, `regions_report.csv`, all of `04_*`)
- one setup script, `setup_gpu.sh`, that installs dependencies and downloads model weights
- the single command that runs stage 1 on the instance
- a reminder that instance storage is lost when the instance is destroyed, and that the instance should be stopped when idle

## Testing

- Synthetic images only (draw ellipses, rectangles and text on a textured background); no manga pages in the repository.
- Cover: polygon rasterisation, both dilations, union and speck removal, region splitting, window computation at page edges, window merging, easy/hard labelling, paste-back leaving unmasked pixels byte-identical, idempotency, `--force`, the CPU fallback that marks regions `diffusion_pending`.
- Stub the detectors and both inpainting backends; the suite must pass with no model downloads and no GPU.

## Acceptance criteria

1. `python run.py --chapter <name>` runs stage 0 and stage 1 in order with no manual steps, on CPU (diffusion pending) and on a GPU instance (diffusion applied).
2. Every page has a final mask, a LaMa result and a final result, same size and format as its input.
3. Outside the mask, results are pixel-identical to the input.
4. On the 5 test pages, at 100% zoom: no bubble outline fragments, no leftover lettering or glow for regions that were detected.
5. Every bubble or piece of text that was not detected is listed in the report back, with the page and location.
6. `regions_report.csv` and all debug images exist for the 5 pages.
7. All tests pass.

## What to report back

- Per page: regions found by source, anything visible that was missed, seconds per step on CPU and on GPU
- Whether rectangular caption boxes were detected by the bubble model, the text model, or neither
- The dilation values you settled on and why
- Every hard region with an honest verdict for the LaMa and the diffusion result: acceptable, or needs another seed or a different approach
- Which diffusion model actually ran, on what GPU, and its memory use
- Licences of all models used
- Anything in this plan that was wrong or outdated
