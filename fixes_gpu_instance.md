# Fixes from the first GPU run on Vast.ai

## Context

Stage 1 was run on a Vast.ai instance (Linux, Python 3.12, RTX PRO 4000 24 GB, container RAM limit about 42.5 GiB, PyTorch template). It got through every step of a 72-page chapter, but only after several fixes that were made by hand on the instance. None of them are in the repository yet. Make all of them here, in the repository, so a fresh instance runs `setup_gpu.sh` and then `run.py` without manual edits.

Do every item in Part A. Part B is for after Part A is pushed and tested.

## Part A: make the pipeline run cleanly on a fresh Linux GPU instance

### A1. `setup_gpu.sh`: model downloads

- The script calls `.venv/bin/huggingface-cli`, which does not exist in current `huggingface_hub` versions. Both calls failed (lines 35 and 41), and the script then printed a misleading "licence not accepted" warning.
- Replace both calls with Python calls to `huggingface_hub.snapshot_download`, which works across library versions.
- For FLUX.1 Fill, pass `ignore_patterns=["flux1-fill-dev.safetensors"]`. That file is a single-file duplicate of the diffusers weights and would double the download.
- Report the two failure cases separately: "download tool or library missing" versus "access denied (licence not accepted or HF_TOKEN not set)". Fall back to SDXL only on a real access error. Exit with an error if neither model downloads.

### A2. `setup_gpu.sh`: environment problems found on the instance

Add these to the main `.venv` setup:

- `pip install "setuptools<81"`. comic-text-detector imports `pkg_resources`, which newer setuptools no longer provides.
- `pip uninstall -y wandb`. A broken `wandb` install made `import diffusers` fail with `ValueError: wandb.__spec__ is None`, which stopped both Flux and the SDXL fallback from loading. Afterwards, check that `python -c "import diffusers"` succeeds and stop with a clear message if it doesn't.
- `--no-cache-dir` on every `pip install`.
- `export HF_HOME=/workspace/hf_cache` (or a folder on the large disk) before downloading models.
- At the start of the script, print free disk space and stop with a clear message if less than 60 GB is free. The first instance had a 16 GB disk and failed halfway through setup.

### A3. `setup_gpu.sh`: Real-ESRGAN on Linux

- `tools/realesrgan/` on the development machine holds the Windows `.exe`, which can't run on the instance.
- In `setup_gpu.sh`, download the Linux build if it isn't already present:
  `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip`
  Unzip it into `tools/realesrgan/`, `chmod +x tools/realesrgan/realesrgan-ncnn-vulkan`, and run it once with `-h` to confirm it starts.
- This build was tested on the instance and upscaled 72 pages at about 3 seconds per page.

### A4. `config.yaml` and `steps/upscale.py`: binary path and missing key

- `steps/upscale.py` reads `up["gpu_id"]` and crashed with `KeyError: 'gpu_id'` because the key isn't in `config.yaml`. Add `gpu_id: auto` under `upscale:`, and make the code use a default when the key is missing instead of crashing.
- `upscale.ncnn_binary` points to the `.exe`. Make the path platform-aware: either give the path without an extension and have the code add `.exe` on Windows, or support separate `ncnn_binary_windows` and `ncnn_binary_linux` keys. The same `config.yaml` must work on both machines without editing.

### A5. `tools/comic-text-detector/`: NumPy 2 compatibility

The vendored code uses aliases that NumPy 2 removed. These were patched by hand on the instance:

- `utils/io_utils.py`: `np.bool8` → `np.bool_`, and remove `np.float_` from `NP_FLOAT_TYPES` (it was an alias for `np.float64`, which is already in the tuple).
- `utils/imgproc_utils.py` line 186: `np.int0` → `np.intp`.

Then search the whole folder for any other removed alias (`np.bool8`, `np.float_`, `np.int0`, `np.object0`, `np.str0`, `np.unicode_`, `np.complex_`, `np.Inf`, `np.NaN`) and fix those too. Exclude `.ipynb_checkpoints/` folders, and delete them from the repository if they're tracked. Add a short note at the top of the folder's README saying it was patched for NumPy 2.

### A6. `steps/diffusion.py`: the run was killed for running out of RAM

- The chapter run stopped with `Killed` after six Flux fills. The GPU was fine. The container's RAM limit is about 42.5 GiB, and the Flux weights are about 34 GB, so loading them through system RAM can exceed it.
- Load the pipeline straight onto the GPU: pass `device_map="cuda"` to `FluxFillPipeline.from_pretrained`, or `low_cpu_mem_usage=True` if the installed diffusers version doesn't accept `device_map` for this pipeline. Remove the later `.to("cuda")` if `device_map` is used. Check the installed diffusers version before choosing.
- Replace the deprecated `torch_dtype=` with `dtype=` if the installed version supports it.
- Apply the same changes to the SDXL fallback.
- Load the model once per run and reuse it for every region (check this is already the case).
- Log peak system RAM during loading, next to the peak GPU memory already logged.

### A7. Atomic writes, so a killed run can't leave half-written files

- After the kill, a page that was being written could be counted as "already done" on the next run.
- In every step, write outputs to a temporary file in the same folder and rename it to the final name only after the write succeeds. The "already done" check must only see completed files.

### A8. Long runs

- Add a section to `README.md` on running a full chapter in the background with `nohup ... > run_ch001.log 2>&1 &` and following it with `tail -f`, so a dropped SSH connection doesn't stop the run.

### A9. Repository hygiene

- Add `data/`, `tools/realesrgan/`, `*.patch` and `.ipynb_checkpoints/` to `.gitignore`.
- The repository is 255 MB, which suggests pages or binaries were committed. Remove `data/` and `tools/realesrgan/` from tracking with `git rm -r --cached`. Report what else is large (`git count-objects -vH`, and the largest files in history). Don't rewrite history without asking me first.

### A10. Tests and report back

- Add tests for: config with and without `gpu_id`, platform-aware binary path on Windows and Linux, atomic writes (a simulated crash leaves no file under the final name), and the NumPy fixes (import comic-text-detector's `io_utils` under NumPy 2). All tests must pass without a GPU or model downloads.
- Report: every file changed, anything in this list that turned out different from described, and the exact commands I should run on the instance after pulling.

## Part B: quality fixes (after Part A is pushed and tested)

These come from reviewing the debug images for pages 06 to 09. Do them in this order, testing on those pages:

1. **Stop erasing text outside bubbles.** Only bubble regions and text inside them go into the mask. Text outside bubbles is kept and drawn in green on the overlay. Exception: small text in the page margin, outside every panel, is removed.
2. **Filter false bubbles.** Keep a bubble detection only if it contains detected text, or if its interior is mostly white and its confidence is high. On page 06 the bus wheel was treated as a bubble, and on page 08 a window area was.
3. **Recover missed small bubbles.** Where detected text sits inside a closed white area that no bubble covers, flood-fill that area and treat it as a bubble (example: the small dots bubble at the bottom left of page 06).
4. **Per-region prompts for Flux.** Flux drew whole new characters into large holes (a girl's face on page 06, a man on page 08, extra people on page 09). Replace the single prompt with one per region: send the panel crop with the region outlined to a vision-language model and ask for one sentence describing only what should continue behind the bubble, ending with "no additional people". Never mention text, letters or speech bubbles in the prompt. Store the prompt in the report.
5. **Guard against invented characters.** After each Flux fill, run a face or character detector. If a face's centre is inside the mask, reject the fill and retry with the next seed, up to 3 times. If all fail, keep the LaMa fill and mark the region `needs_review`.
6. **More context for Flux.** Use the full panel as the window when it fits within the size limit.
7. **Border repair.** After inpainting, where a straight panel border enters and exits a filled region on the same line, redraw it at the original thickness. Skip anything that isn't clearly a straight border.

Jersey names and other text drawn on clothing are not a priority yet.
