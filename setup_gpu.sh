#!/usr/bin/env bash
# Sets up a rented GPU machine (Linux, e.g. Vast.ai) for stages 0 and 1.
# Run it from the repository folder:
#   export HF_TOKEN=hf_...      # needed for FLUX.1 Fill; without it SDXL inpainting is used
#   bash setup_gpu.sh
# Running it again is safe: what is already there is kept.
set -euo pipefail
cd "$(dirname "$0")"

MIN_FREE_GB="${MIN_FREE_GB:-60}"     # models about 41 GB, the two Python environments about 12 GB
PIP="pip install -q --no-cache-dir"  # no pip cache: it only fills the disk

# Model downloads go to the large disk, not to the small home folder.
if [ -z "${HF_HOME:-}" ]; then
    if [ -d /workspace ]; then HF_HOME=/workspace/hf_cache; else HF_HOME="$(pwd)/.hf_cache"; fi
fi
export HF_HOME
mkdir -p "$HF_HOME"

echo "== disk space =="
free_gb() { df -Pk "$1" | awk 'NR==2 {print int($4 / 1048576)}'; }
used_gb() { { du -sk "$@" 2>/dev/null || true; } | awk '{s += $1} END {print int(s / 1048576)}'; }
# What an earlier run of this script already put on the disk counts as available.
have_gb=$(( $(free_gb "$HF_HOME") + $(used_gb "$HF_HOME" .venv .venv-iopaint) ))
echo "Model folder (HF_HOME): $HF_HOME"
echo "Free: $(free_gb "$HF_HOME") GB there, $(free_gb .) GB in $(pwd)"
if [ "$have_gb" -lt "$MIN_FREE_GB" ]; then
    echo "ERROR: not enough disk space. Setup needs about $MIN_FREE_GB GB and only $have_gb GB is available" >&2
    echo "       (free space plus what setup has already downloaded). Rent an instance with a larger disk" >&2
    echo "       (80 GB or more), or point HF_HOME at a larger disk and run this script again." >&2
    exit 1
fi

echo "== system libraries =="
if command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq git curl python3-venv libglib2.0-0 libgl1 libgomp1 libvulkan1 >/dev/null
fi

echo "== main environment (.venv): detectors and diffusion =="
python3 -m venv .venv
.venv/bin/$PIP --upgrade pip
.venv/bin/$PIP -r requirements-gpu.txt
# comic-text-detector imports pkg_resources, which setuptools 81 and later no longer provide.
.venv/bin/$PIP "setuptools<81"
# A broken wandb install makes `import diffusers` fail. Nothing here uses wandb.
.venv/bin/pip uninstall -y -q wandb 2>/dev/null || true
if ! .venv/bin/python -c "import diffusers; print('diffusers', diffusers.__version__)"; then
    echo "ERROR: diffusers cannot be imported in .venv (see the error above). Neither FLUX.1 Fill nor the" >&2
    echo "       SDXL fallback can load until this is fixed. Try: .venv/bin/$PIP --force-reinstall diffusers" >&2
    exit 1
fi

echo "== IOPaint environment (.venv-iopaint): LaMa =="
python3 -m venv .venv-iopaint
.venv-iopaint/bin/$PIP --upgrade pip
.venv-iopaint/bin/$PIP torch torchvision iopaint

echo "== text detector =="
mkdir -p tools/models
[ -d tools/comic-text-detector ] || git clone -q --depth 1 https://github.com/dmMaze/comic-text-detector tools/comic-text-detector
.venv/bin/python scripts/numpy2_patch.py tools/comic-text-detector
for f in comictextdetector.pt comictextdetector.pt.onnx; do
    [ -f "tools/models/$f" ] || curl -fsSL -o "tools/models/$f" \
        "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/$f"
done

echo "== Real-ESRGAN (Linux build) =="
ESRGAN=tools/realesrgan/realesrgan-ncnn-vulkan
if [ ! -f "$ESRGAN" ]; then
    zip="$(mktemp -d)/realesrgan.zip"
    curl -fsSL -o "$zip" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    .venv/bin/python - "$zip" tools/realesrgan <<'PY'
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

archive, target = Path(sys.argv[1]), Path(sys.argv[2])
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp)
    binaries = [p for p in Path(tmp).rglob("realesrgan-ncnn-vulkan") if p.is_file()]
    if not binaries:
        sys.exit("ERROR: the Real-ESRGAN zip does not contain realesrgan-ncnn-vulkan.")
    # The executable needs the models folder that sits next to it: copy the whole folder.
    shutil.copytree(binaries[0].parent, target, dirs_exist_ok=True)
PY
    rm -f "$zip"
fi
chmod +x "$ESRGAN"
# -h prints the usage text and exits with an error code, so the text is what is checked.
esrgan_help="$("$ESRGAN" -h 2>&1 || true)"
if [[ "${esrgan_help,,}" != *usage* ]]; then
    echo "ERROR: $ESRGAN does not start. Its output:" >&2
    echo "${esrgan_help:0:2000}" >&2
    exit 1
fi
echo "realesrgan-ncnn-vulkan starts."

echo "== model weights =="
.venv/bin/python - <<'PY'
import os
import sys

FLUX = "black-forest-labs/FLUX.1-Fill-dev"
SDXL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
BUBBLES = "kitsumed/yolov8m_seg-speech-bubble"
FACES = "deepghs/anime_face_detection"

try:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:  # older huggingface_hub keeps them here
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
except ImportError as e:
    sys.exit(f"ERROR: download tool or library missing. huggingface_hub could not be imported in .venv ({e}). "
             "Run: .venv/bin/pip install --no-cache-dir -r requirements-gpu.txt")

token = os.environ.get("HF_TOKEN") or None


def access_denied(error: Exception) -> bool:
    """True when Hugging Face refused access, as opposed to a network or disk problem."""
    if isinstance(error, (GatedRepoError, RepositoryNotFoundError)):
        return True   # a gated repository looks "not found" to a request without a valid token
    status = getattr(getattr(error, "response", None), "status_code", None)
    return isinstance(error, HfHubHTTPError) and status in (401, 403)


have_flux = have_sdxl = False
try:
    # Skip the single-file copies in the FLUX repository; diffusers does not use them (saves about 24 GB).
    snapshot_download(FLUX, token=token, ignore_patterns=["flux1-fill-dev.safetensors", "ae.safetensors"])
    have_flux = True
    print("FLUX.1 Fill downloaded.")
except Exception as e:
    if not access_denied(e):
        sys.exit(f"ERROR: FLUX.1 Fill download failed, and not because of access ({type(e).__name__}: {e}). "
                 "Check the network and free disk space, then run this script again.")
    why = ("HF_TOKEN is not set" if token is None else
           "the token was rejected, or its account has not accepted the licence at "
           f"https://huggingface.co/{FLUX}")
    print(f"WARNING: access denied to FLUX.1 Fill: {why}. Falling back to SDXL inpainting.")

if not have_flux:
    try:
        snapshot_download(SDXL, token=token, allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])
        have_sdxl = True
        print("SDXL inpainting downloaded.")
    except Exception as e:
        print(f"SDXL inpainting download failed ({type(e).__name__}: {e}).")

if not (have_flux or have_sdxl):
    sys.exit("ERROR: neither FLUX.1 Fill nor SDXL inpainting was downloaded. Hard regions cannot be filled.")

# Not fatal: the steps that need these fetch them on first use.
for repo, patterns in ((BUBBLES, ["model.pt"]), (FACES, ["face_detect_v1.4_s/*"])):
    try:
        snapshot_download(repo, allow_patterns=patterns)
    except Exception as e:
        print(f"WARNING: {repo} not downloaded ({type(e).__name__}: {e}).")
PY

echo "== check =="
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

# The settings every run needs, in one file to load with `source`.
cat > gpu_env.sh <<EOF
export HF_HOME="$HF_HOME"
export IOPAINT_PYTHON="$(pwd)/.venv-iopaint/bin/python"
EOF

cat <<EOF

Setup finished. In every new shell, before running:

    source gpu_env.sh             # sets HF_HOME and IOPAINT_PYTHON
    export HF_TOKEN=hf_...        # only if it is not set already
    export ANTHROPIC_API_KEY=...  # for the prompts step

Then run a whole chapter in the background, so a dropped SSH connection does not stop it:

    nohup .venv/bin/python run.py --chapter ch001 > run_ch001.log 2>&1 &
    tail -f run_ch001.log
EOF
