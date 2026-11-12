#!/usr/bin/env bash
# Sets up a rented GPU machine (Linux, e.g. Vast.ai) for stage 1.
# Run it from the repository folder:
#   export HF_TOKEN=hf_...      # needed for FLUX.1 Fill; without it SDXL inpainting is used
#   bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== system libraries =="
if command -v apt-get >/dev/null; then
    apt-get update -qq && apt-get install -y -qq git curl python3-venv libglib2.0-0 libgl1 >/dev/null
fi

echo "== main environment (.venv): detectors and diffusion =="
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-gpu.txt

echo "== IOPaint environment (.venv-iopaint): LaMa =="
python3 -m venv .venv-iopaint
.venv-iopaint/bin/pip install -q --upgrade pip
.venv-iopaint/bin/pip install -q torch torchvision iopaint

echo "== text detector =="
mkdir -p tools/models
[ -d tools/comic-text-detector ] || git clone -q --depth 1 https://github.com/dmMaze/comic-text-detector tools/comic-text-detector
for f in comictextdetector.pt comictextdetector.pt.onnx; do
    [ -f "tools/models/$f" ] || curl -sL -o "tools/models/$f" \
        "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/$f"
done

echo "== model weights =="
# Skip the single-file copies in the FLUX repository; diffusers does not use them (saves about 24 GB).
if [ -n "${HF_TOKEN:-}" ]; then
    .venv/bin/huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \
        --exclude "flux1-fill-dev.safetensors" "ae.safetensors" --quiet \
        || echo "WARNING: FLUX.1 Fill could not be downloaded. Check that your account has accepted its licence. SDXL inpainting will be used."
else
    echo "WARNING: HF_TOKEN is not set, FLUX.1 Fill is skipped. SDXL inpainting will be used."
fi
.venv/bin/huggingface-cli download diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --include "*.json" "*.txt" "*fp16.safetensors" --quiet
.venv/bin/huggingface-cli download kitsumed/yolov8m_seg-speech-bubble model.pt --quiet

echo "== check =="
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

cat <<EOF

Setup finished. Before running, in every new shell:

    export IOPAINT_PYTHON=$(pwd)/.venv-iopaint/bin/python
    export HF_TOKEN=hf_...        # only if it is not set already

Then run stage 1 inpainting on the test pages:

    .venv/bin/python run.py --chapter ch001 --skip 5 --limit 4 --from inpaint --force
EOF
