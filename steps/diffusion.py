"""Diffusion inpainting for hard regions. Needs a CUDA GPU.

FLUX.1 Fill is tried first. If it cannot be loaded (no access, no token, old
diffusers) or runs out of GPU memory, an SDXL inpainting model takes over and
the switch is printed. The Hugging Face token is read from the HF_TOKEN
environment variable, never from a file.
"""
from __future__ import annotations

import os

from PIL import Image

from steps.common import pick_device

MULTIPLE = 16  # both models want sides divisible by this


def working_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Window size brought within max_side and rounded to a size the models accept."""
    scale = min(1.0, max_side / max(size))
    return tuple(max(MULTIPLE, round(v * scale / MULTIPLE) * MULTIPLE) for v in size)


class DiffusionBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = ""        # the model that actually ran: flux-fill or sdxl-inpaint
        self.pipe = None
        self.switch_reason = ""

    def _load_flux(self):
        import torch
        from diffusers import FluxFillPipeline
        pipe = FluxFillPipeline.from_pretrained(
            self.cfg["flux"]["repo"], torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
        )
        # The transformer and the T5 text encoder together exceed 24 GB; offloading keeps
        # only the part that is working on the GPU.
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_sdxl(self):
        import torch
        from diffusers import AutoPipelineForInpainting
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.cfg["sdxl"]["repo"], torch_dtype=torch.float16, variant="fp16", token=os.environ.get("HF_TOKEN"),
        )
        return pipe.to("cuda")

    def _switch_to_sdxl(self, reason: str) -> None:
        import torch
        self.switch_reason = reason
        print(f"Diffusion: FLUX.1 Fill is not usable ({reason}). Switching to SDXL inpainting.")
        self.pipe = None
        torch.cuda.empty_cache()
        self.pipe, self.name = self._load_sdxl(), "sdxl-inpaint"

    def load(self) -> None:
        if self.pipe is not None:
            return
        if self.cfg["diffusion_model"] == "flux-fill":
            try:
                self.pipe, self.name = self._load_flux(), "flux-fill"
                return
            except Exception as e:
                self._switch_to_sdxl(f"{type(e).__name__}: {str(e)[:300]}")
        else:
            self.pipe, self.name = self._load_sdxl(), "sdxl-inpaint"

    def _run(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int) -> Image.Image:
        import torch
        generator = torch.Generator("cpu").manual_seed(seed)
        width, height = image.size
        if self.name == "flux-fill":
            settings = self.cfg["flux"]
            return self.pipe(
                prompt=prompt, image=image, mask_image=mask, width=width, height=height,
                guidance_scale=settings["guidance"], num_inference_steps=settings["steps"],
                max_sequence_length=512, generator=generator,
            ).images[0]
        settings = self.cfg["sdxl"]
        return self.pipe(
            prompt=prompt, image=image, mask_image=mask, width=width, height=height,
            guidance_scale=settings["guidance"], num_inference_steps=settings["steps"],
            strength=settings["strength"], generator=generator,
        ).images[0]

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Fill the white area of the mask. Returns an image the size of the input window."""
        import torch
        self.load()
        size = working_size(image.size, self.cfg["diffusion_max_side"])
        small, small_mask = image.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
        try:
            result = self._run(small, small_mask, prompt, seed)
        except torch.cuda.OutOfMemoryError:
            if self.name != "flux-fill":
                raise
            self._switch_to_sdxl("out of GPU memory")
            result = self._run(small, small_mask, prompt, seed)
        return result.convert("RGB").resize(image.size, Image.LANCZOS)

    def summary(self) -> str:
        import torch
        peak = torch.cuda.max_memory_allocated() / 2**30
        text = f"{self.name} on {torch.cuda.get_device_name(0)}, peak GPU memory {peak:.1f} GB"
        return text + (f" (switched from FLUX.1 Fill: {self.switch_reason})" if self.switch_reason else "")


def make_diffusion_backend(cfg: dict) -> DiffusionBackend | None:
    """The diffusion backend, or None on a machine without a GPU."""
    return DiffusionBackend(cfg) if pick_device() == "cuda" else None
