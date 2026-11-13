"""Diffusion inpainting for hard regions. Needs a CUDA GPU.

FLUX.1 Fill is tried first. If it cannot be loaded (no access, no token, old
diffusers) or runs out of GPU memory, an SDXL inpainting model takes over and
the switch is printed. The Hugging Face token is read from the HF_TOKEN
environment variable, never from a file.

Memory. The FLUX weights are about 34 GB (transformer 24 GB, T5 text encoder
10 GB), more than a 24 GB card holds and close to the RAM limit of a rented
container, so the two halves are never loaded together:

1. The text encoders are loaded alone, straight onto the GPU, every prompt of
   the run is encoded, and the encoders are released.
2. The transformer and VAE are loaded without text encoders. On a card that
   holds everything (FULL_GPU_GB or more) they go straight onto the GPU and
   never pass through system RAM. On a smaller card they are offloaded: they
   wait in system RAM (about 24 GB) and move to the GPU while they work.

The pipeline is loaded once per run and reused for every region.
"""
from __future__ import annotations

import gc
import inspect
import os

from PIL import Image

from steps.common import pick_device

MULTIPLE = 16       # both models want sides divisible by this
FULL_GPU_GB = 40    # GPU memory from which all of FLUX (weights about 34 GB, plus working memory) stays on the GPU
MAX_SEQUENCE_LENGTH = 512


def working_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Window size brought within max_side and rounded to a size the models accept."""
    scale = min(1.0, max_side / max(size))
    return tuple(max(MULTIPLE, round(v * scale / MULTIPLE) * MULTIPLE) for v in size)


def peak_ram_gb() -> float | None:
    """Most system RAM this process has held so far, in GB. None where the system does not report it."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 2**20      # the value is in kB
    except OSError:
        pass
    return None


def gpu_total_gb() -> float:
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def diffusers_features() -> tuple[bool, bool]:
    """What the installed diffusers accepts in from_pretrained: (device_map="cuda", dtype=).

    device_map="cuda" arrived after 0.32 (which only knows "balanced"), and dtype= replaced
    torch_dtype= later still. An unknown keyword is silently ignored there, which for dtype
    would load the weights in float32 at twice the size, so both are checked, not assumed.
    """
    from diffusers import DiffusionPipeline
    try:
        from diffusers.pipelines.pipeline_utils import SUPPORTED_DEVICE_MAP
    except ImportError:
        SUPPORTED_DEVICE_MAP = []
    try:
        source = inspect.getsource(DiffusionPipeline.from_pretrained.__func__)
    except (OSError, TypeError, AttributeError):
        source = ""
    return "cuda" in SUPPORTED_DEVICE_MAP, 'kwargs.pop("dtype"' in source


def load_arguments(dtype, on_gpu: bool, device_map_ok: bool, dtype_ok: bool) -> dict:
    """Keyword arguments for from_pretrained, for the installed diffusers version."""
    args = {"dtype" if dtype_ok else "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if on_gpu and device_map_ok:
        args["device_map"] = "cuda"     # weights go from disk to the GPU without a copy in system RAM
    return args


def load_pipeline(cls, repo: str, dtype, on_gpu: bool, **components):
    """Load a pipeline. With on_gpu it ends up on the GPU, directly when diffusers can do that."""
    args = load_arguments(dtype, on_gpu, *diffusers_features())
    token = os.environ.get("HF_TOKEN")
    try:
        pipe = cls.from_pretrained(repo, token=token, **args, **components)
    except (ValueError, NotImplementedError) as e:
        if "device_map" not in args:
            raise
        print(f"Diffusion: loading straight onto the GPU was refused ({e}); loading through system RAM instead.")
        del args["device_map"]
        pipe = cls.from_pretrained(repo, token=token, **args, **components)
    if on_gpu and "device_map" not in args:
        pipe = pipe.to("cuda")          # older diffusers: through system RAM, then moved
    return pipe


def check_dtype(pipe, part: str, dtype) -> None:
    model = getattr(pipe, part, None)
    if model is not None and model.dtype != dtype:
        raise RuntimeError(f"{part} was loaded as {model.dtype}, not {dtype}: the dtype argument was ignored")


class DiffusionBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = ""        # the model that actually ran: flux-fill or sdxl-inpaint
        self.pipe = None
        self.placement = ""   # where the loaded model lives, for the summary
        self.switch_reason = ""
        self.embeds: dict[str, tuple] = {}   # FLUX: prompt -> (prompt embedding, pooled embedding), kept on the CPU
        self.loads = 0        # how many times a main pipeline was loaded; stays 1 unless FLUX gives way to SDXL

    def _wants_flux(self) -> bool:
        return self.cfg["diffusion_model"] == "flux-fill" and self.name != "sdxl-inpaint"

    def _release(self) -> None:
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def _encode(self, prompts: list[str]) -> None:
        """Encode prompts with the FLUX text encoders alone, then release the encoders."""
        import torch
        from diffusers import FluxFillPipeline
        encoders = load_pipeline(FluxFillPipeline, self.cfg["flux"]["repo"], torch.bfloat16, on_gpu=True,
                                 transformer=None, vae=None)
        try:
            with torch.no_grad():
                for prompt in prompts:
                    embeds, pooled, _ = encoders.encode_prompt(
                        prompt=prompt, prompt_2=None, device="cuda", max_sequence_length=MAX_SEQUENCE_LENGTH)
                    self.embeds[prompt] = (embeds.to("cpu"), pooled.to("cpu"))
        finally:
            del encoders
            self._release()

    def prepare(self, prompts) -> None:
        """Encode every prompt the run will use, before the image model is loaded.

        Calling this first keeps the text encoders and the transformer from being in memory
        together. fill() calls it too, so a prompt that was not announced still works.
        """
        if not self._wants_flux():
            return
        missing = [p for p in dict.fromkeys(prompts) if p not in self.embeds]
        if not missing:
            return
        try:
            self._encode(missing)
        except Exception as e:
            self._switch_to_sdxl(f"{type(e).__name__}: {str(e)[:300]}")

    def _load_flux(self):
        import torch
        from diffusers import FluxFillPipeline
        on_gpu = gpu_total_gb() >= FULL_GPU_GB
        pipe = load_pipeline(FluxFillPipeline, self.cfg["flux"]["repo"], torch.bfloat16, on_gpu=on_gpu,
                             text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)
        check_dtype(pipe, "transformer", torch.bfloat16)
        if on_gpu:
            self.placement = "all on the GPU"
        else:
            # The transformer alone nearly fills a 24 GB card. Offloading keeps on the GPU only the
            # part that is working; it cannot be combined with device_map.
            pipe.enable_model_cpu_offload()
            self.placement = "offloaded, waits in system RAM"
        return pipe

    def _load_sdxl(self):
        import torch
        from diffusers import AutoPipelineForInpainting
        pipe = load_pipeline(AutoPipelineForInpainting, self.cfg["sdxl"]["repo"], torch.float16, on_gpu=True,
                             variant="fp16")
        check_dtype(pipe, "unet", torch.float16)
        self.placement = "all on the GPU"
        return pipe

    def _loaded(self, pipe, name: str) -> None:
        self.pipe, self.name = pipe, name
        self.loads += 1
        ram = peak_ram_gb()
        print(f"Diffusion: loaded {name} ({self.placement})"
              + (f", peak system RAM so far {ram:.1f} GB" if ram is not None else ""))

    def _switch_to_sdxl(self, reason: str) -> None:
        self.switch_reason = reason
        print(f"Diffusion: FLUX.1 Fill is not usable ({reason}). Switching to SDXL inpainting.")
        self.pipe, self.name = None, "sdxl-inpaint"
        self._release()
        self._loaded(self._load_sdxl(), "sdxl-inpaint")

    def load(self) -> None:
        """Load the model once; later calls return at once."""
        if self.pipe is not None:
            return
        if self._wants_flux():
            try:
                self._loaded(self._load_flux(), "flux-fill")
            except Exception as e:
                self._switch_to_sdxl(f"{type(e).__name__}: {str(e)[:300]}")
        else:
            self._loaded(self._load_sdxl(), "sdxl-inpaint")

    def _run(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int) -> Image.Image:
        import torch
        generator = torch.Generator("cpu").manual_seed(seed)
        width, height = image.size
        if self.name == "flux-fill":
            settings = self.cfg["flux"]
            embeds, pooled = self.embeds[prompt]
            return self.pipe(
                prompt_embeds=embeds.to("cuda"), pooled_prompt_embeds=pooled.to("cuda"),
                image=image, mask_image=mask, width=width, height=height,
                guidance_scale=settings["guidance"], num_inference_steps=settings["steps"],
                max_sequence_length=MAX_SEQUENCE_LENGTH, generator=generator,
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
        self.prepare([prompt])
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
        text = f"{self.name} on {torch.cuda.get_device_name(0)} ({self.placement}), peak GPU memory {peak:.1f} GB"
        ram = peak_ram_gb()
        if ram is not None:
            text += f", peak system RAM {ram:.1f} GB"
        return text + (f" (switched from FLUX.1 Fill: {self.switch_reason})" if self.switch_reason else "")


def make_diffusion_backend(cfg: dict) -> DiffusionBackend | None:
    """The diffusion backend, or None on a machine without a GPU."""
    return DiffusionBackend(cfg) if pick_device() == "cuda" else None
