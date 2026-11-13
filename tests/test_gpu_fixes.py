"""Tests for the fixes from the first GPU run: config defaults, the platform-aware binary path,
atomic writes, the NumPy 2 patch and the memory-aware diffusion loading. No GPU, no downloads."""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from PIL import Image

from conftest import CHAPTER, ROOT, StubDiffusion, make_page
from scripts import numpy2_patch
from steps import common, diffusion, inpaint, upscale
from steps.common import (
    CLEAN_DIR, CLEAN_LAMA_DIR, PNG_DIR, REGIONS_REPORT_NAME, UPSCALED_DIR, load_config, save_png, write_json,
)
from steps.detect_text import placeholder_module
from test_stage1 import pages, run_stage1  # noqa: F401  (pages is a fixture)


# --- A4: gpu_id default and the platform-aware binary path ---

def fake_ncnn(tmp_path, monkeypatch, ctx, exit_code=0):
    """An ncnn 'installation' whose executable is replaced by a function that records its command."""
    bin_dir = tmp_path / "bin"
    (bin_dir / "models").mkdir(parents=True)
    binary = upscale.ncnn_binary_path(tmp_path, "bin/realesrgan-ncnn-vulkan")
    binary.touch()
    (bin_dir / "models" / "realesr-animevideov3-x2.param").touch()
    ctx.cfg["upscale"].update(backend="ncnn", ncnn_binary=str(bin_dir / "realesrgan-ncnn-vulkan"))
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        src, out = Path(cmd[cmd.index("-i") + 1]), Path(cmd[cmd.index("-o") + 1])
        with Image.open(src) as im:
            im.convert("RGB").resize((im.width * 2, im.height * 2)).save(out, format="PNG")
        return types.SimpleNamespace(returncode=exit_code, stderr="vkCreateInstance failed", stdout="")

    monkeypatch.setattr(upscale.subprocess, "run", fake_run)
    return commands


@pytest.fixture
def small_page(ctx):
    folder = ctx.data_dir / PNG_DIR / CHAPTER
    folder.mkdir(parents=True)
    return make_page(folder / "small.png", (500, 800))      # 800 high: needs 2x


def test_the_shipped_config_works_on_both_platforms():
    up = load_config(ROOT / "config.yaml")["upscale"]
    assert up["gpu_id"] == "auto"
    assert not up["ncnn_binary"].lower().endswith(".exe")


@pytest.mark.parametrize("gpu_id, expected", [("missing", None), ("auto", None), (0, "0"), (1, "1")])
def test_gpu_id_is_optional_and_auto_by_default(ctx, small_page, tmp_path, monkeypatch, gpu_id, expected):
    commands = fake_ncnn(tmp_path, monkeypatch, ctx)
    if gpu_id == "missing":
        del ctx.cfg["upscale"]["gpu_id"]
    else:
        ctx.cfg["upscale"]["gpu_id"] = gpu_id
    upscale.run(ctx, [CHAPTER])                              # must not raise KeyError
    with Image.open(ctx.data_dir / UPSCALED_DIR / CHAPTER / "small.png") as out:
        assert out.size == (1000, 1600)
    cmd = commands[0]
    assert (cmd[cmd.index("-g") + 1] if "-g" in cmd else None) == expected


def test_tile_size_is_optional(ctx):
    del ctx.cfg["upscale"]["tile_size"]
    assert upscale.NcnnBackend(ctx.cfg, ROOT).tile_size == 0


@pytest.mark.parametrize("configured", ["tools/realesrgan/realesrgan-ncnn-vulkan", "tools/realesrgan/realesrgan-ncnn-vulkan.exe"])
def test_binary_path_follows_the_platform(configured):
    win = upscale.ncnn_binary_path(PureWindowsPath("C:/proj"), configured, windows=True)
    linux = upscale.ncnn_binary_path(PurePosixPath("/workspace/proj"), configured, windows=False)
    assert win == PureWindowsPath("C:/proj/tools/realesrgan/realesrgan-ncnn-vulkan.exe")
    assert linux == PurePosixPath("/workspace/proj/tools/realesrgan/realesrgan-ncnn-vulkan")


def test_binary_path_absolute_and_empty():
    absolute = upscale.ncnn_binary_path(PurePosixPath("/proj"), "/opt/esrgan/realesrgan-ncnn-vulkan", windows=False)
    assert absolute == PurePosixPath("/opt/esrgan/realesrgan-ncnn-vulkan")
    assert upscale.ncnn_binary_path(PurePosixPath("/proj"), "", windows=False) is None
    assert upscale.ncnn_binary_path(PurePosixPath("/proj"), None, windows=True) is None


def test_binary_path_uses_this_platform_by_default(tmp_path):
    path = upscale.ncnn_binary_path(tmp_path, "bin/realesrgan-ncnn-vulkan")
    assert path.name == ("realesrgan-ncnn-vulkan.exe" if sys.platform == "win32" else "realesrgan-ncnn-vulkan")


# --- A7: atomic writes ---

class Crash(BaseException):
    """Stands in for the process being interrupted in the middle of a write."""


class HalfWrittenImage:
    def save(self, path, format=None):
        Path(path).write_bytes(b"\x89PNG half a file")
        raise Crash()


def no_leftovers(folder: Path) -> bool:
    return not [p for p in folder.rglob("*") if p.name.endswith(".partial") or ".partial." in p.name]


def test_a_crash_while_saving_leaves_no_file_under_the_final_name(tmp_path):
    dst = tmp_path / "out" / "01.png"
    with pytest.raises(Crash):
        save_png(HalfWrittenImage(), dst)
    assert not dst.exists()
    assert no_leftovers(tmp_path)


def test_a_crash_while_saving_keeps_the_previous_complete_file(tmp_path):
    dst = tmp_path / "01.png"
    save_png(Image.new("RGB", (4, 4), (1, 2, 3)), dst)
    before = dst.read_bytes()
    with pytest.raises(Crash):
        save_png(HalfWrittenImage(), dst)
    assert dst.read_bytes() == before


def test_a_crash_while_writing_json_or_copying_leaves_nothing(tmp_path, monkeypatch):
    class Unserialisable:
        pass
    dst = tmp_path / "regions.json"
    with pytest.raises(TypeError):
        write_json(dst, {"regions": Unserialisable()})
    assert not dst.exists()

    def half_copy(src, tmp):
        Path(tmp).write_bytes(b"half")
        raise Crash()
    monkeypatch.setattr(common.shutil, "copyfile", half_copy)
    with pytest.raises(Crash):
        common.copy_file(tmp_path / "missing-src.png", tmp_path / "copy.png")
    assert not (tmp_path / "copy.png").exists()
    assert no_leftovers(tmp_path)


def test_a_file_left_by_a_killed_run_is_not_a_page(tmp_path):
    """A process killed outright cannot clean up; what it leaves must not count as output."""
    make_page(tmp_path / "01.png", (40, 60))
    (tmp_path / ".02.png.partial").write_bytes(b"half a page")
    assert [p.name for p in common.list_pages(tmp_path)[0]] == ["01.png"]


def test_failed_ncnn_upscale_leaves_no_output(ctx, small_page, tmp_path, monkeypatch):
    fake_ncnn(tmp_path, monkeypatch, ctx, exit_code=1)       # wrote its file, then reported failure
    with pytest.raises(SystemExit, match="Upscale failed"):
        upscale.run(ctx, [CHAPTER])
    assert not (ctx.data_dir / UPSCALED_DIR / CHAPTER / "small.png").exists()


def test_crash_in_the_middle_of_inpaint_redoes_only_the_unfinished_page(ctx, pages, monkeypatch):
    """The run dies while writing the second page: the first is complete and reported, the second is redone."""
    real_save = inpaint.save_png
    target = ctx.data_dir / CLEAN_DIR / CHAPTER / "p2.png"

    def dying_save(im, dst):
        if dst == target:
            real_save(HalfWrittenImage(), dst)
        real_save(im, dst)
    monkeypatch.setattr(inpaint, "save_png", dying_save)
    with pytest.raises(Crash):
        run_stage1(ctx, diffusion=StubDiffusion())
    assert (ctx.data_dir / CLEAN_DIR / CHAPTER / "p1.png").exists()
    assert not target.exists()
    assert (ctx.data_dir / CLEAN_LAMA_DIR / CHAPTER / "p2.png").exists()     # one of the pair: not "already done"
    report = (ctx.data_dir / REGIONS_REPORT_NAME).read_text(encoding="utf-8")
    assert "p1.png" in report and "p2.png" not in report                    # finished pages keep their rows

    monkeypatch.setattr(inpaint, "save_png", real_save)
    again = StubDiffusion()
    inpaint.run(ctx, [CHAPTER], diffusion=again)
    assert target.exists()
    assert len(again.seeds) == 1                                             # only p2's hard region was refilled
    report = (ctx.data_dir / REGIONS_REPORT_NAME).read_text(encoding="utf-8")
    assert "p1.png" in report and "p2.png" in report


# --- A5: NumPy 2 ---

OLD_SOURCE = '''import numpy as np
NP_BOOL_TYPES = (np.bool_, np.bool8)
NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)
box = np.int0(points)
big = np.Inf if x is np.NaN else np.float_power(2, 3)
kinds = [np.object0, np.str0, np.unicode_, np.complex_]
keep = np.float32(1) + np.int_(2) + np.bool_(True)
'''


def test_patch_replaces_every_removed_alias():
    patched = numpy2_patch.patch_source(OLD_SOURCE)
    assert "NP_BOOL_TYPES = (np.bool_,)" in patched
    assert "NP_FLOAT_TYPES = (np.float64, np.float16, np.float32)" in patched
    assert "box = np.intp(points)" in patched
    assert "big = np.inf if x is np.nan else np.float_power(2, 3)" in patched      # float_power is not an alias
    assert "kinds = [np.object_, np.str_, np.str_, np.complex128]" in patched
    assert "keep = np.float32(1) + np.int_(2) + np.bool_(True)" in patched         # valid names are untouched
    assert numpy2_patch.patch_source(patched) == patched                            # running it again changes nothing
    namespace = {}
    exec("import numpy as np\n" + "\n".join(patched.splitlines()[1:3]), namespace)  # the tuples are valid Python
    assert namespace["NP_BOOL_TYPES"] == (namespace["np"].bool_,)


def test_patch_folder_skips_and_deletes_checkpoints_and_notes_the_readme(tmp_path):
    repo = tmp_path / "detector"
    (repo / "utils" / ".ipynb_checkpoints").mkdir(parents=True)
    (repo / "utils" / "io_utils.py").write_text(OLD_SOURCE, encoding="utf-8")
    (repo / "utils" / ".ipynb_checkpoints" / "io_utils-checkpoint.py").write_text(OLD_SOURCE, encoding="utf-8")
    (repo / "README.md").write_text("# detector\n", encoding="utf-8")
    assert numpy2_patch.find_aliases(repo)
    assert numpy2_patch.main([str(repo)]) == 0
    assert numpy2_patch.find_aliases(repo) == []
    assert not (repo / "utils" / ".ipynb_checkpoints").exists()
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("> **Patched for NumPy 2**") and readme.endswith("# detector\n")
    assert numpy2_patch.patch_folder(repo) == []                                    # second run: nothing to do


DETECTOR = ROOT / "tools" / "comic-text-detector"
needs_detector = pytest.mark.skipif(not (DETECTOR / "utils" / "io_utils.py").is_file(),
                                    reason="tools/comic-text-detector is not cloned")


@needs_detector
def test_the_detector_clone_has_no_removed_aliases():
    assert numpy2_patch.find_aliases(DETECTOR) == [], "run: python scripts/numpy2_patch.py tools/comic-text-detector"


@needs_detector
@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="opencv is not installed")
def test_detector_utils_import_under_numpy_2():
    """Imports io_utils and imgproc_utils with the aliases NumPy 2 removed taken out of numpy,
    so the test means the same on NumPy 1 and NumPy 2."""
    code = f'''
import sys, json
import numpy as np
for name in {sorted(numpy2_patch.ALIASES)!r}:
    if name in vars(np):
        delattr(np, name)
sys.path.insert(0, {str(DETECTOR)!r})
from utils import io_utils, imgproc_utils
out = json.dumps({{"b": np.bool_(True), "f": np.float32(1.5), "d": np.float64(2.5), "i": np.int64(3), "a": np.arange(2)}},
                 cls=io_utils.NumpyEncoder)
assert json.loads(out) == {{"b": True, "f": 1.5, "d": 2.5, "i": 3, "a": [0, 1]}}, out
print("ok", np.__version__)
'''
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.startswith("ok"), proc.stderr[-2000:]


# --- A2: the wandb placeholder must not break `import diffusers` later in the run ---

def test_placeholder_module_is_removed_after_the_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    with placeholder_module("wandb"):
        import wandb                                         # the detector's import succeeds
        assert wandb.__spec__ is None
    assert "wandb" not in sys.modules
    importlib.util.find_spec("wandb")                        # what diffusers does on import; raised ValueError before


def test_placeholder_module_restores_a_module_that_was_loaded(monkeypatch):
    real = types.ModuleType("wandb")
    monkeypatch.setitem(sys.modules, "wandb", real)
    with placeholder_module("wandb"):
        assert sys.modules["wandb"] is not real
    assert sys.modules["wandb"] is real


# --- A6: diffusion loading ---

def test_load_arguments_follow_the_installed_diffusers():
    assert diffusion.load_arguments("bf16", True, device_map_ok=True, dtype_ok=True) == {
        "dtype": "bf16", "low_cpu_mem_usage": True, "device_map": "cuda"}
    assert diffusion.load_arguments("bf16", True, device_map_ok=False, dtype_ok=False) == {
        "torch_dtype": "bf16", "low_cpu_mem_usage": True}
    assert "device_map" not in diffusion.load_arguments("bf16", False, device_map_ok=True, dtype_ok=True)


class FakeTensor:
    def to(self, device):
        return self


class FakePipeline:
    """Records how diffusers is used; draws nothing."""
    loads = []              # one entry per from_pretrained call
    refuse_device_map = False

    def __init__(self, repo, kwargs):
        import torch
        self.repo, self.kwargs = repo, kwargs
        self.offloaded = self.moved_to = None
        self.calls = []
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype"))
        part = types.SimpleNamespace(dtype=dtype)
        self.transformer = None if "transformer" in kwargs else part
        self.unet = part
        assert dtype in (torch.bfloat16, torch.float16)

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        if cls.refuse_device_map and "device_map" in kwargs:
            raise NotImplementedError("cuda not supported. Supported strategies are: balanced")
        pipe = cls(repo, kwargs)
        FakePipeline.loads.append(pipe)
        return pipe

    def enable_model_cpu_offload(self):
        assert "device_map" not in self.kwargs, "diffusers refuses offloading on a pipeline with a device map"
        self.offloaded = True

    def to(self, device):
        self.moved_to = device
        return self

    def encode_prompt(self, prompt, prompt_2, device, max_sequence_length):
        self.calls.append(prompt)
        return FakeTensor(), FakeTensor(), FakeTensor()

    def __call__(self, image, mask_image, width, height, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(images=[Image.new("RGB", (width, height), (0, 255, 0))])


@pytest.fixture
def fake_diffusers(monkeypatch):
    FakePipeline.loads, FakePipeline.refuse_device_map = [], False
    module = types.ModuleType("diffusers")
    module.FluxFillPipeline = module.AutoPipelineForInpainting = FakePipeline
    monkeypatch.setitem(sys.modules, "diffusers", module)
    monkeypatch.setattr(diffusion, "diffusers_features", lambda: (True, False))
    monkeypatch.setattr(diffusion, "gpu_total_gb", lambda: 24.0)
    monkeypatch.setattr(diffusion.DiffusionBackend, "_release", lambda self: None)
    return FakePipeline


def fill_all(backend, prompts, fills_each=2):
    image, mask = Image.new("RGB", (640, 480)), Image.new("L", (640, 480), 255)
    for prompt in prompts:
        for seed in range(fills_each):
            assert backend.fill(image, mask, prompt, seed).size == (640, 480)


def test_flux_loads_once_and_never_holds_encoders_and_transformer_together(cfg, fake_diffusers):
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    prompts = ["a street", "a wall", "a street"]
    backend.prepare(prompts)
    fill_all(backend, prompts)

    encoders, main = fake_diffusers.loads                       # two loads in all, for six fills
    assert encoders.kwargs["transformer"] is None and encoders.kwargs["vae"] is None
    assert encoders.kwargs["device_map"] == "cuda"              # the text encoders never pass through system RAM
    assert encoders.calls == ["a street", "a wall"]             # each prompt encoded once
    assert main.kwargs["text_encoder"] is None and main.kwargs["text_encoder_2"] is None
    assert "device_map" not in main.kwargs and main.offloaded   # 24 GB card: the transformer is offloaded
    assert len(main.calls) == 6 and all("prompt" not in call and "prompt_embeds" in call for call in main.calls)
    assert backend.name == "flux-fill" and backend.loads == 1
    import torch
    assert main.kwargs["torch_dtype"] is torch.bfloat16 and "dtype" not in main.kwargs


def test_flux_goes_straight_onto_a_gpu_that_holds_it(cfg, fake_diffusers, monkeypatch):
    monkeypatch.setattr(diffusion, "gpu_total_gb", lambda: 48.0)
    monkeypatch.setattr(diffusion, "diffusers_features", lambda: (True, True))
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    fill_all(backend, ["a street"])
    main = fake_diffusers.loads[-1]
    assert main.kwargs["device_map"] == "cuda" and not main.offloaded and main.moved_to is None
    assert "dtype" in main.kwargs and "torch_dtype" not in main.kwargs


def test_older_diffusers_without_cuda_device_map_still_loads(cfg, fake_diffusers, monkeypatch):
    monkeypatch.setattr(diffusion, "diffusers_features", lambda: (False, False))
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    fill_all(backend, ["a street"])
    encoders, main = fake_diffusers.loads
    assert "device_map" not in encoders.kwargs and encoders.moved_to == "cuda"
    assert main.offloaded and backend.name == "flux-fill"


def test_refused_device_map_falls_back_to_loading_through_ram(cfg, fake_diffusers, capsys):
    fake_diffusers.refuse_device_map = True
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    fill_all(backend, ["a street"])
    assert backend.name == "flux-fill" and backend.switch_reason == ""
    assert fake_diffusers.loads[0].moved_to == "cuda"
    assert "loading through system RAM instead" in capsys.readouterr().out


def test_sdxl_loads_once_straight_onto_the_gpu(cfg, fake_diffusers):
    cfg["inpaint"]["diffusion_model"] = "sdxl-inpaint"
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    backend.prepare(["a street"])                               # nothing to encode ahead for SDXL
    fill_all(backend, ["a street", "a wall"])
    (pipe,) = fake_diffusers.loads
    assert pipe.kwargs["device_map"] == "cuda" and pipe.kwargs["variant"] == "fp16" and pipe.moved_to is None
    assert [call["prompt"] for call in pipe.calls] == ["a street", "a street", "a wall", "a wall"]


def test_flux_that_cannot_load_gives_way_to_sdxl_once(cfg, fake_diffusers, monkeypatch, capsys):
    def no_access(self, prompts):
        raise OSError("401 gated repo")
    monkeypatch.setattr(diffusion.DiffusionBackend, "_encode", no_access)
    backend = diffusion.DiffusionBackend(cfg["inpaint"])
    backend.prepare(["a street"])
    fill_all(backend, ["a street", "a wall"])
    assert backend.name == "sdxl-inpaint" and "401 gated repo" in backend.switch_reason
    assert len(fake_diffusers.loads) == 1 and backend.loads == 1
    assert "Switching to SDXL inpainting" in capsys.readouterr().out


def test_a_float32_load_is_refused(cfg, fake_diffusers, monkeypatch):
    """If diffusers ignored the dtype argument the weights would take twice the memory: stop instead."""
    import torch
    pipe = types.SimpleNamespace(transformer=types.SimpleNamespace(dtype=torch.float32))
    with pytest.raises(RuntimeError, match="dtype argument was ignored"):
        diffusion.check_dtype(pipe, "transformer", torch.bfloat16)


def test_inpaint_announces_every_prompt_before_the_first_fill(ctx, pages):
    class Recording(StubDiffusion):
        def __init__(self):
            super().__init__()
            self.events = []

        def prepare(self, prompts):
            self.events.append(("prepare", sorted(set(prompts))))

        def fill(self, image, mask, prompt, seed):
            self.events.append(("fill", prompt))
            return super().fill(image, mask, prompt, seed)

    recording = Recording()
    run_stage1(ctx, diffusion=recording)
    kinds = [kind for kind, _ in recording.events]
    assert kinds[0] == "prepare" and kinds.count("prepare") == 1 and kinds.count("fill") == 2
    assert {prompt for kind, prompt in recording.events if kind == "fill"} <= set(recording.events[0][1])


def test_peak_ram_is_a_number_or_unknown():
    ram = diffusion.peak_ram_gb()
    assert ram is None or ram > 0
    if sys.platform.startswith("linux"):
        assert ram is not None


# --- prompts: a fixed prompt needs no API key and no prompts step ---

def test_fixed_prompt_mode_skips_the_api_and_fills_every_hard_region(ctx, pages):
    ctx.cfg["prompts"]["mode"] = "fixed"
    run_stage1_without_prompts = None  # noqa: F841
    from steps import build_masks, describe_regions, detect_bubbles, detect_text, tag_pages, make_debug
    detect_bubbles.run(ctx)
    detect_text.run(ctx)
    tag_pages.run(ctx)
    build_masks.run(ctx)
    describe_regions.run(ctx)                                  # no describer, no key: returns at once
    assert not (ctx.data_dir / "04_prompts").exists()
    diffusion = StubDiffusion()
    inpaint.run(ctx, diffusion=diffusion)
    assert diffusion.prompts and set(diffusion.prompts) == {ctx.cfg["prompts"]["fixed_prompt"]}
    assert "no additional people" in ctx.cfg["prompts"]["fixed_prompt"]
