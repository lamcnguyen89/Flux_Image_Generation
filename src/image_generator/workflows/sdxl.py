"""SDXL text-to-image loading, scheduler selection, and named LoRA adapters."""

from copy import deepcopy
import gc
from importlib import import_module
import math
from pathlib import Path
import re
import secrets
from time import perf_counter

from .base import GeneratedBatch, GenerationRequest, Workflow
from .common import _nonnegative_number, scheduler_config_snapshot


SAMPLERS = {
    "euler": "EulerDiscreteScheduler",
    "euler_ancestral": "EulerAncestralDiscreteScheduler",
    "dpmpp_2m_sde": "DPMSolverMultistepScheduler",
}


class SDXLWorkflow(Workflow):
    ui_modes = ("txt2img",)
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 1024, "height": 1024,
        "num_inference_steps": 50, "guidance_scale": 5.0,
        "num_images_per_prompt": 1, "seed": None, "clip_skip": None,
        "sampling": {"sampler": "model_default", "noise_schedule": "default"}, "loras": [],
    }

    def __init__(
        self, *, device="cuda:0", dtype=None, cache_dir=None,
        local_files_only=False, token=None,
    ):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
        self.device = device
        self.dtype = dtype or ("float32" if device == "cpu" else "float16")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if device == "cpu" and self.dtype == "float16":
            raise ValueError("CPU generation requires float32 or bfloat16")
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self.local_files_only = local_files_only
        self._token = token
        self._torch = None
        self._diffusers = None
        self._pipeline = None
        self._model_key = None
        self._scheduler_class = None
        self._scheduler_config = None
        self._has_adapters = False

    def validate(self, request: GenerationRequest) -> None:
        model, settings = request.model, request.settings
        if request.inputs:
            raise ValueError("SDXL txt2img does not accept image inputs")
        if model.kind != "model" or model.family != "sdxl" or model.defaults["workflow"] != "sdxl":
            raise ValueError("The SDXL workflow requires an SDXL catalog model")
        if request.mode != "txt2img" or request.mode not in model.defaults["supported_modes"]:
            raise ValueError("This workflow currently supports only declared txt2img mode")
        unknown = settings.keys() - self.defaults.keys()
        if unknown:
            raise ValueError(f"Unsupported SDXL settings: {', '.join(sorted(unknown))}")
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(request.negative_prompt, str):
            raise ValueError("negative_prompt must be a string")
        requirements = model.defaults["requirements"]
        multiple = requirements.get("dimension_multiple", 8)
        if type(multiple) is not int or multiple <= 0:
            raise ValueError("requirements.dimension_multiple must be a positive integer")
        multiple = math.lcm(8, multiple)
        for name in ("width", "height", "num_inference_steps", "num_images_per_prompt"):
            value = settings.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if name in {"width", "height"} and value % multiple:
                raise ValueError(f"{name} must be divisible by {multiple}")
        seed = settings.get("seed")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be null or an integer between 0 and 2**63 - 1")
        if not _nonnegative_number(settings.get("guidance_scale")):
            raise ValueError("guidance_scale must be a finite number >= 0")
        clip_skip = settings.get("clip_skip")
        if clip_skip is not None and (type(clip_skip) is not int or clip_skip < 0):
            raise ValueError("clip_skip must be null or a nonnegative integer")
        sampling = settings.get("sampling")
        if not isinstance(sampling, dict) or set(sampling) != {"sampler", "noise_schedule"}:
            raise ValueError("sampling must contain sampler and noise_schedule")
        if sampling["sampler"] not in {"model_default", *SAMPLERS}:
            raise ValueError(f"Unsupported SDXL sampler {sampling['sampler']!r}")
        if sampling["noise_schedule"] != "default":
            raise ValueError("Only the default noise schedule is currently supported")
        allowed = requirements.get("allowed_sampling")
        if allowed is not None and (not isinstance(allowed, list) or sampling not in allowed):
            raise ValueError(f"Sampling {sampling!r} is prohibited by this checkpoint's allowed_sampling")
        if not isinstance(requirements.get("scheduler_config", {}), dict):
            raise ValueError("requirements.scheduler_config must be an object")
        self._validate_weights(model.asset_paths["weights"], "Checkpoint", optional=True)
        if model.asset_paths["weights"] is None and not model.defaults.get("base_model_id"):
            raise ValueError("A pretrained SDXL model requires base_model_id")
        names, identifiers = set(), set()
        for selection in request.loras:
            asset = selection.asset
            if asset.family != "sdxl" or asset.kind != "lora":
                raise ValueError(f"LoRA {asset.id!r} does not declare SDXL compatibility")
            variant = asset.defaults["compatibility"].get("model_variant")
            model_variant = model.defaults.get("model_variant")
            if variant and model_variant and variant != model_variant:
                raise ValueError(f"LoRA {asset.id!r} targets a different model variant")
            if asset.id in identifiers or selection.adapter_name in names:
                raise ValueError("Duplicate LoRA IDs or adapter names")
            identifiers.add(asset.id)
            names.add(selection.adapter_name)
            if not re.fullmatch(r"[A-Za-z0-9_-]+", selection.adapter_name):
                raise ValueError("Adapter names may contain only letters, digits, underscores, and hyphens")
            if not _nonnegative_number(selection.strength):
                raise ValueError(f"LoRA {asset.id!r} strength must be a finite number >= 0")
            self._validate_weights(asset.asset_paths["weights"], f"LoRA {asset.id!r}")

    def get_ui_definition(self, model, mode):
        from .ui import UIField, common_fields, definition, sampling_fields

        return definition(model, mode, workflow="sdxl", supported_modes=self.ui_modes, fields=(
            *common_fields(model, dimension_base=8),
            *sampling_fields(model, ("model_default", *SAMPLERS)),
            UIField("clip_skip", "Clip skip", "integer", "Advanced", minimum=0, nullable=True,
                    help="Leave blank to use the pipeline default."),
        ))

    @staticmethod
    def _validate_weights(path, label, *, optional=False):
        if path is None and optional:
            return
        if path is None or not path.is_file():
            raise ValueError(f"{label} weights not found: {path}")
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"{label} weights must be a .safetensors file")

    def _load_runtime(self):
        if self._torch is None:
            self._torch = import_module("torch")
        if self.device.startswith("cuda"):
            if not self._torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Run on the configured GPU host, or explicitly use --device cpu.")
            if self.dtype == "bfloat16" and not self._torch.cuda.is_bf16_supported():
                raise RuntimeError("The selected CUDA device does not support bfloat16")
        if self._diffusers is None:
            self._diffusers = import_module("diffusers")

    def _load_pipeline(self, request):
        path = request.model.asset_paths["weights"]
        fingerprint = None
        if path is not None:
            stat = path.stat()
            fingerprint = (str(path), stat.st_size, stat.st_mtime_ns)
        key = (request.model.id, request.model.defaults.get("base_model_id"), fingerprint)
        if self._pipeline is not None and self._model_key == key:
            return
        self.close()
        options = {
            "dtype": getattr(self._torch, self.dtype), "use_safetensors": True,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            options["cache_dir"] = str(self.cache_dir)
        if self._token:
            options["token"] = self._token
        pipeline_class = self._diffusers.StableDiffusionXLPipeline
        if path is None:
            if self.dtype == "float16":
                options["variant"] = "fp16"
            self._pipeline = pipeline_class.from_pretrained(request.model.defaults["base_model_id"], **options)
        else:
            if request.model.defaults.get("base_model_id"):
                options["config"] = request.model.defaults["base_model_id"]
            self._pipeline = pipeline_class.from_single_file(str(path), **options)
        self._pipeline.to(self.device)
        self._scheduler_class = type(self._pipeline.scheduler)
        self._scheduler_config = deepcopy(dict(self._pipeline.scheduler.config))
        self._model_key = key

    def _configure_scheduler(self, request):
        sampler = request.settings["sampling"]["sampler"]
        scheduler_class = self._scheduler_class if sampler == "model_default" else getattr(self._diffusers, SAMPLERS[sampler])
        options = {}
        if sampler == "dpmpp_2m_sde":
            options.update(algorithm_type="sde-dpmsolver++", solver_order=2)
        options.update(request.model.defaults["requirements"].get("scheduler_config", {}))
        # Start from the loader's config, never the previous job's scheduler.
        self._pipeline.scheduler = scheduler_class.from_config(self._scheduler_config, **options)

    def _configure_loras(self, request):
        if self._has_adapters:
            self._pipeline.unload_lora_weights()
            self._has_adapters = False
        for selection in request.loras:
            self._has_adapters = True  # Failed loading may already have modified modules.
            path = selection.asset.asset_paths["weights"]
            try:
                self._pipeline.load_lora_weights(
                    str(path.parent), weight_name=path.name,
                    adapter_name=selection.adapter_name, local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(f"Could not load LoRA {selection.asset.id!r}; verify checkpoint and loader compatibility: {exc}") from exc
        if request.loras:
            self._pipeline.set_adapters(
                [selection.adapter_name for selection in request.loras],
                adapter_weights=[float(selection.strength) for selection in request.loras],
            )

    def generate(self, request: GenerationRequest) -> GeneratedBatch:
        self.validate(request)
        try:
            self._load_runtime()
            self._load_pipeline(request)
            self._configure_scheduler(request)
            self._configure_loras(request)
            settings = request.settings
            seed = settings["seed"]
            seeds = [
                secrets.randbelow(2**63) if seed is None else (seed + index) % (2**63)
                for index in range(settings["num_images_per_prompt"])
            ]
            generators = [self._torch.Generator(device=self.device).manual_seed(value) for value in seeds]
            arguments = {name: settings[name] for name in (
                "width", "height", "num_inference_steps", "guidance_scale", "num_images_per_prompt",
            )}
            if settings["clip_skip"] is not None:
                arguments["clip_skip"] = settings["clip_skip"]
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            start = perf_counter()
            with self._torch.inference_mode():
                images = tuple(self._pipeline(
                    prompt=request.prompt, negative_prompt=request.negative_prompt,
                    generator=generators, **arguments,
                ).images)
            if self.device.startswith("cuda"):
                self._torch.cuda.synchronize(self.device)
            seconds = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The pipeline returned an unexpected number of images")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(images, {
                **request_snapshot(request), "seeds": seeds,
                "scheduler": {
                    "class_name": type(self._pipeline.scheduler).__name__,
                    "config": scheduler_config_snapshot(dict(self._pipeline.scheduler.config)),
                },
                "runtime": {
                    "device": self.device, "dtype": self.dtype,
                    "torch_version": self._torch.__version__,
                    "diffusers_version": self._diffusers.__version__,
                    "generation_seconds": seconds,
                },
            })
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._pipeline = None
        self._model_key = None
        self._scheduler_class = None
        self._scheduler_config = None
        self._has_adapters = False
        gc.collect()
        if self._torch is not None and self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
