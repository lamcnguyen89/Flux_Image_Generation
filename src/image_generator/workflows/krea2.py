"""Krea 2 RAW text-to-image using the working notebook loader and named LoRAs."""

from dataclasses import replace
from contextlib import nullcontext
from time import perf_counter

from .base import GeneratedBatch
from .common import (
    TransformerWorkflow, file_fingerprint,
    image_seeds, validate_common, validate_weights,
)


class Krea2Workflow(TransformerWorkflow):
    ui_modes = ("txt2img",)
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 1024, "height": 1024, "num_images_per_prompt": 1,
        "num_inference_steps": 52, "guidance_scale": 3.5,
        "max_sequence_length": 512, "seed": None,
        "sampling": {"sampler": "flow_match_euler", "noise_schedule": "default"}, "loras": [],
    }

    def validate(self, request):
        validate_common(request, self.defaults, family="krea2", mode="txt2img",
                        count_field="num_images_per_prompt", samplers={"flow_match_euler", "model_default"})
        model, settings = request.model, request.settings
        if request.inputs:
            raise ValueError("Krea2 txt2img does not accept image inputs")
        if model.defaults.get("model_variant") != "krea_2_raw":
            raise ValueError("This Krea2 workflow supports only model_variant krea_2_raw")
        if not model.defaults.get("base_model_id"):
            raise ValueError("Krea2 requires base_model_id for its pipeline components")
        path = model.asset_paths["weights"]
        if path is not None:
            if model.defaults.get("weights_component") != "transformer":
                raise ValueError("A local Krea2 checkpoint must declare weights_component transformer")
            validate_weights(path, "Krea2 transformer")
        requirements = model.defaults["requirements"]
        if requirements.get("scheduler_class", "FlowMatchEulerDiscreteScheduler") != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Krea 2 RAW requires FlowMatchEulerDiscreteScheduler")
        limit = requirements.get("max_sequence_length_limit", 512)
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("requirements.max_sequence_length_limit must be between 1 and 512")
        length = settings["max_sequence_length"]
        if type(length) is not int or not 1 <= length <= limit:
            raise ValueError(f"max_sequence_length must be between 1 and {limit}")

    def get_ui_definition(self, model, mode):
        from .ui import UIField, common_fields, definition, sampling_fields

        limit = model.defaults["requirements"].get("max_sequence_length_limit", 512)
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("requirements.max_sequence_length_limit must be between 1 and 512")
        fields = tuple(
            replace(field, help="Krea 2 CFG: 0 disables guidance; negative prompting applies above 0.")
            if field.name == "guidance_scale" else field for field in common_fields(model)
        )
        return definition(model, mode, workflow="krea2", supported_modes=self.ui_modes, variant="krea_2_raw", fields=(
            *fields,
            *sampling_fields(model, ("flow_match_euler", "model_default")),
            UIField("max_sequence_length", "Maximum sequence length", "integer", "Advanced", minimum=1,
                    maximum=limit, step=1, help="Maximum prompt token sequence length."),
        ))

    def _build_pipeline(self, request):
        model = request.model
        options = self._load_options()
        path = model.asset_paths["weights"]
        transformer = None
        if path is not None:
            transformer = self._diffusers.Krea2Transformer2DModel.from_single_file(
                str(path), config=model.defaults["base_model_id"], subfolder="transformer", **options,
            ).to(self.device)
        cuda = self.device.startswith("cuda")
        try:
            # Match the notebook's direct CUDA loading, avoiding a second CPU
            # copy of supporting components on the shared-memory GPU host.
            if cuda:
                options["device_map"] = "cuda"
            if transformer is not None:
                options["transformer"] = transformer
            with self._torch.cuda.device(self.device) if cuda else nullcontext():
                return self._diffusers.Krea2Pipeline.from_pretrained(
                    model.defaults["base_model_id"], is_distilled=False, **options,
                )
        finally:
            del transformer

    def generate(self, request):
        self.validate(request)
        try:
            self._load_runtime()
            self._ensure_pipeline(request, file_fingerprint(request.model.asset_paths["weights"]))
            self._configure_scheduler(request, self._diffusers.FlowMatchEulerDiscreteScheduler)
            self._configure_loras(request)
            if request.loras:
                self._pipeline.set_lora_device(
                    adapter_names=[item.adapter_name for item in request.loras], device=self.device,
                )
            settings = request.settings
            seeds = image_seeds(settings, settings["num_images_per_prompt"])
            arguments = {name: settings[name] for name in (
                "width", "height", "num_inference_steps", "guidance_scale",
                "max_sequence_length", "num_images_per_prompt",
            )}
            generators = [self._torch.Generator(device=self.device).manual_seed(seed) for seed in seeds]
            steps = settings["num_inference_steps"]
            if getattr(self, "_progress_callback", None) is not None:
                def on_step_end(pipeline, step, timestep, tensors):
                    total = getattr(pipeline, "num_timesteps", steps)
                    if type(total) is not int or total <= 0:
                        total = steps
                    current = min(step + 1, total)
                    self.report_progress("denoising", current, total)
                    if current == total:
                        self.report_progress("decoding")
                    return tensors

                arguments["callback_on_step_end"] = on_step_end
                arguments["callback_on_step_end_tensor_inputs"] = []
            self.report_progress("denoising", 0, steps)
            self._synchronize()
            start = perf_counter()
            with self._torch.inference_mode():
                images = tuple(self._pipeline(
                    prompt=request.prompt, negative_prompt=request.negative_prompt, generator=generators, **arguments,
                ).images)
            self._synchronize()
            elapsed = perf_counter() - start
            if len(images) != len(seeds):
                raise RuntimeError("The Krea2 pipeline returned an unexpected image count")
            from image_generator.generation import request_snapshot

            return GeneratedBatch(images, {
                **request_snapshot(request), "seeds": seeds, "scheduler": self._scheduler_metadata(),
                "cfg_enabled": settings["guidance_scale"] > 0,
                "negative_prompt_enabled": settings["guidance_scale"] > 0 and bool(request.negative_prompt),
                "runtime": self._runtime_metadata(elapsed),
            })
        except Exception:
            self.close()
            raise
