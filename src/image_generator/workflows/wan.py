"""WAN 2.2 I2V-A14B with explicit expert targeting and streamed variations."""

import hashlib
from io import BytesIO
import math
import subprocess
from time import perf_counter

from .base import GeneratedBatch
from .common import TransformerWorkflow, _nonnegative_number, file_fingerprint, image_seeds, validate_common


def check_video_encoder():
    from imageio_ffmpeg import get_ffmpeg_exe

    try:
        subprocess.run([get_ffmpeg_exe(), "-version"], capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"MP4 encoder is unavailable: {exc}") from exc


class WanWorkflow(TransformerWorkflow):
    defaults = {
        "prompt": "", "negative_prompt": "", "positive_prefix": "", "positive_suffix": "",
        "trigger_words": [], "width": 832, "height": 480, "preserve_aspect_ratio": True,
        "num_frames": 161, "fps": 16, "num_videos_per_prompt": 1, "num_inference_steps": 40,
        "guidance_scale": 3.5, "guidance_scale_2": 3.5, "seed": None, "enable_vae_tiling": True,
        "sampling": {"sampler": "model_default", "noise_schedule": "default"}, "loras": [],
    }

    def validate(self, request):
        validate_common(request, self.defaults, family="wan", mode="img2vid",
                        count_field="num_videos_per_prompt", samplers={"model_default"})
        model, settings = request.model, request.settings
        if model.defaults.get("model_variant") != "wan_2_2_i2v_a14b":
            raise ValueError("This WAN workflow supports only model_variant wan_2_2_i2v_a14b")
        if set(request.inputs) != {"input_image"} or not request.inputs["input_image"].is_file():
            raise ValueError("WAN img2vid requires a local input_image file")
        path = model.asset_paths["weights"]
        if path is not None:
            if model.defaults.get("weights_component") != "pipeline" or not path.is_dir() or not (path / "model_index.json").is_file():
                raise ValueError("Local WAN weights must be a complete Diffusers pipeline directory with model_index.json")
        elif not model.defaults.get("base_model_id"):
            raise ValueError("WAN requires base_model_id or a local pipeline directory")
        guidance = settings["guidance_scale_2"]
        if not _nonnegative_number(guidance):
            raise ValueError("guidance_scale_2 must be a finite number >= 0")
        if (settings["guidance_scale"] > 1) != (guidance > 1):
            raise ValueError("Both WAN guidance scales must be > 1 for CFG, or both <= 1")
        if not _nonnegative_number(settings["fps"]) or settings["fps"] == 0:
            raise ValueError("fps must be a finite number > 0")
        for name in ("preserve_aspect_ratio", "enable_vae_tiling"):
            if type(settings[name]) is not bool:
                raise ValueError(f"{name} must be a boolean")
        rule = model.defaults["requirements"].get("frame_count", {"minimum": 5, "step": 4, "offset": 1})
        if not isinstance(rule, dict) or any(type(rule.get(key)) is not int for key in ("minimum", "step", "offset")) or rule["step"] <= 0 or rule["minimum"] < 1:
            raise ValueError("requirements.frame_count needs integer minimum, positive step, and offset")
        count = settings["num_frames"]
        if type(count) is not int or count < max(5, rule["minimum"]) or (count - 1) % 4 or (count - rule["offset"]) % rule["step"]:
            raise ValueError("num_frames must satisfy the checkpoint frame_count rule and be 4*n + 1, at least 5")
        for selection in request.loras:
            compatibility = selection.asset.defaults["compatibility"]
            if compatibility.get("model_variant") != "wan_2_2_i2v_a14b":
                raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit compatible model_variant")
            if compatibility.get("expert_target") not in {"high_noise", "low_noise"}:
                raise ValueError(f"WAN LoRA {selection.asset.id!r} needs an explicit high_noise or low_noise expert_target")

    def _build_pipeline(self, request):
        path = request.model.asset_paths["weights"]
        source = str(path) if path else request.model.defaults["base_model_id"]
        options = self._load_options()
        if path is not None:
            options["local_files_only"] = True
        vae = self._diffusers.AutoencoderKLWan.from_pretrained(
            source, subfolder="vae", **{**options, "dtype": self._torch.float32},
        )
        try:
            return self._diffusers.WanImageToVideoPipeline.from_pretrained(source, vae=vae, **options)
        finally:
            del vae

    @staticmethod
    def _pipeline_fingerprint(path):
        if path is None:
            return None
        return (str(path), tuple(file_fingerprint(item) for item in sorted(path.rglob("*"))
                                 if item.is_file() and item.suffix in {".json", ".safetensors"}))

    def _prepare_conditioning(self, source, request):
        from PIL import Image, ImageOps

        pipe = self._pipeline
        if pipe.transformer is None or pipe.transformer_2 is None:
            raise ValueError("WAN 2.2 requires both transformer and transformer_2 experts")
        boundary = pipe.config.boundary_ratio
        if not _nonnegative_number(boundary) or not 0 < boundary < 1:
            raise ValueError("WAN 2.2 requires a valid expert boundary_ratio between 0 and 1")
        for name in request.model.defaults["requirements"].get("required_components", []):
            if getattr(pipe, name, None) is None:
                raise ValueError(f"The WAN pipeline is missing required component {name!r}")
        temporal = pipe.vae_scale_factor_temporal
        spatial = pipe.vae_scale_factor_spatial
        patch = pipe.transformer.config.patch_size
        if type(spatial) is not int or spatial <= 0:
            raise ValueError("The loaded VAE requires a positive spatial scale factor")
        if type(temporal) is not int or temporal <= 0 or (request.settings["num_frames"] - 1) % temporal:
            raise ValueError("num_frames does not match the loaded VAE temporal alignment")
        if len(patch) != 3 or any(type(value) is not int or value <= 0 for value in patch):
            raise ValueError("The WAN transformer has an invalid patch size")
        if ((request.settings["num_frames"] - 1) // temporal + 1) % patch[0]:
            raise ValueError("num_frames does not match the loaded transformer temporal patch size")
        multiple = request.model.defaults["requirements"].get("dimension_multiple", 16)
        height_multiple = math.lcm(16, multiple, spatial * patch[1])
        width_multiple = math.lcm(16, multiple, spatial * patch[2])
        settings = request.settings
        height, width = settings["height"], settings["width"]
        if settings["preserve_aspect_ratio"]:
            ratio = source.height / source.width
            height = max(height_multiple, int(math.sqrt(height * width * ratio)) // height_multiple * height_multiple)
            width = max(width_multiple, int(math.sqrt(settings["height"] * settings["width"] / ratio)) // width_multiple * width_multiple)
            conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
        else:
            if height % height_multiple or width % width_multiple:
                raise ValueError(f"Loaded WAN model requires height/width multiples of {height_multiple}/{width_multiple}")
            conditioning = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
        return conditioning

    def generate(self, request):
        if request.settings["num_videos_per_prompt"] != 1:
            raise ValueError("Use generate_batches() or run_generation() to stream multiple WAN videos")
        return next(self.generate_batches(request))

    def generate_batches(self, request):
        self.validate(request)
        try:
            from PIL import Image, ImageOps

            payload = request.inputs["input_image"].read_bytes()
            with Image.open(BytesIO(payload)) as image:
                source = ImageOps.exif_transpose(image).convert("RGB")
            input_hash = hashlib.sha256(payload).hexdigest()
            del payload
            check_video_encoder()
            self._load_runtime()
            self._ensure_pipeline(request, self._pipeline_fingerprint(request.model.asset_paths["weights"]))
            conditioning = self._prepare_conditioning(source, request)
            self._configure_loras(request, targeted=True)
            tiling = self._pipeline.vae.enable_tiling if request.settings["enable_vae_tiling"] else self._pipeline.vae.disable_tiling
            tiling()
            arguments = {name: request.settings[name] for name in ("num_frames", "num_inference_steps", "guidance_scale", "guidance_scale_2")}
            arguments.update(height=conditioning.height, width=conditioning.width, num_videos_per_prompt=1, output_type="np")
            seeds = image_seeds(request.settings, request.settings["num_videos_per_prompt"])
            for index, seed in enumerate(seeds):
                self._configure_scheduler(request)
                generator = self._torch.Generator(device=self.device).manual_seed(seed)
                self._synchronize()
                start = perf_counter()
                with self._torch.inference_mode():
                    frames = self._pipeline(
                        image=conditioning, prompt=request.prompt, negative_prompt=request.negative_prompt,
                        generator=generator, **arguments,
                    ).frames[0]
                self._synchronize()
                elapsed = perf_counter() - start
                if len(frames) != request.settings["num_frames"]:
                    raise RuntimeError("The WAN pipeline returned an unexpected frame count")
                from image_generator.generation import request_snapshot

                batch = GeneratedBatch((), {
                    **request_snapshot(request), "seeds": [seed], "video_index": index,
                    "input_image": {"path": str(request.inputs["input_image"]), "sha256": input_hash,
                                    "source_size": list(source.size), "conditioning_size": list(conditioning.size)},
                    "generation_settings": arguments.copy(), "scheduler": self._scheduler_metadata(),
                    "boundary_ratio": self._pipeline.config.boundary_ratio,
                    "frame_count": len(frames), "duration_seconds": len(frames) / request.settings["fps"],
                    "runtime": {**self._runtime_metadata(elapsed), "vae_dtype": str(self._pipeline.vae.dtype)},
                }, videos=(frames,), conditioning_image=conditioning)
                yield batch
                del batch, frames
        except Exception:
            self.close()
            raise
