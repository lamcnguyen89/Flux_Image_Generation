"""Test-only workflows. Never register these in the application's model registry."""

from copy import deepcopy
from pathlib import Path

from image_generator.catalog import CatalogAsset
from image_generator.generation import request_snapshot
from image_generator.workflows.base import GeneratedBatch, GenerationRequest, Workflow


def make_request(root, *, workflow="fake_image", model_id="fake-model", **changes):
    settings = {"prompt": "A small test image", "negative_prompt": "", "width": 16, "height": 16,
                "num_images_per_prompt": 2, "seed": 42, "loras": [], **changes}
    directory = Path(root) / "catalog" / model_id
    model = CatalogAsset("model", directory / "model.defaults.json", directory / "model.user.json", {
        "schema_version": 1, "id": model_id, "display_name": model_id,
        "family": "test", "workflow": workflow, "weights": None, "base_model_id": "test/fake",
        "supported_modes": ["txt2img"], "requirements": {}, "settings": deepcopy(settings),
    }, {"schema_version": 1, "settings": {}})
    return GenerationRequest(model, "txt2img", deepcopy(settings), settings["prompt"], settings["negative_prompt"])


class FakeWorkflow(Workflow):
    defaults = {}

    def __init__(self, name="fake_image", *, events=None, before_generate=None, **options):
        self.name, self.events, self.before_generate = name, events if events is not None else [], before_generate
        self.options = options
        self.events.append(("create", name))
        self.closed = False

    def validate(self, request):
        if request.settings.get("failure") == "validation":
            raise ValueError("fake validation failed")
        self.events.append(("validate", request.model.id))

    def generate(self, request):
        from PIL import Image

        if self.closed:
            raise AssertionError("A closed workflow was reused")
        self.events.append(("generate", request.model.id))
        if self.before_generate:
            self.before_generate(request)
        if request.settings.get("failure") == "generation":
            raise RuntimeError("fake generation failed")
        if request.settings.get("failure") == "interrupt":
            raise KeyboardInterrupt
        count = request.settings["num_images_per_prompt"]
        seed = request.settings["seed"]
        seeds = [(42 if seed is None else seed) + index for index in range(count)]
        return GeneratedBatch(tuple(Image.new("RGB", (16, 16), (seed % 256, 20, 30)) for seed in seeds),
                              {**request_snapshot(request), "seeds": seeds})

    def close(self):
        self.events.append(("close", self.name))
        self.closed = True
