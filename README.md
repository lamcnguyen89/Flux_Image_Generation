# Image Generator

Local image and video generation with a read-only Flask model catalog.
Model assets and experiment notebooks live in `Models/Model_Families/`.
See the [model folder layout](Models/Model_Families/README.md) for the checkpoint
and LoRA folders, SDXL/FLUX/WAN configurations, and reusable JSON examples.

The application is in `src/image_generator/`. SDXL and FLUX text-to-image and
WAN 2.2 image-to-video generation have reusable backends and command-line runners.
The notebooks remain independent experiments.

Catalog discovery, manifest validation, and settings resolution are implemented
without GPU dependencies. Run `python scripts/inspect_catalog.py` to list models
and LoRAs, or add `--json` to inspect their effective settings. See the
[catalog guide](Documents/Catalog.md) for the Python API, override rules, and tests.

Run `.venv/bin/python main.py` and open <http://127.0.0.1:5000> to browse the
catalog and inspect each model's resolved settings and linked LoRAs. See the
[web interface guide](Documents/Web_Interface.md) for configuration and tests.

Generate one image with the catalog's base SDXL model:

```bash
.venv/bin/python scripts/generate_sdxl.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras
```

Images and matching JSON metadata are saved under `Images/`. Add `--dry-run` to
validate and print the request without loading models. See the
[SDXL backend guide](Documents/SDXL_Workflow.md) for checkpoints, schedulers,
multiple LoRAs, cache configuration, and the Python API.

Generate a FLUX image or animate a local image with WAN:

```bash
.venv/bin/python scripts/generate_flux.py \
  --prompt "A mountain lake at sunrise" --seed 42 --no-loras
.venv/bin/python scripts/generate_wan.py \
  --input-image Images/input.png --prompt "A gentle breeze moves across the lake" \
  --frames 49 --seed 42 --no-loras
```

WAN outputs MP4, conditioning PNG, and JSON files under `Videos/`, processing
variations one at a time. Both runners support `--dry-run`. See the
[FLUX guide](Documents/FLUX_Workflow.md) and [WAN guide](Documents/WAN_Workflow.md)
for family-specific settings, adapters, and Python APIs.

See [environment setup](Documents/Setup.md) for the shared requirements, project
virtual environment, GPU dependency setup, and validation commands.
