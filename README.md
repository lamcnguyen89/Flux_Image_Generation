# Image Generator

Local image and video generation with a Flask model catalog and generation forms.
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

Choose **Configure generation** to edit SDXL, FLUX, or WAN settings and preview
the typed request. For a demo without weights or a GPU, run
`IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py`.
See [generation UI testing](Documents/Generation_UI.md) for the workflow contract,
fixture pages, and automated/manual checks. SDXL also supports queue submission
and live progress; settings persistence remains a separate feature.

Catalog loading automatically creates missing model/LoRA defaults from the
family's example JSON and missing user files with empty overrides. Existing
files are preserved. New entries receive unique IDs and detected weight paths;
review template settings and supply actual compatibility metadata before generation.
The reusable `initialize_asset()` helper also supports future upload handlers;
see [manifest initialization](Documents/Catalog.md#automatic-manifest-initialization).

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

MiniMax H3 uses the same catalog, UI contract, LoRA selection, and generation
request API through its native ComfyUI adapter. Start with a validated request:

```bash
.venv/bin/python scripts/generate_minimax.py \
  --input-image Images/input.png --prompt "A gentle camera movement" --seed 42 --dry-run
```

See [MiniMax setup, generation, and tests](Documents/MiniMax_Workflow.md) for its
ComfyUI runtime, component requirements, fixture form, and silent MP4 outputs.

The shared durable queue and worker lifecycle are implemented. Start a dedicated
worker with `.venv/bin/python scripts/run_worker.py`, or test two queued fake
workflows producing small PNGs with `.venv/bin/python tests/demo_queue.py`.
See [queue usage, recovery, and tests](Documents/Worker_Queue.md). SDXL web forms
now submit jobs, show loading and denoising progress, and display saved images.
Run the Flask app and worker in separate terminals; see
[SDXL execution and GPU testing](Documents/SDXL_UI_Execution.md).
