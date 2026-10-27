# Image Generator

Local image and video generation with a read-only Flask model catalog.
Model assets and experiment notebooks live in `Models/Model_Families/`.
See the [model folder layout](Models/Model_Families/README.md) for the checkpoint
and LoRA folders, SDXL/FLUX/WAN configurations, and reusable JSON examples.

The application scaffold is in `src/image_generator/`. Each model family workflow
is currently a TODO placeholder; generation is still explored in the notebooks.

Catalog discovery, manifest validation, and settings resolution are implemented
without GPU dependencies. Run `python scripts/inspect_catalog.py` to list models
and LoRAs, or add `--json` to inspect their effective settings. See the
[catalog guide](Documents/Catalog.md) for the Python API, override rules, and tests.

Run `.venv/bin/python main.py` and open <http://127.0.0.1:5000> to browse the
catalog and inspect each model's resolved settings and linked LoRAs. See the
[web interface guide](Documents/Web_Interface.md) for configuration and tests.

See [environment setup](Documents/Setup.md) for the shared requirements, project
virtual environment, GPU dependency setup, and validation commands.
