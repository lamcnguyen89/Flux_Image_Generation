# Generation form fixtures

`model_families/` contains three checkpoints (SDXL, FLUX.1-dev, WAN 2.2 I2V)
and seven LoRAs with defaults and saved user overrides. Referenced local weights
are deliberately absent. These files support rendering and request previews,
not generation. They are separate from your actual model library.

From the project root, preview all three families without GPU dependencies:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py
```

Open `http://127.0.0.1:5000/` and select **Configure generation**, or open
`/generate/fixture-sdxl`, `/generate/fixture-flux`, or `/generate/fixture-wan`.
Enter a prompt and click **Preview request**. WAN additionally requires selecting
a source image. Previews validate form settings only and do not save images,
files, overrides, or jobs. Reloading restores the fixture's saved settings.
