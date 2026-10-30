# Flask catalog and generation forms

Run from the project root after the [environment setup](Setup.md):

```bash
bash scripts/start.sh
```

This starts the web server and generation worker together using `.venv`. Logs
appear in the same terminal. Ctrl+C stops both, allowing active generation to
finish. See [launcher options and GPU testing](SDXL_UI_Execution.md).

To start only the web server:

```bash
.venv/bin/python main.py
```

Open <http://127.0.0.1:5000>. The server uses Flask's local development server
with debugging off. The installed package also supports Flask's application
factory CLI:

```bash
.venv/bin/python -m flask --app image_generator:create_app run
```

## Pages

- `/` lists discovered models grouped by family, their declared modes, and
  registered LoRA counts. Choose **View settings** to inspect a model.
- `/models/<model_id>` shows model metadata, effective settings, saved linked
  LoRAs and their strengths/selections, defaults, user overrides, requirements,
  and manifest paths. Values retain JSON types: `null`, `false`, lists, and
  nested objects remain distinguishable.
- `/generate/<model_id>` renders the workflow's editable form. POST previews
  typed settings and reports errors without saving settings. SDXL forms
  submit jobs through **Generate images**; WAN forms use **Generate video**
  with a source-image upload. Both execute on the dedicated worker.
  See [generation forms](Generation_UI.md) for fixture startup and testing.
- `/jobs` lists recent jobs; `/jobs/<job_id>` displays live progress, execution
  history, errors, and saved outputs. `/api/jobs/<job_id>` supplies polling data.

The catalog and saved-settings pages support GET and HEAD; writes to those
routes return 405. Generation forms support GET, HEAD, POST previews, and SDXL/WAN
queue submissions. Browsing never loads weights or queues jobs. Catalog browsing keeps workflow
imports lazy; generation forms import only lightweight definitions and validation.
Catalog loading creates missing defaults/user JSON files using
the [initialization rules](Catalog.md#automatic-manifest-initialization), without
editing existing files. The library must be writable when manifests are missing;
initialization errors appear on the catalog diagnostic page.
LoRA family counts are inventory counts,
not claims of checkpoint compatibility. Only explicitly linked adapters appear
on model pages. A checkpoint-specific strength takes precedence over the
adapter's resolved strength; recommendations are displayed separately and
never applied to model settings.

The catalog is discovered once per request. Refresh after editing JSON files
to see the latest saved configuration. Unknown models return 404. Invalid
catalog configuration returns a diagnostic page with status 503, naming each
source file and field. Fix the files and refresh to recover without restarting.

## Application configuration

`create_app()` defaults to `Models/Model_Families` in the source checkout.
Supply another catalog root through application configuration:

```python
from image_generator import create_app

app = create_app({"CATALOG_ROOT": "/path/to/Model_Families"})
```

Or set the environment variable before starting the app:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=/path/to/Model_Families .venv/bin/python main.py
```

Flask's prefixed configuration loader reads `IMAGE_GENERATOR_*` variables;
explicit factory configuration takes precedence. Relative catalog roots resolve
against the working directory. When installed outside an editable checkout,
configure the path to your external model library explicitly.

The factory is in `src/image_generator/__init__.py`. Catalog views live in the
`catalog` blueprint under `web/web.py`; the `generation` blueprint is in
`web/generation_routes.py`. Shared templates, CSS, and JavaScript are under
`web/templates/` and `web/static/`. These assets are included in Python package builds. Flask
imports are deferred until the factory is called, preserving the catalog
inspection script's ability to run without installed web or GPU dependencies.

The `jobs` blueprint lives in `web/job_routes.py`. Database/output configuration,
worker startup, and real GPU testing are described in [SDXL execution](SDXL_UI_Execution.md).
See [WAN execution](WAN_UI_Execution.md) for source storage, video playback,
upload limits, and WAN-specific GPU checks.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_web.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Flask test-client checks cover navigation, resolved values, adapter strengths,
read-only behavior, manifest refresh, HTML escaping, error pages, static assets,
and rendering every model in the repository catalog.
