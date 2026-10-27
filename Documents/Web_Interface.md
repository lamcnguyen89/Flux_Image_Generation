# Read-only Flask catalog

Run from the project root after the [environment setup](Setup.md):

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

Pages support GET and HEAD. There are no editing or generation endpoints;
write requests return 405. Browsing does not load weights, import workflows,
queue jobs, or create user files. LoRA family counts are inventory counts,
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

The factory is in `src/image_generator/__init__.py`. Views live in the `catalog`
blueprint under `web/web.py`; shared templates and CSS are under `web/templates/`
and `web/static/`. These assets are included in Python package builds. Flask
imports are deferred until the factory is called, preserving the catalog
inspection script's ability to run without installed web or GPU dependencies.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_web.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Flask test-client checks cover navigation, resolved values, adapter strengths,
read-only behavior, manifest refresh, HTML escaping, error pages, static assets,
and rendering every model in the repository catalog.
