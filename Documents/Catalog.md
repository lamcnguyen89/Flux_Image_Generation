# Catalog discovery and settings resolution

The catalog reads model and LoRA configuration without importing Torch,
Diffusers, or Flask, opening weights, downloading models, or writing settings.
It is the data layer for the planned web interface and generation workflows.

## Inspect the catalog

From the project root, using Python 3.11 (the project interpreter is
`.venv/bin/python`):

```bash
python scripts/inspect_catalog.py
python scripts/inspect_catalog.py --json
python scripts/inspect_catalog.py --id model_a4278bb0-94f4-4960-ada0-39ca89cec194
python scripts/inspect_catalog.py --root /path/to/Model_Families
```

The default command lists names, families, and permanent IDs. `--id` prints one
asset's manifests, resolved local paths, and effective settings as JSON. `--json`
prints this information for every asset. The default catalog root is relative
to the script, so it also works when invoked by absolute path from another
directory. A custom relative `--root` is relative to the working directory.

Invalid catalog data and unknown IDs exit with status 1. Validation errors name
the source file and field; `--json` returns an `errors` list. Discovery collects
errors across files and never returns a partially valid catalog. An existing,
empty root is a valid empty catalog; a missing root is an error.

## Python API

After installing the local package, or with `PYTHONPATH=src`:

```python
from image_generator.catalog import discover_catalog, CatalogValidationError

try:
    catalog = discover_catalog("Models/Model_Families")
except CatalogValidationError as error:
    for issue in error.issues:
        print(issue.path, issue.field, issue.message)
    raise

model = catalog.get_model("model_a4278bb0-94f4-4960-ada0-39ca89cec194")
settings = model.resolve_settings()
job_settings = model.resolve_settings(request_overrides={"num_inference_steps": 40})

print(model.display_name, model.defaults["workflow"])
print(model.asset_paths["weights"])  # None for this pretrained model
print(settings["sampling"])

# LoRAs have the same asset interface and resolve their settings separately.
for lora_id, lora in catalog.loras.items():
    print(lora_id, lora.resolve_settings()["strength"])
```

`catalog.models` and `catalog.loras` index assets by permanent ID. `get_model()`,
`get_lora()`, and `get_asset()` raise `KeyError` for an unknown ID. Each asset
exposes its kind, defaults manifest, user manifest, manifest paths, and resolved
local asset paths. Treat the source manifest dictionaries as read-only;
`resolve_settings()` returns an independent tree that callers can edit.

## Discovery and validation

Discovery reads only these paths beneath the supplied root:

```text
<family>/checkpoints/<asset>/model.defaults.json
<family>/loras/<asset>/lora.defaults.json
```

Example files and unrelated JSON do not register assets. A corresponding
`model.user.json` or `lora.user.json` is optional: absence means no overrides,
and no file is created. Malformed, unreadable, or unsupported-version user
files are errors, rather than a fallback to defaults.

Validation checks schema version 1, required identity/loading fields, shared
setting types, duplicate IDs across the whole catalog, and model links to
unknown LoRA IDs (including disabled links). JSON duplicate keys and non-finite
numbers are rejected. User manifests may contain only `schema_version` and
`settings`; they cannot replace identity, requirements, compatibility, workflow,
supported modes, or weight paths.

`weights`, `notes_file`, and `preview_image` paths resolve relative to the asset
manifest. Absolute paths remain absolute. Discovery does not check that local
weights exist, hash their contents, or inspect the Hugging Face cache.
`weights: null` is valid for models with a nonempty `base_model_id`; that remote
repository identifier is preserved as metadata, not resolved as a local path.

Family and workflow names are not restricted to today's three implementations.
New metadata and settings are retained. Validation covers shared setting types;
workflow-specific numeric ranges, dimensions, scheduler rules, input requirements,
and actual LoRA compatibility still need validation when generation is implemented.
An empty verified-checkpoint list or unknown WAN expert target remains valid
catalog metadata and does not establish loading compatibility.

## Settings precedence

```text
Workflow defaults → Asset defaults → Saved user overrides → Request overrides
```

The optional workflow and request layers are supplied by callers; the current
workflow placeholders provide no defaults. The resolver merges only the
`settings` objects, never the manifest metadata:

- Objects merge recursively by key. Overriding `sampling.sampler` preserves
  `sampling.noise_schedule` from a lower-priority layer.
- Lists replace as whole values, including an empty list. LoRA links are not
  merged by ID or appended automatically.
- Scalars replace earlier values. Explicit `null` is preserved where supported,
  such as `seed` and `clip_skip`; it does not mean “remove this override.”
- Removing a saved override restores the lower-priority default on the next load.
- Resolution does not mutate the source objects or save anything to disk.

The lower-level `merge_settings(*layers)` and
`resolve_settings(defaults, user_overrides=None, *, workflow_defaults=None,
request_overrides=None)` functions are pure merges of settings dictionaries.
`CatalogAsset.resolve_settings()` also checks the effective shared setting types.

LoRA settings resolve independently of model settings. Model links remain under
`settings.loras` with their saved `enabled` and optional checkpoint-specific
`strength` values. Discovery does not create links, infer compatibility from a
family folder, insert trigger words, or apply `recommended_generation` to model
settings. The future job-preparation layer will combine enabled adapter settings
with checkpoint-specific and request-specific strengths.

## Tests

```bash
python -m unittest discover -s tests -p 'test_catalog.py' -v
python -m unittest discover -s tests -v
```

Tests use temporary manifests for merge behavior, error handling, paths, links,
and CLI output. CLI tests run outside the project with Python's `-S` option to
exclude installed packages. A separate smoke test discovers the repository's
actual manifests without loading any models.
