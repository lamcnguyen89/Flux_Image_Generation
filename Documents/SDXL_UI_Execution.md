# SDXL web submission, execution, and progress

SDXL forms can now submit real jobs to the durable queue. The dedicated worker
reports loading, checkpoint reuse/switching, LoRA configuration, denoising steps,
decoding, and saving. Jobs remain accessible after refreshing or leaving the page.
FLUX, WAN, and MiniMax web forms retain their existing request-preview behavior.

## Run with the actual GPU

From the project root, restart the web app and worker to load the updated code.
Use one terminal for Flask:

```bash
.venv/bin/python main.py
```

Use another terminal for the dedicated SDXL worker:

```bash
.venv/bin/python scripts/run_worker.py --device cuda:0 --dtype float16
```

Open <http://127.0.0.1:5000/>, choose an SDXL checkpoint, and select **Configure
generation**. Enter a prompt and select **Generate images**. Start with one image
at 512×512 and 10 inference steps for a small execution check. Missing pretrained
components can download on first use, following the existing cache/token settings;
use the worker's `--local-files-only` option to require cached components.

Submission validates settings and local references, stores a frozen request, and
redirects to `/jobs/<job_id>`. Flask never loads the GPU pipeline. The worker must
be running to execute jobs; an absent/stale worker is reported on the job page.

The job page updates once per second. It displays the current stage, denoising
step count, execution history, errors, and saved outputs with full-size viewing,
downloads, actual seeds, and metadata links. **Jobs** in the navigation opens the
latest 50 jobs. Without JavaScript, **Refresh status** refreshes the server-rendered
page. Browser disconnection does not cancel a queued/running job.

## Check reuse, switching, and LoRA reset

Keep the same worker process running for these checks:

1. Generate with checkpoint A and an SDXL-compatible LoRA. The first job should
   show **Loading checkpoint** and **Loading selected LoRAs**.
2. Configure another request for A, uncheck all adapters, and generate again.
   Its history should show **Reusing loaded checkpoint**, **Clearing previous
   LoRAs**, and **No adapters selected**. Model weights remain loaded between jobs.
3. Choose another SDXL checkpoint B and submit a request. Its history should show
   **Switching checkpoint** and **Loading checkpoint**. Only one active pipeline
   is retained; returning to A after B requires loading A again.
4. Change adapter selections or strengths for another A/B request. The old adapter
   stack is unloaded before applying the new named adapters and strengths.

Saved JSON metadata includes `runtime.model_reused` and `runtime.loras_reset`.
The latter reports explicit unloading of an adapter stack from the reused
pipeline; switching checkpoints instead closes the entire previous pipeline.
Changing a checkpoint's base configuration or replacing its weight file also
invalidates the loaded pipeline cache. A failed job discards the pipeline before
the next job is attempted.

Progress during downloading/loading and VAE decoding is indeterminate. Denoising
progress uses the pipeline's actual timestep count and is a stage count, not a
whole-job ETA. Fast transitions remain visible in persisted execution history.
SQLite progress writes are throttled to five per second, with stage transitions
and final counts written immediately.

## Queue and output configuration

Flask defaults match the worker defaults:

| Flask configuration | Default | Worker option |
| --- | --- | --- |
| `JOB_DATABASE` | `data/jobs.sqlite3` | `--database` |
| `JOB_OUTPUT_ROOT` | `data/outputs` | `--output-dir` |

Environment overrides use `IMAGE_GENERATOR_JOB_DATABASE` and
`IMAGE_GENERATOR_JOB_OUTPUT_ROOT`. Point both processes at the same database and
output directory. The application serves only registered output IDs whose files
resolve inside that job's configured output directory; missing or outside files
return 404.

Generation POSTs require a same-session form token. An optional configured Flask
`SECRET_KEY` (or `IMAGE_GENERATOR_SECRET_KEY`) preserves sessions across web-app
restarts and must be shared by multiple web processes. The single-process local
default generates a secret at startup; reload an open form after restarting it.

Each form also has a unique submission key. Repeating the same submission returns
the existing job instead of starting duplicate GPU work. Reusing that key with
different values returns 409 and asks for a fresh form. Reloading the job page is
a GET and never resubmits generation. No model/user settings are saved by submission.

Job database schema version 1 upgrades automatically to version 2, preserving
jobs and outputs and adding execution history and submission deduplication.
No catalog rediscovery is needed to view existing jobs or serve their results.

## Automated checks

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_sdxl_execution.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

The integration tests submit the actual HTML form through Flask, execute the
real SDXL workflow/worker with a mocked Diffusers pipeline, and inspect progress
while inference is blocked mid-step. They verify saved PNG/JSON serving,
checkpoint reuse and switching, clearing LoRAs, recovery after inference errors,
full-width seed values, request deduplication, form-token enforcement, storage
errors, version-1 database migration, and output-path confinement.

The supplied fixture library has no real weights. Its **Preview request** action
works without CUDA; **Generate images** reports missing weights. Automated tests
provide fake local assets and a mocked runtime. Use the actual library and the
two-terminal procedure above to verify full GPU execution and real model quality.
