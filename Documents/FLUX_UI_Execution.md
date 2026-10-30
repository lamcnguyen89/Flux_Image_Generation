# FLUX image generation in the web UI

FLUX.1-dev uses the same form layout, generation queue, live job page, image
gallery, and download links as SDXL. Choose a FLUX checkpoint in the catalog,
open **Configure generation**, enter a prompt, and click **Generate images**.

Start the web server and worker from the project root:

```bash
bash scripts/start.sh
```

The worker uses each workflow's default precision (bfloat16 for FLUX on CUDA).
Runtime, cache, checkpoint, and component requirements are described in the
[FLUX backend guide](FLUX_Workflow.md). Opening or previewing a form does not
load model libraries or require checkpoint weights. Generation validates the
selected weights before queueing the request.

## Controls and results

Shared controls include prompts, dimensions, image count, steps, seed, sampling,
multiple LoRAs with individual strengths, and saved model/adapter settings.
FLUX adds true CFG scale and a maximum prompt sequence length constrained by
the checkpoint manifest. The negative prompt takes effect when true CFG is
above 1 and the negative prompt is nonempty; an inline hint shows its status.

Generation snapshots the selected values and adapters without saving changes
to model or adapter settings. **Preview request**, **Save model settings**, and
adapter save/reset actions remain available separately. Repeating the same
submission returns the original job; conflicting changes to a repeated
submission are rejected.

The job page shows runtime/model loading, checkpoint reuse or switching, LoRA
configuration, denoising steps, decoding, and saving. Jobs survive page refreshes
and appear under **Generation jobs**. Failures display the error and release
the pipeline before the next job. Switching adapters restores parameters
overwritten by previous FLUX adapters.

Images and matching JSON metadata are saved under `Images/<job_id>/` by default,
using `flux-seed-<seed>-<uuid>` filenames. The gallery supports full image viewing,
image downloads, and metadata downloads. Metadata includes the actual seed,
resolved prompts/settings, selected LoRAs, true CFG status, scheduler, and runtime
details. Configure `JOB_OUTPUT_ROOT` or use the launcher's `--output-dir` to
select another image root.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_flux_execution.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_sdxl*.py' -v
```

FLUX integration tests use the real form, queue, worker, workflow, and PNG/JSON
saving with synthetic model runtimes. They cover progress while inference is
still running, pipeline reuse, adapter reset, checkpoint switching, failures,
request validation, duplicate submissions, downloads, and output isolation.

For a GPU check, start with one small image, a fixed seed, and no adapters.
Confirm progress and downloads, then generate with a LoRA, clear the selection,
and generate again to verify adapter reset and checkpoint reuse. Change
checkpoints to verify switching. Real GPU generation requires the configured
local weights and pipeline components.
