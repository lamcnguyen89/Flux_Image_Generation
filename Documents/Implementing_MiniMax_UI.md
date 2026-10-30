# Implementing the MiniMax generation UI

Implemented using the shared WAN upload and results infrastructure. See
[MiniMax UI operation and validation](MiniMax_UI_Execution.md) for startup,
supported controls, progress, tests, and the real GPU validation procedure.
The milestones below describe the implementation requirements.

Connect the existing MiniMax H3 `img2vid` workflow to the web form, shared queue,
and video results pages. The workflow runs native ComfyUI Python nodes inside
the generation worker; the UI does not require a separate ComfyUI server.

## Reuse the existing foundation

The repository already has:

- A MiniMax form and UI contract, with source preview, prompts, dimensions,
  frame/video counts, seed, VAE tiling, and multiple H3 LoRA strengths.
- Validation for H3 architecture, components, sampling, and adapter compatibility.
- A native backend that caches checkpoint/VAE/text encoder components, switches
  weights, rebuilds changed adapter stacks from the unpatched base, and streams
  variations through the shared MP4/PNG/JSON exporter.
- A durable queue, worker ownership and recovery, progress/history storage,
  polling, job pages, and registered video output serving.
- A single `scripts/start.sh` launcher for the web app and worker.

MiniMax now saves uploaded images for generation and queues executable requests.
Request previews continue to validate without saving uploads.

## 1. Share WAN's managed uploads and video submission

Implement the upload service described in [the WAN plan](Implementing_WAN_UI.md)
once, and use it for both families. If MiniMax is implemented first, build that
shared service as MiniMax's first milestone.

- Validate image content and upload/decoded-image limits, preserving original
  image bytes for the workflow's existing orientation and conditioning logic.
- Persist the source under an immutable, server-owned path, stable across
  retries of the same submission key.
- Retain sources referenced by queued/running jobs across web/worker restarts;
  clean up unreferenced files after failed submissions.
- Preserve existing CSRF and request-deduplication behavior, including rejecting
  conflicting retries without overwriting another job's source.
- Enable `minimax` + `img2vid` submission in the shared generation route and
  pass `inputs={"input_image": source_path}` to `prepare_generation()`.
- Queue the frozen request and redirect to its existing job page. Keep model
  loading and execution in the worker.
- Update the form to say **Generate video**, with useful source/validation
  feedback. Source preview URLs must be scoped to managed job-associated files.

## 2. Preserve H3-specific controls

Use the existing MiniMax UI definition and server-side validators:

| Setting | Supported behavior |
| --- | --- |
| Dimensions | Multiples of 32, plus stricter manifest requirements |
| Frame count | `17*n + 5`, subject to manifest constraints and the 3600-frame ceiling |
| Playback FPS | Fixed at 24 |
| Guidance | Fixed at 1 through BasicGuider |
| Negative prompt | Empty; unavailable in this implementation |
| Sampling | `res_multistep` with the `simple` schedule |
| LoRAs | Multiple explicitly H3-compatible model adapters with individual strengths |
| Audio | Silent MP4 output |

Keep fixed settings read-only and reject forged values in Python. H3 uses one
transformer, so its adapter selection uses the existing single list. When a
selection/strength changes, rebuild from the cached unpatched base; clearing
adapters returns to that base. A zero-strength adapter is skipped.

Display these model restrictions alongside the form. Generation should use the
existing compatibility declarations and validation, without automatically
marking adapters as checkpoint-verified or applying their recommended settings.

## 3. Add native ComfyUI progress

MiniMax does not use the WAN/SDXL Diffusers step callback. Extend
`workflows/minimax.py` to report:

1. Runtime preparation and text encoder resolution.
2. Checkpoint/VAE/encoder loading or cached component reuse.
3. Adapter stack reuse/reset/application.
4. Source conditioning and prompt encoding.
5. Sampling steps and the current video variation.
6. VAE decoding and video encoding/saving.

The installed ComfyUI checkout exposes
`comfy.utils.set_progress_bar_global_hook`. Its sampler's existing callback
updates `ProgressBar` with completed/total sampling steps. Bridge those numeric
updates to `Workflow.report_progress()` during sampling, then use the existing
worker throttling, queue history, and browser polling.

Capture the previous hook and restore it in `finally`, including on failure.
The hook is process-global, so scope it to the serialized worker operation and
ensure it cannot leak into the next job or model family. Preserve ComfyUI's
existing sampling callback and latent bookkeeping; do not retain tensors or
preview payloads in persisted progress. Verify the hook signature against the
installed ComfyUI revision.

Keep the existing quantized-transfer guards and `inference_mode(False)` /
`no_grad()` execution contexts intact. Loading/decoding should use stage
indicators where meaningful numeric progress is unavailable. Show variation
counts separately from sampling steps, without claiming a whole-job ETA.

## 4. Reuse the video results UI

Share WAN's result enhancements:

- Play each MP4 as it is saved, with seeking, download, and metadata links.
- Show the source and exact conditioning image.
- Display actual dimensions, frames, 24 FPS, duration, actual seed, and silent
  audio status from saved metadata. Aspect preservation can change dimensions.
- Keep completed earlier variations accessible if a later one fails.
- Confine source/output routes to the associated job's managed files.

The current queued output root is `Images/<job_id>/`, including video files.
The standalone MiniMax runner separately defaults to `Videos/`. Any dedicated
video output root needs matching worker and web-serving configuration.

## Runtime setup and GPU validation

The current manifest references an H3 checkpoint and video VAE whose local
files exist. The text encoder uses a documented Qwen3-VL repository reference;
the backend prefers a matching family-local file, then resolves the cache or
downloads during execution, subject to offline settings. Existing file presence
does not establish successful inference or adapter quality.

ComfyUI must include native H3 support. The default checkout is `build/ComfyUI`;
configuration and component details are in [the backend guide](MiniMax_Workflow.md).

```bash
bash scripts/start.sh --device cuda:0 --comfyui-dir build/ComfyUI
```

Omit `--dtype`: MiniMax's checkpoint and ComfyUI manage precision, and its
constructor rejects an explicit dtype. The current runtime supports CUDA device
0. For a worker serving mixed families, leave dtype unset so each workflow uses
its own precision policy.

Start with one video, a fixed seed, no adapters, small model-valid dimensions
and frame count, and a low step count. Then verify component reuse, an enabled
adapter, clearing adapters, checkpoint switching, and cleanup between families.

## Implementation milestones and tests

1. **Shared inputs and queue:** submit a multipart form, persist its image,
   execute a fake video workflow, and display a playable short MP4. Test invalid
   uploads, retry stability/conflicts, failed-submission cleanup, and restart
   persistence before loading large models.
2. **Native execution and progress:** run the real MiniMax workflow with mocked
   ComfyUI nodes, observe progress mid-sampling, and verify hook restoration on
   success/failure, fixed-setting validation, base/component reuse, adapter
   changes/reset, zero strengths, and worker cleanup.
3. **Results and GPU:** check browser playback and byte-range seeking with a
   small real MP4 encoded from synthetic frames; test managed source/conditioning
   links, metadata, and partial results. Then perform the real GPU checks above.

Saving/resetting checkpoint settings can use the shared persistence feature
across families later. Submitting a generation request freezes its settings
without modifying model/user manifests.
