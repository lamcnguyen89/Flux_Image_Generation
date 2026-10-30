# Implementing the WAN generation UI

This plan connects the existing WAN 2.2 I2V-A14B `img2vid` workflow to the web
interface for upload, queued execution, progress, and playable video results.

## What is already implemented

- The WAN form includes a source-image picker/preview, prompts, dimensions,
  aspect-ratio handling, frame count, FPS, video count, steps, two guidance
  scales, seed, VAE tiling, and grouped high-noise/low-noise LoRA controls.
- Form and workflow validation enforce frame counts, guidance compatibility,
  dimensions, and explicit adapter expert targets.
- The backend loads/reuses pipelines, switches checkpoints, unloads previous
  adapters, targets each expert, and streams video variations one at a time.
- The shared queue freezes requests, fingerprints input files, owns the worker
  lifecycle, persists progress/history, and records image/video outputs. The
  launcher already starts the family-independent worker.
- Output saving produces MP4, conditioning PNG, and JSON metadata. Job pages
  already contain a video player; output routes support registered video files.

The current web route enables submission only for SDXL. WAN preview currently
retains an uploaded filename, not the file itself, so it cannot produce a durable
executable image-to-video request.

## 1. Persist source-image uploads

Build a shared upload-storage helper for WAN and later MiniMax:

- Validate the actual image content, enforce upload/decoded-image limits, and
  preserve the original image bytes. The workflow already handles EXIF
  orientation, RGB conversion, resizing, and cropping.
- Save the source under a server-owned immutable path, for example
  `data/inputs/<submission_key>/source.<detected-extension>`.
- Use a stable path across retries of the same submission. A new upload path on
  every POST would change the frozen request and break submission deduplication.
- Reject conflicting retries without overwriting an existing job's source.
  Clean up unreferenced staged files on validation/enqueue failure and retain
  sources referenced by queued/running jobs across restarts.
- Serve source previews only through a job-associated route confined to managed
  input storage. Never use the client filename as a filesystem destination.

This is the largest new shared feature. The queue already supports absolute
input paths and checks input fingerprints before executing a job.

## 2. Connect WAN submission

- Enable `wan` + `img2vid` submission in `web/generation_routes.py`.
- Pass the persisted image as `inputs={"input_image": source_path}` to
  `prepare_generation()`, then enqueue through the existing job store.
- Reuse CSRF checks, frozen settings/LoRA selections, submission deduplication,
  validation errors, and the redirect to `/jobs/<job_id>`.
- Update image-specific form copy to show **Generate video** and appropriate
  count labels. Preserve source selection/previews through validation feedback
  using managed uploads, or clearly request reselection where necessary.
- Reuse current WAN validation and LoRA eligibility rules. Keep GPU imports and
  model execution in the worker.

The current checkpoint manifest declares the supported variant and
`Wan-AI/Wan2.2-I2V-A14B-Diffusers` base model. Its local weights are unset, so
execution uses cached/downloaded pretrained components. Local weights must be
a complete Diffusers pipeline containing both experts.

## 3. Add WAN execution progress

Extend `workflows/wan.py` and shared `TransformerWorkflow` progress reporting:

- Report runtime/model loading, checkpoint reuse/switching, image conditioning,
  adapter reset/loading, denoising, decoding, and video encoding/saving.
- Use the installed WAN pipeline's step-end callback for actual denoising step
  counts, preserving its callback tensor dictionary.
- Show the active video variation and step count separately. Account for
  already-saved outputs when displaying overall completion.
- Reuse worker-side progress throttling and persisted execution history.
- Verify that clearing/changing selections removes adapters from both experts
  before the next job; retain shared pipeline cleanup after failures.

Model loading, decoding, and encoding should use stage indicators until those
operations expose meaningful counts. Denoising steps do not represent a
whole-job time estimate. Exercise shared changes with FLUX regression tests.

## 4. Complete video results

Extend the existing job payload, template, and polling script to show:

- Playable MP4 results as each variation is saved, with seeking and downloads.
- The uploaded source and exact resized/cropped conditioning image.
- Actual output dimensions, frame count, FPS, duration, seed, and metadata.
  Requested dimensions can differ from actual dimensions when preserving aspect
  ratio, so read actual values from saved output metadata.
- Previously saved videos when a later variation fails, alongside the error.

Use the existing configured job-output root initially: currently
`Images/<job_id>/` for queued jobs, including MP4/conditioning PNG/JSON files.
The standalone WAN runner separately defaults to `Videos/`. A dedicated video
output root would require coordinated worker and serving configuration.

## Suggested implementation order and checks

1. **Uploads and submission:** submit an actual multipart form to the queue and
   execute a fake video workflow. Verify input persistence, invalid-image
   feedback, stable retries, conflicting retries, and cleanup after failures.
2. **WAN execution and progress:** use the real WAN workflow with a mocked
   Diffusers pipeline. Observe progress mid-step; verify reuse, switching,
   expert-specific strengths/reset, streamed variations, and recovery after
   inference/export failure.
3. **Results and browser behavior:** produce a small real MP4 from synthetic
   frames without a GPU. Verify browser playback, byte-range seeking,
   downloads, source/conditioning links, actual dimensions, and partial results.
   Check that output/input routes reject paths outside their managed roots.
4. **GPU validation:** run `bash scripts/start.sh --device cuda:0`, keeping WAN's
   default BF16 precision unless explicitly configured otherwise. Start with
   one video, a known source, no adapters, the smallest model-valid dimensions
   and frame count, and a small step count. Repeat with both expert adapter
   groups, then clear adapters and switch checkpoints on the same worker.

The first reviewable milestone is **source upload -> queued fake video job ->
playable saved MP4**. It exercises the new input lifecycle and the existing
shared execution path before involving large model weights.

Saving/resetting checkpoint settings can follow through a shared settings
persistence feature across families; generation submission itself continues to
freeze request settings without changing model/user manifests.
