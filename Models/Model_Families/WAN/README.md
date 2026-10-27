# WAN checkpoint and LoRA JSON

The pretrained WAN checkpoint entry and all six local LoRA folders have populated
default/user JSON pairs. The notebook still uses its existing Python settings;
this change only prepares catalog data. It does not load these JSON files, create
checkpoint/LoRA links, or modify weights.

## Checkpoint

`checkpoints/wan2_2_i2v_a14b/model.defaults.json` identifies the official
`Wan-AI/Wan2.2-I2V-A14B-Diffusers` repository, a complete image-to-video pipeline
with high-noise and low-noise experts. `weights: null` means to load that
pretrained repository; a future local override would need a complete compatible
Diffusers pipeline directory rather than one transformer file.

Its numerical defaults come from the current notebook: 832x480 pixels, source
aspect-ratio preservation, 161 frames, 16 fps export, 40 steps, both guidance
scales at 3.5, one video per prompt, random seeds, and VAE tiling enabled. The
negative prompt is also copied from the notebook. The default positive prompt
is empty, and the `loras` list is empty until adapters are explicitly linked.

Requirements record an input image, both `transformer` and `transformer_2`, a
dimension multiple of 16 for this variant, and the frame-count rule
`(num_frames - offset) % step == 0` with `step: 4`, `offset: 1`, and `minimum: 5`.
For example, 49, 81, and 161 frames satisfy it. This is an application-specific
constraint object, not a JSON Schema definition.

The future workflow must also check the loaded model's VAE/patch alignment and
expert boundary configuration. Both guidance scales must enable CFG together
(both above 1) or disable it together (both at or below 1), matching the current
notebook's validation. `guidance_scale` controls high noise; `guidance_scale_2`
controls low noise. The repository scheduler and expert boundary are preserved
through `sampling: {"sampler": "model_default", "noise_schedule": "default"}`
and an empty `scheduler_config`.

Playback fps, aspect-ratio fitting, and VAE tiling need workflow/export handling;
they should not all be blindly forwarded as pipeline call arguments. Input
images are job inputs, not persistent checkpoint defaults. These settings target
WAN 2.2 I2V A14B, not all WAN variants.

## LoRAs

Each `lora.defaults.json` contains a permanent ID, source/version information when
recorded locally, relative weight and notes paths, compatibility data, strength,
trigger words, optional prompt additions, and optional generation recommendations.
`compatibility.expert_target` is `high_noise`, `low_noise`, or `null` when unknown.
`compatibility.model_variant` records the intended model when established.

| Adapter folder | Variant | Expert | Initial strength | Basis |
|---|---|---|---|---|
| Assertive_Cowgirl | `wan_2_2_i2v_a14b` | `high_noise` | 1.0 | Exact training filename and local notes identify the target. Notes recommend 0.7-1.0, normally 1.0, and high-noise guidance near 6.0. |
| CS | `wan_2_2_i2v_a14b` | `null` | 1.0 | Source naming identifies WAN 2.2 I2V; tensor width supports 14B. No expert metadata or high/low filename designation. Notes recommend at least strength 1.0. |
| DT | `wan_2_2_i2v_a14b` | `null` | 0.8 | Source naming identifies WAN 2.2 I2V; tensor width supports 14B. Notes discuss both models and a low-noise dataset without identifying this file's expert. Strength uses the notebook fallback. |
| Doggy_POV | `null` | `null` | 0.8 | Notes identify 480p 14B I2V training but not this checkpoint variant or a WAN 2.2 expert. Strength uses the notebook fallback. |
| Missionary | `wan_2_2_i2v_a14b` | `high_noise` | 0.8 | Filename identifies WAN 2.2 I2V high noise. Only that file is present; strength uses the notebook fallback. |
| Mouthful | `wan_2_2_i2v_a14b` | `high_noise` | 1.0 | High-noise filename and existing I2V notebook selection; notes recommend at least strength 1.0. |

Assertive_Cowgirl's generic architecture header says `wan2.1/lora`, but the exact
training filename and notes explicitly identify WAN 2.2 I2V high noise; those
more specific fields determine the catalog entry. Doggy_POV includes image
cross-attention projection tensors; compatibility with the current two-expert
checkpoint has not been assumed. DT contains `diff_b` bias-delta tensors, so
loader support requires validation.

The base trigger for Assertive_Cowgirl and the explicit `POVdog` trigger are
copied from local notes. CS and Mouthful store the recurring leading example
token `m0u7hfu11`; this records its documented use rather than asserting that it
is mandatory. Other narrative prompt examples remain in the existing notes and
are not inserted into default prompt fields.

Only Assertive_Cowgirl currently has a native generation recommendation:
`guidance_scale: 6.0`, explicitly targeting high-noise guidance. Referenced
four/six-step wrapper examples are not assumed to apply to the current ordinary
base-model workflow. Recommended presets should only be applied explicitly.
Doggy_POV's mentioned strength of 0.4-0.5 concerns a separate companion adapter
and was not assigned to this file.

Source URLs and source IDs come from each adapter's own primary notes link;
Doggy_POV's companion URL is not used as its origin. Mouthful's notes contain no
origin URL, so its source fields are `null`. No live webpage settings were
retrieved for this change. Hashes/previews are `null`, and all verified-checkpoint
lists are empty because no loading/generation tests were performed. Unknown
expert targets must be resolved before the future workflow loads those adapters.

Each user file starts as `{"schema_version": 1, "settings": {}}`. IDs are assigned
once; retain them when renaming files or moving folders. Checkpoint links and
per-checkpoint strengths belong in the checkpoint's `settings.loras` list.

## Examples

- [Checkpoint defaults](checkpoints/model.defaults.example.json)
- [Checkpoint user overrides](checkpoints/model.user.example.json)
- [LoRA defaults](loras/lora.defaults.example.json)
- [LoRA user overrides](loras/lora.user.example.json)

Copy a pair into the appropriate asset subfolder and remove `.example` from the
filenames. Assign a new permanent model/LoRA UUID, update metadata and asset
paths, and replace illustrative settings and triggers with documented values.
The LoRA example demonstrates a high-noise adapter; do not assign that expert to
an unknown file solely from the template. Register separate files independently
when a creator supplies high- and low-noise versions.

The checkpoint user example links the example LoRA but leaves it disabled. Replace
or remove that placeholder link before use. A fresh user file can simply contain
an empty `settings` object alongside `schema_version: 1`. Examples are not
registered assets; future discovery should exclude `*.example.json` and inspect
exact default filenames inside asset subfolders.
