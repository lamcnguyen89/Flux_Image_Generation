# FLUX-directory LoRA configurations

All 18 adapter folders now contain a populated `lora.defaults.json` and an initial
`lora.user.json` with `{"schema_version": 1, "settings": {}}`. This is catalog
data only: no checkpoint links were created and no notebook was changed to load
these files.

## Example JSON files

- [Default configuration](lora.defaults.example.json)
- [User overrides](lora.user.example.json)

Copy the pair into an adapter's folder and remove `.example` from the filenames.
Generate a new permanent LoRA UUID, update the asset paths and metadata, and
replace illustrative trigger words and settings with documented values. These
examples target FLUX.1-dev. Paths resolve relative to the copied JSON file.

The default example demonstrates optional generation recommendations. Use `{}`
for `recommended_generation` when recommendations are unknown. The user example
demonstrates strength, trigger-word, and step-count overrides; a fresh user file
can instead contain `{"schema_version": 1, "settings": {}}`. Recommendations
are applied explicitly, not automatically when selecting an adapter.

These templates are not registered adapters. Catalog discovery should exclude
`*.example.json` and look for `lora.defaults.json` inside asset subfolders.

## Metadata sources

Default files include permanent `lora_<UUID>` IDs, display names, source URLs and
version IDs, relative weight/notes paths, compatibility information, and editable
settings. Original source URLs are preserved from each folder's notes. Busts has
no notes or source URL, so those fields remain `null`. Where a folder has multiple
notes files, `notes_file` references the main notes file; other notes are retained.

Attempts to retrieve the CivitAI version pages and public API metadata failed
through the available browsing tools. No online recommendations were verified.
Values are therefore based on the existing local notes and safetensors headers.
Each manifest's compatibility notes record this limitation and any conflicts.
Model/version IDs identify the URL recorded in local notes; without a successful
source lookup or file-hash comparison, they do not prove a downloaded file matches
that source version.

`trained_on` preserves a recorded training checkpoint name or filename, including
opaque names such as `691639.safetensors`; these are not assumed to be downloadable
repository identifiers. Generic values such as `models` are left `null`. Training
steps, training guidance values, tag frequencies, and training image sizes were
not converted into inference settings or guessed trigger words.

## Initial settings

- The default strength is the existing FLUX notebook preset, `0.8`, unless local
  notes explicitly recommend another value. This fallback is not a claim about
  the creator's preferred strength.
- MysticHent and Mystic_Toon use their documented strength of `1.0` and expose
  the documented `35` steps under `recommended_generation`. Their DPMPP_2M/BETA
  and CFG 4 recommendations remain in the original notes: no compatible sampler
  mapping or intended guidance control was verified. They are not automatically
  translated to the checkpoint workflow's flow-matching Euler configuration.
- kera2unlocked retains the locally documented trigger words and a `20`-step
  recommendation representing the lower end of `20+`. Its strength of `0.8` is
  within the documented 0.5-0.9 range. Euler/CFG recommendations are not mapped
  while the target architecture remains unconfirmed.
- All other undocumented trigger words and recommendations are empty. Prompt
  additions are empty. Missing preview and file hash values remain `null`.
- Recommendations are optional presets for explicit application, not settings to
  silently merge when selecting multiple adapters.

## Catalog compatibility

All 18 entries are classified as `family: "flux"` and
`compatibility.model_variant: "flux_1_dev"`, as requested by the user. For Master,
Cowgirl, Dog, Missionary, and kera2unlocked, this is an explicit catalog override.
The original metadata evidence below is retained for reference; no weight files
were moved or altered.

| Folder(s) | Recorded family / variant | Evidence or remaining review |
|---|---|---|
| 69, BJ, Ball, Bkk, FluxxTechno, Glory_H, Hanfu, MysticHent, Mystic_Toon, Petite, TF | `flux` / `flux_1_dev` | Header declares `flux-1-dev/lora`. |
| Busts | `flux` / `flux_1_dev` | Header declares `Flux.1-dev/lora`; describes an extracted fcBustyCheckpoint adapter. Includes `diff_b` bias-delta tensors; loader support needs validation. No local origin URL. |
| Fluxxx | `flux` / `flux_1_dev` | Local notes describe unStable Illusion FluXXX with Flux Dev; tensors use FLUX double-block names. No embedded architecture metadata. |
| Master | `flux` / `flux_1_dev` (user override) | Tensor names are FLUX-style single-transformer blocks with 3072-wide attention, but header says `sd_1.5`. |
| Cowgirl, Dog, Missionary | `flux` / `flux_1_dev` (user override) | Original header says `zimage`; tensor names use `diffusion_model.layers` with 3840-wide attention. |
| kera2unlocked | `flux` / `flux_1_dev` (user override) | Original header has no declared architecture; source naming refers to Krea 2 and projections are 6144-wide. |

Hanfu's local source URL/version is identical to TF's despite different weight
titles and files. The source association is preserved but explicitly marked for
review rather than silently corrected.

All `verified_checkpoint_ids` lists remain empty because no loading or generation
tests were performed. The catalog classification reflects the user's requested
compatibility assignment; the original metadata findings remain in each affected
manifest's compatibility notes.

Keep the IDs when moving or renaming adapters. A checkpoint's `settings.loras`
list should contain an adapter's ID only after an explicit link; its saved
strength and the current request can override the adapter-level strength.
