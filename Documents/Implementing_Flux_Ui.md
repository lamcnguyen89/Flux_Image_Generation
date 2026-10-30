# What will it take to implement the Flux UI to Add multiple LoRAs, save/reset settings?

Most of the FLUX form and multiple-LoRA support already exists. The main missing piece is persistent settings editing. This can be completed independently of the queued-generation system.

The current form supports selecting multiple registered LoRAs, individual strengths, compatibility filtering, and optional trigger-word inclusion. The FLUX backend loads those adapters together and resets the previous adapter stack between requests.

I’d complete the feature in four parts.

1. Finish the LoRA editor

Give each configured adapter an explicit row with:
- Enable/disable and remove controls.
- Individual strength.
- “Use adapter default” versus a checkpoint-specific strength override.
- Trigger words and compatibility information.
- An explicit Apply recommended settings action.
Disabled adapters should retain their configured strengths when saved. The current preview parser includes only enabled adapters, so saving needs a serializer that preserves the complete configured list.

2. Add reusable settings persistence

Add a settings service and web/settings_routes.py with save/reset operations for both models and adapters.
Action	Behavior
Save model settings	Save changed generation settings, configured LoRAs, and checkpoint-specific strengths to model.user.json.
Reset model settings	Clear model overrides and reload inherited defaults.
Save adapter defaults	Save shared adapter preferences, such as strength and trigger words, to lora.user.json.
Reset adapter defaults	Clear that adapter’s overrides.
Reload saved settings	Discard unsaved form edits without changing files.


Model-specific strengths should remain separate from shared adapter defaults. Saving a strength for one checkpoint must not unexpectedly change other checkpoints.

3. Define precise save/reset behavior

Your existing resolver already handles inheritance. The writer should save overrides relative to workflow defaults plus model defaults, rather than copying the entire resolved form into the user file.

Important details:
- Removing an override restores inheritance.
- null remains an explicit value—for example, a random seed.
- Lists replace inherited lists. Saving "loras": [] explicitly disables all inherited adapters.
- Resetting a model restores its shipped settings, including any default LoRA links.
- Preserve settings outside the editor’s scope.
- Validate before writing, replace JSON files atomically, and detect stale saves.
- Saving defaults should allow an empty prompt; generation should still require one.
- Saving settings should work without loading weights or requiring a GPU.
4. Add persistence tests

Using temporary fixture manifests, verify that:
- Multiple adapters and their strengths survive saving and refreshing.
- Disabled adapters retain their configuration.
- Model-specific changes remain isolated.
- Shared adapter defaults apply where no model override exists.
- Reset restores inherited values.
- Invalid submissions leave files unchanged.
- Nested settings, empty lists, and nullable seeds round-trip correctly.
I’d implement the settings service first, save/reset routes second, then the LoRA editor and persistence tests. Those components can serve SDXL, WAN, and MiniMax too.

Once the SDXL queue and results infrastructure exists, enabling FLUX generation through it becomes a separate integration step; save/reset functionality does not need to wait for that.