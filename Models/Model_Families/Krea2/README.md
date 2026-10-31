# Krea 2

`Krea2_Image_Gen.ipynb` remains the standalone working experiment. The catalog
workflow and web UI use `workflow: krea2` and `model_variant: krea_2_raw`.
See the [workflow guide](../../../Documents/Krea2_Workflow.md) and
[UI guide](../../../Documents/Krea2_UI_Execution.md).

The RAW model manifest points to the existing local transformer and the matching
`krea/Krea-2-Raw` component repository. Defaults preserve the notebook's 52 steps,
guidance 3.5, 1024×1024 dimensions, 512-token budget, seed 42, and selected
Vérité realism adapter at strength 0.8. Adapter compatibility comes from the local
notes and notebook selection; the verified-checkpoint list remains empty.
User override files start empty.

Checkpoint and LoRA example JSON pairs live directly under `checkpoints/` and
`loras/`. They provide templates for catalog initialization; example files do not
register assets. New assets receive generated IDs and detected weights. Review
their metadata and settings before generation. RAW uses full classifier-free
guidance; Turbo and RAW-to-Turbo refinement need separate implementations.
