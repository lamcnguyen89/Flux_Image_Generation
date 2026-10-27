# Model family layout

This is the first structural step toward the model catalog. Each selectable model
and each LoRA has its own folder and two JSON placeholders. Every JSON file
currently contains only `{}`; its fields and settings will be designed later.
The notebooks still use their existing Python settings and do not read these JSON
files yet.

```text
Model_Families/
├── SDXL/
│   ├── checkpoints/
│   │   ├── bigLust_v16/
│   │   │   ├── BigLust_Image_Gen.ipynb
│   │   │   ├── bigLust_v16.safetensors
│   │   │   ├── bigLust_Notes.md
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   ├── noobaiXLNAIXL_vPred10Version/
│   │   │   ├── NoobAIXL_Image_Gen.ipynb
│   │   │   ├── noobaiXLNAIXL_vPred10Version.safetensors
│   │   │   ├── noobAIXL_Notes.md
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   ├── sdxl_base_1_0/
│   │   │   ├── SDXL_Image_Gen.ipynb
│   │   │   ├── model.defaults.json
│   │   │   └── model.user.json
│   │   └── <other SDXL checkpoint folders>/
│   └── loras/
│       └── ChelXL/
│           ├── ChelXL.safetensors
│           ├── ChelXL_Notes.md
│           ├── lora.defaults.json
│           └── lora.user.json
├── FLUX/
│   ├── Flux_Image_Gen.ipynb
│   ├── checkpoints/
│   │   └── fluxNSFWUNLOCKED_v30FP16/
│   │       ├── fluxNSFWUNLOCKED_v30FP16.safetensors
│   │       ├── fluxNSFWUnlocked_Notes.md
│   │       ├── model.defaults.json
│   │       └── model.user.json
│   └── loras/
│       └── <existing LoRA folders>/
└── WAN/
    ├── WAN_Video_Gen.ipynb
    ├── checkpoints/
    │   └── wan2_2_i2v_a14b/
    │       ├── model.defaults.json
    │       └── model.user.json
    └── loras/
        └── <existing LoRA folder>/
            ├── <existing weights and notes>
            ├── lora.defaults.json
            └── lora.user.json
```

## Organization

- The architecture folders are `SDXL/`, `FLUX/`, and `WAN/`.
- SDXL contains the base model and the former BigLust, IllustriousXL, LustifyXL,
  NoobAIXL, PonyDiffusionV6XL, PyrosSDXL, juggarnautXL, and waiIllustrious models.
- Each local checkpoint and its notes live under
  `checkpoints/<checkpoint filename without its extension>/`.
- Each SDXL notebook lives alongside its own checkpoint and JSON pair. The
  FLUX and WAN notebooks retain their existing locations.
- SDXL LoRAs live under `SDXL/loras/`, including `ChelXL/`. Sharing an architecture
  folder does not establish compatibility with every checkpoint; model-specific
  compatibility will be defined when the JSON contents are designed.
- SDXL's `sdxl_base_1_0` and WAN's `wan2_2_i2v_a14b` folders represent the
  pretrained models already selected by their notebooks. These folders contain
  configuration placeholders, not downloaded model weights. Hugging Face cache
  behavior is unchanged.
- Weight filenames are preserved and remain excluded from Git.

## Notebook paths

Checkpoint paths remain relative to the architecture family's `checkpoints/`
directory and include the checkpoint subfolder. LoRA paths remain relative to
that family's `loras/`. All SDXL notebooks now use `model_family = "SDXL"`.
They find the project root from their nested working directory. Absolute asset
paths continue to work. Paths shown in the notebook descriptions and code
comments are relative to the family directory.

No generation logic or scheduler behavior was changed. In particular, NoobAI's
existing scheduler handling remains in its notebook until it is moved into a
shared SDXL workflow in a later step.

## Adding assets

Create a folder beneath the appropriate family's `checkpoints/` or `loras/`.
Keep that asset's weights, notes, and optional previews together, and add the
matching `model.defaults.json` / `model.user.json` or
`lora.defaults.json` / `lora.user.json` pair. Leave the JSON as `{}` for now.
Models consisting of multiple components can keep those components together in
one model folder.
