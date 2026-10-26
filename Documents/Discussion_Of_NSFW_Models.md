# What are the main base models used for generating NSFW images?

As of October 2026, Stable Diffusion XL (SDXL) and its derivatives remain among the most widely used foundations for locally generating NSFW images, although newer architectures such as FLUX, Z-Image, and Chroma are becoming increasingly relevant.

An important distinction: many popular NSFW models aren't original foundation models. They're specialized fine-tunes of larger models.

## Main model families

| Model family         | Primary strengths                                        | NSFW ecosystem        |
| -------------------- | -------------------------------------------------------- | --------------------- |
| Stable Diffusion 1.5 | Lightweight, mature, photorealistic and anime fine-tunes | Very extensive, older |
| SDXL 1.0             | High-resolution realism, anatomy, flexibility            | Extremely extensive   |
| Pony Diffusion V6 XL | Anime, stylized characters, anatomy, poses               | Extremely extensive   |
| Illustrious XL       | Anime, illustration, detailed characters                 | Extremely extensive   |
| NoobAI XL            | Anime, detailed characters, complex compositions         | Extensive             |
| FLUX.1 Dev           | Photorealism, lighting, prompt adherence                 | Growing, more limited |
| Chroma               | FLUX-derived photorealism, customization                 | Growing               |
| Z-Image              | Efficient, modern photorealistic generation              | Emerging              |
| Qwen-Image           | Prompt understanding, detailed compositions              | Emerging              |

Pony, Illustrious, and NoobAI all belong to the broader SDXL architectural family, although they have substantially different training and LoRA ecosystems.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://andrewaltimit.github.io\&sz=32)

Andrews Notebook

+2



## Popular NSFW-specific checkpoints

### Photorealistic images

- Lustify SDXL: Designed specifically for realistic adult imagery and anatomy.
- CyberRealistic Pony: A photorealistic derivative of the Pony ecosystem.
- BigLust: Another adult-oriented SDXL fine-tune.
- Juggernaut XL: A general photorealistic model often customized with adult-oriented LoRAs.
- RealVisXL: Strong realistic image generation, with optional specialized fine-tunes.

### Anime and illustrated images

- Pony Diffusion V6 XL: One of the most established choices, with an especially large specialized LoRA ecosystem.
- WAI-NSFW-Illustrious: An adult-oriented Illustrious derivative.
- NoobAI XL: Popular for detailed anime illustrations.
- Illustrious XL: A major foundation for newer anime-specific fine-tunes.

These are frequently recommended in community guides, although popularity rankings vary by platform.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2



## Which would I choose for an NVIDIA Jetson AGX Thor?

Given your interest in running generative models locally on the AGX Thor, I'd divide them into three categories:

| Goal                                 | Recommended starting point         |
| ------------------------------------ | ---------------------------------- |
| Maximum NSFW ecosystem compatibility | Pony V6 XL                         |
| High-quality anime                   | WAI Illustrious / NoobAI           |
| Photorealistic adult images          | Lustify SDXL / CyberRealistic Pony |
| Experiment with newer architectures  | Chroma / Z-Image                   |
| Advanced general photorealism        | FLUX.1 Dev                         |

The AGX Thor's 128 GB of unified memory makes it particularly interesting for exploring larger diffusion transformers. However, its ARM64 architecture, JetPack software stack, and available CUDA kernels can complicate installation compared with an x86 NVIDIA workstation.

One important observation: The newest or largest foundation model is not necessarily the best NSFW model. SDXL-based checkpoints can outperform much larger general-purpose models on specialized adult imagery because of their training datasets and mature fine-tuning ecosystem.

For exploring the ecosystem, I'd start with [Civitai](https://civitai.com/?utm_source=chatgpt.com) and [Hugging Face Models](https://huggingface.co/models?utm_source=chatgpt.com).

If your primary interest is photorealistic NSFW generation rather than anime, I would focus first on Lustify SDXL, CyberRealistic Pony, and newer Chroma-based fine-tunes. Those are more directly relevant than the anime-focused Illustrious ecosystem.

# I was just going to build tiny applications for each of the main models used for generating NSFW. Would a flask app be a good container for this?

Yes. Flask is an excellent choice for building small, self-hosted image-generation applications, especially if your goal is to create a separate interface for each major NSFW-capable model family.

Given your experience with Flask, Jetson hardware, and Python-based AI systems, you could build something much simpler than ComfyUI or Automatic1111 while retaining direct control over inference.

I would recommend using Flask as the web interface and API layer, with the actual image-generation models running in separate inference modules.

## 1. Recommended architecture

Browser interface

Prompt · Negative prompt · Seed · Resolution · Generate

HTTP / JSON

Flask Application

Routes · Validation · Job queue · Image gallery

SDXL / Pony

Diffusers pipeline

Illustrious / NoobAI

SDXL pipeline

FLUX / Chroma

Transformer pipeline

Z-Image

Model-specific pipeline

GPU inference / Saved images

Jetson AGX Thor · CUDA · Local storage

Flask handles requests, but PyTorch and the model's inference pipeline do the actual generation.

You don't need Flask running inside a container. You can run the whole application directly on JetPack Linux or package it in a Docker container later.

## 2. One Flask app per model, or one universal app?

There are two reasonable approaches.

| Architecture                        | Advantages                                                    | Disadvantages                              |
| ----------------------------------- | ------------------------------------------------------------- | ------------------------------------------ |
| Separate Flask app per model family | Isolated dependencies, easy debugging, independent deployment | Duplicate interfaces and code              |
| One Flask app with model adapters   | Shared interface, reusable code, easier model switching       | More abstraction and dependency management |

For your use case, I would start with one reusable Flask interface and separate model adapters.

A directory structure could look like this:

```
image-lab/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── gallery/
├── engines/
│   ├── base.py
│   ├── sdxl.py
│   ├── flux.py
│   ├── zimage.py
│   └── qwen_image.py
├── models/
│   ├── sdxl/
│   ├── pony/
│   ├── illustrious/
│   └── flux/
└── outputs/
```

You could use the same SDXL engine for Pony, Illustrious, NoobAI, and other compatible checkpoints. That eliminates a lot of unnecessary development.

## 3. Suggested minimal interface

Local Image Lab

Prototype

Model familySDXL / Pony / Illustrious

Prompt

Negative prompt

Inference steps

30

Output resolution

512px

768px

1024px

&#x20;Copy generation settingsThis is a UI mockup, not a connected image-generation service. Model-specific parameters can be added to the real application.

## 4. Important implementation details for the AGX Thor

I'd prioritize four things:

- Load models on demand. Don't keep multiple large diffusion pipelines in memory simultaneously.
- Run generation in a worker process. Flask should enqueue a task and immediately return a job ID; the browser can poll for completion.
- Use per-family Python environments or containers when necessary. Incompatible PyTorch, Transformers, or attention-library requirements can otherwise become troublesome.
- Save generation metadata. Keep the model, checkpoint, LoRAs, seed, sampler, steps, and prompt alongside every output.

For JetPack 7 on Thor, NVIDIA's newer multi-architecture CUDA and PyTorch containers are a better starting point than older Orin-specific L4T containers. Make sure the PyTorch environment supports Thor's `sm_110` GPU architecture.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 5. Flask versus ComfyUI

There's also a third option worth considering: Flask as your custom frontend, with ComfyUI acting as an inference backend.

That gives you:

- Your own minimal interface.
- Access to ComfyUI's model loaders, schedulers, and optimized workflows.
- Less custom code for diffusion inference.
- A path to advanced models without maintaining every pipeline yourself.

The downside is the extra complexity of installing and maintaining ComfyUI on ARM64.

My recommendation: Build your first application around SDXL using Flask and Hugging Face Diffusers. Once that's stable on the Thor, try FLUX or Z-Image using the same frontend.

That will help you determine whether you actually need separate applications. You may discover that one small Flask application with interchangeable inference backends can support nearly your entire local image-generation collection.

# Is Wan 2.2 the only local model available for generating adult video?

No. Wan 2.2 is not the only locally runnable model capable of generating adult video. However, it is one of the strongest choices because of its image-to-video capabilities, realistic motion, and extensive community ecosystem of NSFW-specific LoRAs and fine-tunes.

As of October 2026, several other open-weight model families are worth considering.

## Main local video-generation models

| Model              | Adult-content ecosystem   | Primary strengths                            |
| ------------------ | ------------------------- | -------------------------------------------- |
| Wan 2.2            | Very extensive            | Realistic motion, image-to-video, NSFW LoRAs |
| Wan 2.1            | Extensive                 | Mature workflows, established fine-tunes     |
| HunyuanVideo / 1.5 | Some community fine-tunes | Photorealism, cinematic movement             |
| LTX-2 / 2.3        | Smaller, developing       | Fast generation, synchronized audio/video    |
| CogVideoX          | Limited                   | Text-to-video, smaller models                |
| Mochi 1            | Limited                   | Natural motion, detailed scenes              |
| SkyReels V2        | Smaller ecosystem         | Human motion, cinematic scenes               |
| AnimateDiff        | Extensive SD ecosystem    | Animating SD1.5/SDXL-generated images        |

These are all locally runnable model families, but adult-content capability varies substantially by checkpoint and inference pipeline.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+2



## Which are most relevant for NSFW generation?

[Alibaba Wan 2.2: Open AI for cinematic video creation](https://images.openai.com/static-rsc-4/vGxBi9HzUCn7YzYmACJTcfS-wjdJn0qocpGcp-4jGZoiC5Sv9e46JgLM0Mc574y-kjRtSiyCKpJE9cIvLICew2QWGCEG0Geg504Nbrlu50jJZrvNsTlMQMp-yCBkgJA6jRGQWTWFFtrXC_e2LFadvGxOeg5I90LbJBcP1JU69l4?purpose=inline)

1\. Wan 2.2 — Best starting point

Strong adult-oriented LoRA ecosystem, both text-to-video and image-to-video workflows, and substantial community support. [Official repository](https://github.com/Wan-Video/Wan2.2?utm_source=chatgpt.com)

[HunyuanVideo 1.5: A leading lightweight video generation model](https://images.openai.com/static-rsc-4/lL93hCZo_-Y1p_ku2WHI8f0qvsEyh_hnob2LNYMBT3_eGxbVKVn7loyg-Ax8InYpNR41smUV8TUCprTA-RazOIIMFOydQvcg_ChbldTJQwLj0jbAOGE9tTRJAuUJw8ve4up7RsiRGadm4pjv3qdMp5AZHIkJIrt1TPCpZGi-iz0?purpose=inline)

2\. HunyuanVideo — Alternative for realism

Interesting for photorealistic characters and motion. The NSFW-focused ecosystem is less developed than Wan's. [Official repository](https://github.com/Tencent-Hunyuan/HunyuanVideo?utm_source=chatgpt.com)

[LTX-2 AI Video Examples (Prompts + Settings) | MaxVideoAI](https://images.openai.com/static-rsc-4/zk9YSUObRjtoKuBCfmNJ67JNZRJ9JkNWsWirGj6NbEmFoa06fz4ZH6sn8HbCDcQIqFlkqzROfoaPxCthFNg-9eTUw4-vtIBMzx72UGxx_1MknlpGPbyNZ9Fvbhzof3_eFHd-oHvz-ikYFpF4PO8gR6dxmM5Gkz_n6yTwTHhDivw?purpose=inline)

3\. LTX-2.3 — Best for experimentation

Notable for native synchronized audio/video and flexible video workflows. It is primarily a general-purpose model rather than an adult-specific one. [Official repository](https://github.com/Lightricks/LTX-Video?utm_source=chatgpt.com)

## What about your Jetson AGX Thor?

Given your 128 GB unified-memory AGX Thor, I would consider these four initial experiments:

1. Wan 2.2 TI2V-5B: Smaller model for validating the inference environment.
2. Wan 2.2 I2V-A14B: More advanced image-to-video generation, with adult-oriented community fine-tunes.
3. HunyuanVideo 1.5: Compare realism and motion against Wan.
4. LTX-2.3: Evaluate generation speed and audio/video capabilities.

The memory capacity is promising, but ARM64 compatibility, CUDA kernels, CPU/GPU memory bandwidth, and inference optimizations still determine practical performance. Having enough memory doesn't guarantee fast inference.

## One important distinction

There are three different things people sometimes call an uncensored video model:

- General-purpose open-weight model: Can run without a hosted service's moderation layer, but may not have learned adult imagery particularly well.
- NSFW fine-tuned checkpoint: Additional training specifically improves adult-content generation.
- NSFW LoRA: A smaller adapter that modifies a compatible base model's output.

This matters because a heavily fine-tuned Wan 2.2 checkpoint could perform substantially better at adult content than a larger, newer general-purpose model.

For your plan to build lightweight Flask applications, I'd prioritize Wan 2.2, HunyuanVideo, and LTX-2.3. They represent three interesting architectures and ecosystems without requiring you to maintain a dozen nearly identical applications.

A particularly useful experiment would be to generate the same non-explicit reference image into a short video with all three models, then compare motion consistency, anatomy, temporal coherence, generation time, and memory consumption.

# Besides flask to make code easy to use by an end user, what is a good way to create a self-contained app for generating adult content which contains the models and LORas?

If your goal is to build a self-contained adult AI image/video generator that an end user can install and run without knowing Python, CUDA, PyTorch, or ComfyUI, I would consider three main approaches:

1. PySide6 + PyInstaller — best for a native Python desktop application.
2. Tauri + Python inference backend — best for a polished, lightweight desktop product.
3. Docker + a web interface — best for deploying a complete AI environment across Linux machines, especially your Jetson AGX Thor.

You can bundle the models, LoRAs, inference code, UI, and Python dependencies into a distributable application.

## 1. Comparing the options

| Approach                 | Interface          | Can bundle models and LoRAs? | Best use                        |
| ------------------------ | ------------------ | ---------------------------- | ------------------------------- |
| PySide6 + PyInstaller    | Native desktop GUI | Yes                          | Simple standalone AI apps       |
| Tauri + Python           | Modern desktop UI  | Yes                          | Commercial-quality applications |
| Docker + Flask           | Browser            | Yes                          | Linux/Jetson deployment         |
| Electron + Python        | Modern desktop UI  | Yes                          | Cross-platform apps             |
| Gradio + packaged Python | Browser            | Yes                          | Rapid prototyping               |
| ComfyUI Portable         | Node-based web UI  | Yes                          | Advanced image/video workflows  |

## 2. My first choice: PySide6 + PyInstaller

[What 4 Months of Solo SaaS Building Taught Me (The Hard Way) - DEV Community](https://images.openai.com/static-rsc-4/4SSXTVIZyRobzjndrIH3WZ4nT4u0jaMeGC3upyr2lo2zUbNM7thr9SjKdUc4RikzlLqPbhU71I8kRiO7GmhpA7ECpLlfaCEtDHdWciXQCSU024c4ZicPYsHAGs7_p3hKSQ4e9309PoVhPB9eOiYudiv1yNFQ_N8v2gkyciDsY8E?purpose=inline)

PySide6

PySide6 is the official Python binding for Qt. You can develop a native desktop application entirely in Python, with buttons, dropdown menus, progress indicators, image viewers, and video playback.

PyInstaller can then package your Python interpreter, libraries, and application code into a distribution that doesn't require users to install Python themselves.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.pyinstaller.org\&sz=32)

PyInstaller 6.22.3 documentation

+1



A typical packaged application would look like:

```
AdultStudio/
├── AdultStudio              # Launchable application
├── runtime/
│   ├── python-libraries/
│   ├── torch/
│   └── diffusers/
├── models/
│   ├── sdxl/
│   │   └── checkpoint.safetensors
│   ├── wan22/
│   │   └── model-weights/
│   └── ltx/
│       └── model-weights/
├── loras/
│   ├── sdxl/
│   └── wan22/
├── config/
└── outputs/
```

The user opens the application, chooses a model, types a prompt, and clicks Generate.

Important: I would use PyInstaller's `--onedir` mode, not `--onefile`, for large AI applications. One-file bundles extract embedded resources to a temporary folder at launch, which is especially undesirable when models occupy tens of gigabytes.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.pyinstaller.org\&sz=32)

PyInstaller 6.22.3 documentation

+1



## 3. Second choice: Tauri + Python

[GitHub - agmmnn/tauri-ui: 🦀 Create modern Tauri desktop apps in just a few simple steps with shadcn/ui. React, Next.js, Sveltekit.](https://images.openai.com/static-rsc-4/DKF-RDSHzbtOkfHYylApEprlQkS0CyWVq26SNbl0xOBQ-VlazAORiZ3f4W5gi4k-CFud4tGPvCQ44VIpn8XQQeOW0HhGl7DY1ABy5-voo8yCfPRIe-SZF0ovcWm8LUs7FwY5tPjq0JbSYFYQzVObXB-QkL78_40WRIgw6pqwa1o?purpose=inline)

Tauri

Tauri lets you create a desktop interface using HTML, CSS, and JavaScript while running Python inference code as a separate process.

This is particularly attractive if you want to commercialize your applications later.

Tauri officially supports bundling external executables, called sidecars, including Python services packaged using PyInstaller.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Its architecture could look like:

Tauri Desktop Application

Prompt editor · Model selection · LoRA controls · Output gallery

Local IPC / HTTP

Python Inference Process

PyTorch · Diffusers · Model management · GPU execution

Bundled Models

SDXL, Wan, LTX

Bundled LoRAs

Adapters by model family

The user never needs to see a terminal, run `pip install`, or launch an inference server manually.

## 4. Third choice: Docker

For your AGX Thor, Docker is especially appealing because it can package the inference environment and isolate dependency conflicts.

You could distribute:

```
AdultStudio/
├── docker-compose.yml
├── image-studio/
│   └── Dockerfile
├── video-studio/
│   └── Dockerfile
├── models/
├── loras/
└── outputs/
```

One command starts the application:

```
docker compose up -d
```

The models and LoRAs can be included in the image or shipped as a separate model package mounted into the container.

For the Thor, you need an ARM64-compatible PyTorch/CUDA container that matches JetPack 7. NVIDIA's current Jetson guidance points toward its multi-architecture CUDA/PyTorch images for Thor, rather than older L4T images intended for Orin.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Developer

+1



Docker is self-contained at the application-environment level, but the host still needs the NVIDIA GPU driver/runtime integration.

## 5. Handling large models and LoRAs

This is probably the most important design decision.

I would bundle models and LoRAs as external data directories inside the application package, rather than embedding them into the executable.

For example:

```
AdultStudio/
├── app/
├── models/
│   ├── sdxl/
│   ├── pony/
│   ├── illustrious/
│   ├── wan22/
│   └── ltx/
├── loras/
│   ├── realism/
│   ├── styles/
│   └── motion/
└── user-data/
```

Benefits include easy model replacement, independent application updates, less complicated debugging, and no need to rebuild the executable when adding a LoRA.

You should also provide a manifest that records each model's architecture, compatible LoRAs, required inference backend, checksum, license, and expected memory requirements.

When distributing an adult-content application, verify that every bundled checkpoint and LoRA permits redistribution and your intended commercial use. Also make sure the application has appropriate safeguards against nonconsensual intimate imagery and sexual content involving minors.

## 6. What I would build in your situation

Given your existing Python and Jetson experience, I'd avoid writing multiple completely independent applications.

Instead, create a single modular desktop application with multiple inference engines.

Desktop frontend: PySide6

Native Python interface without requiring a browser or local web server.

Inference: PyTorch + Diffusers / model-specific backends

Separate adapters for SDXL, Wan, LTX, and other architectures.

Distribution: PyInstaller `--onedir`

Package the Python runtime and required dependencies, with models beside the executable.

Storage: Local model repository

Enable offline inference and optional additional model packs.

My recommendation is to start with PySide6 rather than Tauri. You'll remain entirely in Python, reuse your existing inference code, and create a native application without adding a JavaScript/Rust build chain.

If you later need a sophisticated commercial interface, you can replace the frontend with Tauri while retaining much of your Python inference backend.

The key architectural principle is to separate the user interface, inference engine, and model files. That will allow you to evolve from a small standalone image generator into an application supporting several image and video model families without rewriting everything.

# In terms of architecture, would it be better to create a few mini apps, one for each of the popular image generation models or should I just combine them into one larger one?

I would recommend one unified application with several independent model backends, rather than building a completely separate application for every image-generation model.

However, I would not build one giant Python program either.

For your situation, the best architecture is a modular monolith: one desktop interface, one model manager, and several independently maintained inference engines.

This gives you the simplicity of a single application without tightly coupling everything together.

## 1. Comparing the three architectures

| Architecture          | Advantages                                      | Disadvantages                                          | Verdict                      |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------ | ---------------------------- |
| Separate mini apps    | Simple, isolated, independently deployable      | Duplicated UI, configuration, updates, and storage     | Good for experiments         |
| One large application | Unified interface, shared features              | Can become difficult to maintain; dependency conflicts | Avoid tightly coupled design |
| Modular monolith      | Unified UI with isolated engines, reusable code | Requires some interface design upfront                 | Best overall                 |

An important observation: SDXL, Pony, Illustrious, and many other checkpoints share essentially the same underlying inference architecture.

Creating a separate application for each would duplicate a significant amount of code without providing much benefit.

By contrast, FLUX, Wan, and LTX have substantially different model pipelines and dependencies. Those deserve separate backend modules—and potentially separate runtime processes.

## 2. Architecture I recommend

Unified Desktop Application

PySide6 interface · Prompt editor · Model browser · Image/video gallery

Shared Application Core

Job queue · Model registry · Storage · Settings · Metadata

SDXL Engine

SDXL · Pony · Illustrious

FLUX Engine

FLUX · compatible fine-tunes

Other Image Engines

Z-Image · Qwen-Image

Video Engines

Wan · LTX · Hunyuan

Model & LoRA Repository

External model directories · Versioned manifests · Local outputs

The critical distinction is that an engine is not the same as a model.

For example, one SDXL engine could load 20 different compatible checkpoints, without requiring 20 applications or 20 independent inference implementations.

## 3. How I would structure the source code

```
ImageStudio/
├── app.py
├── ui/
│   ├── main_window.py
│   ├── generation_panel.py
│   ├── model_selector.py
│   └── gallery.py
│
├── core/
│   ├── model_registry.py
│   ├── model_manager.py
│   ├── job_queue.py
│   └── storage.py
│
├── engines/
│   ├── base_engine.py
│   ├── sdxl_engine.py
│   ├── flux_engine.py
│   ├── zimage_engine.py
│   └── qwen_engine.py
│
├── models/
├── loras/
├── outputs/
└── requirements.txt
```

Each engine implements a common interface:

```
from abc import ABC, abstractmethodclass BaseEngine(ABC):    @abstractmethod    def load(self, model_path):        pass    @abstractmethod    def generate(self, prompt, **kwargs):        pass    @abstractmethod    def unload(self):        pass
```

Your model manager selects the appropriate engine. Each engine is responsible for its own inference implementation.

I would also give every engine a capability manifest describing supported parameters: negative prompts, CFG, LoRAs, image-to-image, resolutions, etc. The UI then displays only the controls that the chosen model actually supports.

This fits well with Hugging Face's current Modular Diffusers direction, which supports composable pipelines and lazy loading for architectures including SDXL, FLUX, Wan, Qwen, and Z-Image.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face

+1



## 4. Isolation matters more than the number of applications

The biggest potential problem with combining everything is not the interface. It's dependency compatibility.

For example:

- SDXL might work with one set of PyTorch and Diffusers dependencies.
- A newer FLUX model might require a more recent Transformers version.
- Another pipeline might depend on custom CUDA extensions.
- An upgrade supporting one model could break another.

My solution would be one UI process and separate inference worker processes where needed, each potentially using its own Python environment.

This also means a crashed model worker doesn't necessarily crash the entire application.

On your AGX Thor, I would initially permit only one large model to be active at a time to keep memory usage predictable.

## 5. How to build it incrementally

Phase 1 — SDXL

Build the interface, inference engine, model switcher, and LoRA loading. Support Pony and Illustrious through the same engine where compatible.

Phase 2 — New architecture

Add FLUX or Z-Image as a second engine. This is the real test of whether your abstraction works.

Phase 3 — Distribution

Package the desktop interface and compatible workers. Keep model files in replaceable directories and provide a manifest for installation.

Phase 4 — Video

Introduce Wan or LTX as a separate worker, while reusing the model manager, gallery, and job infrastructure.

## My recommendation

Build one application for users, multiple engines for developers, and isolated execution environments only where necessary.

That gives you three substantial benefits: one interface to maintain, one collection of shared functionality, and independent model architectures that can evolve without breaking the whole application.

It also preserves the option to distribute a particular engine as a tiny standalone application later. You don't lose the benefits of mini apps—you simply avoid duplicating their foundations.

For your eventual goal of commercializing software, that's a much better foundation than developing a collection of unrelated programs.