# Application environment setup

The application will use one project-local `.venv` for its SDXL, FLUX, and WAN
workflows. The notebooks remain independent experiments. All workflow modules in
`src/image_generator/workflows/` are TODO placeholders.

## Current status

This is an initial scaffold. No environment is created by adding these files.
Runtime pins were read from the existing `huggingface` Conda environment, except
Flask, which is a new application dependency. The baseline is Python 3.11 on Linux
aarch64 with `torch==2.14.1+cu132` and `torchvision==0.29.1+cu132`.
The pinned Transformers version uses torchvision for WAN's CLIP image processor.
These packages have not been validated together in a fresh environment. Keep the
existing Conda environment during migration.

`requirements.txt` currently pins direct dependencies only. It is **not** a
complete generated lock of transitive dependencies. Generate that lock after
confirming the matching PyTorch wheel source and resolving the runtime together
on the target machine.

## Files

- `requirements.in`: direct application dependencies and compatibility ranges.
- `requirements.txt`: initial exact direct pins; later the generated runtime lock.
- `requirements-dev.txt`: optional pip-tools and notebook kernel tooling.
- `.python-version`: the selected Python minor version.
- `pyproject.toml`: installs the package from `src/`; reads dependencies from
  `requirements.in` so package metadata shares the same declarations.
- `scripts/setup.sh`: creates `.venv`, installs dependencies and the local package,
  then runs environment checks.
- `scripts/check_environment.py`: checks imports, FFmpeg, CUDA, BF16, small matrix
  multiplications, and PyTorch SDPA. It does not download or load model weights.

## First setup

Use a Python 3.11 interpreter compatible with the selected PyTorch wheel. To
eliminate the Conda dependency entirely, choose an interpreter installed outside
Conda. Creating a venv from a Conda interpreter still depends on that interpreter.

The existing PyTorch and torchvision installations have no recorded wheel URLs.
Obtain the matching wheels from the source used for your Thor-compatible builds.
The setup script takes explicit wheel paths or URLs and resolves them with the
pinned runtime requirements; it does not silently select a generic PyTorch build.

```bash
bash scripts/setup.sh \
  --python /path/to/python3.11 \
  --torch-wheel /path/to/torch-2.14.1+cu132-cp311-cp311-linux_aarch64.whl \
  --torchvision-wheel /path/to/torchvision-0.29.1+cu132-cp311-cp311-linux_aarch64.whl
```

The paths above are examples; substitute the interpreter and actual wheel you
have verified. A conflicting wheel version will fail dependency resolution.
`PYTHON_BIN`, `TORCH_WHEEL`, and `TORCHVISION_WHEEL` can also supply these arguments.

To install optional development tools too, append `--dev`. The script installs
only into the project's `.venv`; it does not modify the original Conda environment
or install operating-system packages. JetPack, drivers, and required system
libraries must already be available.

After successful setup:

```bash
source .venv/bin/activate
python scripts/check_environment.py
```

Subsequent setup runs can reuse the exact PyTorch/torchvision builds already
present in `.venv`, without specifying their wheels again. All runtime installs
should continue to use the same requirements file.

## Generate the complete runtime lock

Install the optional development requirements. Run compilation using the target
Python 3.11 environment on the target platform, with the compatible torch and
torchvision wheels available to the resolver. Place both wheels in a local wheel
directory:

```bash
.venv/bin/python -m piptools compile \
  --find-links /path/to/verified-wheels \
  --output-file requirements.txt \
  requirements.in
```

Keep that wheel directory available when recreating the environment, or use
stable wheel URLs in the requirements input. A local wheel directory is a
machine-specific source; do not describe a lock referencing it as portable.
For another machine architecture, validate and generate a separate lock.

Commit the requirements files together after reviewing the resolved versions.
Check the environment and perform short real generations for SDXL, FLUX, and WAN,
including their checkpoint/LoRA loading paths, before treating a lock as validated.

## Attention and optional packages

The baseline uses PyTorch's native SDPA. Standalone `flash-attn`, xformers,
bitsandbytes, TensorFlow, and evaluation packages are not included. If an
optimization later proves necessary, install it through a separate documented
optional step and test it against the exact Python/PyTorch/CUDA stack.

Do not install dependencies when a workflow is imported or a model is selected.
Model manifests contain model configuration; package installation stays in setup.

## Checks without installing dependencies

```bash
bash -n scripts/setup.sh
python -m unittest discover -s tests -v
```

The checker supports `--allow-cpu` for dependency inspection on a machine without
CUDA. Passing that mode does not establish that GPU generation works.

References: [pip-tools](https://pip-tools.pypa.io/en/stable/),
[Flask releases](https://pypi.org/project/Flask/), and
[Diffusers attention backends](https://huggingface.co/docs/diffusers/optimization/attention_backends).
