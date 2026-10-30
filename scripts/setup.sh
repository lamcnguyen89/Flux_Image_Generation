#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="$project_root/.venv"
venv_python="$venv_dir/bin/python"
python_bin="${PYTHON_BIN:-}"
comfyui_dir="$project_root/build/ComfyUI"
# Native MiniMax H3 baseline; existing checkouts are validated and preserved.
comfyui_revision="fd7639f477050e5c25b68fc38a9c2a6b6d988211"
# Official CUDA 13.2 wheels for the Python 3.11 / Linux aarch64 baseline.
# Keep these versions aligned with requirements.txt and requirements.in.
torch_source="${TORCH_WHEEL:-https://download-r2.pytorch.org/whl/cu132/torch-2.14.1%2Bcu132-cp311-cp311-manylinux_2_28_aarch64.whl}"
torchvision_source="${TORCHVISION_WHEEL:-https://download-r2.pytorch.org/whl/cu132/torchvision-0.29.1%2Bcu132-cp311-cp311-manylinux_2_28_aarch64.whl}"
install_dev=false

usage() {
    cat <<'USAGE'
Usage: bash scripts/setup.sh [options]

  --python PATH        Standalone, non-Conda Python 3.11 (or set PYTHON_BIN).
  --torch-wheel SOURCE Override the default torch wheel with a path or URL
                       (or set TORCH_WHEEL).
  --torchvision-wheel SOURCE
                       Override the default torchvision wheel with a path or URL
                       (or set TORCHVISION_WHEEL).
  --dev                Also install requirements-dev.txt.
  -h, --help           Show this help without changing the environment.

The initial dependency baseline targets Linux aarch64 / Jetson Thor.
Creates one shared .venv for SDXL, FLUX, WAN, and MiniMax H3.
Downloads ComfyUI to build/ComfyUI when absent; preserves existing checkouts.
Install Python 3.11 outside Conda first; see Documents/Setup.md.
Downloads the pinned torch and torchvision CUDA 13.2 wheels from PyTorch by default.
Wheel overrides must match requirements.txt. No FlashAttention is installed.
USAGE
}

fail() {
    printf 'Setup error: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --python|--torch-wheel|--torchvision-wheel)
            (($# >= 2)) && [[ "$2" != --* && -n "$2" ]] \
                || fail "$1 requires a value."
            if [[ "$1" == --python ]]; then
                python_bin="$2"
            elif [[ "$1" == --torch-wheel ]]; then
                torch_source="$2"
            else
                torchvision_source="$2"
            fi
            shift 2
            ;;
        --dev) install_dev=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1. Use --help for usage." ;;
    esac
done

if [[ ! -e "$comfyui_dir" ]]; then
    command -v git >/dev/null 2>&1 || fail "Git is required to download MiniMax's ComfyUI runtime."
elif [[ ! -f "$comfyui_dir/requirements.txt" || ! -f "$comfyui_dir/comfy_extras/nodes_minimax_h3.py" ]]; then
    fail "ComfyUI at $comfyui_dir is incomplete or lacks native MiniMax H3 support. Move it aside or update it, then rerun setup."
fi

if [[ -z "$python_bin" ]]; then
    if [[ -x "$venv_python" ]]; then
        python_bin="$venv_python"
    else
        python_bin=python3.11
    fi
fi
command -v "$python_bin" >/dev/null 2>&1 \
    || fail "Python executable not found: $python_bin. Install standalone Python 3.11 (see Documents/Setup.md), then pass --python /path/to/python3.11. Do not use a Conda interpreter."

check_python() {
    "$1" - "$project_root/.python-version" <<'PY'
from pathlib import Path
import platform
import sys

# A venv retains its base interpreter, including a dependency on Conda when
# created from one. Check the base prefix as well as the executable itself.
prefixes = (Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).resolve().parent.parent)
if any((prefix / "conda-meta").is_dir() for prefix in prefixes):
    raise SystemExit(
        f"Setup error: {sys.executable} is backed by Conda. "
        "Use standalone Python 3.11; see Documents/Setup.md. "
        "If .venv was created from Conda, move it aside and rerun setup."
    )
expected = Path(sys.argv[1]).read_text().strip()
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual != expected:
    raise SystemExit(f"Expected Python {expected}; got {actual}. Select a compatible interpreter.")
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise SystemExit("This initial requirements baseline targets Linux aarch64. Validate separate pins for another platform.")
PY
}

check_python "$python_bin"
# Validate before reusing installed wheels or changing an existing environment,
# even when --python explicitly selects a different, standalone interpreter.
if [[ -x "$venv_python" && "$python_bin" != "$venv_python" ]]; then
    check_python "$venv_python"
fi

wheel_sources=()
for package in torch torchvision; do
    if [[ "$package" == torch ]]; then
        wheel_source="$torch_source"
    else
        wheel_source="$torchvision_source"
    fi
    case "$wheel_source" in
        https://*|http://*) ;;
        *) [[ -f "$wheel_source" && "$wheel_source" == *.whl ]] \
               || fail "--$package-wheel must name an existing .whl file or a wheel URL." ;;
    esac
    wheel_sources+=("$wheel_source")
done

if [[ ! -x "$venv_python" ]]; then
    "$python_bin" -m venv "$venv_dir"
fi

check_python "$venv_python"

if [[ ! -e "$comfyui_dir" ]]; then
    mkdir -p "$(dirname -- "$comfyui_dir")"
    git init "$comfyui_dir"
    git -C "$comfyui_dir" remote add origin https://github.com/Comfy-Org/ComfyUI.git
    git -C "$comfyui_dir" fetch --depth 1 origin "$comfyui_revision"
    git -C "$comfyui_dir" checkout --detach FETCH_HEAD
fi
[[ -f "$comfyui_dir/requirements.txt" && -f "$comfyui_dir/comfy_extras/nodes_minimax_h3.py" ]] \
    || fail "ComfyUI at $comfyui_dir lacks native MiniMax H3 support."

# Resolve all families together, constraining upstream ComfyUI requirements to
# the project baseline so they cannot replace the platform-specific GPU wheels.
install_args=(-c "$project_root/requirements.txt"
              -r "$project_root/requirements.txt"
              -r "$comfyui_dir/requirements.txt" "${wheel_sources[@]}")
"$venv_python" -m pip install "${install_args[@]}"
if [[ "$install_dev" == true ]]; then
    "$venv_python" -m pip install -r "$project_root/requirements-dev.txt"
fi

# Dependencies were installed above; install the src package without resolving them again.
"$venv_python" -m pip install --no-deps -e "$project_root"
"$venv_python" -m pip check
"$venv_python" "$script_dir/check_environment.py"
printf '\nSetup complete. Activate with: source "%s/bin/activate"\n' "$venv_dir"
printf 'Model families: SDXL, FLUX, WAN, MiniMax H3 (ComfyUI: %s)\n' "$comfyui_dir"
