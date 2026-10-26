#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_dir="$project_root/.venv"
venv_python="$venv_dir/bin/python"
python_bin="${PYTHON_BIN:-}"
torch_source="${TORCH_WHEEL:-}"
torchvision_source="${TORCHVISION_WHEEL:-}"
install_dev=false

usage() {
    cat <<'USAGE'
Usage: bash scripts/setup.sh [options]

  --python PATH        Python 3.11 executable (or set PYTHON_BIN).
  --torch-wheel SOURCE Matching CUDA torch wheel path or URL (or set TORCH_WHEEL).
  --torchvision-wheel SOURCE
                       Matching torchvision wheel path or URL (or set TORCHVISION_WHEEL).
  --dev                Also install requirements-dev.txt.
  -h, --help           Show this help without changing the environment.

The initial dependency baseline targets Linux aarch64 / Jetson Thor.
Matching torch and torchvision wheels are required on the first run. Later runs
can reuse the exact builds already installed in .venv. No FlashAttention is installed.
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

if [[ -z "$python_bin" ]]; then
    if [[ -x "$venv_python" ]]; then
        python_bin="$venv_python"
    else
        python_bin=python3.11
    fi
fi
command -v "$python_bin" >/dev/null 2>&1 \
    || fail "Python executable not found: $python_bin. Pass --python /path/to/python3.11."

"$python_bin" - "$project_root/.python-version" <<'PY'
from pathlib import Path
import platform
import sys

expected = Path(sys.argv[1]).read_text().strip()
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual != expected:
    raise SystemExit(f"Expected Python {expected}; got {actual}. Select a compatible interpreter.")
if platform.system() != "Linux" or platform.machine() != "aarch64":
    raise SystemExit("This initial requirements baseline targets Linux aarch64. Validate separate pins for another platform.")
PY

wheel_sources=()
for package in torch torchvision; do
    if [[ "$package" == torch ]]; then
        wheel_source="$torch_source"
    else
        wheel_source="$torchvision_source"
    fi
    if [[ -n "$wheel_source" ]]; then
        case "$wheel_source" in
            https://*|http://*) ;;
            *) [[ -f "$wheel_source" && "$wheel_source" == *.whl ]] \
                   || fail "--$package-wheel must name an existing .whl file or a wheel URL." ;;
        esac
        wheel_sources+=("$wheel_source")
        continue
    fi
    [[ -x "$venv_python" ]] \
        || fail "First setup requires --$package-wheel /path/to/the/compatible/$package.whl (or its URL). See Documents/Setup.md."
    "$venv_python" - "$project_root/requirements.txt" "$package" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

package = sys.argv[2]
pins = [line.strip().split("==", 1)[1] for line in Path(sys.argv[1]).read_text().splitlines() if line.strip().startswith(package + "==")]
if len(pins) != 1:
    raise SystemExit(f"Expected one exact {package} pin in requirements.txt.")
try:
    installed = version(package)
except PackageNotFoundError:
    installed = None
if installed != pins[0]:
    raise SystemExit(f"Supply --{package}-wheel for the exact build pinned in requirements.txt.")
PY
done

if [[ ! -x "$venv_python" ]]; then
    "$python_bin" -m venv "$venv_dir"
fi

# Check an existing environment too, even when --python selects another interpreter.
"$venv_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), ".venv must use Python 3.11"'

install_args=(-r "$project_root/requirements.txt" "${wheel_sources[@]}")
"$venv_python" -m pip install "${install_args[@]}"
if [[ "$install_dev" == true ]]; then
    "$venv_python" -m pip install -r "$project_root/requirements-dev.txt"
fi

# Dependencies were installed above; install the src package without resolving them again.
"$venv_python" -m pip install --no-deps -e "$project_root"
"$venv_python" -m pip check
"$venv_python" "$script_dir/check_environment.py"
printf '\nSetup complete. Activate with: source "%s/bin/activate"\n' "$venv_dir"
