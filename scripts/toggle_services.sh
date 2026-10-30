#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/toggle_services.sh

If Image Generator is installed and running, stop it, disable startup at boot,
and remove /etc/systemd/system/image-generator.service.
Otherwise, install the project unit, enable startup at boot, and start it.
Uses sudo for system changes when run as a normal user.
USAGE
}

fail() {
    printf 'Service toggle error: %s\n' "$*" >&2
    exit 1
}

if (($#)); then
    if (($# == 1)) && [[ "$1" == --help || "$1" == -h ]]; then
        usage
        exit 0
    fi
    fail 'Unexpected arguments. Use --help for usage.'
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
service_name=image-generator.service
source_unit="$project_root/systemd/$service_name"
installed_unit="/etc/systemd/system/$service_name"

command -v systemctl >/dev/null 2>&1 || fail 'systemctl is required.'
properties="$(systemctl --system show "$service_name" \
    --property=LoadState --property=ActiveState --property=FragmentPath)" \
    || fail 'Cannot query systemd. No changes were made.'

load_state= active_state= fragment_path=
while IFS='=' read -r property value; do
    case "$property" in
        LoadState) load_state="$value" ;;
        ActiveState) active_state="$value" ;;
        FragmentPath) fragment_path="$value" ;;
    esac
done <<< "$properties"

case "$load_state" in
    loaded|not-found) ;;
    masked) fail "The service is masked. Unmask it with sudo systemctl unmask $service_name first." ;;
    *) fail "Cannot manage service with LoadState=${load_state:-unknown}. Check systemctl status $service_name." ;;
esac
case "$active_state" in
    active|reloading|activating|inactive|failed) ;;
    deactivating) fail 'The service is still stopping. Wait for shutdown to finish, then retry.' ;;
    *) fail "Unexpected ActiveState=${active_state:-unknown}. No changes were made." ;;
esac
if [[ -n "$fragment_path" && "$fragment_path" != "$installed_unit" ]]; then
    fail "The service is loaded from $fragment_path instead of $installed_unit. Manage that installation separately."
fi

privilege=()
if ((EUID != 0)); then
    command -v sudo >/dev/null 2>&1 || fail 'Root privileges or sudo are required.'
    privilege=(sudo)
fi
trap 'printf "Service toggle failed. Inspect: journalctl -u image-generator.service -n 50\n" >&2' ERR

if [[ "$fragment_path" == "$installed_unit" \
    && ( "$active_state" == active || "$active_state" == reloading || "$active_state" == activating ) ]]; then
    printf 'Stopping %s; an active generation may take time to finish…\n' "$service_name"
    "${privilege[@]}" systemctl --system stop "$service_name"
    "${privilege[@]}" systemctl --system disable "$service_name"
    "${privilege[@]}" rm -f -- "$installed_unit"
    "${privilege[@]}" systemctl --system daemon-reload
    printf '%s stopped, disabled, and removed.\n' "$service_name"
else
    [[ -f "$source_unit" ]] || fail "Missing service file: $source_unit"
    [[ -x "$project_root/.venv/bin/python" ]] \
        || fail 'Project Python is missing. Run bash scripts/setup.sh first.'
    printf 'Installing and starting %s…\n' "$service_name"
    "${privilege[@]}" install -m 0644 -- "$source_unit" "$installed_unit"
    "${privilege[@]}" systemctl --system daemon-reload
    if [[ "$active_state" == failed ]]; then
        "${privilege[@]}" systemctl --system reset-failed "$service_name"
    fi
    "${privilege[@]}" systemctl --system enable --now "$service_name"
    systemctl --system is-active --quiet "$service_name"
    printf '%s started and enabled at boot.\n' "$service_name"
fi
