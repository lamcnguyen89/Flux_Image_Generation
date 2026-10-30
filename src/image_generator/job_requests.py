"""Versioned executable requests, independent of live catalog files and GPU imports."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .catalog import CatalogAsset
from .catalog.validation import validate_defaults, validate_settings, validate_user
from .workflows.base import GenerationRequest, LoRASelection


def _local_components(value, directory):
    if isinstance(value, list):
        return [_local_components(item, directory) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for name, item in value.items():
        if name == "weights" and isinstance(item, str) and item.strip():
            result[name] = str((directory / Path(item).expanduser()).resolve())
        else:
            result[name] = _local_components(item, directory)
    return result


def _asset_document(asset):
    defaults = deepcopy(asset.defaults)
    # Freeze relative/symlink paths now, without relying on later manifest edits.
    for name, path in asset.asset_paths.items():
        defaults[name] = str(path) if path is not None else None
    if "components" in defaults:
        defaults["components"] = _local_components(defaults["components"], asset.manifest_path.parent)
    return {"kind": asset.kind, "manifest_path": str(asset.manifest_path.resolve()),
            "user_path": str(asset.user_path.resolve()), "defaults": defaults,
            "user_overrides": deepcopy(asset.user_overrides)}


def _asset_from_document(document, kind):
    if not isinstance(document, dict) or document.get("kind") != kind:
        raise ValueError(f"Invalid queued {kind} asset")
    for name in ("manifest_path", "user_path"):
        if not isinstance(document.get(name), str) or not Path(document[name]).is_absolute():
            raise ValueError(f"Queued asset {name} must be an absolute path")
    path = Path(document["manifest_path"])
    issues = validate_defaults(document.get("defaults"), path, kind)
    issues.extend(validate_user(document.get("user_overrides"), Path(document["user_path"])))
    if issues:
        raise ValueError("Invalid queued asset: " + "; ".join(str(issue) for issue in issues))
    return CatalogAsset(kind, path, Path(document["user_path"]),
                        deepcopy(document["defaults"]), deepcopy(document["user_overrides"]))


def encode_request(request: GenerationRequest) -> dict:
    """Capture resolved values and complete assets; never serialize runtime options."""
    document = {
        "schema_version": 1, "model": _asset_document(request.model), "mode": request.mode,
        "settings": deepcopy(request.settings), "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "inputs": {name: str(path.expanduser().resolve()) for name, path in request.inputs.items()},
        "loras": [{"asset": _asset_document(item.asset), "settings": deepcopy(item.settings),
                   "strength": item.strength, "adapter_name": item.adapter_name} for item in request.loras],
    }
    # Exercise the actual disk format and return a detached, strict JSON tree.
    result = json.loads(json.dumps(document, allow_nan=False))
    decode_request(result)
    return result


def decode_request(document: dict) -> GenerationRequest:
    """Restore an executable request without consulting or creating catalog files."""
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported queued request schema version")
    json.dumps(document, allow_nan=False)
    model = _asset_from_document(document.get("model"), "model")
    for name in ("mode", "prompt", "negative_prompt"):
        if not isinstance(document.get(name), str):
            raise ValueError(f"Queued {name} must be a string")
    if document["mode"] not in model.defaults["supported_modes"] or not document["prompt"].strip():
        raise ValueError("Queued requests need a declared mode and nonempty resolved prompt")
    issues = validate_settings(document.get("settings"), model.manifest_path)
    if issues:
        raise ValueError("Invalid queued settings: " + "; ".join(str(issue) for issue in issues))
    inputs = document.get("inputs")
    if not isinstance(inputs, dict) or any(not isinstance(name, str) or not isinstance(path, str)
                                          or not Path(path).is_absolute() for name, path in inputs.items()):
        raise ValueError("Queued inputs must map names to absolute paths")
    selections = document.get("loras")
    if not isinstance(selections, list):
        raise ValueError("Queued loras must be a list")
    loras = []
    for item in selections:
        if not isinstance(item, dict):
            raise ValueError("Invalid queued LoRA selection")
        asset = _asset_from_document(item.get("asset"), "lora")
        issues = validate_settings(item.get("settings"), asset.manifest_path)
        if issues or type(item.get("strength")) not in (int, float) or not isinstance(item.get("adapter_name"), str):
            raise ValueError("Invalid queued LoRA settings, strength, or adapter name")
        loras.append(LoRASelection(asset, deepcopy(item["settings"]), item["strength"], item["adapter_name"]))
    return GenerationRequest(model, document["mode"], deepcopy(document["settings"]), document["prompt"],
                             document["negative_prompt"], tuple(loras), {name: Path(path) for name, path in inputs.items()})


def _component_paths(value):
    if isinstance(value, dict):
        for name, item in value.items():
            if name == "weights" and isinstance(item, str):
                yield Path(item)
            else:
                yield from _component_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _component_paths(item)


def fingerprint_path(path: Path, *, hash_contents=False) -> dict:
    """Detect local replacements without hashing or loading multi-GB tensors."""
    def stat(file):
        value = file.stat()
        return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    if path.is_file():
        result = {"kind": "file", "stat": stat(path)}
        if hash_contents:
            with path.open("rb") as stream:
                result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result
    if path.is_dir():
        return {"kind": "directory", "files": {
            file.relative_to(path).as_posix(): stat(file)
            for file in sorted(path.rglob("*")) if file.is_file()
        }}
    raise ValueError(f"Queued asset or input not found: {path}")


def request_fingerprints(request: GenerationRequest) -> dict:
    inputs = set(request.inputs.values())
    paths = set(inputs)
    for asset in (request.model, *(item.asset for item in request.loras)):
        if asset.asset_paths["weights"] is not None:
            paths.add(asset.asset_paths["weights"])
        paths.update(_component_paths(asset.defaults.get("components", {})))
    return {str(path): fingerprint_path(path, hash_contents=path in inputs) for path in sorted(paths)}


def verify_fingerprints(fingerprints: dict) -> None:
    for filename, expected in fingerprints.items():
        path = Path(filename)
        if fingerprint_path(path, hash_contents="sha256" in expected) != expected:
            raise ValueError(f"Queued asset or input changed after submission: {path}")
