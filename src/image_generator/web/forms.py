"""Translate HTML form values to workflow settings, without running generation."""

from copy import deepcopy
import math

from image_generator.workflows.ui import set_setting, setting_value


def display_values(ui, settings, form=None):
    values = {}
    for field in ui.fields:
        if field.scope != "settings":
            continue
        if form is not None:
            value = form.get(field.name, "") if field.type != "boolean" else field.name in form
        else:
            value = setting_value(settings, field.name)
            if field.type == "string_list":
                value = "\n".join(value or [])
        values[field.name] = "" if value is None else value
    return values


def parse_settings(ui, settings, form):
    result, errors = deepcopy(settings), {}
    for field in ui.fields:
        if field.scope != "settings":
            continue
        raw = form.get(field.name, "")
        try:
            if field.type == "boolean":
                if raw not in {"", "on"}:
                    raise ValueError("Invalid checkbox value")
                value = raw == "on"
            elif not raw.strip() and field.nullable:
                value = None
            elif field.type == "integer":
                value = int(raw)
            elif field.type == "number":
                value = float(raw)
            elif field.type == "string_list":
                value = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                value = raw
            error = field.validate(value)
            if error:
                errors[field.name] = error
            else:
                set_setting(result, field.name, value)
        except (ValueError, OverflowError):
            errors[field.name] = "Enter a whole number." if field.type == "integer" else "Enter a valid value."
    return result, errors


def lora_options(catalog, model, settings, form=None):
    links = {link["id"]: link for link in settings.get("loras", [])}
    selected = set(form.getlist("lora")) if form is not None else None
    options = []
    for asset in sorted(catalog.loras.values(), key=lambda item: (item.display_name.casefold(), item.id)):
        compatibility = asset.defaults["compatibility"]
        variant = compatibility.get("model_variant")
        target = compatibility.get("expert_target")
        reason = ""
        if asset.family != model.family:
            reason = "This adapter belongs to a different family."
        elif variant and model.defaults.get("model_variant") and variant != model.defaults["model_variant"]:
            reason = "This adapter targets a different model variant."
        elif model.defaults["workflow"] == "wan":
            if variant != model.defaults.get("model_variant") or target not in {"high_noise", "low_noise"}:
                reason = "Set the WAN model variant and high-noise or low-noise expert target before use."
        if reason and asset.id not in links and not (asset.family == model.family and model.family == "wan"):
            continue
        resolved = asset.resolve_settings()
        link = links.get(asset.id, {})
        strength = link.get("strength", resolved.get("strength", 0.8))
        options.append({
            "asset": asset, "settings": resolved, "eligible": not reason, "reason": reason,
            "enabled": (asset.id in selected if selected is not None else bool(link.get("enabled"))) and not reason,
            "strength": form.get(f"strength:{asset.id}", strength) if form is not None else strength,
            "verified": model.id in compatibility.get("verified_checkpoint_ids", []),
            "expert": target if target in {"high_noise", "low_noise"} and not reason else "unknown",
        })
    return options


def parse_loras(options, form):
    selected = form.getlist("lora")
    errors, links = {}, []
    if len(selected) != len(set(selected)):
        errors["loras"] = "Select each adapter only once."
    by_id = {item["asset"].id: item for item in options}
    for identifier in dict.fromkeys(selected):
        item = by_id.get(identifier)
        if item is None or not item["eligible"]:
            errors["loras"] = "A selected adapter is unavailable for this model."
            continue
        try:
            strength = float(form.get(f"strength:{identifier}", ""))
            if not math.isfinite(strength) or strength < 0:
                raise ValueError
        except (ValueError, OverflowError):
            errors[f"strength:{identifier}"] = "Strength must be a finite number at least zero."
            continue
        links.append({"id": identifier, "enabled": True, "strength": strength})
    return links, errors
