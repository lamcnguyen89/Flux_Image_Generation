"""Generation forms and stateless request previews; no jobs or settings writes."""

from flask import Blueprint, abort, render_template, request
from werkzeug.utils import secure_filename

from .web import get_catalog


generation_bp = Blueprint("generation", __name__)


@generation_bp.route("/generate/<model_id>", methods=["GET", "POST"])
def configure(model_id):
    # Catalog browsing keeps workflow imports lazy; these modules are CPU-only.
    from image_generator.workflows.registry import create_workflow
    from image_generator.workflows.ui import FormValidationError
    from .forms import display_values, lora_options, parse_loras, parse_settings

    catalog = get_catalog()
    try:
        model = catalog.get_model(model_id)
    except KeyError:
        abort(404)
    try:
        workflow = create_workflow(model.defaults["workflow"])
        modes = tuple(mode for mode in workflow.ui_modes if mode in model.defaults["supported_modes"])
        if not modes:
            raise ValueError("This model has no modes with an implemented generation form.")
        mode = request.form.get("mode", modes[0]) if request.method == "POST" else request.args.get("mode", modes[0])
        if mode not in modes:
            return render_template("error.html", title="Unsupported mode",
                                   message="Choose a mode supported by this model and workflow."), 400
        ui = workflow.get_ui_definition(model, mode)
    except ValueError as error:
        return render_template("error.html", title="Generation form unavailable", message=str(error)), 422

    settings = model.resolve_settings(workflow_defaults=workflow.defaults)
    form = request.form if request.method == "POST" else None
    values = display_values(ui, settings, form)
    options = lora_options(catalog, model, settings, form)
    errors, preview = {}, None
    include_triggers = form is not None and form.get("include_trigger_words") == "on"
    if form is not None:
        submitted, errors = parse_settings(ui, settings, form)
        links, lora_errors = parse_loras(options, form)
        errors.update(lora_errors)
        submitted["loras"] = links
        if not errors:
            try:
                workflow.validate_form_settings(model, mode, submitted)
            except FormValidationError as error:
                errors.update(error.errors)
        input_files = {}
        for field in ui.fields:
            if field.scope == "inputs":
                upload = request.files.get(field.name)
                if upload is None or not upload.filename:
                    if field.required:
                        errors[field.name] = "Select a source image."
                else:
                    input_files[field.name] = secure_filename(upload.filename) or "selected-image"
        if not errors:
            preview = {"model_id": model.id, "mode": mode, "overrides": submitted,
                       "include_trigger_words": include_triggers, "input_files": input_files}

    return render_template(
        "generate.html", model=model, ui=ui, values=values, loras=options,
        errors=errors, preview=preview, include_triggers=include_triggers,
    ), 422 if errors else 200
