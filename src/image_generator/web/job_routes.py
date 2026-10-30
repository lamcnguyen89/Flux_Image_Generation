"""Persistent job pages, polling, and registered output serving; no GPU imports."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, url_for

from image_generator.jobs import JobStore


jobs_bp = Blueprint("jobs", __name__)
TERMINAL = {"completed", "failed", "interrupted"}
STAGE_LABELS = {
    "queued": "Waiting in queue", "validating": "Checking job assets", "preparing": "Preparing workflow",
    "generating": "Generating", "loading_runtime": "Preparing GPU runtime",
    "loading_model": "Loading checkpoint", "switching_model": "Switching checkpoint",
    "reusing_model": "Reusing loaded checkpoint", "resetting_loras": "Clearing previous LoRAs",
    "loading_loras": "Loading selected LoRAs", "loras_ready": "LoRAs configured",
    "denoising": "Denoising", "decoding": "Decoding images", "saving": "Saving outputs",
    "completed": "Completed", "failed": "Failed", "interrupted": "Interrupted",
}


@jobs_bp.app_errorhandler(sqlite3.Error)
def unavailable_queue(error):
    message = "The generation queue is unavailable. Check database access and try again."
    if request.path.startswith("/api/jobs/"):
        return jsonify({"error": message}), 503
    return render_template("error.html", title="Queue unavailable", message=message), 503


def get_job_store():
    path = Path(current_app.config["JOB_DATABASE"]).expanduser().resolve()
    store = current_app.extensions.get("generation_job_store")
    if store is None or store.database != path:
        store = JobStore(path)
        current_app.extensions["generation_job_store"] = store
    return store


def get_job(job_id):
    try:
        return get_job_store().get(job_id)
    except KeyError:
        abort(404)


def stage_label(stage):
    return STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize())


def progress_text(stage, current, total):
    if current is None:
        return ""
    if stage == "denoising" and total:
        return f"Step {current} of {total}"
    if stage in {"loading_loras", "loras_ready"}:
        return f"{current} of {total} adapters" if total else "No adapters selected"
    if stage in {"saving", "completed", "failed", "interrupted"}:
        return f"{current} output(s) saved"
    return ""


def worker_notice(store, job):
    if job.status not in {"queued", "running"}:
        return ""
    worker = store.worker_status()
    if worker is None or worker["status"] != "running":
        return "The worker is stopped. This job will wait until a worker starts."
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(worker["heartbeat_at"])).total_seconds()
    if age > 15:
        return "The worker heartbeat is stale. An interrupted job will be recovered when the worker restarts."
    return ""


def job_payload(job):
    settings = job.request["settings"]
    model = job.request["model"]["defaults"]
    width, height = settings.get("width"), settings.get("height")
    dimensions = {"width": width, "height": height} if (
        type(width) is int and type(height) is int and width > 0 and height > 0
    ) else {}

    def output_payload(output):
        return {
            "id": output.id, "kind": output.kind, "seed": str(output.seed),
            **dimensions,
            "url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="media"),
            "download_url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="media", download=1),
            "metadata_url": url_for("jobs.output", job_id=job.id, output_id=output.id, part="metadata", download=1),
        }
    return {
        "id": job.id, "model": {"id": model["id"], "name": model["display_name"]},
        "workflow": job.workflow, "mode": job.request["mode"], "status": job.status,
        "terminal": job.status in TERMINAL, "error": job.error,
        "progress": {"stage": job.stage, "label": stage_label(job.stage), "current": job.current, "total": job.total,
                     "text": progress_text(job.stage, job.current, job.total)},
        "worker_notice": worker_notice(get_job_store(), job),
        "events": [{"stage": event.stage, "label": stage_label(event.stage),
                    "current": event.current, "total": event.total,
                    "text": progress_text(event.stage, event.current, event.total)} for event in job.events],
        "outputs": [output_payload(output) for output in job.outputs],
        "created_at": job.created_at, "started_at": job.started_at, "finished_at": job.finished_at,
        "image_count": settings.get("num_images_per_prompt"),
    }


@jobs_bp.get("/jobs")
def index():
    return render_template("jobs.html", jobs=get_job_store().list_jobs(limit=50), stage_label=stage_label)


@jobs_bp.get("/jobs/<job_id>")
def detail(job_id):
    job = get_job(job_id)
    return render_template("job.html", job=job, status=job_payload(job))


@jobs_bp.get("/api/jobs/<job_id>")
def status(job_id):
    response = jsonify(job_payload(get_job(job_id)))
    response.cache_control.no_store = True
    return response


@jobs_bp.get("/jobs/<job_id>/outputs/<output_id>/<part>")
def output(job_id, output_id, part):
    job = get_job(job_id)
    record = next((item for item in job.outputs if item.id == output_id), None)
    if record is None or part not in {"media", "metadata", "conditioning"}:
        abort(404)
    path = {"media": record.file_path, "metadata": record.metadata_path, "conditioning": record.conditioning_path}[part]
    root = Path(current_app.config["JOB_OUTPUT_ROOT"]).expanduser().resolve()
    directory = (root / job.id).resolve()
    if path is None:
        abort(404)
    path = path.resolve()
    if not directory.is_relative_to(root) or not path.is_relative_to(directory) or not path.is_file():
        abort(404)
    mimetype = "application/json" if part == "metadata" else (
        "video/mp4" if part == "media" and record.kind == "video" else "image/png")
    response = send_file(path, mimetype=mimetype, as_attachment=request.args.get("download") == "1",
                         download_name=path.name, conditional=True, max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
