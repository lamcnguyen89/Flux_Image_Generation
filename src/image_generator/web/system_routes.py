"""Host shared-memory monitoring and worker-owned model cleanup; no GPU imports."""

import sqlite3

import psutil
from flask import Blueprint, jsonify, request

from .job_routes import get_job_store
from .security import csrf_token, valid_csrf


system_bp = Blueprint("system", __name__)


@system_bp.app_context_processor
def system_context():
    return {"system_csrf_token": csrf_token()}


@system_bp.get("/api/system/memory")
def memory():
    # Unified-memory hosts share physical RAM between CPU and GPU. Available RAM
    # excludes reclaimable caches from usage, unlike Linux's unrelated shmem field.
    stats = psutil.virtual_memory()
    used = max(0, stats.total - stats.available)
    response = jsonify({"total_bytes": stats.total, "used_bytes": used,
                        "percent": round(100 * used / stats.total, 1) if stats.total else 0})
    response.cache_control.no_store = True
    return response


@system_bp.errorhandler(sqlite3.Error)
def unavailable_queue(error):
    return jsonify({"error": "The generation queue is unavailable. Try again."}), 503


@system_bp.post("/api/system/models/unload")
def unload_models():
    if not valid_csrf(request.headers.get("X-CSRF-Token")):
        return jsonify({"error": "Session expired. Refresh the page and try again."}), 400
    try:
        result = get_job_store().request_model_unload()
    except ValueError as error:
        return jsonify({"error": str(error)}), 409
    return jsonify(result), 202


@system_bp.get("/api/system/models/unload/<identifier>")
def unload_status(identifier):
    try:
        result = get_job_store().model_unload_status(identifier)
    except KeyError:
        return jsonify({"error": "Model cleanup request was not found."}), 404
    response = jsonify(result)
    response.cache_control.no_store = True
    return response
