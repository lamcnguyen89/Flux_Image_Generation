#!/usr/bin/env python3
"""Run a standalone fake-image queue demo; retain its database and PNGs in /tmp."""

import json
from pathlib import Path
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.jobs import JobStore
from image_generator.worker import Worker
from queue_fakes import FakeWorkflow, make_request


def main():
    directory = Path(mkdtemp(prefix="image-generator-queue-demo-"))
    store = JobStore(directory / "jobs.sqlite3")
    submitted = [store.enqueue(make_request(directory, model_id="first")),
                 store.enqueue(make_request(directory, workflow="fake_other", model_id="second", seed=100))]
    Worker(store, directory / "outputs", workflow_factory=FakeWorkflow).run(until_idle=True)
    reopened = JobStore(store.database)
    results = []
    for original in submitted:
        job = reopened.get(original.id)
        assert job.status == "completed" and len(job.outputs) == 2
        results.append({"id": job.id, "status": job.status,
                        "images": [str(output.file_path) for output in job.outputs]})
    assert not any(name in sys.modules for name in ("torch", "diffusers", "comfy"))
    print(json.dumps({"directory": str(directory), "database": str(store.database), "jobs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
