"""Memory reporting and cross-process model cleanup without model runtimes."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.jobs import JobStore
from image_generator.worker import Worker, WorkerCleanupError
from queue_fakes import FakeWorkflow, make_request


class SystemTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "jobs.sqlite3"
        self.store = JobStore(self.database)
        self.app = create_app({"TESTING": True, "JOB_DATABASE": self.database,
                               "CATALOG_ROOT": ROOT / "tests/fixtures/model_families"})
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as session:
            self.token = session["generation_csrf"]
        self.workflows = []

    def factory(self, name, **options):
        workflow = FakeWorkflow(name, **options)
        self.workflows.append(workflow)
        return workflow

    def worker(self):
        return Worker(self.store, self.root / "outputs", workflow_factory=self.factory)

    def unload(self):
        return self.client.post("/api/system/models/unload", headers={"X-CSRF-Token": self.token})

    def test_memory_uses_available_ram_and_is_not_cached(self):
        gib = 1024 ** 3
        with patch("image_generator.web.system_routes.psutil.virtual_memory",
                   return_value=SimpleNamespace(total=128 * gib, available=32 * gib)):
            response = self.client.get("/api/system/memory")
        self.assertEqual(response.json, {"total_bytes": 128 * gib, "used_bytes": 96 * gib, "percent": 75.0})
        self.assertTrue(response.cache_control.no_store)

    def test_header_controls_render_on_catalog_history_and_error_pages(self):
        for url in ("/", "/jobs", "/missing", "/generate/fixture-sdxl"):
            response = self.client.get(url)
            html = response.get_data(as_text=True)
            self.assertIn("Shared Memory", html)
            self.assertIn("Clear all models", html)
            self.assertIn('aria-label="System shared memory usage"', html)
            self.assertIn('data-csrf-token="' + self.token + '"', html)
        response = self.client.get("/catalog-assets/memory.js")
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_unload_requires_post_and_same_session_token(self):
        self.assertEqual(self.client.get("/api/system/models/unload").status_code, 405)
        self.assertEqual(self.client.post("/api/system/models/unload").status_code, 400)
        self.assertEqual(self.client.post("/api/system/models/unload", headers={"X-CSRF-Token": "wrong"}).status_code, 400)
        self.assertEqual(self.unload().status_code, 409)

    def test_idle_worker_clears_models_and_next_job_loads_a_new_workflow(self):
        self.store.enqueue(make_request(self.root))
        with self.worker() as worker:
            worker.run_once()
            response = self.unload()
            self.assertEqual(response.status_code, 202)
            identifier = response.json["id"]
            self.assertEqual(self.unload().json["id"], identifier)
            self.assertFalse(self.workflows[0].closed)
            self.assertIsNone(worker.run_once())
            status = self.client.get(f"/api/system/models/unload/{identifier}")
            self.assertEqual(status.json["status"], "completed")
            self.assertTrue(status.cache_control.no_store)
            self.assertTrue(self.workflows[0].closed)
            self.store.enqueue(make_request(self.root))
            self.assertEqual(worker.run_once().status, "completed")
            self.assertEqual(len(self.workflows), 2)

    def test_request_during_generation_waits_until_job_finishes(self):
        identifiers = []
        def before_generate(request):
            # Another connection represents the separate web server process.
            identifiers.append(JobStore(self.database).request_model_unload()["id"])
            self.assertEqual(self.store.model_unload_status(identifiers[0])["status"], "pending")
            self.assertFalse(self.workflows[0].closed)
        worker = self.worker()
        worker.workflow_factory = lambda name, **options: self.factory(name, before_generate=before_generate)
        self.store.enqueue(make_request(self.root))
        following = self.store.enqueue(make_request(self.root))
        with worker:
            self.assertEqual(worker.run_once().status, "completed")
            self.assertTrue(self.workflows[0].closed)
            self.assertEqual(self.store.model_unload_status(identifiers[0])["status"], "completed")
            self.assertEqual(self.store.get(following.id).status, "queued")

    def test_cleanup_failure_is_reported_and_stops_worker(self):
        self.store.enqueue(make_request(self.root))
        with self.worker() as worker:
            worker.run_once()
            identifier = self.store.request_model_unload()["id"]
            with patch.object(self.workflows[0], "close", side_effect=RuntimeError("unload failed")):
                with self.assertRaises(WorkerCleanupError):
                    worker.run_once()
            self.assertEqual(self.store.model_unload_status(identifier)["status"], "failed")
            self.assertTrue(worker._stop.is_set())

    def test_stale_worker_and_shutdown_do_not_report_success(self):
        with self.worker() as worker:
            identifier = self.store.request_model_unload()["id"]
            with self.store._transaction() as connection:
                connection.execute("UPDATE worker_state SET heartbeat_at='2000-01-01T00:00:00+00:00'")
            self.assertEqual(self.unload().status_code, 409)
        self.assertEqual(self.store.model_unload_status(identifier)["status"], "failed")
        with self.worker():
            new_identifier = self.store.request_model_unload()["id"]
        self.assertNotEqual(identifier, new_identifier)

    def test_restart_marks_unconfirmed_requests_failed_and_unknown_status_is_404(self):
        self.store._start_worker("old-worker", 123)
        identifier = self.store.request_model_unload()["id"]
        with self.worker():
            self.assertEqual(self.store.model_unload_status(identifier)["status"], "failed")
        self.assertEqual(self.client.get("/api/system/models/unload/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
