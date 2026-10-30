"""Exercise worker lifecycle with real SQLite and small fake images, without CUDA."""

from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.jobs import JobStore
from image_generator.worker import Worker, WorkerAlreadyRunning, WorkerCleanupError
from queue_fakes import FakeWorkflow, make_request


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        Event().wait(0.01)
    raise AssertionError("Timed out waiting for worker state")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = JobStore(self.root / "jobs.sqlite3")
        self.events, self.workflows = [], []

    def factory(self, name, **options):
        instance = FakeWorkflow(name, events=self.events, **options)
        self.workflows.append(instance)
        return instance

    def worker(self, **options):
        return Worker(self.store, self.root / "outputs", workflow_factory=self.factory, **options)

    def enqueue(self, **settings):
        return self.store.enqueue(make_request(self.root, **settings))

    def test_fake_images_complete_with_metadata_progress_and_persisted_outputs(self):
        queued = self.enqueue()
        with self.worker() as worker:
            completed = worker.run_once()
            self.assertEqual(completed.id, queued.id)
            self.assertEqual(completed.status, "completed")
            self.assertEqual((completed.current, completed.total), (2, 2))
            self.assertIsNotNone(completed.started_at)
            self.assertIsNotNone(completed.finished_at)
            self.assertIsNone(worker.run_once())
        reloaded = JobStore(self.store.database).get(queued.id)
        self.assertEqual([output.seed for output in reloaded.outputs], [42, 43])
        for output in reloaded.outputs:
            self.assertEqual(output.kind, "image")
            self.assertEqual(output.file_path.parent, self.root / "outputs" / queued.id)
            self.assertIsNone(output.conditioning_path)
            with Image.open(output.file_path) as image:
                self.assertEqual(image.size, (16, 16))
                self.assertEqual(image.getpixel((0, 0)), (output.seed, 20, 30))
            self.assertEqual(json.loads(output.metadata_path.read_text())["seed"], output.seed)
        self.assertTrue(self.workflows[0].closed)
        self.assertEqual(self.store.worker_status()["status"], "stopped")

    def test_fifo_same_workflow_reuse_and_family_switch_close_before_loading(self):
        jobs = [self.enqueue(model_id="one"), self.enqueue(model_id="two"),
                self.enqueue(workflow="fake_other", model_id="three")]
        self.worker().run(until_idle=True)
        self.assertTrue(all(self.store.get(job.id).status == "completed" for job in jobs))
        self.assertEqual([event for event in self.events if event[0] == "generate"],
                         [("generate", "one"), ("generate", "two"), ("generate", "three")])
        self.assertEqual(len(self.workflows), 2)
        self.assertLess(self.events.index(("close", "fake_image")), self.events.index(("create", "fake_other")))

    def test_validation_generation_and_saving_failures_do_not_poison_next_job(self):
        for failure in ("validation", "generation", "saving"):
            with self.subTest(failure=failure):
                failed = self.enqueue(failure=failure)
                following = self.enqueue()
                with self.worker() as worker:
                    if failure == "saving":
                        with patch("image_generator.worker.save_batch", side_effect=OSError("disk full")):
                            result = worker.run_once()
                    else:
                        result = worker.run_once()
                    self.assertEqual(result.status, "failed")
                    self.assertIn("failed" if failure != "saving" else "disk full", result.error)
                    self.assertEqual(result.outputs, ())
                    self.assertTrue(self.workflows[-1].closed)
                    self.assertEqual(worker.run_once().id, following.id)
                self.assertEqual(self.store.get(failed.id).status, "failed")
                self.assertEqual(self.store.get(following.id).status, "completed")

    def test_changed_or_missing_input_fails_before_model_creation(self):
        source = self.root / "input.png"
        source.write_bytes(b"original input")
        job = self.store.enqueue(replace(make_request(self.root), inputs={"input_image": source}))
        source.write_bytes(b"replaced input")
        with self.worker() as worker:
            self.assertEqual(worker.run_once().status, "failed")
        self.assertIn("changed after submission", self.store.get(job.id).error)
        self.assertEqual(self.workflows, [])

    def test_stop_request_finishes_current_job_and_leaves_next_queued(self):
        first, following = self.enqueue(), self.enqueue()
        worker = self.worker()
        def factory(name, **options):
            return FakeWorkflow(name, before_generate=lambda request: worker.request_stop())
        worker.workflow_factory = factory
        worker.run()
        self.assertEqual(self.store.get(first.id).status, "completed")
        self.assertEqual(self.store.get(following.id).status, "queued")

    def test_keyboard_interrupt_records_interrupted_and_releases_lock(self):
        job = self.enqueue(failure="interrupt")
        with self.assertRaises(KeyboardInterrupt):
            self.worker().run(once=True)
        self.assertEqual(self.store.get(job.id).status, "interrupted")
        self.assertTrue(self.workflows[-1].closed)
        with self.worker():
            self.assertEqual(self.store.get(job.id).status, "interrupted")

    def test_cleanup_failure_stops_worker_before_loading_another_family(self):
        self.enqueue()
        second = self.enqueue(workflow="fake_other")
        third = self.enqueue()
        with self.worker() as worker:
            self.assertEqual(worker.run_once().status, "completed")
            self.workflows[0].close = lambda: (_ for _ in ()).throw(RuntimeError("cannot unload"))
            with self.assertRaises(WorkerCleanupError):
                worker.run_once()
        self.assertEqual(len(self.workflows), 1)
        self.assertEqual(self.store.get(second.id).status, "failed")
        self.assertEqual(self.store.get(third.id).status, "queued")

    def test_live_worker_lock_prevents_recovery_and_double_claims(self):
        queued = self.enqueue()
        with self.worker() as first:
            second = self.worker()
            with self.assertRaises(WorkerAlreadyRunning):
                second.start()
            self.assertEqual(self.store.get(queued.id).status, "queued")
            self.assertEqual(first.run_once().status, "completed")
        with self.worker() as following:
            self.assertIsNone(following.run_once())

    def test_heartbeat_updates_during_idle_and_failure_stops_claiming(self):
        with self.worker(heartbeat_interval=0.02) as worker:
            initial = self.store.worker_status()["heartbeat_at"]
            wait_until(lambda: self.store.worker_status()["heartbeat_at"] != initial)
            with patch.object(self.store, "_heartbeat", side_effect=RuntimeError("database unavailable")):
                wait_until(lambda: worker._heartbeat_failure is not None)
            with self.assertRaisesRegex(RuntimeError, "heartbeat failed"):
                worker.run_once()

    def test_runtime_options_remain_outside_persisted_requests(self):
        job = self.enqueue()
        self.worker(runtime_options={"fake_image": {"token": "private-worker-token"}}).run(once=True)
        self.assertEqual(self.workflows[0].options["token"], "private-worker-token")
        self.assertNotIn("private-worker-token", json.dumps(self.store.get(job.id).request))
        self.assertNotIn("private-worker-token", self.store.get(job.id).outputs[0].metadata_path.read_text())

    def test_streamed_batches_retain_recorded_outputs_when_a_later_batch_fails(self):
        class StreamingFake(FakeWorkflow):
            def generate_batches(self, request):
                yield self.generate(request)
                raise RuntimeError("later batch failed")
        job = self.enqueue()
        worker = self.worker()
        worker.workflow_factory = lambda name, **options: StreamingFake(name)
        worker.run(once=True)
        result = self.store.get(job.id)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.outputs), 2)
        self.assertTrue(all(output.file_path.exists() for output in result.outputs))

    def test_saved_video_outputs_use_the_same_queue_and_typed_references(self):
        from image_generator.workflows.base import GeneratedBatch
        class VideoFake(FakeWorkflow):
            def generate(self, request):
                batch = super().generate(request)
                return GeneratedBatch((), batch.metadata, videos=([batch.images[0]] * 5, [batch.images[1]] * 5),
                                      conditioning_image=batch.images[0])
        job = self.enqueue(fps=24)
        worker = self.worker()
        worker.workflow_factory = lambda name, **options: VideoFake(name)
        with patch("image_generator.generation.export_video", side_effect=lambda frames, path, fps: path.write_bytes(b"fake video")):
            worker.run(once=True)
        result = self.store.get(job.id)
        self.assertEqual(result.status, "completed")
        self.assertTrue(all(output.kind == "video" and output.conditioning_path.is_file() for output in result.outputs))

    def test_unknown_workflow_fails_without_importing_arbitrary_code(self):
        job = self.enqueue(workflow="unregistered")
        Worker(self.store, self.root / "outputs").run(once=True)
        self.assertEqual(self.store.get(job.id).status, "failed")
        self.assertIn("not implemented", self.store.get(job.id).error)

    def test_hard_process_death_recovers_running_job_without_retrying_it(self):
        running, pending = self.enqueue(), self.enqueue()
        ready = self.root / "ready"
        code = f"""
import sys
from pathlib import Path
from threading import Event
sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT / 'tests')!r}]
from image_generator.jobs import JobStore
from image_generator.worker import Worker
from queue_fakes import FakeWorkflow
def block(request):
    Path({str(ready)!r}).write_text('generating')
    Event().wait(30)
worker = Worker(JobStore({str(self.store.database)!r}), {str(self.root / 'outputs')!r},
                workflow_factory=lambda name, **options: FakeWorkflow(name, before_generate=block), heartbeat_interval=0.02)
worker.run()
"""
        process = subprocess.Popen([sys.executable, "-c", code], cwd="/tmp", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            wait_until(lambda: ready.exists() or process.poll() is not None)
            self.assertTrue(ready.exists(), process.stderr.read().decode() if process.poll() is not None else "worker did not generate")
            self.assertEqual(self.store.get(running.id).status, "running")
            with self.assertRaises(WorkerAlreadyRunning):
                self.worker().start()
            self.assertEqual(self.store.get(running.id).status, "running")
            process.kill()
            process.wait(timeout=5)
            with self.worker() as recovered:
                self.assertEqual(self.store.get(running.id).status, "interrupted")
                self.assertIn("not retried", self.store.get(running.id).error)
                self.assertEqual(recovered.run_once().id, pending.id)
            self.assertEqual(self.store.get(pending.id).status, "completed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()

    def test_cli_empty_queue_works_without_site_packages_or_model_imports(self):
        code = f"""
import runpy, sys
sys.argv = [{str(ROOT / 'scripts/run_worker.py')!r}, '--database', {str(self.store.database)!r},
            '--output-dir', {str(self.root / 'outputs')!r}, '--once']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('torch', 'diffusers', 'comfy', 'flask', 'PIL'))
"""
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.store.worker_status()["status"], "stopped")

    def test_cli_sigterm_stops_an_idle_worker_cleanly(self):
        process = subprocess.Popen([sys.executable, "-S", str(ROOT / "scripts/run_worker.py"),
                                    "--database", str(self.store.database), "--poll-interval", "0.02"],
                                   cwd="/tmp", stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            wait_until(lambda: self.store.worker_status() is not None or process.poll() is not None)
            self.assertIsNone(process.poll())
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(self.store.worker_status()["status"], "stopped")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
