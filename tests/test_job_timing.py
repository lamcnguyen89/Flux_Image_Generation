"""Persisted timing and shared generation UI checks without loading models."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.jobs import Job, JobEvent, JobStore
from image_generator.web.job_timing import format_duration, job_timing
from queue_fakes import make_request


START = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def stamp(seconds):
    return (START + timedelta(seconds=seconds)).isoformat()


def event(stage, seconds, current=None, total=None):
    return JobEvent(stage, current, total, stamp(seconds))


class JobTimingTests(unittest.TestCase):
    def job(self, **changes):
        job = Job(
            id="timed-job", workflow="sdxl", status="running", request={"settings": {}},
            fingerprints={}, created_at=stamp(-500), updated_at=stamp(40),
            started_at=stamp(0), finished_at=None, worker_id="test-worker",
            stage="denoising", current=2, total=10, error=None, outputs=(),
            events=(event("loading_model", 0), event("denoising", 20, 2, 10)),
        )
        return replace(job, **changes)

    def timing(self, job):
        return job_timing(job, now=START + timedelta(seconds=50))

    def test_elapsed_excludes_queue_wait_and_estimate_excludes_model_loading(self):
        timing = self.timing(self.job())
        self.assertEqual(timing["elapsed_seconds"], 50)
        self.assertEqual(timing["remaining_seconds"], 80)
        self.assertEqual(timing["elapsed_text"], "00:00:50")
        self.assertEqual(timing["remaining_text"], "00:01:20")

    def test_polling_without_new_steps_does_not_change_sampling_rate(self):
        job = self.job()
        later = job_timing(job, now=START + timedelta(seconds=65))
        self.assertEqual(later["remaining_seconds"], self.timing(job)["remaining_seconds"])
        self.assertEqual(later["elapsed_seconds"], 65)

    def test_remaining_videos_are_counted_and_rate_resets_for_next_variation(self):
        for workflow in ("wan", "minimax"):
            with self.subTest(workflow=workflow):
                job = self.job(
                    workflow=workflow, request={"settings": {"num_videos_per_prompt": 3}},
                    updated_at=stamp(140), events=(
                        event("video_variation", 10, 1, 3), event("denoising", 20, 10, 10),
                        event("decoding_video", 70), event("saving", 90, 1),
                        event("video_variation", 100, 2, 3), event("denoising", 120, 2, 10),
                    ),
                )
                timing = job_timing(job, now=START + timedelta(seconds=145))
                self.assertEqual(timing["remaining_seconds"], 180)

    def test_image_batch_is_not_mistaken_for_sequential_videos(self):
        for workflow in ("sdxl", "flux"):
            with self.subTest(workflow=workflow):
                job = self.job(workflow=workflow, request={"settings": {"num_images_per_prompt": 4}})
                self.assertEqual(self.timing(job)["remaining_seconds"], 80)

    def test_loading_and_first_step_have_no_estimate_yet(self):
        for job in (self.job(stage="loading_model", current=None, total=None, events=()),
                    self.job(current=0, events=(event("denoising", 20, 0, 10),)),
                    self.job(updated_at=stamp(20))):
            with self.subTest(job=job):
                timing = self.timing(job)
                self.assertIsNone(timing["remaining_seconds"])
                self.assertEqual(timing["remaining_text"], "Estimating…")

    def test_queued_job_is_not_timed_as_generation(self):
        timing = self.timing(self.job(status="queued", started_at=None, stage="queued", events=()))
        self.assertIsNone(timing["elapsed_seconds"])
        self.assertEqual(timing["elapsed_text"], "Not started")
        self.assertEqual(timing["remaining_text"], "Waiting in queue")

    def test_terminal_elapsed_time_is_frozen_at_finish(self):
        for status in ("completed", "failed", "interrupted"):
            with self.subTest(status=status):
                timing = self.timing(self.job(status=status, stage=status, finished_at=stamp(45)))
                self.assertEqual(timing["elapsed_seconds"], 45)
                self.assertEqual(timing["remaining_seconds"], 0 if status == "completed" else None)
                self.assertEqual(timing["remaining_text"], "00:00:00" if status == "completed" else "Stopped")

    def test_final_processing_does_not_claim_job_is_complete(self):
        events = (event("denoising", 20, 10, 10),)
        for stage in ("denoising", "decoding", "decoding_video", "encoding_video", "saving"):
            with self.subTest(stage=stage):
                timing = self.timing(self.job(stage=stage, current=10, events=events))
                self.assertIsNone(timing["remaining_seconds"])
                self.assertEqual(timing["remaining_text"], "Finishing…")

    def test_decoding_before_later_videos_is_not_final_processing(self):
        job = self.job(stage="decoding_video", request={"settings": {"num_videos_per_prompt": 2}},
                       events=(event("video_variation", 0, 1, 2), event("denoising", 20, 10, 10)))
        self.assertEqual(self.timing(job)["remaining_text"], "Estimating…")

    def test_unavailable_worker_has_no_live_estimate(self):
        timing = job_timing(self.job(), now=START + timedelta(seconds=50), worker_available=False)
        self.assertIsNone(timing["remaining_seconds"])
        self.assertEqual(timing["remaining_text"], "Worker unavailable")

    def test_duration_supports_hours_and_clamps_clock_skew(self):
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertEqual(format_duration(360000), "100:00:00")
        timing = job_timing(self.job(), now=START - timedelta(seconds=1))
        self.assertEqual(timing["elapsed_seconds"], 0)


class JobTimingWebTests(unittest.TestCase):
    def test_all_workflows_render_timing_below_progress_and_refresh_from_persisted_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "jobs.sqlite3"
            app = create_app({"TESTING": True, "JOB_DATABASE": database,
                              "JOB_OUTPUT_ROOT": root / "outputs", "JOB_INPUT_ROOT": root / "inputs"})
            client = app.test_client()
            store = JobStore(database)
            store._start_worker("timing-worker", 123)
            base = datetime.now(timezone.utc) - timedelta(seconds=60)
            for workflow in ("sdxl", "flux", "wan", "minimax"):
                with self.subTest(workflow=workflow):
                    count = 3 if workflow in {"wan", "minimax"} else 1
                    job = store.enqueue(make_request(root, workflow=workflow, model_id=workflow,
                                                     num_videos_per_prompt=count))
                    queued = client.get(f"/api/jobs/{job.id}").json
                    self.assertEqual(queued["timing"]["remaining_text"], "Waiting in queue")
                    with patch("image_generator.jobs._now", return_value=base.isoformat()):
                        store._claim_next("timing-worker")
                    with patch("image_generator.jobs._now", return_value=(base + timedelta(seconds=40)).isoformat()):
                        store._progress(job.id, "timing-worker", "video_variation", 1, count)
                        store._progress(job.id, "timing-worker", "denoising", 0, 8)
                    with patch("image_generator.jobs._now", return_value=(base + timedelta(seconds=50)).isoformat()):
                        store._progress(job.id, "timing-worker", "denoising", 2, 8)
                    payload = client.get(f"/api/jobs/{job.id}").json
                    self.assertGreaterEqual(payload["timing"]["elapsed_seconds"], 60)
                    self.assertEqual(payload["timing"]["remaining_seconds"], 30 if count == 1 else 110)
                    # Reloading the page uses the persisted stage start, not the page-open time.
                    html = client.get(f"/jobs/{job.id}").get_data(as_text=True)
                    self.assertLess(html.index('id="job-progress"'), html.index('class="job-timing"'))
                    self.assertIn('id="job-elapsed"', html)
                    self.assertIn(f'id="job-remaining">{payload["timing"]["remaining_text"]}', html)
                    store._finish(job.id, "timing-worker", "interrupted")


if __name__ == "__main__":
    unittest.main()
