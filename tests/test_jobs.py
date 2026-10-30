"""Durable queue requests and storage without GPU libraries or live catalog reads."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator.catalog import CatalogAsset
from image_generator.job_requests import decode_request, encode_request, verify_fingerprints
from image_generator.jobs import JobStore
from image_generator.workflows.base import LoRASelection
from queue_fakes import make_request


class JobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = JobStore(self.root / "data/jobs.sqlite3")

    def test_enqueue_survives_reopening_with_detached_settings_and_timestamps(self):
        request = make_request(self.root)
        job = self.store.enqueue(request)
        request.settings["width"] = 512
        request.model.defaults["requirements"]["changed"] = True
        job.request["settings"]["height"] = 1024
        restored = JobStore(self.store.database).get(job.id)
        self.assertEqual(restored.status, "queued")
        self.assertEqual(restored.stage, "queued")
        self.assertIsNone(restored.started_at)
        self.assertIsNone(restored.finished_at)
        self.assertEqual(restored.request["settings"]["width"], 16)
        self.assertEqual(restored.request["settings"]["height"], 16)
        self.assertEqual(restored.request["model"]["defaults"]["requirements"], {})
        self.assertEqual(restored.outputs, ())
        self.assertEqual(restored.created_at, restored.updated_at)

    def test_executable_snapshot_restores_model_lora_and_prepared_prompt_without_catalog(self):
        request = make_request(self.root)
        directory = self.root / "adapter"
        directory.mkdir()
        weights = directory / "adapter.safetensors"
        weights.write_bytes(b"test adapter")
        asset = CatalogAsset("lora", directory / "lora.defaults.json", directory / "lora.user.json", {
            "schema_version": 1, "id": "style", "display_name": "Style", "family": "test",
            "adapter_type": "lora", "weights": "adapter.safetensors", "compatibility": {},
            "settings": {"strength": 0.5, "trigger_words": ["style"]},
        }, {"schema_version": 1, "settings": {"strength": 0.7}})
        selection = LoRASelection(asset, {"strength": 0.7, "trigger_words": ["style"]}, 0.3, "adapter_0")
        request = replace(request, prompt="prefix style resolved prompt", loras=(selection,))
        queued = self.store.enqueue(request)
        weights_before = weights.read_bytes()
        request.model.manifest_path.parent.mkdir(parents=True)
        request.model.manifest_path.write_text("invalid live manifest")
        asset.manifest_path.write_text("invalid live adapter manifest")
        restored = decode_request(self.store.get(queued.id).request)
        self.assertEqual(restored.prompt, "prefix style resolved prompt")
        self.assertEqual(restored.loras[0].strength, 0.3)
        self.assertEqual(restored.loras[0].settings["strength"], 0.7)
        self.assertEqual(restored.loras[0].asset.asset_paths["weights"], weights)
        self.assertEqual(weights.read_bytes(), weights_before)

    def test_component_paths_inputs_and_pipeline_files_are_fingerprinted(self):
        request = make_request(self.root)
        directory = request.model.manifest_path.parent
        (directory / "pipeline").mkdir(parents=True)
        (directory / "pipeline/model_index.json").write_text("{}")
        (directory / "component.bin").write_bytes(b"component")
        image = self.root / "input.png"
        image.write_bytes(b"input")
        request.model.defaults["weights"] = "pipeline"
        request.model.defaults["components"] = {"vae": {"weights": "component.bin"}}
        job = self.store.enqueue(replace(request, inputs={"input_image": image}))
        self.assertEqual(len(job.fingerprints), 3)
        verify_fingerprints(job.fingerprints)
        self.assertEqual(job.request["model"]["defaults"]["components"]["vae"]["weights"], str(directory / "component.bin"))
        (directory / "pipeline/new-config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            verify_fingerprints(job.fingerprints)

    def test_missing_assets_are_rejected_before_a_job_is_inserted(self):
        request = make_request(self.root)
        request.model.defaults["weights"] = "missing.safetensors"
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.enqueue(request)
        self.assertEqual(self.store.list_jobs(), ())

    def test_request_versions_types_and_nonfinite_values_are_rejected(self):
        document = encode_request(make_request(self.root))
        for changes in ({"schema_version": 2}, {"schema_version": True}, {"mode": "unsupported"},
                        {"settings": {"width": "16"}}, {"inputs": {"input_image": "relative.png"}},
                        {"loras": {}}, {"prompt": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode_request({**deepcopy(document), **changes})
        request = make_request(self.root, custom=float("nan"))
        with self.assertRaises(ValueError):
            self.store.enqueue(request)

    def test_unknown_ids_filters_and_database_versions(self):
        with self.assertRaises(KeyError):
            self.store.get("missing")
        with self.assertRaises(ValueError):
            self.store.list_jobs(status="unknown")
        with self.assertRaises(ValueError):
            self.store.list_jobs(limit=0)
        with sqlite3.connect(self.store.database) as connection:
            connection.execute("PRAGMA user_version=99")
        with self.assertRaisesRegex(ValueError, "schema version"):
            JobStore(self.store.database)

    def test_multiple_submitters_create_unique_durable_jobs(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = list(executor.map(lambda index: self.store.enqueue(make_request(self.root, model_id=f"model-{index}")), range(20)))
        self.assertEqual(len({job.id for job in jobs}), 20)
        self.assertEqual(len(JobStore(self.store.database).list_jobs()), 20)
        self.assertEqual(len(self.store.list_jobs(status="queued", limit=3)), 3)

    def test_concurrent_retries_with_the_same_submission_key_create_one_job(self):
        request = make_request(self.root)
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = list(executor.map(lambda _: self.store.enqueue(request, submission_key="ab" * 16), range(12)))
        self.assertEqual(len({job.id for job in jobs}), 1)
        self.assertEqual(len(self.store.list_jobs()), 1)


if __name__ == "__main__":
    unittest.main()
