"""Exercise real manifest persistence with absent weights and no GPU runtime."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_generator import create_app
from image_generator.catalog import discover_catalog
from image_generator.catalog.persistence import (
    SettingsConflict, reset_settings, save_adapter_settings, save_model_settings, settings_revision,
)
from test_generation_ui import FormPage


class SettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "families"
        shutil.copytree(ROOT / "tests/fixtures/model_families", self.root)
        self.app = create_app({"TESTING": True, "CATALOG_ROOT": self.root})
        self.client = self.app.test_client()
        self.user = self.root / "FLUX/checkpoints/demo/model.user.json"
        self.defaults = self.user.with_name("model.defaults.json")
        self.style_user = self.root / "FLUX/loras/style/lora.user.json"

    def write(self, path, settings):
        path.write_text(json.dumps({"schema_version": 1, "settings": settings}))

    def catalog(self):
        return discover_catalog(self.root, initialize_missing=False)

    def form(self, model="fixture-flux"):
        response = self.client.get(f"/generate/{model}")
        self.assertEqual(response.status_code, 200)
        return FormPage(response.get_data(as_text=True)).data()

    def post(self, data, action="save_model", model="fixture-flux"):
        data["action"] = action
        return self.client.post(f"/settings/models/{model}", data=data)

    def saved(self, path=None):
        return json.loads((path or self.user).read_text())["settings"]

    def test_multiple_and_disabled_adapters_round_trip_and_preview_only_enables_active(self):
        data = self.form()
        data.setlist("configured_lora", ["fixture-flux-style", "fixture-flux-detail"])
        data.setlist("lora", ["fixture-flux-style"])
        data["strength:fixture-flux-style"] = "0.35"
        data["strength:fixture-flux-detail"] = "0.7"
        response = self.post(data)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.saved()["loras"], [
            {"id": "fixture-flux-style", "enabled": True, "strength": 0.35},
            {"id": "fixture-flux-detail", "enabled": False, "strength": 0.7},
        ])
        refreshed = self.form()
        self.assertEqual(refreshed.getlist("configured_lora"), ["fixture-flux-detail", "fixture-flux-style"])
        self.assertEqual(refreshed["strength:fixture-flux-detail"], "0.7")
        self.assertEqual(refreshed.getlist("lora"), ["fixture-flux-style"])
        refreshed["prompt"] = "A lake"
        response = self.client.post("/generate/fixture-flux", data=refreshed)
        preview = FormPage(response.get_data(as_text=True)).preview
        self.assertEqual(preview["overrides"]["loras"], [
            {"id": "fixture-flux-style", "enabled": True, "strength": 0.35}])

    def test_shared_defaults_followed_only_without_checkpoint_override(self):
        second = self.defaults.parent.parent / "other"
        second.mkdir()
        manifest = json.loads(self.defaults.read_text())
        manifest["id"] = "fixture-flux-other"
        manifest["settings"]["loras"] = [{"id": "fixture-flux-style", "enabled": True}]
        (second / "model.defaults.json").write_text(json.dumps(manifest))
        self.write(second / "model.user.json", {})
        data = self.form()
        data.add("lora", "fixture-flux-style")
        data["strength:fixture-flux-style"] = "0.3"
        self.assertEqual(self.post(data).status_code, 303)
        data = self.form()
        data["adapter_strength:fixture-flux-style"] = "0.95"
        data["adapter_trigger_words:fixture-flux-style"] = "new style\nsecond trigger"
        data["action"] = "save_adapter"
        response = self.client.post("/settings/adapters/fixture-flux-style", data=data)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(self.form()["strength:fixture-flux-style"], "0.3")
        other = self.form("fixture-flux-other")
        self.assertEqual(other["strength:fixture-flux-style"], "0.95")
        self.assertEqual(other["strength_source:fixture-flux-style"], "default")
        self.assertEqual(self.saved(self.style_user)["trigger_words"], ["new style", "second trigger"])
        data = self.form()
        data["strength_source:fixture-flux-style"] = "default"
        self.assertEqual(self.post(data).status_code, 303)
        self.assertNotIn("strength", self.saved()["loras"][0])
        self.assertEqual(self.form()["strength:fixture-flux-style"], "0.95")

    def test_reset_restores_shipped_links_and_empty_list_explicitly_disables_them(self):
        manifest = json.loads(self.defaults.read_text())
        manifest["settings"]["loras"] = [{"id": "fixture-flux-style", "enabled": True, "strength": 0.4}]
        self.defaults.write_text(json.dumps(manifest))
        data = self.form()
        data.setlist("configured_lora", [])
        data.setlist("lora", [])
        self.assertEqual(self.post(data).status_code, 303)
        self.assertEqual(self.saved()["loras"], [])
        data = self.form()
        self.assertEqual(data.getlist("lora"), [])
        self.assertEqual(self.post(data, "reset_model").status_code, 303)
        self.assertEqual(self.saved(), {})
        self.assertEqual(self.form().getlist("lora"), ["fixture-flux-style"])
        self.assertEqual(self.form()["strength:fixture-flux-style"], "0.4")

    def test_nested_diff_nullable_seed_and_unedited_settings_survive(self):
        manifest = json.loads(self.defaults.read_text())
        manifest["settings"]["seed"] = 42
        self.defaults.write_text(json.dumps(manifest))
        self.write(self.user, {"sampling": {"custom": {"x": 1}}, "extension": ["keep"]})
        data = self.form()
        data["seed"] = ""
        data["guidance_scale"] = "3.5"  # back to inherited: delete the override
        self.assertEqual(self.post(data).status_code, 303)
        self.assertEqual(self.saved(), {"sampling": {"custom": {"x": 1}}, "extension": ["keep"], "seed": None})
        self.assertEqual(self.form()["seed"], "")

    def test_service_nested_changes_and_removing_override_restore_inheritance(self):
        catalog = self.catalog()
        model = catalog.get_model("fixture-sdxl")
        desired = model.resolve_settings()
        desired["sampling"]["sampler"] = "euler"
        saved = save_model_settings(catalog, model.id, desired, scope=["sampling.sampler"],
                                    expected_revision=settings_revision(catalog))
        self.assertNotIn("sampling", saved["settings"])
        catalog = self.catalog()
        save_model_settings(catalog, "fixture-flux", {}, scope=[], remove=["guidance_scale"],
                            expected_revision=settings_revision(catalog))
        self.assertEqual(self.saved(), {})

    def test_adapter_save_preserves_unedited_settings_and_reset_clears_all_overrides(self):
        self.write(self.style_user, {"strength": 0.6, "recommended_generation": {"num_inference_steps": 19}, "custom": True})
        data = self.form()
        data["adapter_strength:fixture-flux-style"] = "0.8"
        data["action"] = "save_adapter"
        self.assertEqual(self.client.post("/settings/adapters/fixture-flux-style", data=data).status_code, 303)
        self.assertNotIn("strength", self.saved(self.style_user))
        self.assertEqual(self.saved(self.style_user)["recommended_generation"], {"num_inference_steps": 19})
        data = self.form()
        data["action"] = "reset_adapter"
        self.assertEqual(self.client.post("/settings/adapters/fixture-flux-style", data=data).status_code, 303)
        self.assertEqual(self.saved(self.style_user), {})

    def test_stale_model_adapter_and_reset_submissions_do_not_write(self):
        for target in (self.user, self.style_user, self.defaults):
            with self.subTest(target=target):
                data = self.form()
                document = json.loads(target.read_text())
                document["settings"]["seed"] = 5
                target.write_text(json.dumps(document))
                before = self.user.read_bytes()
                self.assertEqual(self.post(data).status_code, 409)
                self.assertEqual(self.post(data, "reset_model").status_code, 409)
                data["action"] = "reset_adapter"
                self.assertEqual(self.client.post("/settings/adapters/fixture-flux-style", data=data).status_code, 409)
                self.assertEqual(self.user.read_bytes(), before)

    def test_validation_and_csrf_failures_leave_files_unchanged(self):
        for name, value in (("width", "770"), ("guidance_scale", "nan"), ("seed", "-1"),
                            ("strength:fixture-flux-style", "-0.5"), ("strength_source:fixture-flux-style", "bad")):
            with self.subTest(name=name):
                data = self.form()
                data.add("configured_lora", "fixture-flux-style")
                data[name] = value
                before = self.user.read_bytes()
                response = self.post(data)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(self.user.read_bytes(), before)
                self.assertIn("Check these values", response.get_data(as_text=True))
        data = self.form()
        del data["csrf_token"]
        before = self.user.read_bytes()
        self.assertEqual(self.post(data).status_code, 400)
        self.assertEqual(self.user.read_bytes(), before)
        for strength in ("nan", "inf", "-1", "bad"):
            data = self.form()
            data["action"] = "save_adapter"
            data["adapter_strength:fixture-flux-style"] = strength
            before = self.style_user.read_bytes()
            self.assertEqual(self.client.post("/settings/adapters/fixture-flux-style", data=data).status_code, 422)
            self.assertEqual(self.style_user.read_bytes(), before)

    def test_unknown_and_duplicate_configured_adapters_rejected(self):
        for configured, enabled in ((["missing"], []), (["fixture-sdxl-style"], []),
                                     (["fixture-flux-style"] * 2, []),
                                     (["fixture-flux-style"], ["fixture-flux-style"] * 2)):
            data = self.form()
            data.setlist("configured_lora", configured)
            data.setlist("lora", enabled)
            before = self.user.read_bytes()
            self.assertEqual(self.post(data).status_code, 422)
            self.assertEqual(self.user.read_bytes(), before)

    def test_atomic_replace_failure_preserves_original_file_and_cleans_temporary(self):
        data = self.form()
        data["width"] = "512"
        before = self.user.read_bytes()
        with patch("image_generator.catalog.persistence.os.replace", side_effect=OSError("disk failure")):
            self.assertEqual(self.post(data).status_code, 422)
        self.assertEqual(self.user.read_bytes(), before)
        self.assertEqual(list(self.user.parent.glob(".model.user.json.*")), [])

    def test_service_detects_conflict_under_lock_and_rejects_out_of_scope_changes(self):
        catalog = self.catalog()
        revision = settings_revision(catalog)
        self.write(self.user, {"width": 512})
        with self.assertRaises(SettingsConflict):
            reset_settings(catalog, "fixture-flux", expected_revision=revision)
        catalog = self.catalog()
        with self.assertRaises(ValueError):
            save_model_settings(catalog, "fixture-flux", {"weights": "bad"}, scope=["weights"],
                                expected_revision=settings_revision(catalog))
        with self.assertRaises(ValueError):
            save_adapter_settings(catalog, "fixture-flux-style", {"recommended_generation": {}},
                                  expected_revision=settings_revision(catalog))

    def test_concurrent_saves_accept_only_one_editor_revision(self):
        catalog = self.catalog()
        revision = settings_revision(catalog)

        def save(width):
            try:
                save_model_settings(catalog, "fixture-flux", {"width": width}, scope=["width"],
                                    expected_revision=revision)
                return "saved"
            except SettingsConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(save, [512, 768]))
        self.assertCountEqual(results, ["saved", "conflict"])
        self.assertIn(self.saved()["width"], [512, 768])

    def test_save_empty_prompt_requires_no_weights_and_preview_still_requires_prompt(self):
        data = self.form()
        self.assertEqual(data["prompt"], "")
        with patch("image_generator.workflows.flux.FluxWorkflow.validate", side_effect=AssertionError("weights checked")), \
             patch("image_generator.workflows.flux.FluxWorkflow.generate", side_effect=AssertionError("GPU used")):
            self.assertEqual(self.post(data).status_code, 303)
        self.assertEqual(self.client.post("/generate/fixture-flux", data=self.form()).status_code, 422)

    def test_other_workflows_can_save_defaults_without_prompts_or_source_uploads(self):
        for family in ("sdxl", "wan", "minimax"):
            with self.subTest(family=family):
                model = f"fixture-{family}"
                data = self.form(model)
                data["prompt"] = ""
                response = self.post(data, model=model)
                self.assertEqual(response.status_code, 303)

    def test_cross_field_save_errors_preserve_edits_without_requiring_source_or_prompt(self):
        model = "fixture-wan"
        data = self.form(model)
        data["guidance_scale_2"] = "0.5"
        before = self.root / "WAN/checkpoints/demo/model.user.json"
        original = before.read_bytes()
        response = self.post(data, model=model)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(before.read_bytes(), original)
        page = FormPage(response.get_data(as_text=True))
        self.assertEqual(page.controls["guidance_scale_2"][0]["value"], "0.5")
        html = response.get_data(as_text=True)
        self.assertNotIn("This field is required.", html)
        self.assertNotIn("Select a source image.", html)


if __name__ == "__main__":
    unittest.main()
