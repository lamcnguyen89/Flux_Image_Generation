"""MiniMax notebook validation and LoRA stack checks without CUDA or model weights."""

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

NOTEBOOK = Path(__file__).resolve().parents[1] / "Models/Model_Families/MiniMax/MiniMax_Video_Gen.ipynb"


class MiniMaxNotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        self.namespace = {}
        exec(compile(self.cells["minimax-settings"], str(NOTEBOOK), "exec"), self.namespace)
        exec(compile(self.cells["minimax-helpers"], str(NOTEBOOK), "exec"), self.namespace)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.family = self.root / "MiniMax"
        self.write_header(self.family / "checkpoints/model.safetensors", {
            "__metadata__": {"config": json.dumps({"transformer": {"image_model": "minimax_h3"}})},
        })
        # Include the original extensionless filename as a supported VAE override.
        self.write_header(self.family / "vae", {
            "latents_mean": {}, "decoder.transformer_blocks.0.scale1": {},
        })
        (self.root / "source.png").touch()
        self.namespace.update(
            project_root=self.root, family_dir=self.family, checkpoint="model.safetensors",
            vae_weights="vae", input_image="source.png", output_dir="Videos", loras=[],
            text_encoder_weights="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            download_text_encoder=True,
        )

    @staticmethod
    def write_header(path, header):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(data)) + data)

    def validate(self):
        return self.namespace["validate_settings"]()

    def test_every_code_cell_compiles(self):
        for cell in json.loads(NOTEBOOK.read_text())["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_defaults_and_extensionless_vae_validate_without_loading_weights(self):
        paths, selected = self.validate()
        self.assertEqual(paths["vae"], self.family / "vae")
        self.assertEqual(selected, [])
        self.assertEqual(self.namespace["num_frames"] % 17, 5)
        self.assertEqual(self.namespace["fps"], 24)

    def test_wan_frame_count_and_bad_dimensions_are_rejected(self):
        for key, value in (("num_frames", 161), ("num_frames", 81), ("width", 848), ("fps", 16)):
            with self.subTest(key=key, value=value):
                previous = self.namespace[key]
                self.namespace[key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.namespace[key] = previous

    def test_wrong_checkpoint_and_wrong_vae_are_rejected(self):
        checkpoint = self.family / "checkpoints/model.safetensors"
        original = checkpoint.read_bytes()
        self.write_header(checkpoint, {"__metadata__": {"config": "{}"}})
        with self.assertRaisesRegex(ValueError, "MiniMax H3 checkpoint"):
            self.validate()
        checkpoint.write_bytes(original)
        self.write_header(self.family / "vae", {"audio_decoder": {}})
        with self.assertRaisesRegex(ValueError, "video VAE"):
            self.validate()

    def test_missing_text_encoder_can_be_downloaded_or_required_locally(self):
        self.validate()
        self.namespace["download_text_encoder"] = False
        # Disabling downloads still permits a shared-cache lookup at resolution time.
        self.validate()
        encoder = self.family / "text_encoders/local.safetensors"
        encoder.parent.mkdir(parents=True)
        encoder.touch()
        self.namespace["text_encoder_weights"] = "local.safetensors"
        self.validate()

    def test_encoder_uses_ui_cache_with_downloads_enabled_or_disabled(self):
        blob = self.root / "hub/blobs/encoder-hash"
        blob.parent.mkdir(parents=True)
        blob.touch()
        encoder = self.root / "hub/snapshots/revision/text_encoders/encoder.safetensors"
        encoder.parent.mkdir(parents=True)
        encoder.symlink_to(blob)
        for cache_location, expected_root in (
            ("", self.root / "Models"),
            ("Models/huggingface", self.root / "Models/huggingface"),
            (str(self.root / "external-cache"), self.root / "external-cache"),
            ("~/encoder-cache", Path.home() / "encoder-cache"),
        ):
            for download in (True, False):
                with self.subTest(cache_location=cache_location, download=download):
                    self.namespace["download_text_encoder"] = download
                    downloader = Mock(return_value=str(encoder))
                    dotenv = SimpleNamespace(load_dotenv=Mock())
                    with patch.dict("os.environ", {"CACHE_LOCATION": cache_location,
                                                  "HF_TOKEN": "fallback-token"}, clear=True), patch.dict(
                        sys.modules, {"dotenv": dotenv,
                                      "huggingface_hub": SimpleNamespace(hf_hub_download=downloader)}
                    ), redirect_stdout(io.StringIO()):
                        paths, _ = self.validate()
                        resolved = self.namespace["resolve_text_encoder"](paths["text_encoder"])
                    self.assertEqual(resolved, encoder)
                    dotenv.load_dotenv.assert_called_once_with(self.root / ".env")
                    downloader.assert_called_once_with(
                        repo_id="Comfy-Org/MiniMax-H3",
                        filename="text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        cache_dir=str(expected_root / "hub"), local_files_only=not download,
                        token="fallback-token",
                    )

    def test_encoder_prefers_family_local_file_over_cache(self):
        encoder = self.family / "text_encoders/local.safetensors"
        encoder.parent.mkdir(parents=True)
        encoder.touch()
        downloader = Mock()
        with patch.dict(sys.modules, {
            "huggingface_hub": SimpleNamespace(hf_hub_download=downloader)
        }):
            self.assertEqual(self.namespace["resolve_text_encoder"](encoder), encoder)
        downloader.assert_not_called()

    def test_custom_missing_encoder_is_not_downloaded_from_arbitrary_path(self):
        self.namespace["text_encoder_weights"] = "../custom.safetensors"
        with self.assertRaisesRegex(ValueError, "custom text encoder"):
            self.validate()

    def test_multiple_loras_resolve_independent_strengths_and_skip_zero(self):
        for name in ("first", "second", "disabled"):
            path = self.family / f"loras/{name}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.namespace["loras"] = [
            {"path": "first.safetensors", "strength": 0.8},
            {"path": str(self.family / "loras/second.safetensors"), "strength": 0.5},
            {"path": "disabled.safetensors", "strength": 0.0},
        ]
        _, selected = self.validate()
        self.assertEqual([item["strength"] for item in selected], [0.8, 0.5])
        self.assertTrue(all(item["path"].is_absolute() for item in selected))

    def test_invalid_or_duplicate_loras_are_rejected(self):
        path = self.family / "loras/first.safetensors"
        path.parent.mkdir(parents=True)
        path.touch()
        entry = {"path": "first.safetensors", "strength": 0.8}
        self.namespace["loras"] = [entry, entry]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate()
        for strength in (float("nan"), float("inf"), -1, True):
            self.namespace["loras"] = [{**entry, "strength": strength}]
            with self.assertRaisesRegex(ValueError, "strength"):
                self.validate()
        self.namespace["loras"] = [{"path": "missing.safetensors", "strength": 1.0}]
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_stack_passes_each_strength_and_restarts_from_base(self):
        base = SimpleNamespace(patches={})

        def load(model, clip, weights, strength, clip_strength, **kwargs):
            patches = {k: list(v) for k, v in model.patches.items()}
            patches.setdefault("weight", []).append((weights, strength))
            return SimpleNamespace(patches=patches), None

        loader = Mock(side_effect=load)
        self.namespace["comfy"] = SimpleNamespace(
            sd=SimpleNamespace(load_lora_for_models=loader),
            utils=SimpleNamespace(load_torch_file=Mock(side_effect=lambda p, **kw: (p, {}))),
        )
        selected = [{"path": Path("first.safetensors"), "strength": 0.8},
                    {"path": Path("second.safetensors"), "strength": 0.5}]
        stack = self.namespace["stack_loras"]
        model = stack(base, selected)
        self.assertEqual(model.patches["weight"], [("first.safetensors", 0.8), ("second.safetensors", 0.5)])
        self.assertEqual(base.patches, {})
        self.assertEqual(len(stack(base, selected[:1]).patches["weight"]), 1)
        self.assertIs(stack(base, []), base)
        self.assertTrue(all(call.args[1] is None and call.args[4] == 0.0 for call in loader.call_args_list))

    def test_unmatched_lora_fails_instead_of_silently_generating_base_model(self):
        base = SimpleNamespace(patches={})
        self.namespace["comfy"] = SimpleNamespace(
            sd=SimpleNamespace(load_lora_for_models=Mock(return_value=(base, None))),
            utils=SimpleNamespace(load_torch_file=Mock(return_value=({}, {}))),
        )
        with self.assertRaisesRegex(ValueError, "did not match"):
            self.namespace["stack_loras"](base, [{"path": Path("wan.safetensors"), "strength": 1.0}])


class MiniMaxSetupTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "minimax-setup"))
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "check_notebook_environment")
        self.run = Mock()
        namespace = {"subprocess": SimpleNamespace(run=self.run), "sys": sys}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
        self.check = namespace["check_notebook_environment"]
        self.report = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n"
        self.package = SimpleNamespace(
            version="0.8.1", read_text=Mock(return_value="Tag: py3-none-manylinux2014_sbsa\n"),
        )
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def run_check(self, report=None, status=1):
        from unittest.mock import patch

        self.run.side_effect = [subprocess.CompletedProcess([], status, report or self.report, ""),
                                subprocess.CompletedProcess([], 0)]
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="aarch64"), \
                patch("importlib.metadata.distribution", return_value=self.package):
            self.check()

    def test_clean_pip_check_needs_no_exception_or_probe(self):
        self.run_check(report="No broken requirements found.\n", status=0)
        self.assertEqual(self.run.call_count, 1)

    def test_exact_sbsa_issue_requires_fresh_cuda_probe(self):
        self.run_check()
        self.assertEqual(self.run.call_count, 2)
        command = self.run.call_args.args[0]
        self.assertEqual(command[:2], [sys.executable, "-c"])
        self.assertIn("torch.backends.cusparselt.version() != 801", command[2])
        self.assertTrue(self.run.call_args.kwargs["check"])

    def test_other_dependency_conflicts_are_not_suppressed(self):
        with self.assertRaisesRegex(RuntimeError, "pip check failed"):
            self.run_check(report=self.report + "other-package requires missing-package\n")
        self.assertEqual(self.run.call_count, 1)

    def test_wrong_wheel_tag_is_not_suppressed(self):
        self.package.read_text.return_value = "Tag: py3-none-manylinux2014_x86_64\n"
        with self.assertRaisesRegex(RuntimeError, "pip check failed"):
            self.run_check()
        self.assertEqual(self.run.call_count, 1)

    def test_same_diagnostic_on_another_platform_is_not_suppressed(self):
        from unittest.mock import patch

        self.run.return_value = subprocess.CompletedProcess([], 1, self.report, "")
        with patch("platform.machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "pip check failed"):
                self.check()
        self.assertEqual(self.run.call_count, 1)

    def test_failed_cuda_probe_stops_setup(self):
        from unittest.mock import patch

        self.run.side_effect = [subprocess.CompletedProcess([], 1, self.report, ""),
                                subprocess.CalledProcessError(1, "CUDA probe")]
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="aarch64"), \
                patch("importlib.metadata.distribution", return_value=self.package):
            with self.assertRaises(subprocess.CalledProcessError):
                self.check()


class MiniMaxGenerationContextTests(unittest.TestCase):
    def test_weight_transfers_keep_normal_tensors_without_recording_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in this test environment")

        notebook = json.loads(NOTEBOOK.read_text())
        source = "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "minimax-generate"))
        context = next(n for n in ast.parse(source).body if isinstance(n, ast.With)
                       and any(isinstance(child, ast.For) for child in n.body))
        # Exercise the notebook's real execution context with tiny model transfers.
        probe = ast.parse('''
assert not torch.is_grad_enabled()
embedding.to(dtype=torch.float64)
assert not embedding.weight.is_inference(), "Model transfers must preserve tensor version counters"
output = embedding(torch.tensor([0, 1]))
assert not output.requires_grad
assert not output.is_inference()
embedding.to(dtype=torch.float32)
assert not embedding.weight.is_inference()
''').body
        context.body = probe
        tree = ast.fix_missing_locations(ast.Module(body=[context], type_ignores=[]))
        embedding = torch.nn.Embedding(8, 8)
        previous_swap = torch.__future__.get_swap_module_params_on_conversion()
        try:
            torch.__future__.set_swap_module_params_on_conversion(True)
            exec(compile(tree, "minimax-generation-context", "exec"), {"torch": torch, "embedding": embedding})
        finally:
            torch.__future__.set_swap_module_params_on_conversion(previous_swap)


class MiniMaxQuantizedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import comfy_kitchen.tensor
        except ImportError:
            raise unittest.SkipTest("Requires the optional ComfyUI inference environment")
        runtime = NOTEBOOK.parents[3] / "build/ComfyUI"
        if not (runtime / "comfy/ops.py").is_file():
            raise unittest.SkipTest("Requires the local ComfyUI checkout")
        sys.path.insert(0, str(runtime))
        import comfy.ops
        import comfy.model_patcher

        cls.torch, cls.ops, cls.patcher = torch, comfy.ops, comfy.model_patcher
        notebook = json.loads(NOTEBOOK.read_text())
        source = "".join(next(c["source"] for c in notebook["cells"] if c["id"] == "minimax-helpers"))
        cls.namespace = {"torch": torch}
        exec(compile(source, "minimax-helpers", "exec"), cls.namespace)

    def embedding(self):
        torch = self.torch
        operations = self.ops.mixed_precision_ops(compute_dtype=torch.float16)
        embedding = operations.Embedding(8, 8, device="cpu", dtype=torch.float16)
        embedding.load_state_dict({
            "weight": torch.ones((8, 8), dtype=torch.int8),
            "weight_scale": torch.ones((8, 1)),
            "comfy_quant": torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()),
                                        dtype=torch.uint8),
        })
        return embedding

    def test_actual_int8_embedding_same_device_offload_inside_inference_mode(self):
        torch = self.torch
        root = torch.nn.Sequential(self.embedding())
        data_ptr = root[0].weight._qdata.data_ptr()
        self.assertEqual(self.namespace["protect_quantized_transfers"](root), 1)
        patcher = self.patcher.ModelPatcher(root, load_device=torch.device("cpu"),
                                            offload_device=torch.device("cpu"))
        with torch.inference_mode():
            patcher.unpatch_model(torch.device("cpu"))
            root.to("cpu")  # A no-op move is the case the old NVFP4 smoke test missed.
            self.assertFalse(root[0].weight.is_inference())
            self.assertGreaterEqual(root[0].weight._version, 0)
        self.assertEqual(root[0].weight._qdata.data_ptr(), data_ptr)

    def test_cached_component_guard_is_idempotent_and_leaves_plain_modules_alone(self):
        torch = self.torch
        embedding = self.embedding()
        plain = torch.nn.Linear(8, 8)
        plain_apply = plain._apply
        cache = {"base": SimpleNamespace(model=torch.nn.Sequential(embedding, plain)),
                 "clip": SimpleNamespace(cond_stage_model=embedding),
                 "vae": SimpleNamespace(first_stage_model=torch.nn.Identity())}
        self.assertEqual(self.namespace["protect_cached_transfers"](cache), 1)
        guarded = embedding._apply
        self.assertEqual(self.namespace["protect_cached_transfers"](cache), 0)
        self.assertEqual(plain._apply, plain_apply)
        self.assertIs(embedding._apply, guarded)


if __name__ == "__main__":
    unittest.main()
