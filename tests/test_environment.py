"""Exercise environment diagnostics without installing GPU dependencies."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_environment", ROOT / "scripts/check_environment.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.stdout = redirect_stdout(io.StringIO())
        self.stderr = redirect_stderr(io.StringIO())
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_lazy_pipeline_import_failure_is_reported(self):
        class BrokenPipelineModule:
            def __getattr__(self, name):
                raise ImportError("pipeline dependency unavailable")

        with patch.object(CHECKER, "IMPORTS", {"diffusers": ("diffusers", ("WanImageToVideoPipeline",))}):
            with patch.object(CHECKER.importlib, "import_module", return_value=BrokenPipelineModule()):
                modules, errors = CHECKER.check_imports()
        self.assertEqual(modules, {})
        self.assertIn("pipeline dependency unavailable", errors[0])

    def test_missing_cuda_fails_by_default(self):
        torch = self.fake_torch(cuda_available=False)
        errors = CHECKER.check_cuda(torch)
        self.assertIn("CUDA is unavailable", errors[0])

    def test_cpu_dependency_inspection_is_explicitly_allowed(self):
        torch = self.fake_torch(cuda_available=False)
        self.assertEqual(CHECKER.check_cuda(torch, allow_cpu=True), [])

    def test_bf16_is_required_for_current_workflows(self):
        torch = self.fake_torch(bf16_supported=False)
        self.assertIn("BF16", CHECKER.check_cuda(torch)[0])

    def test_cuda_runtime_failure_is_reported(self):
        torch = self.fake_torch()
        torch.inference_mode = nullcontext
        torch.float16, torch.bfloat16 = "fp16", "bf16"
        torch.ones = Mock(side_effect=RuntimeError("no kernel image is available"))
        errors = CHECKER.check_cuda(torch)
        self.assertIn("no kernel image", errors[0])

    def test_unusable_ffmpeg_binary_is_reported(self):
        module = SimpleNamespace(get_ffmpeg_exe=lambda: "/fake/ffmpeg")
        with patch.object(CHECKER.subprocess, "run", side_effect=OSError("cannot execute binary")):
            errors = CHECKER.check_ffmpeg(module)
        self.assertIn("cannot execute binary", errors[0])

    def test_failed_imports_produce_nonzero_exit(self):
        with patch.object(CHECKER, "check_python", return_value=[]):
            with patch.object(CHECKER, "check_imports", return_value=({}, ["missing torch"])):
                self.assertEqual(CHECKER.main([]), 1)

    @staticmethod
    def fake_torch(cuda_available=True, bf16_supported=True):
        return SimpleNamespace(
            __version__="test-build",
            version=SimpleNamespace(cuda="test-cuda"),
            cuda=SimpleNamespace(
                is_available=lambda: cuda_available,
                get_device_name=lambda index: "Test GPU",
                is_bf16_supported=lambda: bf16_supported,
            ),
        )


class SetupArgumentTests(unittest.TestCase):
    def run_setup(self, *arguments):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/setup.sh"), *arguments],
            cwd="/tmp", capture_output=True, text=True, timeout=10,
        )

    def test_help_works_outside_project_without_an_interpreter(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--torch-wheel", result.stdout)

    def test_unknown_option_fails_before_setup(self):
        result = self.run_setup("--invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown option", result.stderr)

    def test_missing_wheel_argument_fails_before_setup(self):
        result = self.run_setup("--torch-wheel")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires a value", result.stderr)


if __name__ == "__main__":
    unittest.main()
