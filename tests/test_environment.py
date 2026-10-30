"""Exercise environment diagnostics without installing GPU dependencies."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote


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
                with patch.object(CHECKER, "check_minimax", return_value=[]):
                    self.assertEqual(CHECKER.main([]), 1)

    def test_missing_minimax_fails_the_overall_environment_check(self):
        with patch.object(CHECKER, "check_python", return_value=[]):
            with patch.object(CHECKER, "check_imports", return_value=({}, [])):
                with patch.object(CHECKER, "check_minimax", return_value=["missing native H3 runtime"]):
                    self.assertEqual(CHECKER.main([]), 1)

    def test_missing_minimax_checkout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            errors = CHECKER.check_minimax(directory)
        self.assertIn("MiniMax ComfyUI runtime missing", errors[0])

    def test_minimax_native_node_import_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            node = Path(directory) / "comfy_extras/nodes_minimax_h3.py"
            node.parent.mkdir()
            node.touch()
            with patch.object(CHECKER.importlib, "import_module", side_effect=ImportError("missing scipy")):
                errors = CHECKER.check_minimax(directory)
            self.assertNotIn(str(Path(directory).resolve()), sys.path)
        self.assertIn("missing scipy", errors[0])

    def test_cpu_inspection_imports_minimax_nodes_without_enabling_cuda(self):
        modules = {
            "comfy.options": SimpleNamespace(enable_args_parsing=Mock()),
            "comfy.cli_args": SimpleNamespace(args=SimpleNamespace(cpu=False)),
            "comfy.sd": SimpleNamespace(CLIPType=SimpleNamespace(MINIMAX="minimax")),
            "comfy.samplers": SimpleNamespace(SAMPLER_NAMES=["res_multistep"], SCHEDULER_NAMES=["simple"]),
        }
        def import_module(name):
            return modules.get(name, Mock())
        with tempfile.TemporaryDirectory() as directory:
            node = Path(directory) / "comfy_extras/nodes_minimax_h3.py"
            node.parent.mkdir()
            node.touch()
            with patch.object(CHECKER.importlib, "import_module", side_effect=import_module) as importer:
                self.assertEqual(CHECKER.check_minimax(directory, allow_cpu=True), [])
            self.assertTrue(modules["comfy.cli_args"].args.cpu)
            self.assertIn(unittest.mock.call("comfy_extras.nodes_minimax_h3"), importer.call_args_list)

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


class SetupInterpreterTests(unittest.TestCase):
    """Run setup with isolated interpreters and simulated pip/venv operations."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "scripts").mkdir()
        self.script = self.root / "scripts/setup.sh"
        shutil.copyfile(ROOT / "scripts/setup.sh", self.script)
        self.comfyui = self.root / "build/ComfyUI"
        (self.comfyui / "comfy_extras").mkdir(parents=True)
        (self.comfyui / "comfy_extras/nodes_minimax_h3.py").touch()
        (self.comfyui / "requirements.txt").write_text("torch\ntorchvision\n")
        (self.root / ".python-version").write_text(
            f"{sys.version_info.major}.{sys.version_info.minor}\n"
        )
        self.conda = self.root / "conda"
        (self.conda / "conda-meta").mkdir(parents=True)
        self.environment = os.environ.copy()
        for name in ("PYTHON_BIN", "TORCH_WHEEL", "TORCHVISION_WHEEL"):
            self.environment.pop(name, None)

    def interpreter(self, prefix, base_prefix=None, simulate_install=False):
        executable = prefix / "bin/python"
        executable.parent.mkdir(parents=True, exist_ok=True)
        commands = ""
        if simulate_install:
            commands = (
                "from pathlib import Path\n"
                "import json, shutil\n"
                "if sys.argv[1:3] == ['-m', 'venv']:\n"
                "    target = Path(sys.argv[3]) / 'bin/python'\n"
                "    target.parent.mkdir(parents=True)\n"
                "    shutil.copy2(__file__, target)\n"
                "    raise SystemExit(0)\n"
                "if sys.argv[1:3] == ['-m', 'pip']:\n"
                f"    with open({str(self.root / 'pip-arguments.jsonl')!r}, 'a') as log:\n"
                "        log.write(json.dumps(sys.argv[3:]) + '\\n')\n"
                "    raise SystemExit(0)\n"
                f"if sys.argv[1] == {str(self.root / 'scripts/check_environment.py')!r}:\n"
                "    raise SystemExit(0)\n"
            )
        # Model interpreter prefixes without needing a real Conda installation.
        # Execute setup's actual Python preflight against that runtime metadata.
        executable.write_text(
            f"#!{sys.executable}\n"
            "import platform, sys\n"
            f"sys.prefix = {str(prefix)!r}\n"
            f"sys.base_prefix = {str(base_prefix or prefix)!r}\n"
            f"sys.executable = {str(executable)!r}\n"
            "platform.system = lambda: 'Linux'\n"
            "platform.machine = lambda: 'aarch64'\n"
            + commands +
            "assert sys.argv[1] == '-', 'Setup attempted an unexpected operation'\n"
            "sys.argv = sys.argv[1:]\n"
            "exec(compile(sys.stdin.read(), '<setup preflight>', 'exec'))\n"
        )
        executable.chmod(0o755)
        return executable

    def run_setup(self, *arguments):
        return subprocess.run(
            ["bash", str(self.script), *map(str, arguments)],
            cwd="/tmp", env=self.environment,
            capture_output=True, text=True, timeout=10,
        )

    def test_missing_python_explains_standalone_requirement(self):
        result = self.run_setup("--python", self.root / "missing-python")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Install standalone Python 3.11", result.stderr)

    def test_conda_interpreter_is_rejected(self):
        python = self.interpreter(self.conda)
        result = self.run_setup("--python", python)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("backed by Conda", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def test_existing_venv_with_conda_base_is_rejected(self):
        self.interpreter(self.root / ".venv", base_prefix=self.conda)
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("backed by Conda", result.stderr)
        self.assertIn("move it aside", result.stderr)

    def test_explicit_standalone_python_does_not_bypass_existing_conda_venv(self):
        python = self.interpreter(self.root / "standalone")
        self.interpreter(self.root / ".venv", base_prefix=self.conda)
        result = self.run_setup("--python", python)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("backed by Conda", result.stderr)

    def test_standalone_python_is_accepted_with_conda_active_in_shell(self):
        python = self.interpreter(self.root / "standalone")
        self.environment["CONDA_PREFIX"] = str(self.conda)
        result = self.run_setup("--python", python, "--torch-wheel", self.root / "missing.whl")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--torch-wheel must name an existing .whl file", result.stderr)
        self.assertNotIn("backed by Conda", result.stderr)
        self.assertFalse((self.root / ".venv").exists())

    def install_arguments(self):
        return json.loads((self.root / "pip-arguments.jsonl").read_text().splitlines()[0])

    def test_first_setup_uses_official_wheels_matching_requirements(self):
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        result = self.run_setup("--python", python)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.install_arguments()
        wheels = [arg for arg in arguments if arg.startswith("https://")]
        self.assertEqual(len(wheels), 2)
        for package, wheel in zip(("torch", "torchvision"), wheels):
            pin = next(
                line.split("==")[1] for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.startswith(package + "==")
            )
            self.assertEqual(
                unquote(wheel),
                f"https://download-r2.pytorch.org/whl/cu132/{package}-{pin}-cp311-cp311-manylinux_2_28_aarch64.whl",
            )
        self.assertIn(str(self.root / "requirements.txt"), arguments)
        self.assertIn(str(self.comfyui / "requirements.txt"), arguments)
        self.assertEqual(arguments[arguments.index("-c") + 1], str(self.root / "requirements.txt"))

    def test_fresh_setup_fetches_pinned_comfyui_and_rerun_preserves_it(self):
        shutil.rmtree(self.comfyui)
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        git = fake_bin / "git"
        git.write_text(
            f"#!{sys.executable}\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            f"with open({str(self.root / 'git-arguments.jsonl')!r}, 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] == 'init':\n"
            "    Path(sys.argv[2]).mkdir(parents=True)\n"
            "if 'checkout' in sys.argv:\n"
            "    checkout = Path(sys.argv[2])\n"
            "    (checkout / 'comfy_extras').mkdir()\n"
            "    (checkout / 'comfy_extras/nodes_minimax_h3.py').touch()\n"
            "    (checkout / 'requirements.txt').write_text('torch\\ntorchvision\\n')\n"
        )
        git.chmod(0o755)
        self.environment["PATH"] = str(fake_bin) + os.pathsep + self.environment["PATH"]
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        result = self.run_setup("--python", python)
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.root / "git-arguments.jsonl"
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(commands[0], ["init", str(self.comfyui)])
        self.assertEqual(commands[1][-1], "https://github.com/Comfy-Org/ComfyUI.git")
        self.assertEqual(commands[2][3:6], ["--depth", "1", "origin"])
        self.assertRegex(commands[2][-1], r"^[0-9a-f]{40}$")
        self.assertEqual(commands[3][-3:], ["checkout", "--detach", "FETCH_HEAD"])
        marker = self.comfyui / "local-changes.txt"
        marker.write_text("preserve this checkout")
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(log.read_text().splitlines()), 4)
        self.assertEqual(marker.read_text(), "preserve this checkout")

    def test_existing_checkout_without_h3_fails_without_installing(self):
        (self.comfyui / "comfy_extras/nodes_minimax_h3.py").unlink()
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        result = self.run_setup("--python", python)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks native MiniMax H3 support", result.stderr)
        self.assertFalse((self.root / "pip-arguments.jsonl").exists())
        self.assertFalse((self.root / ".venv").exists())

    def test_dev_tools_install_after_complete_runtime_with_shared_constraints(self):
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        result = self.run_setup("--python", python, "--dev")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [json.loads(line) for line in (self.root / "pip-arguments.jsonl").read_text().splitlines()]
        self.assertIn(str(self.comfyui / "requirements.txt"), commands[0])
        self.assertEqual(commands[1], ["install", "-r", str(self.root / "requirements-dev.txt")])
        self.assertIn("--no-deps", commands[2])
        self.assertEqual(commands[3], ["check"])

    def test_explicit_wheel_overrides_environment_and_preserves_other_default(self):
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        self.environment["TORCH_WHEEL"] = "https://example.com/environment.whl"
        wheel = self.root / "local wheels/torch.whl"
        wheel.parent.mkdir()
        wheel.touch()
        result = self.run_setup("--python", python, "--torch-wheel", wheel)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.install_arguments()
        self.assertIn(str(wheel), arguments)
        self.assertNotIn(self.environment["TORCH_WHEEL"], arguments)
        self.assertTrue(any("/torchvision-" in arg for arg in arguments))

    def test_environment_can_override_both_wheels(self):
        python = self.interpreter(self.root / "standalone", simulate_install=True)
        self.environment["TORCH_WHEEL"] = "https://example.com/torch.whl"
        self.environment["TORCHVISION_WHEEL"] = "https://example.com/torchvision.whl"
        result = self.run_setup("--python", python)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self.install_arguments()
        self.assertIn(self.environment["TORCH_WHEEL"], arguments)
        self.assertIn(self.environment["TORCHVISION_WHEEL"], arguments)


if __name__ == "__main__":
    unittest.main()
