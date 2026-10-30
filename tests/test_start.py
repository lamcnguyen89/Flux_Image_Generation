"""Exercise combined-service shutdown with real, isolated child processes."""

import os
from pathlib import Path
import json
import shutil
import signal
import subprocess
import sys
import tempfile
from threading import Event
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts/start.sh"

SERVICE = '''
from pathlib import Path
import json
import os
import signal
import sys
import time
root = Path(os.environ['LAUNCHER_TEST_ROOT'])
if name == 'worker' and os.environ.get('LAUNCHER_TEST_FAIL_WORKER'):
    deadline = time.monotonic() + 5
    while not (root / 'web.ready').exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raise SystemExit(2)
def stop(number, frame):
    if name == 'worker':
        time.sleep(0.15)  # Finish an active generation before exiting.
    (root / (name + '.finished')).write_text('saved and stopped')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / (name + '.arguments')).write_text(json.dumps({
    'argv': sys.argv[1:], 'cwd': str(Path.cwd()),
    'database': os.environ.get('IMAGE_GENERATOR_JOB_DATABASE'),
    'outputs': os.environ.get('IMAGE_GENERATOR_JOB_OUTPUT_ROOT'),
}))
(root / (name + '.ready')).write_text(str(os.getpid()))
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    trigger = root / (name + '.exit')
    if trigger.exists():
        raise SystemExit(int(trigger.read_text()))
    time.sleep(0.01)
'''


class StartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "application with spaces"
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        self.launcher = scripts / "start.sh"
        shutil.copy2(START, self.launcher)
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        # Exercise the complete shell entry point with isolated CPU-only services.
        flask = self.root / "fakes/flask"
        flask.mkdir(parents=True)
        (flask / "__init__.py").write_text("")
        (flask / "cli.py").write_text("def load_dotenv(): pass\n")
        (flask / "__main__.py").write_text("name = 'web'\n" + SERVICE)
        worker = self.root / "src/image_generator"
        worker.mkdir(parents=True)
        (worker / "__init__.py").write_text("")
        (worker / "worker.py").write_text("name = 'worker'\n" + SERVICE)

    def wait_ready(self, names, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all((self.root / f"{name}.ready").exists() for name in names):
                return
            if process.poll() is not None:
                self.fail(f"Service exited before becoming ready: {process.returncode}")
            Event().wait(0.01)
        self.fail("Timed out waiting for child services")

    def start(self, *, wait_for=("web", "worker"), fail_worker=False):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.root / "fakes")
        environment["LAUNCHER_TEST_ROOT"] = str(self.root)
        if fail_worker:
            environment["LAUNCHER_TEST_FAIL_WORKER"] = "1"
        process = subprocess.Popen(["bash", str(self.launcher), "--database", str(self.root / "queue.sqlite3"),
                                    "--output-dir", str(self.root / "generated images"),
                                    "--device", "cuda:3", "--dtype", "float32"],
                                   cwd="/tmp", env=environment, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.close, process)
        self.wait_ready(wait_for, process)
        return process

    @staticmethod
    def close(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=12)

    def assert_reaped(self, name):
        pid = int((self.root / f"{name}.ready").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_ctrl_c_stops_web_and_waits_for_worker_to_finish(self):
        process = self.start()
        for name in ("web", "worker"):
            settings = json.loads((self.root / f"{name}.arguments").read_text())
            self.assertEqual(settings["database"], str(self.root / "queue.sqlite3"))
            self.assertEqual(settings["outputs"], str(self.root / "generated images"))
            self.assertEqual(settings["cwd"], str(self.root))
        self.assertEqual(settings["argv"][-4:], ["--device", "cuda:3", "--dtype", "float32"])
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130, stderr)
        self.assertIn("active generation", stdout)
        for name in ("web", "worker"):
            self.assertTrue((self.root / f"{name}.finished").is_file())
            self.assert_reaped(name)

    def test_service_failure_stops_companion_and_preserves_failure_status(self):
        process = self.start()
        (self.root / "web.exit").write_text("7")
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 7, stderr)
        self.assertIn("Web server exited", stderr)
        self.assertTrue((self.root / "worker.finished").is_file())
        self.assert_reaped("web")
        self.assert_reaped("worker")

    def test_worker_startup_failure_stops_web_server(self):
        process = self.start(wait_for=("web",), fail_worker=True)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2, stderr)
        self.assertIn("Generation worker exited", stderr)
        self.assertTrue((self.root / "web.finished").is_file())
        self.assert_reaped("web")


if __name__ == "__main__":
    unittest.main()
