"""Exercise service toggling without invoking the host's administrative tools."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UNIT = "image-generator.service"
DESTINATION = f"/etc/systemd/system/{UNIT}"

FAKE_TOOL = r'''
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == 'sudo':
    os.execv(str(Path(sys.argv[0]).parent / args[0]), args)
root = Path(os.environ['TOGGLE_TEST_ROOT'])
state_path = root / 'state.json'
state = json.loads(state_path.read_text())
args = [arg for arg in args if arg != '--system']
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps([name, *args]) + '\n')
operation = args[0] if name == 'systemctl' else name
if state.get('fail') == operation:
    sys.exit(1)
if name == 'systemctl':
    if operation == 'show':
        print('LoadState=' + state['load'])
        print('ActiveState=' + state['active'])
        print('FragmentPath=' + state['fragment'])
    elif operation == 'stop':
        state['active'] = 'inactive'
    elif operation == 'disable':
        state['enabled'] = False
    elif operation == 'enable':
        assert args == ['enable', '--now', 'image-generator.service'], args
        state['enabled'] = True
        state['active'] = 'active'
    elif operation == 'is-active':
        sys.exit(0 if state['active'] == 'active' else 3)
    elif operation == 'reset-failed':
        state['active'] = 'inactive'
    elif operation == 'daemon-reload':
        pass
    else:
        raise AssertionError(args)
elif name == 'install':
    assert args[:3] == ['-m', '0644', '--'], args
    assert args[-1] == '/etc/systemd/system/image-generator.service', args
    state['unit_contents'] = Path(args[-2]).read_text()
    state['fragment'] = args[-1]
    state['load'] = 'loaded'
elif name == 'rm':
    assert args == ['-f', '--', '/etc/systemd/system/image-generator.service'], args
    state['fragment'] = ''
    state['load'] = 'not-found'
    state.pop('unit_contents', None)
else:
    raise AssertionError(name)
state_path.write_text(json.dumps(state))
'''


class ToggleServicesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project with spaces"
        scripts = self.project / "scripts"
        scripts.mkdir(parents=True)
        self.script = scripts / "toggle_services.sh"
        shutil.copy2(ROOT / "scripts/toggle_services.sh", self.script)
        source_unit = self.project / "systemd" / UNIT
        source_unit.parent.mkdir()
        shutil.copy2(ROOT / "systemd" / UNIT, source_unit)
        python = self.project / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("systemctl", "sudo", "install", "rm"):
            tool = bin_dir / name
            tool.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
            tool.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["PATH"] = str(bin_dir) + os.pathsep + os.defpath
        self.environment["TOGGLE_TEST_ROOT"] = str(self.root)

    def run_toggle(self, *, load="loaded", active="inactive", fragment=DESTINATION,
                   fail=None, arguments=()):
        state_path = self.root / "state.json"
        state_path.write_text(json.dumps({
            "load": load, "active": active, "fragment": fragment,
            "enabled": True, "fail": fail,
        }))
        result = subprocess.run(
            ["/bin/bash", str(self.script), *arguments], cwd="/tmp",
            env=self.environment, text=True, capture_output=True, timeout=10,
        )
        calls_path = self.root / "calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] \
            if calls_path.exists() else []
        return result, json.loads(state_path.read_text()), calls

    def test_missing_stopped_and_failed_services_are_installed_and_started(self):
        for load, active, fragment in (
            ("not-found", "inactive", ""),
            ("loaded", "inactive", DESTINATION),
            ("loaded", "failed", DESTINATION),
        ):
            with self.subTest(active=active, load=load):
                result, state, calls = self.run_toggle(
                    load=load, active=active, fragment=fragment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(state["active"], "active")
                self.assertTrue(state["enabled"])
                self.assertEqual(state["unit_contents"], (ROOT / "systemd" / UNIT).read_text())
                self.assertFalse(any(call[0] == "rm" for call in calls))
                if active == "failed":
                    self.assertIn(["systemctl", "reset-failed", UNIT], calls)

    def test_running_or_starting_service_is_stopped_disabled_and_removed_in_order(self):
        for active in ("active", "activating", "reloading"):
            with self.subTest(active=active):
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                result, state, calls = self.run_toggle(active=active)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(state["active"], "inactive")
                self.assertFalse(state["enabled"])
                self.assertEqual(state["fragment"], "")
                self.assertEqual(calls[1:], [
                    ["systemctl", "stop", UNIT],
                    ["systemctl", "disable", UNIT],
                    ["rm", "-f", "--", DESTINATION],
                    ["systemctl", "daemon-reload"],
                ])
                self.assertTrue((self.project / "systemd" / UNIT).exists())

    def test_failed_stop_or_disable_keeps_installed_unit(self):
        for operation in ("stop", "disable"):
            with self.subTest(operation=operation):
                result, state, calls = self.run_toggle(active="active", fail=operation)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(state["fragment"], DESTINATION)
                self.assertFalse(any(call[0] == "rm" for call in calls))

    def test_failed_start_is_reported_and_unit_is_kept(self):
        result, state, calls = self.run_toggle(fail="enable")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(state["fragment"], DESTINATION)
        self.assertIn("journalctl", result.stderr)
        self.assertFalse(any(call[0] == "rm" for call in calls))

    def test_unavailable_manager_prevents_changes(self):
        result, state, calls = self.run_toggle(fail="show")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("Cannot query systemd", result.stderr)

    def test_masked_foreign_and_stopping_services_are_left_alone(self):
        for overrides in (
            {"load": "masked"},
            {"fragment": f"/usr/lib/systemd/system/{UNIT}", "active": "active"},
            {"active": "deactivating"},
        ):
            with self.subTest(overrides=overrides):
                (self.root / "calls.jsonl").unlink(missing_ok=True)
                result, state, calls = self.run_toggle(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(calls), 1)

    def test_uninstall_does_not_require_source_unit_or_virtual_environment(self):
        (self.project / "systemd" / UNIT).unlink()
        (self.project / ".venv/bin/python").unlink()
        result, state, calls = self.run_toggle(active="active")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state["fragment"], "")

    def test_missing_python_prevents_installation(self):
        (self.project / ".venv/bin/python").unlink()
        result, state, calls = self.run_toggle()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("setup.sh", result.stderr)

    def test_help_and_bad_arguments_do_not_call_system_tools(self):
        for arguments, expected_code in ((("--help",), 0), (("--unknown",), 1)):
            with self.subTest(arguments=arguments):
                result, state, calls = self.run_toggle(arguments=arguments)
                self.assertEqual(result.returncode, expected_code)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
