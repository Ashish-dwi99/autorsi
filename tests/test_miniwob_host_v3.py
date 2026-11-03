"""Bounded fake lifecycle tests; no Modal, provider, model, or browser import."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "proposals/launch_private_miniwob_v2_host_v3.py"
spec = importlib.util.spec_from_file_location("host_v3_under_test", SOURCE)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class FakeCall:
    object_id = "fc-fixture"
    def __init__(self, admission):
        self.gets, self.cancels = [], []
        self.get_error = self.cancel_error = None
        self.result = {"function_call_id": self.object_id, "run": host.REMOTE_RUN_ID,
                       "source_manifest": admission["source_manifest"],
                       "status": "completed", "child_status": "completed",
                       "container_termination_required": False}
    def get(self, timeout):
        self.gets.append(timeout)
        if self.get_error:
            raise self.get_error
        return self.result
    def cancel(self, **kwargs):
        self.cancels.append(kwargs)
        if self.cancel_error:
            raise self.cancel_error


class FakeFunction:
    def __init__(self, admission):
        self.object_id = host.EXPECTED_FUNCTION_ID
        self.hydrates, self.spawns = 0, []
        self.hydrate_error = self.spawn_error = None
        self.call = FakeCall(admission)
    def hydrate(self):
        self.hydrates += 1
        if self.hydrate_error:
            raise self.hydrate_error
        return self
    def spawn(self, admission):
        self.spawns.append(admission)
        if self.spawn_error:
            raise self.spawn_error
        return self.call


class HostLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runs = Path(self.temporary.name)
        self.admission = {"run_id": host.REMOTE_RUN_ID, "source_manifest": {"frozen": "a" * 64}}
        self.deployment = {"app_name": host.APP_NAME, "app_id": host.EXPECTED_APP_ID,
                           "function_id": host.EXPECTED_FUNCTION_ID,
                           "source_manifest": self.admission["source_manifest"]}
        self.function = FakeFunction(self.admission)
        self.lookups = []
    def factory(self, *args, **kwargs):
        self.lookups.append((args, kwargs))
        return self.function
    def run_once(self, **kwargs):
        return host.dispatch_once(self.admission, self.deployment, self.runs, self.factory,
            sdk_receipt={"fake": True}, source_sha256="b" * 64,
            now=lambda: host.NOT_AFTER_UNIX - 1, clock=lambda: 0, **kwargs)
    def failure(self):
        return json.loads((self.runs / (host.LOCAL_ATTEMPT_ID + ".failure.json")).read_text())

    def test_exact_identity_one_call_retains_distinct_host_and_remote_ids(self):
        result = self.run_once()
        self.assertEqual(self.lookups, [((host.APP_NAME, "benchmark"), {"environment_name": "main"})])
        self.assertEqual(self.function.hydrates, 1)
        self.assertEqual(self.function.spawns, [self.admission])
        self.assertEqual(self.function.call.cancels, [])
        self.assertEqual(result["local_attempt_id"], host.LOCAL_ATTEMPT_ID)
        self.assertEqual(result["remote_run_id"], host.REMOTE_RUN_ID)
        self.assertEqual(result["new_reservation_usd"], 0)
        self.assertLessEqual(self.function.call.gets[0], host.HOST_DEADLINE_SECONDS)

    def test_hydrated_identity_mismatch_refuses_before_spawn(self):
        self.function.object_id = "fu-unexpected"
        with self.assertRaisesRegex(ValueError, "hydrated function ID"):
            self.run_once()
        self.assertEqual(self.function.spawns, [])
        self.assertEqual(self.failure()["submission_disposition"], "pre_submit_no_spawn_entered")

    def test_hydration_failure_is_pre_submit_and_never_retried(self):
        self.function.hydrate_error = RuntimeError("lookup unavailable")
        with self.assertRaisesRegex(RuntimeError, "lookup unavailable"):
            self.run_once()
        self.assertEqual((len(self.lookups), self.function.hydrates, len(self.function.spawns)), (1, 1, 0))
        self.assertFalse(self.failure()["spawn_entered"])

    def test_entered_spawn_failure_remains_unknown_without_retry(self):
        self.function.spawn_error = TimeoutError("no returned handle")
        with self.assertRaises(TimeoutError):
            self.run_once()
        self.assertEqual(len(self.function.spawns), 1)
        self.assertEqual(self.failure()["submission_disposition"], "entered_unknown_no_returned_handle")
        self.assertFalse(self.failure()["cancellation"]["attempted"])
        self.assertFalse(self.failure()["retry_authorized"])

    def test_call_receipt_write_failure_still_cancels_known_call(self):
        def save(path, value):
            if path.name.endswith(".call.json"):
                raise OSError("disk write failed")
            host.write_once(path, value)
        with self.assertRaisesRegex(OSError, "disk write failed"):
            self.run_once(save=save)
        self.assertEqual(self.function.call.cancels, [{"terminate_containers": True}])
        self.assertEqual(self.function.call.gets, [])
        self.assertEqual(self.failure()["submission_disposition"], "returned_handle")

    def test_mismatched_result_cancels_once(self):
        self.function.call.result["run"] = "other-run"
        with self.assertRaisesRegex(RuntimeError, "exact call/run/source"):
            self.run_once()
        self.assertEqual(len(self.function.call.cancels), 1)
        self.assertTrue((self.runs / (host.LOCAL_ATTEMPT_ID + ".remote-report.json")).exists())

    def test_aborted_remote_result_is_saved_and_cancellation_requested(self):
        self.function.call.result.update(status="aborted", container_termination_required=True)
        with self.assertRaisesRegex(RuntimeError, "container termination"):
            self.run_once()
        self.assertEqual(self.function.call.cancels, [{"terminate_containers": True}])
        saved = json.loads((self.runs / (host.LOCAL_ATTEMPT_ID + ".remote-report.json")).read_text())
        self.assertEqual(saved["status"], "aborted")

    def test_get_timeout_and_cancel_error_preserve_original_failure(self):
        self.function.call.get_error = TimeoutError("original result timeout")
        self.function.call.cancel_error = RuntimeError("cancel unavailable")
        with self.assertRaisesRegex(TimeoutError, "original result timeout"):
            self.run_once()
        self.assertEqual(len(self.function.call.cancels), 1)
        self.assertFalse(self.failure()["cancellation"]["request_returned"])

    def test_failed_deadline_setup_is_not_an_attempted_cancel_request(self):
        self.function.call.get_error = TimeoutError("original result timeout")
        def setup_failure():
            raise RuntimeError("could not arm cancellation deadline")
        with self.assertRaisesRegex(TimeoutError, "original result timeout"):
            self.run_once(before_cancel=setup_failure)
        self.assertEqual(self.function.call.cancels, [])
        self.assertFalse(self.failure()["cancellation"]["attempted"])

    def test_existing_host_marker_refuses_without_lookup(self):
        self.run_once()
        self.lookups.clear()
        with self.assertRaises(FileExistsError):
            self.run_once()
        self.assertEqual(self.lookups, [])
        self.assertEqual(len(self.function.spawns), 1)

    def test_invalid_remote_identity_refuses_without_lookup_or_marker(self):
        self.admission["run_id"] = "silently-renamed-runtime"
        with self.assertRaises(ValueError):
            self.run_once()
        self.assertEqual(self.lookups, [])
        self.assertEqual(list(self.runs.iterdir()), [])


class AlarmPhaseTests(unittest.TestCase):
    def test_first_alarm_during_cancel_does_not_grant_second_grace(self):
        arms = []
        handler, cancel = host.alarm_handlers(arms.append)
        cancel()
        self.assertEqual(arms, [20])
        with self.assertRaises(TimeoutError):
            handler(None, None)
        self.assertEqual(arms, [20])

    def test_main_deadline_grants_only_one_cancellation_phase(self):
        arms = []
        handler, cancel = host.alarm_handlers(arms.append)
        with self.assertRaises(TimeoutError):
            handler(None, None)
        self.assertEqual(arms, [20])
        with self.assertRaises(TimeoutError):
            handler(None, None)
        self.assertEqual(arms, [20])


if __name__ == "__main__":
    unittest.main()
