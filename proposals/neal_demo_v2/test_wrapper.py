"""Pure fake bridge/lifecycle/host tests; no provider, model, or site calls."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import neal_bridge as bridge
import neal_control as control
import neal_lifecycle as lifecycle
from neal_limits import *
from neal_protocol import make_observation
from neal_worker import merge_harness_progress


def observation(seq=1):
    return make_observation(session_id=SESSION_ID, actor="assistant", level=1, seq=seq,
        frame_id=SESSION_ID + ":1:assistant:fixture", png=b"\x89PNG\r\n\x1a\nfixture",
        visible_dom={"text": "visible fixture", "elements": []}, remaining_actions=32,
        remaining_wall_s=90, prior_action_result=None)


def command(obs):
    return {**{key: obs[key] for key in bridge.IDENTITY_FIELDS},
            "native_action_json": '{"action":[{"done":{"text":"fixture","success":false}}]}'}


class FakeStore:
    def __init__(self, data=None): self.data, self.puts, self.gets = dict(data or {}), [], []
    def put(self, key, value, *, skip_if_exists):
        self.puts.append((key, value, skip_if_exists))
        if key in self.data: return False
        self.data[key] = value
        return True
    def get(self, key):
        self.gets.append(key)
        return self.data.get(key)


class FakeBroker:
    async def call(self, operation, timeout=5): return operation()


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_exact_command_consumed_and_duplicate_sequence_refused(self):
        obs = observation()
        store = FakeStore({SESSION_ID + ":action:1": command(obs)})
        actor = bridge.AssistantBridge(store, "a" * 64, broker=FakeBroker())
        result = await actor.exchange(obs)
        self.assertEqual(result, command(obs))
        self.assertEqual(actor.last_seq, 1)
        with self.assertRaises(ValueError): await actor.exchange(obs)
        self.assertEqual(len(store.puts), 1)

    async def test_stale_frame_latches_failure_without_next_observation(self):
        obs = observation()
        wrong = command(obs); wrong["frame_id"] += "-stale"
        store = FakeStore({SESSION_ID + ":action:1": wrong})
        actor = bridge.AssistantBridge(store, "a" * 64, broker=FakeBroker())
        with self.assertRaises(ValueError): await actor.exchange(obs)
        with self.assertRaises(ValueError): await actor.exchange(observation(2))
        self.assertTrue(actor.failed)
        self.assertEqual(len(store.puts), 1)

    async def test_occupied_observation_key_is_never_overwritten(self):
        obs = observation()
        store = FakeStore({SESSION_ID + ":observation:1": "prior evidence"})
        actor = bridge.AssistantBridge(store, "a" * 64, broker=FakeBroker())
        with self.assertRaises(ValueError): await actor.exchange(obs)
        self.assertEqual(store.data[SESSION_ID + ":observation:1"], "prior evidence")
        self.assertEqual(store.gets, [])

    async def test_matching_readiness_required(self):
        ready = {"session_id": SESSION_ID, "actor": "assistant", "source_manifest_sha256": "a" * 64,
                 "ready": True, "expires_unix": NOT_AFTER_UNIX}
        actor = bridge.AssistantBridge(FakeStore({SESSION_ID + ":ready": ready}), "a" * 64, broker=FakeBroker())
        with patch("neal_bridge.time.time", return_value=NOT_AFTER_UNIX - 1):
            self.assertEqual(await actor.wait_ready(), ready)
        ready["source_manifest_sha256"] = "b" * 64
        actor = bridge.AssistantBridge(FakeStore({SESSION_ID + ":ready": ready}), "a" * 64, broker=FakeBroker())
        with self.assertRaises(ValueError): await actor.wait_ready()

    async def test_broker_exception_latches_no_second_request(self):
        broker = bridge.SerialBroker()
        calls = []
        def failed():
            calls.append(1)
            raise RuntimeError("fake SDK failure")
        with self.assertRaises(bridge.BrokerUnknown): await broker.call(failed, timeout=1)
        with self.assertRaises(bridge.BrokerUnknown): await broker.call(lambda: calls.append(2), timeout=1)
        self.assertEqual(calls, [1])


class GuardTests(unittest.TestCase):
    def test_hash_and_identity_types_are_checked(self):
        obs = observation()
        bridge.validate_observation(obs)
        changed = copy.deepcopy(obs); changed["visible_dom"]["text"] = "changed"
        with self.assertRaises(ValueError): bridge.validate_observation(changed)
        action = command(obs); action["seq"] = True
        with self.assertRaises(ValueError): bridge.validate_action_identity(action, obs)
        bridge.validate_observation(observation(96))
        with self.assertRaises(ValueError): observation(97)

    def test_concurrent_assistant_progress_preserves_generation_watchdog(self):
        report = {"native_pending": True, "stage": "generation", "stage_started_monotonic": 10}
        merge_harness_progress(report, {"phase": "finalizing", "browser_close_returned": True}, 100)
        self.assertEqual((report["stage"], report["stage_started_monotonic"]), ("generation", 10))
        report["native_pending"] = False
        merge_harness_progress(report, {"phase": "finalizing", "browser_close_returned": True}, 101)
        self.assertEqual((report["stage"], report["stage_started_monotonic"]), ("finalizing", 101))

    def admission(self):
        source = {"fixture": "a" * 64}
        return {"run_id": CPU_RUN_ID, "session_id": SESSION_ID, "levels": list(LEVELS),
                "source_manifest": source, "source_manifest_sha256": lifecycle.manifest_sha(source),
                "budget_allocation": {"source_manifest_sha256": lifecycle.manifest_sha(source),
                    "allocated_usd": 3, "total_committed_usd": 21, "ledger_sha256": "b" * 64}}, source

    def test_zero_or_mismatched_allocation_refuses(self):
        admission, source = self.admission()
        with patch("neal_lifecycle.time.time", return_value=NOT_AFTER_UNIX - 1):
            lifecycle.validate_admission(admission, "preflight", source)
            admission["budget_allocation"]["allocated_usd"] = 0
            with self.assertRaises(ValueError): lifecycle.validate_admission(admission, "preflight", source)

    def test_video_configuration_does_not_substitute_for_decode_receipt(self):
        admission, source = self.admission()
        admission["run_id"] = GPU_RUN_ID
        video = {"sha256": "c" * 64, "bytes": 10}
        child = {"status": "completed", "model_calls": 0, "cuda_initialized": False,
            "runtime_versions": {"fixture": "1"}, "browser_runtime_manifest": {"fixture": "sha"},
            "demo": {"status": "completed", "browser_close_returned": True, "planned_levels": list(LEVELS),
                "levels": [1, 3, 5], "artifacts": [video], "blocked_browser_requests": []}}
        admission.update(cpu_receipt={"status": "completed", "child_exitcode": 0, "child_reaped": True,
            "source_manifest": source, "child": child}, runtime_versions=child["runtime_versions"],
            browser_runtime_manifest=child["browser_runtime_manifest"])
        with patch("neal_lifecycle.time.time", return_value=NOT_AFTER_UNIX - 1):
            with self.assertRaisesRegex(ValueError, "decode"):
                lifecycle.validate_admission(admission, "demo", source)
            child["media_inspection"] = {"status": "decoded_frames_verified", "width": 970, "height": 650,
                "frames": 1, "video_sha256": video["sha256"], "video_bytes": 10,
                "ffprobe_returncode": 0, "stderr_bytes": 0}
            lifecycle.validate_admission(admission, "demo", source)

    def test_artifact_symlinks_and_second_video_refuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "first.webm").write_bytes(b"fixture")
            self.assertEqual(len(lifecycle.artifact_inventory(root, hash_files=True)), 1)
            (root / "second.webm").write_bytes(b"fixture")
            with self.assertRaises(ValueError): lifecycle.artifact_inventory(root)
            (root / "second.webm").unlink()
            (root / "unsafe.png").symlink_to(root / "first.webm")
            with self.assertRaises(ValueError): lifecycle.artifact_inventory(root)

    def test_local_mutation_intent_prevents_duplicate_delivery_after_unknown_rpc(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(control, "RUNS", Path(folder)):
            calls = []
            def unknown():
                calls.append(1)
                raise TimeoutError("fake unknown delivery")
            with self.assertRaises(TimeoutError): control.once_mutation("fixture", {"action": "fixture"}, unknown)
            with self.assertRaises(FileExistsError): control.once_mutation("fixture", {"action": "fixture"}, unknown)
            self.assertEqual(calls, [1])
            failed = json.loads((Path(folder) / "fixture.failure.json").read_text())
            self.assertFalse(failed["retry_authorized"])


class HostLifecycleTests(unittest.TestCase):
    def setup_fake(self, folder, *, actual_id="fu-fixed", spawn_error=None):
        calls = []
        handle = SimpleNamespace(object_id="fc-fixed")
        handle.cancel = lambda **kwargs: calls.append(("cancel", kwargs))
        handle.get = lambda **kwargs: calls.append(("get", kwargs))
        function = SimpleNamespace(object_id=actual_id)
        function.hydrate = lambda: calls.append(("hydrate",))
        def spawn(admission):
            calls.append(("spawn",))
            if spawn_error: raise spawn_error
            return handle
        function.spawn = spawn
        sdk = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *args, **kwargs: function))
        admission = {"run_id": CPU_RUN_ID, "source_manifest": {"fixture": "a" * 64}}
        deployment = {"functions": {"cpu_preflight": "fu-fixed"}, "app_id": "ap-fixed"}
        return calls, sdk, admission, deployment

    def test_actual_handle_receipt_write_failure_is_cancelled(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, sdk, admission, deployment = self.setup_fake(folder)
            original_save = control.save_once
            def fail_call_receipt(path, value):
                if str(path).endswith(".call.json"): raise OSError("fake disk failure")
                return original_save(path, value)
            with patch.object(control, "RUNS", Path(folder)), patch.object(control, "check_sdk", return_value=sdk), \
                    patch.object(control, "prepare_admission", return_value=(admission, deployment)), \
                    patch.object(control, "save_once", side_effect=fail_call_receipt), \
                    patch.object(control.signal, "setitimer"):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    control.launch("preflight", None, None, None)
            self.assertEqual([event[0] for event in calls], ["hydrate", "spawn", "cancel"])
            receipt = json.loads((Path(folder) / "preflight.failure.json").read_text())
            self.assertEqual(receipt["submission_disposition"], "returned_handle")
            self.assertTrue(receipt["cancellation"]["attempted"])

    def test_wrong_function_identity_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, sdk, admission, deployment = self.setup_fake(folder, actual_id="fu-other")
            with patch.object(control, "RUNS", Path(folder)), patch.object(control, "check_sdk", return_value=sdk), \
                    patch.object(control, "prepare_admission", return_value=(admission, deployment)):
                with self.assertRaisesRegex(ValueError, "ID mismatch"):
                    control.launch("preflight", None, None, None)
            self.assertEqual(calls, [("hydrate",)])
            receipt = json.loads((Path(folder) / "preflight.failure.json").read_text())
            self.assertEqual(receipt["submission_disposition"], "pre_submit")

    def test_unknown_spawn_is_once_only_and_does_not_guess_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, sdk, admission, deployment = self.setup_fake(folder, spawn_error=TimeoutError("unknown submission"))
            with patch.object(control, "RUNS", Path(folder)), patch.object(control, "check_sdk", return_value=sdk), \
                    patch.object(control, "prepare_admission", return_value=(admission, deployment)):
                with self.assertRaises(TimeoutError): control.launch("preflight", None, None, None)
                with self.assertRaises(FileExistsError): control.launch("preflight", None, None, None)
            self.assertEqual([event[0] for event in calls], ["hydrate", "spawn"])
            receipt = json.loads((Path(folder) / "preflight.failure.json").read_text())
            self.assertEqual(receipt["submission_disposition"], "entered_unknown")
            self.assertNotIn("cancellation", receipt)


if __name__ == "__main__": unittest.main()
