"""Local fake browser/lifecycle checks; no provider or public-site calls."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import nav_common as common
import nav_control as control
import nav_worker as worker

HERE = Path(__file__).resolve().parent


class FakeFrame:
    def __init__(self, name, url, parent=None):
        self.name, self.url, self.parent_frame = name, url, parent
    def is_detached(self): return False


class FakePage:
    def __init__(self, context):
        self.context = context
        self.main_frame = FakeFrame("", "about:blank")
        self.frames = [self.main_frame]
        self.listeners = {}
        self.navigation_evaluations = 0
    def on(self, kind, callback): self.listeners[kind] = callback
    def set_default_timeout(self, timeout): self.timeout = timeout
    async def set_content(self, source): self.wrapper = source
    async def evaluate(self, source, argument=None):
        if source == worker.CREATE_FRAMES:
            self.navigation_evaluations += 1
            responses=[]
            for actor in ("adam", "assistant"):
                frame = FakeFrame("navigation-diagnostic:" + actor, "about:blank", self.main_frame)
                self.frames.append(frame)
                request = SimpleNamespace(url=common.INITIAL, resource_type="document", frame=frame,
                    redirected_from=None, failure=None)
                route = SimpleNamespace(request=request)
                async def continued(): self.context.continued += 1
                async def aborted(_): self.context.aborted += 1
                route.continue_, route.abort = continued, aborted
                self.listeners["request"](request)
                await self.context.routing(route)
                response = SimpleNamespace(request=request, url=common.INITIAL, status=403,
                    headers={"x-frame-options": "SAMEORIGIN", "content-type": "text/html", "set-cookie": "must not persist"})
                responses.append(response)
            for response in responses:
                self.listeners["response"](response)
            return {"created_ms": 1}
        if source == "window.__navDiagnostic.loads.slice(0,16)":
            return [{"actor": "adam", "time_ms": 2}, {"actor": "assistant", "time_ms": 3}]
        raise AssertionError("No extra page code or navigation admitted")
    async def screenshot(self, **kwargs):
        Path(kwargs["path"]).write_bytes(b"fake test PNG")


class FakeContext:
    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self.continued = self.aborted = 0
        self.page = FakePage(self)
    async def route(self, pattern, callback): self.routing = callback
    async def new_page(self): return self.page
    async def close(self): (self.video_dir / "fake.webm").write_bytes(b"fake video bytes")


class FakeBrowser:
    version = "fake-browser-version"
    async def new_context(self, **kwargs):
        self.settings = kwargs
        self.context = FakeContext(kwargs["record_video_dir"])
        return self.context
    async def close(self): self.closed = True


class FakePlaywright:
    def __init__(self):
        self.browser = FakeBrowser()
        self.chromium = SimpleNamespace(launch=self.launch)
    async def start(self): return self
    async def launch(self, **kwargs):
        self.launch_settings = kwargs
        return self.browser
    async def stop(self): self.stopped = True


class NavigationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_error_receipt_stops_same_navigation_without_bypass_or_retry(self):
        fake = FakePlaywright()
        fake_module = SimpleNamespace(async_playwright=lambda: fake)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"playwright.async_api": fake_module}), \
                patch("importlib.metadata.version", return_value="1.60.0"):
            root = Path(folder); artifacts = root / "artifacts"; artifacts.mkdir(); (artifacts / "video").mkdir()
            report = {"events": [], "trace_limit_reached": False, "document_error_observed": False,
                "blocked_requests": 0, "browser_close_returned": False}
            await worker.diagnose(report, root / "progress.json", artifacts)
            self.assertEqual(report["observation_stop_reason"], "actual_document_refusal")
            self.assertEqual(fake.browser.context.page.navigation_evaluations, 1)
            self.assertEqual(fake.browser.context.continued, 2)
            self.assertEqual(fake.browser.context.aborted, 0)
            self.assertEqual(fake.launch_settings, {"headless": True})
            self.assertNotIn("ignore_https_errors", fake.browser.settings)
            responses = [e["data"] for e in report["events"] if e["kind"] == "document_response"]
            self.assertEqual(len(responses), 2)
            self.assertTrue(all(e["status"] == 403 for e in responses))
            self.assertEqual([e["frame"]["retained_frame_slot"] for e in responses],[1,2])
            self.assertNotIn("set-cookie", json.dumps(report))
            self.assertTrue(report["browser_close_returned"])
            self.assertEqual(len(report["artifacts"]), 1)
            self.assertTrue(report["screenshot_skipped_after_observed_stop"])
            self.assertEqual(worker.CREATE_FRAMES.count("frame.src="), 1)

    async def test_stalled_first_evaluation_expires_without_second_navigation(self):
        fake = FakePlaywright()
        original = FakePage.evaluate
        entries = []
        async def stall(page, source, argument=None):
            if source == worker.CREATE_FRAMES:
                entries.append(source)
                await asyncio.Future()
            return await original(page, source, argument)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {"playwright.async_api": SimpleNamespace(async_playwright=lambda: fake)}), \
                patch("importlib.metadata.version", return_value="1.60.0"), \
                patch.object(FakePage, "evaluate", stall), patch.object(worker, "OBSERVATION_SECONDS", 0.01):
            root=Path(folder); artifacts=root/"artifacts";artifacts.mkdir();(artifacts/"video").mkdir()
            report={"events":[],"trace_limit_reached":False,"document_error_observed":False,
                "blocked_requests":0,"browser_close_returned":False}
            await worker.diagnose(report,root/"progress.json",artifacts)
            self.assertEqual(len(entries),1)
            self.assertEqual(report["observation_stop_reason"],"original30s_observation_deadline")
            self.assertTrue(report["browser_close_returned"])


class PureTests(unittest.TestCase):
    def test_route_gates_only_initials_and_observer_records_natural_redirects(self):
        gate=worker.DocumentAdmission()
        observer=worker.RedirectObservation()
        main=FakeFrame("","about:blank")
        frame=FakeFrame("","about:blank",main)
        first=SimpleNamespace(frame=frame,url=common.INITIAL,redirected_from=None,resource_type="document")
        self.assertIsNone(observer.observe(first)[1])
        self.assertTrue(gate.accept(first,main)[0])
        redirected=SimpleNamespace(frame=frame,url=common.CANONICAL,redirected_from=first,resource_type="document")
        # Real Chromium skips route for this hop; only the event observer sees it.
        evidence,reason=observer.observe(redirected)
        self.assertIsNone(reason)
        self.assertTrue(evidence["network_contact_may_already_have_occurred"])
        loop=SimpleNamespace(frame=frame,url=common.INITIAL,redirected_from=redirected,resource_type="document")
        self.assertEqual(observer.observe(loop)[1],"observed_redirect_loop")
        self.assertFalse(gate.accept(SimpleNamespace(frame=frame,url=common.INITIAL,redirected_from=None),main)[0])
        self.assertFalse(gate.accept(SimpleNamespace(frame=frame,url=common.INITIAL,redirected_from=redirected),main)[0])
        foreign=FakeFrame("navigation-diagnostic:assistant","about:blank",frame)
        self.assertFalse(gate.accept(SimpleNamespace(frame=foreign,url=common.INITIAL,redirected_from=None),main)[0])
        second=FakeFrame("","about:blank",main)
        self.assertTrue(gate.accept(SimpleNamespace(frame=second,url=common.INITIAL,redirected_from=None),main)[0])
        third=FakeFrame("","about:blank",main)
        self.assertFalse(gate.accept(SimpleNamespace(frame=third,url=common.INITIAL,redirected_from=None),main)[0])

    def test_asset_redirect_is_observed_after_possible_contact_not_preventively_routed(self):
        observer=worker.RedirectObservation()
        asset=SimpleNamespace(url="https://embed.neal.fun/asset.js",resource_type="script",redirected_from=None)
        self.assertIsNone(observer.observe(asset)[1])
        redirected=SimpleNamespace(url="https://unexpected.example/asset.js",resource_type="script",redirected_from=asset)
        evidence,reason=observer.observe(redirected)
        self.assertEqual(reason,"observed_unlisted_origin")
        self.assertTrue(evidence["network_contact_may_already_have_occurred"])

    def test_trace_limit_and_oversized_event_are_bounded(self):
        report = {"events": [], "trace_limit_reached": False}
        for _ in range(129): worker.append_event(report, "fixture", {"text": "x" * 10000}, now=1)
        self.assertEqual(len(report["events"]), 128)
        self.assertTrue(report["trace_limit_reached"])
        self.assertTrue(report["events"][0]["data"]["truncated"])

    def test_url_metadata_does_not_expose_query_or_fragment(self):
        record = worker.safe_url("https://embed.neal.fun/path?token=hidden#secret")
        self.assertEqual(record["url_without_query_or_fragment"], "https://embed.neal.fun/path")
        self.assertNotIn("hidden", json.dumps(record))
        self.assertTrue(record["had_query_or_fragment"])
        self.assertNotIn("hidden",worker.diagnostic_text("Refused https://embed.neal.fun/path?token=hidden nonce-ABCDEF"))
        self.assertNotIn("ABCDEF",worker.diagnostic_text("nonce-ABCDEF"))

    def test_large_framing_header_keeps_status_and_source_identity_structured(self):
        report={"events":[],"trace_limit_reached":False}
        data={"status":403,"frame":{"name":"actor-fixture"},
            "headers":{"content-security-policy":worker.header_evidence("default-src 'none'; "+"x"*100000)},
            **worker.safe_url(common.INITIAL)}
        worker.append_event(report,"document_response",data,now=1)
        receipt=report["events"][0]["data"]
        self.assertEqual(receipt["status"],403)
        self.assertEqual(receipt["frame"]["name"],"actor-fixture")
        self.assertGreater(receipt["headers"]["content-security-policy"]["raw_bytes"],100000)

    def test_only_existing_shared_envelope_and_exact_origins_admit(self):
        source = {"fixture": "a" * 64}
        admission = {"run_id": common.RUN, "source_manifest": source,
            "source_manifest_sha256": common.canonical_sha(source), "allowed_origins": list(common.ORIGINS),
            "budget_allocation": {"source_manifest_sha256": common.canonical_sha(source), "allocated_usd": 3,
                "shared_existing_neal_envelope": True, "total_committed_usd": 21, "ledger_sha256": "b" * 64}}
        with patch.object(common.time, "time", return_value=common.EXPIRES - 1):
            common.validate_admission(admission, source)
            admission["budget_allocation"]["shared_existing_neal_envelope"] = False
            with self.assertRaises(ValueError): common.validate_admission(admission, source)

    def test_known_handle_receipt_write_failure_is_cancelled(self):
        calls = []
        handle = SimpleNamespace(object_id="fc-fixture", cancel=lambda **kw: calls.append("cancel"))
        function = SimpleNamespace(object_id="fu-fixture", hydrate=lambda: None,
            spawn=lambda admission: calls.append("spawn") or handle)
        fake = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *a, **k: function))
        deployment = {"app_id": "ap-fixture", "functions": {common.FUNCTION: "fu-fixture"}}
        with tempfile.TemporaryDirectory() as folder:
            save = control.save_once
            def fail_call(path, value):
                if Path(path).name == "call.json": raise OSError("fake disk failure")
                return save(path, value)
            with patch.object(control, "RUNS", Path(folder)), patch.object(control, "sdk", return_value=fake), \
                    patch.object(control, "admission_from_saved", return_value=({"source_manifest": {}}, deployment)), \
                    patch.object(control, "save_once", side_effect=fail_call), patch.object(control.signal, "setitimer"):
                with self.assertRaises(OSError): control.launch(None, None, None)
        self.assertEqual(calls, ["spawn", "cancel"])

    def test_full_parent_remount_and_actual_imported_bytes_join(self):
        if importlib.util.find_spec("modal") is None:
            self.skipTest("Use reviewed Modal1.4.3 interpreter")
        code = r'''
import sys,types,json,pathlib
import nav_common,nav_worker
source_path=sys.argv[1]
module=types.ModuleType("nav_shallow_probe");module.__file__=source_path
sys.modules[module.__name__]=module
exec(compile(open(source_path).read(),source_path,"exec"),module.__dict__)
assert module.PACKAGED
assert type(module.diagnose_navigation).__name__=="Function"
actual=nav_common.manifest(pathlib.Path(nav_worker.__file__).parent)
bindings=module.executing_bindings(actual)
assert bindings[nav_common.SOURCE_PREFIX+"modal_navigation.py"]["actual_file_path"]==source_path
assert "torch" not in sys.modules and "vllm" not in sys.modules
print(json.dumps({"bindings":bindings,"provider_calls":0}))
'''
        descriptor, temporary = tempfile.mkstemp(prefix="nav-shallow-", suffix=".py", dir="/tmp")
        try:
            with os.fdopen(descriptor, "wb") as stream: stream.write((HERE / "modal_navigation.py").read_bytes())
            env = dict(os.environ, NEAL_NAV_PACKAGED="1", PYTHONPATH=str(HERE), PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-c", code, temporary], env=env,
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)["bindings"]), 3)
        finally:
            Path(temporary).unlink()


if __name__ == "__main__": unittest.main()
