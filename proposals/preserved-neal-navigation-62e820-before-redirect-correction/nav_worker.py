"""One level1 navigation per frame; observation only, normal browser security."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from nav_common import (RUN, ORIGINS, INITIAL, DOCUMENTS, OBSERVATION_SECONDS,
    atomic_json, inventory)

WRAPPER = """<!doctype html><html><head><style>
html,body{margin:0;width:970px;height:650px;overflow:hidden;background:#121824;color:white;font:14px sans-serif}
main{display:flex;gap:20px;padding:10px}section{width:465px}header{height:30px;font-size:13px}
iframe{width:465px;height:600px;border:0;display:block;background:white}
</style></head><body><main>
<section><header>Adam pane · navigation diagnostic only</header><div id="pane-adam"></div></section>
<section><header>Assistant pane · navigation diagnostic only</header><div id="pane-assistant"></div></section>
</main></body></html>"""

CREATE_FRAMES = r"""({initial})=>{
 window.__navDiagnostic={created_ms:performance.now(),loads:[]};
 for(const actor of ['adam','assistant']) {
   const frame=document.createElement('iframe');
   frame.name='navigation-diagnostic:'+actor;
   frame.title=actor+' official game';frame.referrerPolicy='no-referrer';
   frame.addEventListener('load',()=>window.__navDiagnostic.loads.push({actor,time_ms:performance.now()}));
   frame.src=initial;document.getElementById('pane-'+actor).appendChild(frame);
 }
 return {created_ms:window.__navDiagnostic.created_ms};
}"""

HEADERS = frozenset(("content-security-policy", "content-security-policy-report-only",
    "x-frame-options", "cross-origin-embedder-policy", "cross-origin-opener-policy",
    "cross-origin-resource-policy", "content-type", "server", "location"))


class DocumentAdmission:
    """One initial request per fresh actor frame; actual acyclic redirects only."""
    def __init__(self):
        self.frames = []

    def slot(self, frame):
        return next((index + 1 for index, entry in enumerate(self.frames) if entry["frame"] == frame), None)

    def accept(self, request, main_frame):
        try:
            frame, previous = request.frame, request.redirected_from
            if frame.parent_frame != main_frame or request.url not in DOCUMENTS:
                return False, "foreign_frame_or_document"
            retained = next((entry for entry in self.frames if entry["frame"] == frame), None)
            if retained is None:
                # The owned wrapper creates exactly two direct child frames.
                # A precommit frame name is observation, not admission proof.
                if previous is not None or request.url != INITIAL or len(self.frames) >= 2:
                    return False, "no_owned_initial_request"
                self.frames.append({"frame": frame, "last": request, "urls": {request.url}})
                return True, "first_owned_navigation"
            if (frame != retained["frame"] or previous is not retained["last"]
                    or request.url in retained["urls"] or len(retained["urls"]) >= len(DOCUMENTS)):
                return False, "reload_replacement_loop_or_unjoined_redirect"
            retained["last"] = request
            retained["urls"].add(request.url)
            return True, "actual_original_redirect_chain"
        except Exception as exc:
            return False, "frame_or_lineage_unavailable:" + type(exc).__name__


def safe_url(url):
    """Keep URL origin/path and digest; do not persist query/cookie tokens."""
    parsed = urlsplit(url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
    return {"url_without_query_or_fragment": clean[:1024],
        "complete_url_sha256": hashlib.sha256(url.encode()).hexdigest(),
        "had_query_or_fragment": bool(parsed.query or parsed.fragment)}


def diagnostic_text(text, cap=512):
    """Only bounded diagnostic text; redact URL queries and CSP nonces."""
    text = re.sub(r"https?://[^\s\"'<>]+", lambda match: safe_url(match.group())["url_without_query_or_fragment"], str(text))
    return re.sub(r"nonce-[A-Za-z0-9+/_=-]+", "nonce-REDACTED", text)[:cap]


def header_evidence(value):
    raw = value.encode()
    return {"preview": diagnostic_text(value, 256), "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "preview_may_be_redacted_or_truncated": True}


def append_event(report, kind, data, *, now=None):
    if len(report["events"]) >= 128:
        report["trace_limit_reached"] = True
        return
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
    if kind == "document_response" and len(encoded) > 8192:
        # Preserve the discriminating status/URL/frame fields even with very
        # long policy headers; header lengths/digests remain structured.
        data = dict(data)
        data["headers"] = {key: {k: v for k, v in value.items() if k != "preview"}
            for key, value in data.get("headers", {}).items()}
        data["header_previews_omitted_for_bound"] = True
    elif kind != "document_response" and len(encoded) > 2048:
        data = {"truncated": True, "original_json_bytes": len(encoded),
            "original_json_sha256": hashlib.sha256(encoded).hexdigest(),
            "preview": encoded[:1700].decode(errors="replace")}
    report["events"].append({"kind": kind, "monotonic": time.monotonic() if now is None else now, "data": data})


async def diagnose(report, progress_path, artifacts):
    from playwright.async_api import async_playwright
    playwright = browser = context = page = None
    document_admission = DocumentAdmission()
    started = time.monotonic()
    report.update(stage="browser_setup", stage_started_monotonic=started)
    atomic_json(progress_path, report)

    recording = True
    def event(kind, data):
        if not recording:
            return
        append_event(report, kind, data)
        atomic_json(progress_path, report)

    def frame_info(frame):
        return {"name": frame.name[:128], "retained_frame_slot": document_admission.slot(frame),
            **safe_url(frame.url), "detached": frame.is_detached()}

    def request_frame_info(request):
        try:
            return frame_info(request.frame)
        except Exception as exc:
            return {"frame_unavailable": True, "error_type": type(exc).__name__}

    async def route_request(route):
        request = route.request
        parsed = urlsplit(request.url)
        allowed = f"{parsed.scheme}://{parsed.netloc}" in ORIGINS
        reason = "allowed_asset_origin" if allowed else "unadmitted_asset_origin"
        if request.resource_type == "document":
            document_allowed, reason = document_admission.accept(request, page.main_frame)
            allowed = allowed and document_allowed
        event("route", {**safe_url(request.url), "resource_type": request.resource_type,
            "allowed": bool(allowed), "reason": reason, "frame": request_frame_info(request)})
        if allowed:
            await route.continue_()
        else:
            report["blocked_requests"] += 1
            await route.abort("blockedbyclient")

    def response_received(response):
        request = response.request
        if request.resource_type == "document":
            selected = {key.lower(): header_evidence(value)
                for key, value in response.headers.items() if key.lower() in HEADERS}
            event("document_response", {**safe_url(response.url), "status": response.status,
                "headers": selected, "frame": request_frame_info(request),
                "redirected_from": safe_url(request.redirected_from.url) if request.redirected_from else None})
            if response.status >= 400:
                report["document_error_observed"] = True

    def request_failed(request):
        event("request_failed", {**safe_url(request.url), "resource_type": request.resource_type,
            "failure": diagnostic_text(request.failure), "frame": request_frame_info(request)})
        if request.resource_type == "document":
            report["document_error_observed"] = True

    def console_message(message):
        if message.type in ("error", "warning") and any(word in message.text.lower()
                for word in ("frame", "content security", "refused", "net::", "failed to load", "cross-origin")):
            # Browser diagnostics only; no response body or arbitrary game state.
            event("console", {"type": message.type, "text": diagnostic_text(message.text, 1536)})

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        import importlib.metadata
        report["browser_version"] = browser.version
        report["playwright_version"] = importlib.metadata.version("playwright")
        context = await browser.new_context(viewport={"width": 970, "height": 650}, device_scale_factor=1,
            record_video_dir=str(artifacts / "video"), record_video_size={"width": 970, "height": 650},
            accept_downloads=False)
        await context.route("**/*", route_request)
        page = await context.new_page()
        page.set_default_timeout(3000)
        page.on("response", response_received)
        page.on("requestfailed", request_failed)
        page.on("framenavigated", lambda frame: event("frame_navigated", frame_info(frame)))
        page.on("console", console_message)
        page.on("pageerror", lambda error: event("page_error", {"type": type(error).__name__, "message": diagnostic_text(error, 1536)}))
        await page.set_content(WRAPPER)
        navigation_started = time.monotonic()
        report.update(stage="navigation_observation", stage_started_monotonic=navigation_started)
        atomic_json(progress_path, report)
        async def observe_original_navigation():
            report["creation"] = await page.evaluate(CREATE_FRAMES, {"initial": INITIAL})
            report["navigation_assignments"] = 2
            previous = None
            stop = "observation_deadline"
            while time.monotonic() - navigation_started < OBSERVATION_SECONDS:
                frames = [frame_info(frame) for frame in page.frames if frame != page.main_frame]
                loads = await page.evaluate("window.__navDiagnostic.loads.slice(0,16)")
                current = {"frames": frames, "loads": loads}
                if current != previous:
                    event("frame_and_load_snapshot", current)
                    previous = current
                report["latest_snapshot"] = current
                report["observation_elapsed_s"] = time.monotonic() - navigation_started
                atomic_json(progress_path, report)
                if report["trace_limit_reached"]:
                    stop = "bounded_trace_full"
                    break
                if report["document_error_observed"]:
                    stop = "actual_document_error_observed"
                    break
                await asyncio.sleep(min(0.2, max(0, OBSERVATION_SECONDS - (time.monotonic() - navigation_started))))
            report["observation_stop_reason"] = stop
            report["final_frames"] = [frame_info(frame) for frame in page.frames if frame != page.main_frame]
            try:
                await page.screenshot(path=str(artifacts / "final-wrapper.png"), type="png", full_page=False, timeout=3000)
                report["final_screenshot_returned"] = True
            except Exception as exc:
                report["screenshot_error"] = {"type": type(exc).__name__, "message": str(exc)[:512]}
        try:
            await asyncio.wait_for(observe_original_navigation(),
                max(0, OBSERVATION_SECONDS - (time.monotonic() - navigation_started)))
        except asyncio.TimeoutError:
            report["observation_stop_reason"] = "original30s_observation_deadline"
            report["observation_request_disposition"] = "Cancelled await; no new observation; browser close still required"
            report["final_frames"] = report.get("latest_snapshot", {}).get("frames", [])
        finally:
            recording = False
        report["status"] = "diagnostic_observed"
    finally:
        recording = False
        report.update(stage="finalizing", stage_started_monotonic=time.monotonic())
        atomic_json(progress_path, report)
        if context is not None:
            await context.close()
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()
        report["browser_close_returned"] = True
        report["artifacts"] = inventory(artifacts)
        report.update(stage="completed", stage_started_monotonic=time.monotonic())
        atomic_json(progress_path, report)


def worker_entry(admission, progress_path, artifact_dir):
    os.setsid()
    started = time.monotonic()
    report = {"schema": 1, "run": RUN, "status": "running", "stage": "initializing",
        "stage_started_monotonic": started, "events": [], "blocked_requests": 0,
        "document_error_observed": False, "trace_limit_reached": False, "navigation_assignments": 0,
        "model_calls": 0, "game_actions": 0, "gpu_requested": False,
        "browser_close_returned": False, "source_manifest": admission["source_manifest"],
        "not_a_preflight_or_gpu_admission": True,
        "security": "Default isolated Chromium; no UA/proxy/cookie/SSL/CSP modification or challenge interaction"}
    artifacts = Path(artifact_dir)
    artifacts.mkdir(exist_ok=False)
    (artifacts / "video").mkdir()
    try:
        asyncio.run(diagnose(report, progress_path, artifacts))
    except BaseException as exc:
        report.update(status="diagnostic_failed", stage="failed", stage_started_monotonic=time.monotonic(),
            error={"type": type(exc).__name__, "message": str(exc)[:2000]})
    report["elapsed_s"] = time.monotonic() - started
    atomic_json(progress_path, report)
