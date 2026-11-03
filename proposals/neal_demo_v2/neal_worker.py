"""Sole CUDA/browser child for the proposed private live demo.

Only worker_entry enters a session or optional runtime. Model code is imported
after source admission and explicit assistant readiness. No paid teacher API is
called. The separate supervisor owns original wall/native/artifact deadlines.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import tempfile
import time
import math

from neal_limits import (CPU_RUN_ID, GPU_RUN_ID, SESSION_ID, LEVELS, MAX_MODEL_CALLS,
    CHILD_WALL_SECONDS, NOT_AFTER_UNIX)
from weight_space_lab.diagnostic import atomic_json


EXPECTED_VERSIONS = {"vllm": "0.18.0", "torch": "2.10.0", "torchvision": "0.25.0",
    "torchaudio": "2.10.0", "transformers": "4.57.6", "safetensors": "0.7.0",
    "pillow": "12.2.0", "playwright": "1.60.0", "modal": "1.4.3"}
REVISION = "28f795cd9ec73aedf2970a8914bbbf0d18accd82"
MODEL_PINS = {
    "merged_config": "9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0",
    "merged_index": "1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0",
    "adapter_config": "52e05adc5f8595942879747f1cb4df8f45c06f547005bf7dc3094e6233723e92",
    "adapter": "49311fe8c20663926e6474cc3e2a8c9e86eff2821259e22e85a533eb5ba08933"}


def merge_harness_progress(report, demo, now):
    """Assistant/browser progress never resets a pending native-call clock."""
    report["demo"] = demo
    report["browser_close_returned"] = demo.get("browser_close_returned") is True
    if not report["native_pending"]:
        phase = demo.get("phase", "browser_step")
        if phase != report["stage"]:
            report["stage"], report["stage_started_monotonic"] = phase, now


def runtime_versions():
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if any(actual[name].split("+", 1)[0] != expected for name, expected in EXPECTED_VERSIONS.items()):
        raise ValueError("runtime package version differs from frozen policy stack")
    return actual


def browser_runtime_manifest():
    """Hash only the owned remote image's Playwright browser installation."""
    root = Path("/root/.cache/ms-playwright")
    paths = [path for path in sorted(root.rglob("*")) if path.is_file()]
    if not paths or len(paths) > 10000 or sum(path.stat().st_size for path in paths) > 2 * 1024 ** 3:
        raise ValueError("bounded owned browser installation required")
    manifest = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("browser installation symlink refused")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        manifest[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return manifest


def inspect_video(artifact_dir, artifacts, progress_path):
    """One bounded actual CPU decode/count, not a configured-size assumption."""
    import subprocess
    if len(artifacts) != 1:
        raise ValueError("exactly one finalized video required")
    item = artifacts[0]
    path = Path(artifact_dir) / item["relative_path"]
    if (path.is_symlink() or not path.resolve().is_relative_to(Path(artifact_dir).resolve())
            or path.stat().st_size != item["bytes"]):
        raise ValueError("finalized owned video identity required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("video changed before media inspection")
    stdout = Path(progress_path).with_name("ffprobe-output.json")
    stderr = Path(progress_path).with_name("ffprobe-error.txt")
    executable = Path("/usr/bin/ffprobe")
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run([str(executable), "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,nb_read_frames:format=duration", "-of", "json", str(path)],
            stdout=out, stderr=err, timeout=20, check=False)
    if result.returncode != 0 or stderr.stat().st_size != 0 or stdout.stat().st_size > 128 * 1024:
        raise ValueError("bounded CPU media inspection did not complete without decode errors")
    parsed = json.loads(stdout.read_text())
    streams = parsed.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise ValueError("one video stream and no unrelated audio required")
    video = streams[0]
    frames, duration = int(video["nb_read_frames"]), float(parsed["format"]["duration"])
    if (video.get("width") != 970 or video.get("height") != 650 or frames <= 0
            or not math.isfinite(duration) or not 0 < duration <= 180):
        raise ValueError("actual frame geometry/duration differs from fixed preflight")
    return {"status": "decoded_frames_verified", "video_sha256": item["sha256"], "video_bytes": item["bytes"],
            "width": video["width"], "height": video["height"], "codec": video["codec_name"],
            "frames": frames, "duration_s": duration, "ffprobe_returncode": result.returncode,
            "stderr_bytes": stderr.stat().st_size,
            "ffprobe_binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}


def prepare_policy(admission, report, save, stage, admit_work):
    """Reviewed vLLM0.18 single-worker load and exact merged-checkpoint edit.

    This is a new composition requiring its own review. The underlying loader,
    attention mapping and immutable-original editor remain imported unchanged.
    """
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    from weight_space_lab.eval.runner import PolicyReply
    from weight_space_lab.merged_patch import MergedWeightEditor
    from weight_space_lab.vllm_attention import AttentionView
    from weight_space_lab.vllm018_compat import normalize_config, install_single_process_loader
    from weight_space_lab.patch import file_sha256
    from weight_space_lab.policy_io import processor_manifest
    from weight_space_lab.diagnostic import reconcile_fixture_admission

    if admission.get("candidate_scale") != 0.5 or admission.get("revision") != REVISION:
        raise ValueError("only previously selected exact0.5 edit and original base revision are admitted")
    if admission.get("model_pins") != MODEL_PINS:
        raise ValueError("model identity differs from frozen selected artifact")
    merged, adapter = Path("/merged/bu-30b-ft"), Path("/adapter/adapter")
    base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / REVISION
    files = {"merged_config": merged / "config.json", "merged_index": merged / "model.safetensors.index.json",
             "adapter_config": adapter / "adapter_config.json", "adapter": adapter / "adapter_model.safetensors"}
    for name, path in files.items():
        if file_sha256(path)["sha256"] != MODEL_PINS[name]:
            raise ValueError("model source changed: " + name)
    if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
        raise ValueError("processor artifacts changed")
    if reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"]) != admission["runner_reconciliation"]:
        raise ValueError("frozen model helper reconciliation changed")
    if len(admission["prior_source"].get("full_merged_shard_hashes", {})) != 13:
        raise ValueError("prior complete merged-source hash receipt required; no full shard rehash")
    overlay = Path(tempfile.mkdtemp(prefix="neal-owned-model-metadata-"))
    normalized = json.dumps(normalize_config((merged / "config.json").read_bytes()), sort_keys=True).encode()
    (overlay / "config.json").write_bytes(normalized)
    index_bytes = (merged / "model.safetensors.index.json").read_bytes()
    for name in {"model.safetensors.index.json", *json.loads(index_bytes)["weight_map"].values()}:
        if Path(name).name != name:
            raise ValueError("invalid checkpoint shard basename")
        (overlay / name).symlink_to(merged / name)
    for name in list(admission["fixtures"]["processor_manifest"]) + ["processor_config.json"]:
        if name != "config.json" and (base / name).is_file():
            (overlay / name).symlink_to(base / name)
    report["loader_hook"] = install_single_process_loader(index_bytes)
    report["normalized_config_sha256"] = hashlib.sha256(normalized).hexdigest()
    torch.manual_seed(20261103)
    torch.set_num_threads(4)
    admit_work()
    stage("model_load")
    load_started = time.monotonic()
    engine = LLM(model=str(overlay), tokenizer=str(base), trust_remote_code=False,
        dtype="bfloat16", quantization=None, tensor_parallel_size=1, pipeline_parallel_size=1,
        distributed_executor_backend="uni", gpu_memory_utilization=0.92, max_model_len=8192,
        max_num_seqs=1, max_num_batched_tokens=8192, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=False, limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 1100 * 850}, generation_config="vllm",
        seed=20261103, disable_log_stats=True)
    report["model_load_s"] = time.monotonic() - load_started
    report["gpu"] = torch.cuda.get_device_name(0)
    report["load_receipts"] = engine.apply_model(lambda model: model._bu_merged_compat_receipt)
    if len(report["load_receipts"]) != 1 or not report["load_receipts"][0].get("complete"):
        raise ValueError("complete single-worker load receipt required")

    def prepare_editor(model):
        model.eval()
        model._bu_weight_editor = MergedWeightEditor(AttentionView(model),
            load_file(str(adapter / "adapter_model.safetensors"), device="cpu"),
            json.loads((adapter / "adapter_config.json").read_text()))
        return {"retained_original_attention_matrices": 192}

    admit_work()
    stage("retain_original_attention")
    report["editor"] = engine.apply_model(prepare_editor)
    stage("weight_edit")
    audit = engine.apply_model(lambda model: model._bu_weight_editor.apply(0.5))
    if len(audit) != 1:
        raise ValueError("unexpected model worker count")
    engine.reset_prefix_cache()
    engine.reset_mm_cache()
    report["patch_audit"] = audit[0]
    report["current_scale"] = 0.5
    report["weight_formula"] = "cast_bfloat16(float32(exact_merged_weight) + (0.5 - 1) * 2 * (B @ A)); original attention retained; no raw-base equivalence claim"
    save()

    async def infer(messages, cap=192):
        admit_work()
        if (report["native_pending"] or report["model_calls"] >= MAX_MODEL_CALLS
                or cap not in (64, 192)
                or cap == 64 and report["model_calls"] != 0
                or cap == 192 and report["model_calls"] == 0):
            raise RuntimeError("native pending/request cap/exact one initial warmup barrier")
        report["model_calls"] += 1
        report["native_pending"] = True
        stage("generation")
        request_started = time.monotonic()
        try:
            results = await asyncio.to_thread(engine.chat, messages,
                SamplingParams(temperature=0, top_p=1, max_tokens=cap, n=1), use_tqdm=False)
        except BaseException:
            report["native_inference_disposition"] = "unknown; no new call; preserve original generation watchdog timestamp"
            save()
            raise
        report["native_pending"] = False
        if len(results) != 1 or not results[0].finished or len(results[0].outputs) != 1:
            raise RuntimeError("complete native return required")
        output = results[0].outputs[0]
        report["generation_records"].append({"call": report["model_calls"], "scope": "neutral_warmup" if cap == 64 else "scored_demo",
            "prompt_tokens": len(results[0].prompt_token_ids), "completion_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason, "stop_reason": output.stop_reason,
            "max_tokens": cap, "elapsed_s": time.monotonic() - request_started})
        stage("browser_step")
        return PolicyReply(output.text, len(results[0].prompt_token_ids), len(output.token_ids))

    async def warmup():
        blank = io.BytesIO()
        Image.new("RGB", (160, 210), (255, 255, 255)).save(blank, format="PNG")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": 'Unscored neutral warmup; blank page, no task. Return {"action":[{"done":{"text":"ready","success":false}}]}.'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(blank.getvalue()).decode()}}]}]
        reply = await asyncio.wait_for(infer(messages, 64), 40)
        report["neutral_warmup"] = {"text": reply.text[:4096], "prompt_tokens": reply.prompt_tokens,
                                     "completion_tokens": reply.completion_tokens}
        save()

    return infer, warmup


def worker_entry(kind, admission, progress_path, artifact_dir):
    os.setsid()
    started = time.monotonic()
    report = {"schema": 1, "run": CPU_RUN_ID if kind == "preflight" else GPU_RUN_ID,
        "status": "running", "stage": "source_admission", "stage_started_monotonic": started,
        "pid": os.getpid(), "process_group": os.getpgrp(), "source_manifest": admission.get("source_manifest"),
        "native_pending": False, "model_calls": 0, "generation_records": [], "bridge_events": [],
        "browser_close_returned": False, "model_call_counter_scope": "Admitted chat request attempts before asyncio.to_thread, including one neutral warmup; not proof of native entry; returned generation records and startup/profile forwards are separate",
        "assistant_model_calls": None, "assistant_tokens": None, "assistant_cost_usd": None,
        "model_exported": False, "claimed_superiority": False}

    def save():
        report["elapsed_s"] = time.monotonic() - started
        atomic_json(progress_path, report)

    def stage(name):
        report["stage"], report["stage_started_monotonic"] = name, time.monotonic()
        save()

    def admit_work():
        if time.monotonic() - started >= CHILD_WALL_SECONDS - 60 or not time.time() < NOT_AFTER_UNIX:
            raise TimeoutError("new work refused within original child wall; no extra finalization tail")

    def publish(demo):
        merge_harness_progress(report, demo, time.monotonic())
        save()

    async def execute():
        from neal_harness import preflight, run_demo
        if kind == "preflight":
            import torch
            if torch.cuda.is_initialized():
                raise ValueError("CPU preflight unexpectedly initialized CUDA")
            report["cuda_initialized"] = False
            result = await preflight(admission, progress_path, artifact_dir, publish=publish)
            if result.get("status") == "completed" and result.get("browser_close_returned") is True:
                stage("media_inspection")
                report["media_inspection"] = inspect_video(artifact_dir, result.get("artifacts", []), progress_path)
                save()
        elif kind == "demo":
            from neal_bridge import AssistantBridge, private_store
            bridge = AssistantBridge(private_store(), admission["source_manifest_sha256"],
                publish=lambda events: (report.update(bridge_events=events), save()))
            stage("waiting_assistant_ready")
            report["assistant_ready"] = await bridge.wait_ready()
            infer, warmup = prepare_policy(admission, report, save, stage, admit_work)
            await warmup()
            result = await run_demo(admission, progress_path, artifact_dir,
                adam_policy=infer, assistant_bridge=bridge, publish=publish)
        else:
            raise ValueError("unsupported child phase")
        if result.get("status") != "completed" or result.get("browser_close_returned") is not True:
            raise RuntimeError("browser preflight/demo failed; no success inference")
        if report["native_pending"]:
            raise RuntimeError("pending inference cannot become completed")
        report["status"] = "completed"
        stage("completed")

    try:
        if admission.get("session_id") != SESSION_ID or tuple(admission.get("levels", ())) != LEVELS:
            raise ValueError("fixed session and three-level protocol required")
        report["runtime_versions"] = runtime_versions()
        report["browser_runtime_manifest"] = browser_runtime_manifest()
        if kind == "demo" and (report["runtime_versions"] != admission.get("runtime_versions")
                or report["browser_runtime_manifest"] != admission.get("browser_runtime_manifest")):
            raise ValueError("actual image/browser identity changed after CPU preflight")
        save()
        asyncio.run(execute())
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if report["native_pending"]:
            save()
        else:
            stage("failed")
