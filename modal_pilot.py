"""Bounded private Modal BU experiment. No endpoint or model launches on import.

Run fixtures first, then the separately admitted pilot. Existing model volumes
are read-only; result artifacts go to a new volume. The entire GPU experiment
has one call, one container pool, no user-level retries, and a local wall limit.
Provider startup replacement can still occur and is recorded as a limitation.
"""
import json
import pathlib
import time
import modal

ROOT = pathlib.Path(__file__).parent
REVISION = "28f795cd9ec73aedf2970a8914bbbf0d18accd82"
ADAPTER_SHA = "49311fe8c20663926e6474cc3e2a8c9e86eff2821259e22e85a533eb5ba08933"
ADAPTER_CONFIG_SHA = "52e05adc5f8595942879747f1cb4df8f45c06f547005bf7dc3094e6233723e92"
CONFIG_SHA = "9743d159f38b383b746c54c9d286da79da71768e7997c73b37024da307ffcaa0"
INDEX_SHA = "1bb83a632abd9efef296e6394e35c705195f5b8d7cec394742753196b952aec0"
app = modal.App("bu-weight-space-pilot-v1")
image = (modal.Image.debian_slim(python_version="3.11")
         .run_commands("timeout --signal=TERM --kill-after=10s 900s python -m pip install --no-cache-dir "
                       "torch==2.9.1 torchvision==0.24.1 transformers==5.10.2 accelerate==1.12.0 "
                       "safetensors==0.7.0 pillow==12.2.0 playwright==1.60.0")
         .run_commands("timeout --signal=TERM --kill-after=10s 600s python -m playwright install --with-deps chromium")
         .add_local_dir(ROOT / "src", "/opt/weight-space/src", copy=True)
         .env({"PYTHONPATH": "/opt/weight-space/src", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}))
outputs = modal.Volume.from_name("bu-weight-space-results-v1", create_if_missing=True, version=2)
originals = {
    "/base": modal.Volume.from_name("hf-cache").with_mount_options(read_only=True),
    "/adapter": modal.Volume.from_name("bu-lora").with_mount_options(read_only=True),
    "/merged": modal.Volume.from_name("bu-merged").with_mount_options(read_only=True),
    "/results": outputs,
}


@app.function(image=image, volumes={"/base": originals["/base"]}, cpu=(1, 2), memory=(1024, 4096), timeout=300,
              startup_timeout=180, retries=0, max_containers=1, min_containers=0,
              buffer_containers=0, single_use_containers=True, block_network=True)
def verify_fixtures():
    import asyncio
    from playwright.async_api import async_playwright
    from weight_space_lab.eval import build_suite, run_task, PolicyReply
    from weight_space_lab.policy_io import encode_messages, fixture_manifest, processor_manifest
    from transformers import AutoProcessor
    base = pathlib.Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / REVISION
    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, trust_remote_code=False, use_fast=True)

    async def check():
        records = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                for split in ("development", "acceptance"):
                    for task in build_suite(split):
                        context = await browser.new_context(service_workers="block")
                        page = await context.new_page()
                        actions = iter(task.reference_actions)
                        encoded_shapes = []
                        async def reference(messages, max_tokens):
                            if not encoded_shapes:
                                inputs = encode_messages(processor, messages)
                                encoded_shapes.append({k: list(v.shape) for k, v in inputs.items()})
                            return PolicyReply(json.dumps(next(actions)), 0, 0)
                        result = await run_task(page, task, reference)
                        await context.close()
                        if not result.success:
                            raise RuntimeError(json.dumps(result.to_dict()))
                        records.append({"task": task.task_id, "fixture_sha256": task.fixture_sha256,
                                        "success": result.success, "actions": result.executed_actions,
                                        "input_shapes": encoded_shapes})
            finally:
                await browser.close()
        return {"schema": 1, "model_calls": 0, "records": records,
                "fixture_manifest": fixture_manifest(), "processor_manifest": processor_manifest(base),
                "processor_class": type(processor).__name__, "image_processor_class": type(processor.image_processor).__name__}
    return asyncio.run(check())


def summarize(rows):
    return {"tasks": len(rows), "successes": sum(r["success"] for r in rows),
            "invalid_actions": sum(r["action_errors"] for r in rows),
            "policy_calls": sum(r["policy_calls"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "elapsed_s": sum(r["elapsed_s"] for r in rows),
            "policy_elapsed_s": sum(r["policy_elapsed_s"] for r in rows)}


@app.function(image=image, gpu="A100-80GB", cpu=(2, 4), memory=(8192, 98304),
              volumes=originals, timeout=3600, startup_timeout=600, retries=0,
              max_containers=1, min_containers=0, buffer_containers=0,
              single_use_containers=True, block_network=True)
def pilot(admission):
    import asyncio
    import base64
    import gc
    import hashlib
    import io
    import importlib.metadata
    import random
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from playwright.async_api import async_playwright
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    from weight_space_lab.eval import build_suite, run_task, PolicyReply, PolicyDispositionUnknown
    from weight_space_lab.patch import file_sha256
    from weight_space_lab.merged_patch import MergedWeightEditor
    from weight_space_lab.policy_io import encode_messages, fixture_manifest, processor_manifest

    started = time.monotonic()
    run_dir = pathlib.Path("/results/pilot-v1")
    run_dir.mkdir(parents=True, exist_ok=True)
    claim = run_dir / "started.json"
    with claim.open("x") as f:
        json.dump({"unix": time.time(), "recipe": "merged_attention_delta_v1"}, f)
    outputs.commit()
    report = {"schema": 1, "status": "running", "variants": {}, "acceptance": {},
              "patch_audits": {}, "limits": {"global_s": 3300, "max_calls": 384},
              "versions": {k: importlib.metadata.version(k) for k in
                           ("torch", "transformers", "safetensors", "playwright")},
              "benchmark": "synthetic_smoke_v1", "astra_comparison": "NOT_PERFORMED"}

    def save():
        report["elapsed_s"] = time.monotonic() - started
        with (run_dir / "report.json").open("w") as f:
            json.dump(report, f, indent=2)
        outputs.commit()

    def verify_file(path, expected):
        actual = file_sha256(path)["sha256"]
        if actual != expected:
            raise RuntimeError(f"Changed source artifact: {path}")
        return actual

    try:
        merged = pathlib.Path("/merged/bu-30b-ft")
        adapter = pathlib.Path("/adapter/adapter")
        base = pathlib.Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / REVISION
        verify_file(merged / "config.json", CONFIG_SHA)
        verify_file(merged / "model.safetensors.index.json", INDEX_SHA)
        verify_file(adapter / "adapter_model.safetensors", ADAPTER_SHA)
        verify_file(adapter / "adapter_config.json", ADAPTER_CONFIG_SHA)
        if admission["fixture_manifest"] != fixture_manifest() or admission["processor_manifest"] != processor_manifest(base):
            raise RuntimeError("Fixture or processor content changed after CPU admission")
        report["source"] = {"merged_config_sha256": CONFIG_SHA, "merged_index_sha256": INDEX_SHA,
                            "adapter_sha256": ADAPTER_SHA, "cached_base_revision": REVISION,
                            "full_merged_shard_hashes": {}}
        index = json.loads((merged / "model.safetensors.index.json").read_text())
        for filename in sorted(set(index["weight_map"].values())):
            path = merged / filename
            if path.parent != merged or not filename.endswith(".safetensors"):
                raise RuntimeError("Unexpected shard path")
            digest = hashlib.sha256()
            count = 0
            with path.open("rb") as f:
                while chunk := f.read(8 * 1024 * 1024):
                    if time.monotonic() - started >= 300:
                        raise RuntimeError("Original shard hashing exceeded its 300-second admission budget")
                    count += len(chunk)
                    digest.update(chunk)
            report["source"]["full_merged_shard_hashes"][filename] = {"bytes": count, "sha256": digest.hexdigest()}
            print(json.dumps({"phase": "source_hash", "shard": filename, "bytes": count}), flush=True)
        save()
        torch.manual_seed(20261103)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load_start = time.monotonic()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            str(merged), dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(base), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        if type(processor.image_processor).__name__ != admission["image_processor_class"]:
            raise RuntimeError("Processor implementation changed")
        state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
        config = json.loads((adapter / "adapter_config.json").read_text())
        editor = MergedWeightEditor(model, state, config)
        report["load_seconds"] = time.monotonic() - load_start
        report["gpu"] = torch.cuda.get_device_name(0)
        calls = 0
        report["model_calls"] = 0

        def infer(messages, cap):
            nonlocal calls
            if calls >= 384 or time.monotonic() - started >= 3300:
                raise RuntimeError("Global experiment limit reached")
            calls += 1
            report["model_calls"] = calls
            inputs = encode_messages(processor, messages).to("cuda:0")
            count = inputs["input_ids"].shape[-1]
            if count > 8192:
                raise ValueError("Input limit reached; no silent truncation")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=min(cap, 192), do_sample=False,
                                           use_cache=True, max_time=38)
            new = generated[0, count:]
            response = processor.decode(new, skip_special_tokens=True)
            return PolicyReply(response, count, int(new.numel()))

        async def policy(messages, cap):
            # A timeout is fatal to the WHOLE experiment; run_task propagates
            # PolicyDispositionUnknown and no subsequent inference is admitted.
            return await asyncio.to_thread(infer, messages, cap)

        async def evaluate():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    for scale in (1.0, 0.5, 0.75, 0.0):
                        label = str(scale)
                        report["patch_audits"][label] = editor.apply(scale)
                        rows = report["variants"][label] = []
                        for task in build_suite("development"):
                            ctx = await browser.new_context(service_workers="block")
                            page = await ctx.new_page()
                            try:
                                result = await run_task(page, task, policy)
                                rows.append(result.to_dict())
                                print(json.dumps({"phase": "development", "scale": scale,
                                                  "task": task.task_id, "success": result.success,
                                                  "calls": result.policy_calls}), flush=True)
                                save()
                                if scale == 1.0 and len(rows) == 1 and result.executed_actions == 0:
                                    report["status"] = "baseline_interface_check_failed"
                                    report["acceptance_status"] = "UNOPENED"
                                    return
                            finally:
                                await ctx.close()
                    baseline = summarize(report["variants"]["1.0"])
                    choices = [(summarize(report["variants"][str(s)]), s) for s in (0.5, 0.75)]
                    choices.sort(key=lambda v: (-v[0]["successes"], v[0]["invalid_actions"], -v[1]))
                    winner, selected = choices[0]
                    if winner["successes"] <= baseline["successes"]:
                        report["selected_scale"] = 1.0
                        report["status"] = "no_development_improvement"
                        report["acceptance_status"] = "UNOPENED"
                        return
                    report["selected_scale"] = selected
                    # One frozen paired acceptance pass, order alternated by task.
                    for i, task in enumerate(build_suite("acceptance")):
                        order = (1.0, selected) if i % 2 == 0 else (selected, 1.0)
                        for scale in order:
                            editor.apply(scale)
                            ctx = await browser.new_context(service_workers="block",
                                record_video_dir=str(run_dir / "videos"),
                                record_video_size={"width": 1100, "height": 850})
                            page = await ctx.new_page()
                            try:
                                result = await run_task(page, task, policy)
                                row = result.to_dict()
                                video = page.video
                            finally:
                                await ctx.close()
                            if video:
                                row["video_path"] = await video.path()
                            report["acceptance"].setdefault(str(scale), []).append(row)
                            save()
                    original = summarize(report["acceptance"]["1.0"])
                    candidate = summarize(report["acceptance"][str(selected)])
                    original_tasks = {r["task_id"]: r for r in report["acceptance"]["1.0"]}
                    regressions = [r["task_id"] for r in report["acceptance"][str(selected)]
                                   if (original_tasks[r["task_id"]]["success"] and not r["success"])
                                   or r["action_errors"] > original_tasks[r["task_id"]]["action_errors"]]
                    passes = candidate["successes"] > original["successes"] and not regressions
                    report["paired_regressions"] = regressions
                    report["status"] = "accepted_smoke_improvement" if passes else "acceptance_not_improved"
                    report["eligible_for_broader_benchmark"] = passes
                    report["production_promotion"] = "NOT_ESTABLISHED"
                except PolicyDispositionUnknown as exc:
                    # Save BEFORE asyncio.run waits for executor shutdown. Do not
                    # continue any task/candidate after a timed-out native call.
                    report["status"] = "aborted"
                    report["partial_task"] = exc.result.to_dict()
                    report["native_inference_disposition"] = "unknown; terminate this container"
                    save()
                    raise
                finally:
                    await browser.close()
        asyncio.run(evaluate())
        report["model_calls"] = calls
        report["summary"] = {k: summarize(v) for k, v in report["variants"].items()}
        report["acceptance_summary"] = {k: summarize(v) for k, v in report["acceptance"].items()}
        report["max_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save()
        return report
    except BaseException as exc:
        report["status"] = "aborted"
        report["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        if isinstance(exc, PolicyDispositionUnknown):
            report["partial_task"] = exc.result.to_dict()
            report["native_inference_disposition"] = "unknown; terminate this container"
        save()
        raise


@app.local_entrypoint()
def main(phase: str = "fixtures"):
    if phase not in {"fixtures", "pilot"}:
        raise ValueError("phase must be fixtures or pilot")
    output = ROOT / "runs" / f"{phase}-v1.json"
    marker = ROOT / "runs" / f"{phase}-v1.started.json"
    if marker.exists():
        raise RuntimeError("This invocation is already consumed; inspect saved results/logs")
    if phase == "pilot":
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from weight_space_lab.policy_io import fixture_manifest
        fixture_result = json.loads((ROOT / "runs/fixtures-v1.json").read_text())
        if len(fixture_result.get("records", [])) != 16 or not all(r["success"] for r in fixture_result["records"]):
            raise RuntimeError("Fixture runtime validation is incomplete")
        if fixture_result.get("fixture_manifest") != fixture_manifest():
            raise RuntimeError("Fixture/source manifest differs from validated CPU receipt")
    with marker.open("x") as f:
        json.dump({"phase": phase, "started_unix": time.time(), "app_id": app.app_id,
                   "reservation_usd": 0.5 if phase == "fixtures" else 12}, f)
    function = verify_fixtures if phase == "fixtures" else pilot
    call = function.spawn(fixture_result) if phase == "pilot" else function.spawn()
    with (ROOT / "runs" / f"{phase}-v1.call.json").open("x") as f:
        json.dump({"function_call_id": call.object_id, "app_id": app.app_id, "spawned_unix": time.time()}, f)
    try:
        result = call.get(timeout=450 if phase == "fixtures" else 4500)
    except BaseException:
        cancelled = call.cancel(terminate_containers=True)
        with (ROOT / "runs" / f"{phase}-v1.cancel.json").open("x") as f:
            json.dump({"request_returned": True, "unix": time.time(),
                       "physical_termination_or_billing_stop": "Requires provider terminal/billing evidence"}, f)
        raise
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"saved": str(output), "status": result.get("status", "fixtures_passed"),
                      "summary": result.get("summary")}))
