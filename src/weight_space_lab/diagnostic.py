"""One-call diagnostic child. CUDA and model imports occur only in this child.

This is an instrumented latency/interface diagnostic, not a quality benchmark.
Hooks synchronize CUDA and progress writes add overhead. No browser action or
weight edit is performed. The enclosing process owns the hard deadline.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import time


MAX_NEW_TOKENS = 64
GENERATION_SOFT_SECONDS = 120.0
_SERIALIZATION_ONLY_ADDITION = """

    def __reduce__(self):
        # BaseException's default reduction reconstructs from .args, which is
        # the display string rather than the retained result this constructor
        # requires. Preserve only our constructor payload across trusted worker
        # transport; this is not an interface for unpickling external input.
        return type(self), (self.result,)"""


def reconcile_fixture_admission(previous):
    """Join old CPU receipt to a literal, separately reviewed pickle-only delta."""
    from .policy_io import fixture_manifest
    current = fixture_manifest()
    if current["tasks"] != previous["tasks"]:
        raise ValueError("fixture task inventory changed")
    if set(current["sources"]) != set(previous["sources"]):
        raise ValueError("fixture source inventory changed")
    for name in ("eval/fixtures.py", "policy_io.py"):
        if current["sources"][name] != previous["sources"][name]:
            raise ValueError("previous fixture admission does not cover changed " + name)
    runner = Path(__file__).parent / "eval/runner.py"
    raw = runner.read_text()
    if raw.count(_SERIALIZATION_ONLY_ADDITION) != 1:
        raise ValueError("expected exact reviewed exception-serialization addition")
    reconstructed = raw.replace(_SERIALIZATION_ONLY_ADDITION, "", 1).encode()
    reconstructed_sha = hashlib.sha256(reconstructed).hexdigest()
    if reconstructed_sha != previous["sources"]["eval/runner.py"]:
        raise ValueError("runner changes exceed the reviewed serialization-only delta")
    return {"previous_runner_sha256": reconstructed_sha,
            "current_runner_sha256": current["sources"]["eval/runner.py"],
            "delta": "exact PolicyDispositionUnknown.__reduce__ addition only",
            "unchanged_tasks_and_fixture_encoder_sources": True,
            "previous_receipt_rewritten": False}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, indent=2, allow_nan=False)
    with temporary.open("w") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


async def capture_first_visitor():
    """Reproduce the first observation only; close Chromium before inference."""
    from playwright.async_api import async_playwright
    from .eval import build_suite
    from .eval.runner import OBSERVE_JS, SYSTEM_PROMPT

    task = build_suite("development")[0]
    if task.task_id != "development-visitor":
        raise ValueError("unexpected diagnostic fixture")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(service_workers="block")
            page = await context.new_page()
            async def deny_network(route):
                await route.abort()
            await page.route("**/*", deny_network)
            await page.set_viewport_size({"width": 1100, "height": 850})
            await page.set_content(task.html, wait_until="load", timeout=3000)
            observation = await page.evaluate(OBSERVE_JS)
            screenshot = await page.screenshot(type="png", full_page=False)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Goal: " + task.goal + "\nCurrent browser observation:\n" + json.dumps(observation, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(screenshot).decode("ascii")}},
                ]},
            ]
            await context.close()
            return task, messages, hashlib.sha256(screenshot).hexdigest()
        finally:
            await browser.close()


def run_cuda_diagnostic(admission, progress_path):
    """Spawn target; all errors are saved as JSON data, never remote exceptions."""
    # This fresh child owns a separate process group, including its temporary
    # Chromium descendants. No inherited CUDA context exists under spawn.
    os.setsid()
    started = time.monotonic()
    report = {
        "schema": 1, "run": "diagnostic-v2", "status": "running", "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "model_calls": 0, "browser_actions": 0, "weight_edits": 0,
        "limits": {"max_new_tokens": MAX_NEW_TOKENS, "generation_soft_seconds": GENERATION_SOFT_SECONDS},
        "measurements_include_instrumentation_overhead": True,
        "source_manifest": admission["source_manifest"],
        "prior_full_shard_hash_receipt_sha256": admission["prior_report_sha256"],
        "shard_rehash_performed": False, "timings": {}, "language_forwards": [], "vision_forwards": [],
        "token_times_s": [], "actual_backends": [], "native_generation_returned": False,
    }
    def save():
        report["elapsed_s"] = time.monotonic() - started
        atomic_json(progress_path, report)
    def stage(name):
        report["stage"] = name
        report["stage_started_monotonic"] = time.monotonic()
        save()

    try:
        stage("source_admission")
        from .patch import file_sha256
        from .policy_io import processor_manifest, encode_messages
        base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / admission["revision"]
        merged = Path("/merged/bu-30b-ft")
        reconciliation = reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"])
        if reconciliation != admission["runner_reconciliation"]:
            raise ValueError("reviewed runner reconciliation changed")
        report["runner_reconciliation"] = reconciliation
        if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
            raise ValueError("processor bytes changed")
        for filename, pin in admission["merged_metadata_pins"].items():
            if filename not in {"config.json", "model.safetensors.index.json"}:
                raise ValueError("unexpected merged metadata file")
            if file_sha256(merged / filename)["sha256"] != pin:
                raise ValueError("merged metadata changed")
        report["original_source"] = admission["prior_source"]
        stage("fixture_capture")
        import asyncio
        begin = time.monotonic()
        task, messages, screenshot_sha = asyncio.run(capture_first_visitor())
        report["browser_close_returned"] = True
        report["timings"]["fixture_capture_s"] = time.monotonic() - begin
        report["fixture"] = {"task_id": task.task_id, "html_sha256": task.fixture_sha256,
                             "screenshot_sha256": screenshot_sha,
                             "prompt_construction": "same system prompt, Goal/observation JSON, screenshot data URL as run_task step 1",
                             "prior_screenshot_byte_equality": "Not established; earlier screenshot bytes were not retained"}
        report["fixture"]["message_sha256"] = hashlib.sha256(
            json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        stage("runtime_import")
        import importlib.metadata
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLMoeForConditionalGeneration
        from transformers.generation.streamers import BaseStreamer
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "safetensors", "playwright")}
        torch.manual_seed(20261103)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        stage("model_load")
        begin = time.monotonic()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            str(merged), dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
        model.eval()
        torch.cuda.synchronize()
        report["timings"]["model_load_s"] = time.monotonic() - begin
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "capability": list(torch.cuda.get_device_capability(0))}
        stage("processor_load")
        begin = time.monotonic()
        processor = AutoProcessor.from_pretrained(str(base), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True)
        if type(processor.image_processor).__name__ != admission["fixtures"]["image_processor_class"]:
            raise ValueError("processor implementation changed")
        report["timings"]["processor_load_s"] = time.monotonic() - begin
        stage("processor_encode")
        begin = time.monotonic()
        inputs = encode_messages(processor, messages)
        report["timings"]["processor_encode_s"] = time.monotonic() - begin
        report["input_shapes"] = {key: list(value.shape) for key, value in inputs.items()}
        cpu_records = [item for item in admission["fixtures"]["records"] if item["task"] == task.task_id]
        if len(cpu_records) != 1 or cpu_records[0]["input_shapes"] != [report["input_shapes"]]:
            raise ValueError("encoded first-visitor shapes differ from the admitted CPU observation")
        report["input_shapes_match_prior_cpu_receipt"] = True
        count = int(inputs["input_ids"].shape[-1])
        if not 1 <= count <= 8192:
            raise ValueError("invalid prompt length; no truncation")
        report["prompt_tokens"] = count
        stage("host_to_device")
        torch.cuda.synchronize()
        begin = time.monotonic()
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize()
        report["timings"]["host_to_device_s"] = time.monotonic() - begin

        def backends(where):
            text_config = model.config.text_config
            expert = model.model.language_model.layers[0].mlp.experts
            return {"where": where, "root_experts": getattr(model.config, "_experts_implementation", None),
                    "text_experts": getattr(text_config, "_experts_implementation", None),
                    "actual_layer0_experts": getattr(expert.config, "_experts_implementation", None),
                    "text_attention": getattr(text_config, "_attn_implementation", None),
                    "vision_attention": getattr(model.config.vision_config, "_attn_implementation", None),
                    "experts_class": type(expert).__name__, "num_experts": getattr(expert, "num_experts", None),
                    "torch_grouped_mm_present": hasattr(torch, "_grouped_mm"),
                    "torch_functional_grouped_mm_present": hasattr(torch.nn.functional, "grouped_mm")}

        report["actual_backends"].append(backends("before_generate"))
        report["expert_weight_shapes"] = {
            "gate_up_proj": list(model.model.language_model.layers[0].mlp.experts.gate_up_proj.shape),
            "down_proj": list(model.model.language_model.layers[0].mlp.experts.down_proj.shape),
        }
        generation_started = None
        active = {}
        handles = []
        def install_hooks(module, label):
            def pre(module, args, kwargs):
                torch.cuda.synchronize()
                begin = time.monotonic()
                tensors = {key: list(value.shape) for key, value in kwargs.items() if torch.is_tensor(value)}
                active[label] = {"start_s": begin - generation_started, "shapes": tensors, "begin": begin}
                if label == "language":
                    if len(report["language_forwards"]) < 2:
                        report["actual_backends"].append(backends("prefill" if not report["language_forwards"] else "first_decode"))
                report["active_forward"] = {"module": label, "start_s": begin - generation_started}
                if not report[label + "_forwards"]:
                    save()
            def post(module, args, kwargs, output):
                torch.cuda.synchronize()
                now = time.monotonic()
                entry = active.pop(label)
                entry["duration_s"] = now - entry.pop("begin")
                entry["end_s"] = now - generation_started
                report[label + "_forwards"].append(entry)
                report["active_forward"] = None
                if len(report[label + "_forwards"]) <= 2:
                    save()
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(module.register_forward_hook(post, with_kwargs=True))
        install_hooks(model.model.visual, "vision")
        install_hooks(model.model.language_model, "language")

        class TokenTimes(BaseStreamer):
            prompt_seen = False
            def put(self, value):
                if not self.prompt_seen:
                    self.prompt_seen = True
                    return
                # HF transfers selected token IDs to CPU before this callback.
                report["token_times_s"].append(time.monotonic() - generation_started)
                if len(report["token_times_s"]) == 1 or len(report["token_times_s"]) % 8 == 0:
                    save()
            def end(self):
                report["streamer_end_seen"] = True

        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                elapsed = time.monotonic() - generation_started
                reached = elapsed >= GENERATION_SOFT_SECONDS
                report["last_stopping_check"] = {"elapsed_s": elapsed, "new_tokens": int(input_ids.shape[-1]) - count,
                                                  "soft_deadline_reached": reached}
                if reached:
                    report["soft_deadline_triggered"] = True
                return torch.full((input_ids.shape[0],), reached, device=input_ids.device, dtype=torch.bool)

        # Use explicit stopping controls so returned stop reasons are attributable.
        # This changes only this diagnostic's cap/deadline, never checkpoint bytes.
        config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True, num_beams=1,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.generation_config.pad_token_id,
            bos_token_id=model.generation_config.bos_token_id,
        )
        report["generation_config"] = config.to_dict()
        report["diagnostic_decoding_note"] = "Explicit greedy GenerationConfig; 64-token cap and recorded 120s soft criterion. Not a repeat of the earlier 192-token/38s evaluator."
        stage("generation")
        generation_started = time.monotonic()
        report["model_calls"] = 1
        save()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=config, streamer=TokenTimes(),
                                       stopping_criteria=StoppingCriteriaList([Deadline()]))
        torch.cuda.synchronize()
        report["timings"]["generation_s"] = time.monotonic() - generation_started
        report["native_generation_returned"] = True
        for handle in handles:
            handle.remove()
        new = generated[0, count:]
        token_ids = new.tolist()
        report["completion_tokens"] = len(token_ids)
        report["completion_token_ids"] = token_ids
        report["response"] = processor.decode(new, skip_special_tokens=True)
        eos = config.eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        reasons = []
        if token_ids and token_ids[-1] in eos:
            reasons.append("eos")
        if len(token_ids) >= MAX_NEW_TOKENS:
            reasons.append("max_new_tokens")
        if report.get("soft_deadline_triggered", False):
            reasons.append("soft_generation_deadline")
        if not reasons:
            reasons.append("unexplained_return_refuse_interpretation")
        report["stopping_reasons"] = reasons
        times = report["token_times_s"]
        report["timings"]["first_token_s"] = times[0] if times else None
        report["timings"]["decode_intervals_s"] = [later - earlier for earlier, later in zip(times, times[1:])]
        report["actual_backends"].append(backends("after_generate"))
        report["max_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        from .eval.runner import parse_action
        try:
            report["parsed_action"] = parse_action(report["response"], task.allowed_ids)
            report["valid_action_schema"] = True
        except ValueError as exc:
            report["valid_action_schema"] = False
            report["schema_error"] = str(exc)
        report["status"] = "diagnostic_returned"
        stage("completed")
    except BaseException as exc:
        report["status"] = "diagnostic_error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        stage("failed")
