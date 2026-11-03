"""Paired BU strength experiment using a pinned, compatible vLLM loader.

Invoked only in a supervised child. All model operations are serialized; the
parent enforces a native-generation deadline. No original files are modified.
"""
from pathlib import Path
import hashlib
import json
import time

from .diagnostic import atomic_json


def run_vllm_pilot(admission, progress_path):
    import asyncio
    import importlib.metadata
    import tempfile
    import torch
    from safetensors.torch import load_file
    from playwright.async_api import async_playwright
    from vllm import LLM, SamplingParams
    from .vllm018_compat import normalize_config, install_single_process_loader
    from .vllm_attention import AttentionView
    from .merged_patch import MergedWeightEditor
    from .patch import file_sha256
    from .policy_io import processor_manifest
    from .diagnostic import reconcile_fixture_admission
    from .eval import build_suite, run_task, PolicyReply, PolicyDispositionUnknown
    from .eval.bu_policy import make_native_policy

    started = time.monotonic()
    report = {"schema": 1, "run": "vllm-pilot-v4", "status": "running",
              "model_calls": 0, "variants": {}, "acceptance": {}, "patch_audits": {},
              "task_policy_calls_scope": "Action dispatch callbacks, including zero-inference queued actions; model_calls counts actual chat requests",
              "model_call_counter_scope": "Explicit policy chat requests; vLLM startup/profiling forwards are not counted",
              "source_manifest": admission["source_manifest"],
              "benchmark": "synthetic_smoke_native_bu_batch_v4", "astra_comparison": "not performed",
              "prior_full_shard_receipt_sha256": admission["prior_report_sha256"],
              "quality_comparison_to_pilot_v1": "not valid: backend and protocol changed; v4 additionally supports bounded native batches",
              "generation_records": [], "browser_close_returned": False,
              "limits": {"model_calls": 384, "child_wall_s": 1800, "call_hard_s": 60}}

    def save():
        report["elapsed_s"] = time.monotonic() - started
        atomic_json(progress_path, report)

    def stage(name):
        report["stage"] = name
        report["stage_started_monotonic"] = time.monotonic()
        save()

    def summaries(rows):
        return {"tasks": len(rows), "successes": sum(r["success"] for r in rows),
                "invalid_actions": sum(r["action_errors"] for r in rows),
                "policy_dispatches": sum(r["policy_calls"] for r in rows),
                "actual_model_calls": sum(r["native_dispatch_counters"]["actual_model_calls"] for r in rows),
                "elapsed_s": sum(r["elapsed_s"] for r in rows)}

    try:
        stage("source_admission")
        merged = Path("/merged/bu-30b-ft")
        adapter = Path("/adapter/adapter")
        base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / admission["revision"]
        if reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"]) != admission["runner_reconciliation"]:
            raise ValueError("fixture reconciliation changed")
        if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
            raise ValueError("base processor artifacts changed")
        for name, expected in admission["merged_metadata_pins"].items():
            if file_sha256(merged / name)["sha256"] != expected:
                raise ValueError("merged metadata changed")
        for name, expected in admission["adapter_pins"].items():
            if file_sha256(adapter / name)["sha256"] != expected:
                raise ValueError("adapter artifact changed")
        report["versions"] = {k: importlib.metadata.version(k) for k in
                              ("vllm", "torch", "transformers", "safetensors", "playwright")}
        report["source"] = admission["prior_source"]
        overlay = Path(tempfile.mkdtemp(prefix="bu-vllm-metadata-"))
        normalized = normalize_config((merged / "config.json").read_bytes())
        normalized_bytes = json.dumps(normalized, sort_keys=True).encode()
        (overlay / "config.json").write_bytes(normalized_bytes)
        report["normalized_config_sha256"] = hashlib.sha256(normalized_bytes).hexdigest()
        index_bytes = (merged / "model.safetensors.index.json").read_bytes()
        index = json.loads(index_bytes)
        for name in {"model.safetensors.index.json", *index["weight_map"].values()}:
            if Path(name).name != name:
                raise ValueError("invalid checkpoint filename")
            (overlay / name).symlink_to(merged / name)
        processor_files = list(admission["fixtures"]["processor_manifest"])
        for name in processor_files + ["processor_config.json"]:
            if name == "config.json" or not (base / name).is_file():
                continue
            (overlay / name).symlink_to(base / name)
        report["loader_hook"] = install_single_process_loader(index_bytes)
        torch.manual_seed(20261103)
        torch.set_num_threads(4)
        stage("model_load")
        load_start = time.monotonic()
        engine = LLM(model=str(overlay), tokenizer=str(base), trust_remote_code=False,
                     dtype="bfloat16", quantization=None, tensor_parallel_size=1,
                     pipeline_parallel_size=1, distributed_executor_backend="uni",
                     gpu_memory_utilization=0.92, max_model_len=8192,
                     max_num_seqs=1, max_num_batched_tokens=8192,
                     enforce_eager=True, enable_prefix_caching=False,
                     enable_chunked_prefill=False, limit_mm_per_prompt={"image": 1},
                     mm_processor_kwargs={"max_pixels": 1100 * 850},
                     generation_config="vllm", seed=20261103,
                     disable_log_stats=True)
        report["model_load_s"] = time.monotonic() - load_start
        report["gpu"] = torch.cuda.get_device_name(0)
        report["image_budget"] = {"max_pixels": 935000, "viewport": [1100, 850],
                                  "same_for_all_v4_variants": True}
        report["load_receipts"] = engine.apply_model(lambda model: model._bu_merged_compat_receipt)
        if len(report["load_receipts"]) != 1 or not report["load_receipts"][0].get("complete"):
            raise ValueError("complete single-worker load receipt required")
        parameters = SamplingParams(temperature=0, top_p=1, max_tokens=192, n=1)

        def prepare_editor(model):
            model.eval()
            model._bu_weight_editor = MergedWeightEditor(
                AttentionView(model), load_file(str(adapter / "adapter_model.safetensors"), device="cpu"),
                json.loads((adapter / "adapter_config.json").read_text()))
            return {"retained_original_attention_matrices": 192}
        stage("retain_original_attention")
        report["editor"] = engine.apply_model(prepare_editor)

        def apply(scale):
            stage("weight_edit")
            result = engine.apply_model(lambda model: model._bu_weight_editor.apply(scale))
            if len(result) != 1:
                raise ValueError("unexpected worker count")
            # No retained request/KV prefix may be reused after an in-place edit.
            engine.reset_prefix_cache()
            engine.reset_mm_cache()
            report["current_scale"] = scale
            report["patch_audits"][str(scale)] = result[0]
            save()

        async def infer(messages, cap):
            if cap != 192 or report["model_calls"] >= 384 or time.monotonic() - started > 1700:
                raise RuntimeError("experiment generation admission exceeded")
            report["model_calls"] += 1
            stage("generation")
            call_start = time.monotonic()
            try:
                rows = await asyncio.to_thread(engine.chat, messages, parameters, use_tqdm=False)
            except BaseException:
                # Preserve stage='generation' for the supervisor. Async task
                # cancellation does not imply native inference has stopped.
                save()
                raise
            elapsed = time.monotonic() - call_start
            if len(rows) != 1 or len(rows[0].outputs) != 1 or not rows[0].finished:
                raise RuntimeError("incomplete generation receipt")
            output = rows[0].outputs[0]
            report["generation_records"].append({"call": report["model_calls"],
                "scale": report["current_scale"], "elapsed_s": elapsed,
                "prompt_tokens": len(rows[0].prompt_token_ids), "output_tokens": len(output.token_ids),
                "finish_reason": output.finish_reason, "stop_reason": output.stop_reason})
            stage("browser_step")
            return PolicyReply(output.text, len(rows[0].prompt_token_ids), len(output.token_ids))

        async def evaluate():
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                try:
                    async def episode(task, scale):
                        context = await browser.new_context(service_workers="block")
                        page = await context.new_page()
                        events = []
                        policy = make_native_policy(task, infer, events=events)
                        model_calls_before = report["model_calls"]
                        row = None
                        try:
                            result = await run_task(page, task, policy)
                            row = result.to_dict()
                            row["native_protocol_events"] = events
                            return row
                        except PolicyDispositionUnknown as exc:
                            # Save while the timed-out native thread may still
                            # be running; asyncio.run can wait for that thread.
                            report["status"] = "failed"
                            report["partial_task"] = exc.result.to_dict()
                            report["partial_native_events"] = events
                            report["native_inference_disposition"] = "unknown; parent generation watchdog must terminate child"
                            save()
                            raise
                        finally:
                            counters = policy.finish()
                            counters["native_policy_invocations"] = counters.pop("actual_model_calls")
                            counters["actual_model_calls"] = report["model_calls"] - model_calls_before
                            counters["model_call_scope"] = "Admitted chat-request attempts; native entry/return are separate receipts"
                            if row is not None:
                                row["native_dispatch_counters"] = counters
                            elif report.get("partial_task") is not None:
                                report["partial_task"]["native_dispatch_counters"] = counters
                                save()
                            await context.close()
                    for scale in (1.0, 0.5, 0.75, 0.0):
                        apply(scale)
                        rows = report["variants"][str(scale)] = []
                        for task in build_suite("development"):
                            row = await episode(task, scale)
                            rows.append(row)
                            save()
                            if scale == 1.0 and len(rows) == 1 and row["executed_actions"] == 0:
                                report["status"] = "baseline_native_interface_failed"
                                report["acceptance_status"] = "unopened"
                                return
                    baseline = summaries(report["variants"]["1.0"])
                    candidates = [(summaries(report["variants"][str(s)]), s) for s in (0.5, 0.75)]
                    candidates.sort(key=lambda item: (-item[0]["successes"], item[0]["invalid_actions"], -item[1]))
                    winner, scale = candidates[0]
                    if winner["successes"] <= baseline["successes"]:
                        report["status"] = "no_development_improvement"
                        report["acceptance_status"] = "unopened"
                        return
                    report["selected_scale"] = scale
                    for i, task in enumerate(build_suite("acceptance")):
                        for value in ((1.0, scale) if i % 2 == 0 else (scale, 1.0)):
                            apply(value)
                            row = await episode(task, value)
                            report["acceptance"].setdefault(str(value), []).append(row)
                            save()
                    original = {r["task_id"]: r for r in report["acceptance"]["1.0"]}
                    candidate = report["acceptance"][str(scale)]
                    regressions = [r["task_id"] for r in candidate
                        if (original[r["task_id"]]["success"] and not r["success"])
                        or r["action_errors"] > original[r["task_id"]]["action_errors"]]
                    passed = summaries(candidate)["successes"] > summaries(list(original.values()))["successes"] and not regressions
                    report["paired_regressions"] = regressions
                    report["status"] = "accepted_smoke_improvement" if passed else "acceptance_not_improved"
                    report["eligible_for_broader_benchmark"] = passed
                    report["checkpoint_exported"] = False
                    report["promotion"] = "Requires broader benchmark; no production deployment from smoke alone"
                finally:
                    await browser.close()
                    report["browser_close_returned"] = True
        asyncio.run(evaluate())
        report["summary"] = {key: summaries(rows) for key, rows in report["variants"].items()}
        report["acceptance_summary"] = {key: summaries(rows) for key, rows in report["acceptance"].items()}
        stage("completed")
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if isinstance(exc, PolicyDispositionUnknown):
            report.setdefault("partial_task", exc.result.to_dict())
            report["native_inference_disposition"] = "unknown; supervisor must terminate child"
        stage("failed")
