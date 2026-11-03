"""Unlaunched private MiniWoB subset experiment; optional runtimes load at call.

The original checkpoint and one preselected edited checkpoint receive all24
fixed cases. This is a limited, DOM-assisted MiniWoB subset, not the full suite,
not an Astra comparison, and not automatic model promotion. The original native
ten-second task timers continue during inference. Nothing is published here.
"""
from pathlib import Path
import hashlib
import json
import time

from weight_space_lab.diagnostic import atomic_json


SELECTION_RULE = "successes descending; action errors ascending; .75 tie"


def validate_candidate(admission):
    """Recompute the frozen rule from host-verified v4 receipt summaries."""
    evidence = admission["candidate_evidence"]
    if evidence.get("frozen_selection_rule") != SELECTION_RULE:
        raise ValueError("candidate selection rule changed")
    scores = evidence["candidate_scores"]
    if not isinstance(scores, list) or len(scores) != 2:
        raise ValueError("both edited candidate summaries are required")
    by_scale = {}
    for item in scores:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("malformed candidate summary")
        successes, errors, scale = item
        if (type(successes) is not int or not 0 <= successes <= 8
                or type(errors) is not int or errors < 0
                or type(scale) not in (int, float) or scale not in (0.5, 0.75)
                or scale in by_scale):
            raise ValueError("invalid candidate development counts")
        by_scale[scale] = (successes, errors, scale)
    chosen = sorted(by_scale.values(), key=lambda x: (-x[0], x[1], -x[2]))[0][2]
    if admission.get("candidate_scale") != chosen:
        raise ValueError("candidate differs from pre-benchmark rule")
    selection = admission["candidate_selection"]
    expected_development = {str(value): {"tasks": 8, "successes": score[0], "action_errors": score[1]}
                            for value, score in by_scale.items()}
    if selection.get("development") != expected_development:
        raise ValueError("candidate summaries disagree with actual v4 development evidence")
    digest = evidence.get("v4_report_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("actual v4 receipt hash required")
    if selection.get("v4_report_sha256") != digest or selection.get("v4_status") != evidence.get("v4_status"):
        raise ValueError("v4 selection receipt identity differs")
    if type(selection.get("eligible_for_broader_benchmark")) is not bool:
        raise ValueError("retain an explicit prior smoke promotion disposition")
    return chosen


def validate_preflight(receipt):
    from miniwob_native import cases
    if (receipt.get("schema") != 1 or receipt.get("native_timer_ms") != 10000
            or receipt.get("model_calls") != 0 or receipt.get("environment_steps") != 48
            or receipt.get("fresh_resets") != 48):
        raise ValueError("exact native paired CPU preflight required")
    records = receipt.get("cases", [])
    if len(records) != 24:
        raise ValueError("all24 preflight cases required")
    output = {}
    for case, record in zip(cases(), records, strict=True):
        if (record.get("case_id") != case.case_id or record.get("env_id") != case.env_id
                or record.get("seed") != case.seed or record.get("native_timer_ms") != 10000):
            raise ValueError("preflight frozen case order changed")
        initial = record.get("initial", {})
        if set(initial) != {"screenshot_sha256", "visible_dom_sha256", "screen_shape"}:
            raise ValueError("complete paired initial fingerprints required")
        for key in ("screenshot_sha256", "visible_dom_sha256"):
            value = initial[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid initial fingerprint")
        output[case.case_id] = initial
    return output


def summarize(rows):
    """Always retain the planned24 denominator; missing episodes cannot vanish."""
    return {"planned_episodes": 24, "completed_episodes": len(rows),
            "missing_episodes": 24 - len(rows), "native_successes": sum(row["success"] for row in rows),
            "success_fraction_of_planned": sum(row["success"] for row in rows) / 24,
            "coverage_complete": len(rows) == 24,
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "actual_model_calls": sum(row["model_calls"] for row in rows),
            "environment_steps": sum(row["environment_steps"] for row in rows),
            "elapsed_s": sum(row["elapsed_s"] for row in rows)}


def run_miniwob_benchmark(admission, progress_path):
    import asyncio
    import base64
    import io
    import tempfile
    from PIL import Image
    import torch
    from safetensors.torch import load_file
    from vllm import LLM, SamplingParams
    from weight_space_lab.vllm018_compat import normalize_config, install_single_process_loader
    from weight_space_lab.vllm_attention import AttentionView
    from weight_space_lab.merged_patch import MergedWeightEditor
    from weight_space_lab.patch import file_sha256
    from weight_space_lab.policy_io import processor_manifest
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    from weight_space_lab.eval.runner import PolicyReply
    from miniwob_process_entry_v1 import runtime_manifest
    from miniwob_native import (cases, make_environment, run_episode,
                                BenchmarkInferenceUnknown, BenchmarkBudgetStop)

    started = time.monotonic()
    report = {"schema": 1, "run": "private-miniwob-v1", "status": "running", "model_calls": 0,
              "planned_cases": 24, "planned_episodes": 48, "variants": {}, "warmups": [],
              "generation_records": [], "patch_audits": {}, "patch_events": [],
              "candidate_evidence": admission["candidate_evidence"], "candidate_scale": admission["candidate_scale"],
              "source_manifest": admission["source_manifest"],
              "prior_full_shard_receipt_sha256": admission["prior_report_sha256"],
              "benchmark": "Farama MiniWoB1.1.0 fixed12families x2seeds; native10s; DOM-assisted subset",
              "astra_comparison": "not performed", "automatic_promotion": False,
              "selection_uses_miniwob_results": False, "checkpoint_exported": False,
              "model_call_counter_scope": "Explicit chat requests, including2 unscored neutral warmups; startup/profiling forwards excluded",
              "limits": {"model_calls": 386, "scored_model_calls": 384, "warmup_calls": 2,
                         "native_timer_ms": 10000, "actions_per_episode": 8,
                         "child_wall_s": 1500, "new_work_before_s": 1440, "generation_hard_s": 45},
              "browser_close_returned": False, "environments_constructed": 0, "environment_close_returns": 0,
              "browser_disposition_limit": "MiniWoB close may suppress driver.quit errors; return is not independent descendant termination evidence"}
    native_pending = False

    def save():
        report["elapsed_s"] = time.monotonic() - started
        report["summary"] = {key: summarize(rows) for key, rows in report["variants"].items()}
        atomic_json(progress_path, report)

    def stage(name):
        report["stage"] = name
        report["stage_started_monotonic"] = time.monotonic()
        save()

    def admit_work():
        if time.monotonic() - started >= 1440:
            raise BenchmarkBudgetStop("new work refused at child1440s; original1500s outer wall remains")

    try:
        stage("source_admission")
        scale = validate_candidate(admission)
        expected_initials = validate_preflight(admission["miniwob_preflight"])
        if runtime_manifest() != admission["runtime_manifest"]:
            raise ValueError("MiniWoB source/browser/dependency runtime changed after CPU preflight")
        report["runtime_manifest"] = admission["runtime_manifest"]
        report["native_preflight"] = admission["miniwob_preflight"]
        report["candidate_selection"] = admission["candidate_selection"]
        report["prior_smoke_promotion_eligibility"] = admission["candidate_selection"]["eligible_for_broader_benchmark"]
        report["comparison_status"] = ("exploratory_unpromoted_edit" if not report["prior_smoke_promotion_eligibility"]
                                        else "smoke_improved_candidate_requiring_broader_evidence")
        report["variants"] = {"1.0": [], str(scale): []}
        merged, adapter = Path("/merged/bu-30b-ft"), Path("/adapter/adapter")
        base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / admission["revision"]
        if reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"]) != admission["runner_reconciliation"]:
            raise ValueError("existing source reconciliation changed")
        if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
            raise ValueError("base processor artifacts changed")
        for name, expected in admission["merged_metadata_pins"].items():
            if file_sha256(merged / name)["sha256"] != expected:
                raise ValueError("merged metadata changed")
        for name, expected in admission["adapter_pins"].items():
            if file_sha256(adapter / name)["sha256"] != expected:
                raise ValueError("adapter artifact changed")
        report["source"] = admission["prior_source"]
        overlay = Path(tempfile.mkdtemp(prefix="miniwob-bu-metadata-"))
        normalized = json.dumps(normalize_config((merged / "config.json").read_bytes()), sort_keys=True).encode()
        (overlay / "config.json").write_bytes(normalized)
        report["normalized_config_sha256"] = hashlib.sha256(normalized).hexdigest()
        index_bytes = (merged / "model.safetensors.index.json").read_bytes()
        for name in {"model.safetensors.index.json", *json.loads(index_bytes)["weight_map"].values()}:
            if Path(name).name != name:
                raise ValueError("invalid checkpoint filename")
            (overlay / name).symlink_to(merged / name)
        for name in list(admission["fixtures"]["processor_manifest"]) + ["processor_config.json"]:
            if name != "config.json" and (base / name).is_file():
                (overlay / name).symlink_to(base / name)
        report["loader_hook"] = install_single_process_loader(index_bytes)
        torch.manual_seed(20261103)
        torch.set_num_threads(4)
        stage("model_load")
        load_start = time.monotonic()
        engine = LLM(model=str(overlay), tokenizer=str(base), trust_remote_code=False,
                     dtype="bfloat16", quantization=None, tensor_parallel_size=1,
                     pipeline_parallel_size=1, distributed_executor_backend="uni",
                     gpu_memory_utilization=0.92, max_model_len=8192, max_num_seqs=1,
                     max_num_batched_tokens=8192, enforce_eager=True, enable_prefix_caching=False,
                     enable_chunked_prefill=False, limit_mm_per_prompt={"image": 1},
                     mm_processor_kwargs={"max_pixels": 1100 * 850}, generation_config="vllm",
                     seed=20261103, disable_log_stats=True)
        report["model_load_s"] = time.monotonic() - load_start
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
        stage("retain_original_attention")
        report["editor"] = engine.apply_model(prepare_editor)

        def apply(value):
            admit_work()
            if report.get("current_scale") == value:
                return
            if native_pending:
                raise RuntimeError("cannot edit while generation disposition is unknown")
            stage("weight_edit")
            result = engine.apply_model(lambda model: model._bu_weight_editor.apply(value))
            if len(result) != 1:
                raise ValueError("unexpected worker count")
            engine.reset_prefix_cache()
            engine.reset_mm_cache()
            report["current_scale"] = value
            report["patch_audits"].setdefault(str(value), result[0])
            report["patch_events"].append({"scale": value, "audit_sha256": hashlib.sha256(
                json.dumps(result[0], sort_keys=True, allow_nan=False).encode()).hexdigest()})
            save()

        async def infer(request_messages, cap):
            nonlocal native_pending
            admit_work()
            if native_pending or report["model_calls"] >= 386 or cap not in (64, 192):
                raise BenchmarkBudgetStop("model request cap or native pending barrier")
            report["model_calls"] += 1
            native_pending = True
            stage("generation")
            call_start = time.monotonic()
            try:
                result = await asyncio.to_thread(engine.chat, request_messages,
                    SamplingParams(temperature=0, top_p=1, max_tokens=cap, n=1), use_tqdm=False)
            except BaseException:
                report["native_inference_disposition"] = "unknown; no next request; parent must terminate CUDA-owning child"
                save()  # Keep generation stage and original watchdog start.
                raise
            native_pending = False
            if len(result) != 1 or not result[0].finished or len(result[0].outputs) != 1:
                raise RuntimeError("incomplete native generation result")
            output = result[0].outputs[0]
            report["generation_records"].append({"call": report["model_calls"],
                "scope": report["request_scope"], "case_id": report.get("current_case"),
                "scale": report["current_scale"], "elapsed_s": time.monotonic() - call_start,
                "prompt_tokens": len(result[0].prompt_token_ids), "output_tokens": len(output.token_ids),
                "max_tokens": cap, "finish_reason": output.finish_reason, "stop_reason": output.stop_reason})
            stage("browser_step")
            return PolicyReply(output.text, len(result[0].prompt_token_ids), len(output.token_ids))

        async def evaluate():
            # One fixed neutral call per scale before ANY scored reset; no task
            # previews, quality selection, or repeat warmup on a failed result.
            blank = io.BytesIO()
            Image.new("RGB", (160, 210), (255, 255, 255)).save(blank, format="PNG")
            warmup_messages = [{"role": "user", "content": [
                {"type": "text", "text": 'Unscored runtime warmup. This blank page has no task. Return {"action":[{"done":{"text":"ready","success":false}}]}.'},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(blank.getvalue()).decode()}}]}]
            for value in (1.0, scale):
                apply(value)
                report["request_scope"] = "unscored_neutral_warmup"
                report["current_case"] = None
                reply = await asyncio.wait_for(infer(warmup_messages, 64), 40)
                report["warmups"].append({"scale": value, "native_response": reply.text[:4096],
                    "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens})
                save()
            for index, case in enumerate(cases()):
                for value in ((1.0, scale) if index % 2 == 0 else (scale, 1.0)):
                    admit_work()
                    apply(value)
                    report["current_case"], report["request_scope"] = case.case_id, "scored_miniwob_episode"
                    stage("browser_step")
                    env = make_environment(case, admission["miniwob_versions"])
                    report["environments_constructed"] += 1
                    def partial(row):
                        report["partial_episode"] = row
                        save()
                    try:
                        row = await run_episode(env, case, infer, expected_initials[case.case_id], partial)
                        report["variants"][str(value)].append(row)
                        report.pop("partial_episode", None)
                        save()
                    finally:
                        env.close()
                        report["environment_close_returns"] += 1
                        save()
            report["browser_close_returned"] = report["environment_close_returns"] == report["environments_constructed"] == 48

        async def guarded_evaluate():
            try:
                await evaluate()
            except BaseException as exc:
                report["status"] = "incomplete"
                report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                if isinstance(exc, BenchmarkInferenceUnknown):
                    report["partial_episode"] = exc.partial
                if native_pending:
                    report["native_inference_disposition"] = "unknown; save before asyncio executor shutdown; parent45s watchdog required"
                    save()
                else:
                    stage("failed")
                raise

        asyncio.run(guarded_evaluate())
        baseline = {row["case_id"]: row for row in report["variants"]["1.0"]}
        edited = {row["case_id"]: row for row in report["variants"][str(scale)]}
        report["paired_gains"] = [key for key in baseline if edited[key]["success"] and not baseline[key]["success"]]
        report["paired_regressions"] = [key for key in baseline if baseline[key]["success"] and not edited[key]["success"]]
        report["status"] = "completed_frozen_subset"
        report["complete_coverage"] = all(len(rows) == 24 for rows in report["variants"].values())
        stage("completed")
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if native_pending:
            save()
        else:
            stage("failed")
