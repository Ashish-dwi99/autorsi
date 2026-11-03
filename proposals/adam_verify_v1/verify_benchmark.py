"""Unlaunched private MiniWoB subset experiment; optional runtimes load at call.

The original and one previously frozen candidate receive12 reserved seeded cases each. This is a limited, DOM-assisted MiniWoB subset, not the full suite,
not an Astra comparison, and not automatic model promotion. The original native
task-specific timers continue during inference. Nothing is published here.
"""
from pathlib import Path
import hashlib
import json
import time

from weight_space_lab.diagnostic import atomic_json


from verify_contract import (scales, GPU_RUN, MAX_MODEL_REQUESTS, NEW_WORK_SECONDS,
    validate_spec, variant_order, compare_verification, save_observation, reset_variant_caches)


def validate_preflight(receipt):
    from verify_native import (cases, VERIFICATION_SEED, expected_timer_ms, NATIVE_TIMER_MS_BY_FAMILY,
                                  NATIVE_SOURCE_PINS, V1_MINIWOB_FILES_SHA256, V1_MINIWOB_FILE_COUNT)
    if (receipt.get("schema") != 3 or receipt.get("phase") != "verification" or receipt.get("status") != "completed"
            or receipt.get("verification_seed") != VERIFICATION_SEED
            or receipt.get("native_timer_ms_by_family") != NATIVE_TIMER_MS_BY_FAMILY
            or receipt.get("model_calls") != 0 or receipt.get("environment_steps") != 24
            or receipt.get("fresh_resets") != 24 or receipt.get("fresh_reset_attempts") != 24):
        raise ValueError("exact native paired CPU preflight required")
    if receipt.get("source_pins") != {"package_files_sha256": V1_MINIWOB_FILES_SHA256,
            "package_file_count": V1_MINIWOB_FILE_COUNT, "selected_source_files": NATIVE_SOURCE_PINS}:
        raise ValueError("exact frozen installed source pins required")
    records = receipt.get("cases", [])
    if len(records) != 12:
        raise ValueError("all12 reserved verification preflight cases required")
    output = {}
    for case, record in zip(cases(), records, strict=True):
        if (record.get("case_id") != case.case_id or record.get("env_id") != case.env_id
                or record.get("seed") != case.seed or record.get("native_timer_ms") != expected_timer_ms(case)
                or record.get("status") != "completed"):
            raise ValueError("preflight frozen case order changed")
        initial = record.get("initial", {})
        if set(initial) != {"screenshot_sha256", "visible_dom_sha256", "screen_shape"}:
            raise ValueError("complete paired initial fingerprints required")
        for key in ("screenshot_sha256", "visible_dom_sha256"):
            value = initial[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid initial fingerprint")
        resets = record.get("resets", [])
        if len(resets) != 2:
            raise ValueError("both actual reset records required")
        for member, reset in enumerate(resets):
            timer = reset.get("timer", {})
            if (reset.get("pair_member") != member or reset.get("status") != "completed"
                    or reset.get("close_returned") is not True or reset.get("initial") != initial
                    or reset.get("none_step") != {"reward": 0.0, "terminated": False, "truncated": False}
                    or timer.get("case_id") != case.case_id or timer.get("env_id") != case.env_id
                    or timer.get("seed") != case.seed or timer.get("status") != "accepted"
                    or timer.get("expected_native_timer_ms") != expected_timer_ms(case)
                    or timer.get("observed_timer_type") != "int"
                    or timer.get("observed_timer_encoding") != "json_scalar"
                    or type(timer.get("observed_timer_value")) is not int
                    or timer.get("observed_timer_value") != expected_timer_ms(case)):
                raise ValueError("paired unchanged native clocks/reset/NONE/close evidence differs")
        output[case.case_id] = initial
    return output


def summarize(rows):
    """Always retain the planned12 denominator; missing episodes cannot vanish."""
    return {"planned_episodes": 12, "completed_episodes": len(rows),
            "missing_episodes": 12 - len(rows), "native_successes": sum(row["success"] for row in rows),
            "success_fraction_of_planned": sum(row["success"] for row in rows) / 12,
            "coverage_complete": len(rows) == 12,
            "invalid_actions": sum(row["invalid_actions"] for row in rows),
            "actual_model_calls": sum(row["model_calls"] for row in rows),
            "environment_steps": sum(row["environment_steps"] for row in rows),
            "elapsed_s": sum(row["elapsed_s"] for row in rows)}


def prepare_variant(engine, value, candidate, current_scale, edit):
    """Always clear caches for the next episode, even for repeated pair scales."""
    if type(value) not in (int,float) or value not in scales(candidate):
        raise ValueError('only the fixed original/candidate pair is executable')
    idle=getattr(engine.llm_engine,'has_unfinished_requests',None)
    if not callable(idle) or idle() is not False:
        raise RuntimeError('actual idle engine required before variant admission')
    changed=current_scale!=value
    audit=edit(value) if changed else None
    cache=reset_variant_caches(engine)
    if idle() is not False:
        raise RuntimeError('idle engine required after variant cache reset')
    return {'weight_edit_performed':changed,'audit':audit,'cache_reset':cache}


def run_verification(admission, progress_path):
    import asyncio
    import base64
    import io
    import tempfile
    from PIL import Image
    import torch
    from safetensors.torch import load_file
    from vllm import LLM
    from action_schema import (engine_config, sampling_parameters, schema_receipt,
                               validate_cpu_grammar, runtime_backend_identity)
    from weight_space_lab.vllm018_compat import normalize_config, install_single_process_loader
    from weight_space_lab.vllm_attention import AttentionView
    from verify_patch import MergedWeightEditor
    from weight_space_lab.patch import file_sha256
    from weight_space_lab.policy_io import processor_manifest
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    from weight_space_lab.eval.runner import PolicyReply
    from verify_entry import runtime_manifest
    from verify_native import (cases, make_environment, run_episode,
                                BenchmarkInferenceUnknown, BenchmarkBudgetStop, NATIVE_TIMER_MS_BY_FAMILY)

    started = time.monotonic()
    report = {"schema": 1, "run": GPU_RUN, "status": "running", "model_calls": 0,
              "planned_cases": 12, "planned_episodes": 24, "variants": {}, "warmups": [],
              "generation_records": [], "patch_audits": {}, "patch_events": [],
              "experiment_spec": admission["experiment_spec"], "artifacts": [],
              "source_manifest": admission["source_manifest"], "structural_decoding": schema_receipt(),
              "prior_full_shard_receipt_sha256": admission["prior_report_sha256"],
              "benchmark": "Reserved12-family paired verification subset; fixed structural JSON grammar and original+one frozen candidate; native10s/15s; DOM-assisted",
              "astra_comparison": "not performed", "automatic_promotion": False,
              "candidate_selection_frozen_before_verification": True, "verification_started":False, "candidate_reselected":False, "official_prior_score_unchanged": "21/24", "checkpoint_exported": False,
              "model_call_counter_scope": "Admitted chat requests before to_thread, including2 neutral warmups; not native entry proof; startup/profiling forwards excluded",
              "limits": {"model_calls": 194, "warmup_calls": 2, "scored_model_calls_with_full_warmups": 192,
                         "native_timer_ms_by_family": dict(NATIVE_TIMER_MS_BY_FAMILY), "actions_per_episode": 8,
                         "child_wall_s": 1200, "new_work_before_s": 1140, "generation_hard_s": 45},
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
        if time.monotonic() - started >= NEW_WORK_SECONDS:
            raise BenchmarkBudgetStop("new work refused at child1140s; original1200s outer wall remains")

    try:
        stage("source_admission")
        candidate=validate_spec(admission)
        pair=scales(candidate)
        report["candidate_selection"]=admission["candidate_selection"]
        report["candidate_selection_sha256"]=admission["candidate_selection_sha256"]
        report["quality_review"]=admission["quality_review"]
        report["quality_review_sha256"]=admission["quality_review_sha256"]
        from verify_contract import executing_source_join
        import verify_patch, verify_native, verify_entry, action_schema, development_contract
        report["executing_source_join"] = executing_source_join(admission["source_manifest"], {
            "verify_benchmark.py": __file__, "verify_patch.py": verify_patch.__file__,
            "verify_native.py": verify_native.__file__, "verify_entry.py": verify_entry.__file__,
            "action_schema.py": action_schema.__file__, "development_contract.py": development_contract.__file__})
        validate_cpu_grammar(admission["grammar_preflight"])
        if runtime_backend_identity() != admission["grammar_preflight"]["runtime"]:
            raise ValueError("structured decoder runtime changed after fresh CPU admission")
        report["grammar_preflight"] = admission["grammar_preflight"]
        artifact_dir = Path(progress_path).parent / "artifacts"
        artifact_dir.mkdir(exist_ok=False)
        expected_initials = validate_preflight(admission["miniwob_preflight"])
        if runtime_manifest() != admission["runtime_manifest"]:
            raise ValueError("MiniWoB source/browser/dependency runtime changed after CPU preflight")
        report["runtime_manifest"] = admission["runtime_manifest"]
        report["native_preflight"] = admission["miniwob_preflight"]
        report["variants"] = {str(value): [] for value in pair}
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
                     seed=20261103, disable_log_stats=True, structured_outputs_config=engine_config())
        actual_structured = engine.llm_engine.vllm_config.structured_outputs_config
        if any(getattr(actual_structured, key) != value for key, value in engine_config().items()):
            raise ValueError("actual engine structured decoding differs from frozen config")
        report["actual_structured_output_config"] = {key: getattr(actual_structured, key) for key in engine_config()}
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
        stage("actual_weight_value_inspection")
        inspections = engine.apply_model(lambda model: model._bu_weight_editor.inspect_values())
        if len(inspections) != 1 or len(inspections[0].get("entries", [])) != 192:
            raise ValueError("complete192-matrix actual value inspection required")
        report["weight_value_inspection"] = inspections[0]
        save()  # Persist numeric observations before any non-baseline edit.

        def apply(value):
            admit_work()
            if native_pending:
                raise RuntimeError("cannot edit while generation disposition is unknown")
            stage("variant_admission")
            def edit(scale):
                result=engine.apply_model(lambda model:model._bu_weight_editor.apply(scale))
                if len(result)!=1:raise ValueError('unexpected worker count')
                return result[0]
            previous=report.get('current_scale')
            receipt=prepare_variant(engine,value,candidate,previous,edit)
            report["current_scale"] = value
            if receipt['weight_edit_performed']:
                report["patch_audits"][str(value)]=receipt['audit']
            report["patch_events"].append({"scale":value,"previous_scale":previous,
                "weight_edit_performed":receipt['weight_edit_performed'],
                "cache_reset":receipt['cache_reset'],"audit_sha256":hashlib.sha256(
                json.dumps(report['patch_audits'][str(value)],sort_keys=True,allow_nan=False).encode()).hexdigest(),
                "audit_scope":"This edit receipt if performed, otherwise retained prior actual edit receipt"})
            save()

        async def infer(request_messages, cap):
            nonlocal native_pending
            admit_work()
            if native_pending or report["model_calls"] >= MAX_MODEL_REQUESTS or cap not in (64, 96):
                raise BenchmarkBudgetStop("model request cap or native pending barrier")
            report["model_calls"] += 1
            native_pending = True
            stage("generation")
            call_start = time.monotonic()
            try:
                result = await asyncio.to_thread(engine.chat, request_messages,
                    sampling_parameters(cap), use_tqdm=False)
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
                "max_tokens": cap, "schema_sha256": schema_receipt()["sha256"], "finish_reason": output.finish_reason, "stop_reason": output.stop_reason})
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
            for value in pair:
                apply(value)
                report["request_scope"] = "unscored_neutral_warmup"
                report["current_case"] = None
                reply = await asyncio.wait_for(infer(warmup_messages, 64), 40)
                report["warmups"].append({"scale": value, "native_response": reply.text[:4096],
                    "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens})
                save()
            report["verification_started"]=True;save()
            for index, case in enumerate(cases()):
                for value in variant_order(index,candidate):
                    admit_work()
                    apply(value)
                    report["current_case"], report["request_scope"] = case.case_id, "scored_miniwob_episode"
                    stage("browser_step")
                    env = make_environment(case, admission["miniwob_versions"])
                    report["environments_constructed"] += 1
                    def partial(row):
                        report["partial_episode"] = row
                        save()
                    def capture(observation, dispatch):
                        if len(report["artifacts"]) >= 192:
                            raise BenchmarkBudgetStop("bounded observation capture count reached")
                        name = f"case-{index:02d}-scale-{value}-step-{dispatch}.png"
                        receipt = save_observation(observation, artifact_dir / name,
                            case_id=case.case_id, scale=value, dispatch=dispatch,
                            elapsed_s=time.monotonic() - started)
                        report["artifacts"].append(receipt)
                        save()
                        return receipt
                    try:
                        row = await run_episode(env, case, infer, expected_initials[case.case_id], partial,
                                                capture_observation=capture)
                        report["variants"][str(value)].append(row)
                        report.pop("partial_episode", None)
                        save()
                    finally:
                        env.close()
                        report["environment_close_returns"] += 1
                        save()
            report["browser_close_returned"] = report["environment_close_returns"] == report["environments_constructed"] == 24

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
        report["verification_comparison"] = compare_verification(report["variants"],candidate)
        report["status"] = "completed_verification_only"
        report["complete_coverage"] = all(len(rows) == 12 for rows in report["variants"].values())
        stage("completed")
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        if native_pending:
            save()
        else:
            stage("failed")
