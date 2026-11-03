"""Fixed structural decoding constraint; never an action or success oracle.

The grammar deliberately leaves string length/ASCII, visible target identity,
remaining quota and done ordering to the unchanged strict action parser. Plain
JSON strings preserve escapes; xgrammar0.2.8's bounded-string grammar excludes
some escapes, so maxLength/pattern are not falsely claimed here. No model code
or tensor runtime is imported until the explicit CPU/backend or sampling calls.
"""
from __future__ import annotations
import hashlib
import json

XGRAMMAR_VERSION = "0.2.8"
VLLM_VERSION = "0.18.0"


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else list(required),
            "additionalProperties": False}


def action_schema():
    """A fresh constant schema, independent of observations, goals and variants."""
    index = {"type": "integer", "minimum": 1}
    branches = [
        _object({"click": _object({"index": index})}),
        _object({"input": _object({"index": index, "text": {"type": "string"},
                                  "clear": {"type": "boolean"}}, ("index", "text"))}),
        _object({"done": _object({"text": {"type": "string"}, "success": {"type": "boolean"},
                                 "files_to_display": {"anyOf": [{"type": "null"},
                                     {"type": "array", "maxItems": 0}]}}, ("text",))}),
    ]
    return _object({"action": {"type": "array", "minItems": 1, "maxItems": 8,
                               "items": {"anyOf": branches}}})


def schema_json():
    # Preserve declared object-key order: xgrammar uses it when any_order=False.
    return json.dumps(action_schema(), separators=(",", ":"), ensure_ascii=True)


def schema_receipt():
    return {"id": "adam-miniwob-structural-json-v2",
            "sha256": hashlib.sha256(schema_json().encode()).hexdigest(),
            "vllm": VLLM_VERSION, "xgrammar": XGRAMMAR_VERSION,
            "engine_config": engine_config(), "static_for_all_requests": True,
            "parser_remains_authoritative": True, "response_repair": False,
            "max_completion_tokens": 96, "warmup_tokens": 64}


def engine_config():
    # No automatic backend selection or unconstrained reasoning phase.
    return {"backend": "xgrammar", "disable_any_whitespace": False,
            "disable_additional_properties": False,
            "reasoning_parser": "", "reasoning_parser_plugin": "",
            "enable_in_reasoning": True}


def sampling_parameters(cap, *, sampling_class=None, constraint_class=None):
    if type(cap) is not int or cap not in (64, 96):
        raise ValueError("only frozen warmup/scored token bounds admitted")
    if sampling_class is None or constraint_class is None:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        sampling_class, constraint_class = SamplingParams, StructuredOutputsParams
    return sampling_class(temperature=0, top_p=1, max_tokens=cap, n=1,
        structured_outputs=constraint_class(json=schema_json()))


def conformance_examples():
    """Static synthetic protocol cases. No benchmark task, answer or DOM data."""
    def encoded(value):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    positive = [
        ("click", {"action": [{"click": {"index": 3}}]}),
        ("input_default_clear", {"action": [{"input": {"index": 1, "text": "hello"}}]}),
        ("escaped_ascii", {"action": [{"input": {"index": 1, "text": 'quote" slash\\', "clear": False}}]}),
        ("done", {"action": [{"done": {"text": "ready", "success": False}}]}),
        ("done_files_null", {"action": [{"done": {"text": "ready", "files_to_display": None}}]}),
        ("done_files_empty", {"action": [{"done": {"text": "ready", "files_to_display": []}}]}),
        ("eight_actions", {"action": [{"click": {"index": 3}}] * 8}),
        ("unicode_done", {"action": [{"done": {"text": "caf\u00e9"}}]}),
        # Grammar acceptance is intentional; unchanged parser rejects these.
        ("parser_rejects_unicode_input", {"action": [{"input": {"index": 1, "text": "\u00e9"}}]}),
        ("parser_rejects_257_chars", {"action": [{"input": {"index": 1, "text": "x" * 257}}]}),
        ("parser_rejects_missing_ref", {"action": [{"click": {"index": 999999}}]}),
        ("parser_rejects_done_not_last", {"action": [{"done": {"text": "done"}}, {"click": {"index": 3}}]}),
    ]
    negative = [
        ("missing_envelope", {"click": {"index": 3}}),
        ("wrong_envelope", {"actions": [{"click": {"index": 3}}]}),
        ("extra_reasoning", {"memory": "reasoning", "action": [{"click": {"index": 3}}]}),
        ("empty_actions", {"action": []}),
        ("nine_actions", {"action": [{"click": {"index": 3}}] * 9}),
        ("missing_index", {"action": [{"click": {}}]}),
        ("boolean_index", {"action": [{"click": {"index": True}}]}),
        ("zero_index", {"action": [{"click": {"index": 0}}]}),
        ("unknown_tool", {"action": [{"evaluate": {"script": "return true"}}]}),
        ("two_tools", {"action": [{"click": {"index": 3}, "done": {"text": "yes"}}]}),
        ("extra_tool_field", {"action": [{"click": {"index": 3, "text": "no"}}]}),
        ("missing_text", {"action": [{"input": {"index": 1}}]}),
        ("nonboolean_clear", {"action": [{"input": {"index": 1, "text": "x", "clear": "yes"}}]}),
        ("files_disallowed", {"action": [{"done": {"text": "yes", "files_to_display": ["x"]}}]}),
    ]
    rows = [{"name": name, "text": encoded(value), "expected_complete": True} for name, value in positive]
    rows += [{"name": name, "text": encoded(value), "expected_complete": False} for name, value in negative]
    rows += [
        {"name": "truncated_prefix", "text": '{"action":[{"click":{"index":3}', "expected_complete": False},
        {"name": "trailing_prose", "text": encoded(positive[0][1]) + " done", "expected_complete": False},
        {"name": "duplicate_key", "text": '{"action":[{"click":{"index":3,"index":1}}]}', "expected_complete": False},
    ]
    return rows


def validate_cpu_grammar(receipt):
    expected = conformance_examples()
    if (receipt.get("status") != "passed" or receipt.get("schema") != schema_receipt()
            or receipt.get("vocab_size") != 151936 or receipt.get("model_calls") != 0
            or receipt.get("model_instantiations") != 0 or receipt.get("cuda_initialized") is not False
            or receipt.get("runtime", {}).get("versions") != {"vllm": VLLM_VERSION, "xgrammar": XGRAMMAR_VERSION}
            or not receipt.get("runtime", {}).get("python_sources")
            or not receipt.get("tokenizer_class") or not receipt.get("tokenizer_path")
            or len(receipt.get("cases", [])) != len(expected)):
        raise ValueError("fresh exact tokenizer/backend grammar admission required")
    for observed, case in zip(receipt["cases"], expected, strict=True):
        if (observed.get("name") != case["name"]
                or observed.get("text_sha256") != hashlib.sha256(case["text"].encode()).hexdigest()
                or observed.get("complete") is not case["expected_complete"]
                or observed.get("token_complete") is not case["expected_complete"]):
            raise ValueError("backend grammar conformance differs from frozen cases")


def runtime_backend_identity():
    """Installed package versions and bounded Python source identity; no model."""
    import importlib
    import importlib.metadata
    from pathlib import Path
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "xgrammar")}
    if versions != {"vllm": VLLM_VERSION, "xgrammar": XGRAMMAR_VERSION}:
        raise ValueError("unreviewed structured decoder package version")
    sources = {}
    for name in ("vllm.sampling_params", "vllm.config.structured_outputs",
                 "vllm.v1.structured_output", "vllm.v1.structured_output.backend_xgrammar",
                 "xgrammar.grammar", "xgrammar.compiler", "xgrammar.matcher"):
        path = Path(importlib.import_module(name).__file__)
        if not path.is_file() or path.stat().st_size > 1024**2:
            raise ValueError("unexpected structured decoder Python source size")
        sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"versions": versions, "python_sources": sources,
            "scope": "Installed Python wrapper identity; not a native-library binary/ABI attestation"}


def cpu_backend_preflight(base_path, vocab_size, *, save_partial):
    """CPU grammar + actual tokenizer compilation, no model/CUDA construction."""
    import importlib.metadata
    import time
    import torch
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from vllm.config import StructuredOutputsConfig
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
    if (importlib.metadata.version("vllm") != VLLM_VERSION
            or importlib.metadata.version("xgrammar") != XGRAMMAR_VERSION
            or type(vocab_size) is not int or vocab_size != 151936 or torch.cuda.is_initialized()):
        raise ValueError("reviewed exact CPU backend, vocab and uninitialized CUDA required")
    started = time.monotonic()
    report = {"status": "running", "schema": schema_receipt(), "vocab_size": vocab_size,
              "model_calls": 0, "model_instantiations": 0, "cuda_initialized": False,
              "runtime": runtime_backend_identity(), "tokenizer_path": str(base_path), "cases": [],
              "stage": "validating_vllm_schema"}
    save_partial(report)
    config = StructuredOutputsConfig(**engine_config())
    if any(getattr(config, key) != value for key, value in engine_config().items()):
        raise ValueError("structured output config was not accepted exactly")
    validate_xgrammar_grammar(sampling_parameters(96))
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False)
    report.update(tokenizer_class=type(tokenizer).__name__, tokenizer_length=len(tokenizer),
                  stage="compiling_actual_tokenizer_grammar")
    save_partial(report)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(info, max_threads=4, cache_enabled=True, cache_limit_bytes=64 * 1024**2)
    compiled = compiler.compile_json_schema(schema_json(), any_whitespace=True)
    report["stage"] = "grammar_conformance"
    save_partial(report)
    for case in conformance_examples():
        matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        accepted = matcher.accept_string(case["text"])
        complete = bool(accepted and matcher.is_completed() and matcher.is_terminated())
        tokens = tokenizer.encode(case["text"], add_special_tokens=False)
        token_matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
        token_accepted = all(token_matcher.accept_token(token) for token in tokens)
        token_complete = bool(token_accepted and token_matcher.is_completed() and token_matcher.is_terminated())
        row = {"name": case["name"], "text_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
               "accepted_prefix": bool(accepted), "complete": complete, "token_complete": token_complete,
               "synthetic_token_count": len(tokens)}
        report["cases"].append(row)
        report["elapsed_s"] = time.monotonic() - started
        save_partial(report)  # Preserve actual counterexample before refusal.
        if complete is not case["expected_complete"] or token_complete is not case["expected_complete"]:
            report.update(status="failed", failed_case=case["name"])
            save_partial(report)
            raise ValueError("CPU grammar conformance failure: " + case["name"])
    report.update(status="passed", stage="completed", elapsed_s=time.monotonic()-started,
        cuda_initialized=torch.cuda.is_initialized(),
        limitation="CPU compiler is separate from vLLM's compiler; identical-schema GPU warmups precede scored resets")
    validate_cpu_grammar(report)
    save_partial(report)
    return report
