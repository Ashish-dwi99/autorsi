"""Import-safe spawn target shared by CPU packaging admission and GPU pilot.

No Modal declarations, torch imports, model loading, or provider calls occur on
module import. CPU admission reads model metadata/headers and executes the
separately bounded paired screenshot-fixture render/video preflight, without
loading model tensor payloads or performing model inference.
"""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from weight_space_lab.diagnostic import atomic_json


EXPECTED_VERSIONS = {"vllm": "0.18.0", "torch": "2.10.0", "torchvision": "0.25.0",
                     "torchaudio": "2.10.0", "transformers": "4.57.6", "safetensors": "0.7.0",
                     "pillow": "12.2.0", "playwright": "1.60.0", "xgrammar": "0.2.8"}
def _header(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular safetensors file required")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("incomplete safetensors prefix")
        length = int.from_bytes(prefix, "little")
        if not 2 <= length <= 2 * 1024 * 1024:
            raise ValueError("safetensors header exceeds 2 MiB")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("incomplete safetensors header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    return tensors, {"header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
                     "header_bytes_read": 8 + length, "file_bytes": path.stat().st_size,
                     "tensor_count": len(tensors)}


def _verify_imports(admission, progress_path):
    from screen_contract import validate_spec,validate_attention_selection
    validate_spec(admission)
    validate_attention_selection(admission["attention_selection"],admission["attention_development_report"],admission["attention_development_report_sha256"], quality_review=admission["attention_quality_review"], quality_sha256=admission["attention_quality_review_sha256"])
    from screen_contract import executing_join
    executing_join(admission["source_manifest"], {"proposals/adam_screen_v2/screen_entry.py": __file__})
    started = time.monotonic()
    report = {"schema": 1, "run": "adam-screen-imports-20261104-v2", "status": "running", "stage": "imports",
              "stage_started_monotonic": started, "model_calls": 0, "model_instantiations": 0,
              "tensor_payloads_loaded": 0, "pid": os.getpid(), "process_group": os.getpgrp(),
              "source_manifest": admission["source_manifest"], "versions": {}, "imported_modules": [],
              "experiment_spec": admission["experiment_spec"], "attention_selection_sha256":admission["attention_selection_sha256"]}
    atomic_json(progress_path, report)
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual.split("+", 1)[0] != expected:
            raise ValueError(f"unreviewed {package} version: {actual}")
        report["versions"][package] = actual
    modules = ("torch", "torchvision", "torchaudio", "transformers", "vllm", "safetensors.torch",
               "playwright.async_api", "PIL.Image", "screen_worker", "screen_cpu", "screen_cases",
               "screen_protocol", "screen_harness", "vision_edit", "screen_schema", "dev_patch", "dev_contract", "action_schema", "xgrammar",
               "weight_space_lab.vllm018_compat", "weight_space_lab.vllm_attention",
               "weight_space_lab.policy_io", "modal_screen")
    for name in modules:
        module=importlib.import_module(name)
        if name in {"screen_worker","screen_cpu","modal_screen"}:
            executing_join(admission["source_manifest"],{"proposals/adam_screen_v2/"+name+".py":module.__file__})
        elif name in {"screen_cases","screen_protocol","screen_harness","vision_edit","screen_schema"}:
            executing_join(admission["source_manifest"],{"proposals/adam_screen_v2/"+name+".py":module.__file__})
        elif name in {"dev_patch","dev_contract","action_schema"}:
            executing_join(admission["source_manifest"],{"proposals/adam_schema_v2/"+name+".py":module.__file__})
        report["imported_modules"].append(name);atomic_json(progress_path,report)
    # Resolve the same lazily exported interfaces used by the GPU child, but do
    # not instantiate them or install a model/loader hook during CPU admission.
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
    from transformers import Qwen3VLMoeConfig
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU import admission unexpectedly initialized CUDA")
    report["cuda_initialized"] = False
    report["resolved_interfaces"] = [item.__name__ for item in (LLM, SamplingParams, Qwen3VLMoeForConditionalGeneration)]
    from screen_harness import ScreenSession
    from screen_worker import run_screen_development
    required_callables = {
        "LLM.chat": getattr(LLM, "chat", None),
        "LLM.apply_model": getattr(LLM, "apply_model", None),
        "LLM.reset_prefix_cache": getattr(LLM, "reset_prefix_cache", None),
        "LLM.reset_mm_cache": getattr(LLM, "reset_mm_cache", None),
        "LLMEngine.has_unfinished_requests": getattr(LLMEngine, "has_unfinished_requests", None),
        "LLMEngine.reset_encoder_cache": getattr(LLMEngine, "reset_encoder_cache", None),
        "Qwen3VLMoeForConditionalGeneration.load_weights": getattr(Qwen3VLMoeForConditionalGeneration, "load_weights", None),
        "ScreenSession.run_case": ScreenSession.run_case,
        "run_screen_development": run_screen_development,
    }
    if not all(callable(value) for value in required_callables.values()):
        raise ValueError("required runtime interface is missing or non-callable")
    report["static_callable_interfaces"] = sorted(required_callables)
    from weight_space_lab.vllm018_compat import normalize_config, admitted_weight_names
    from weight_space_lab.patch import file_sha256, validate_adapter_config, discover_lora_pairs
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    from weight_space_lab.policy_io import processor_manifest
    report["stage"] = "metadata_and_headers"
    report["stage_started_monotonic"] = time.monotonic()
    atomic_json(progress_path, report)
    merged, adapter = Path("/merged/bu-30b-ft"), Path("/adapter/adapter")
    for filename, expected in admission["merged_metadata_pins"].items():
        if filename not in {"config.json", "model.safetensors.index.json"} or file_sha256(merged / filename)["sha256"] != expected:
            raise ValueError("merged metadata identity changed")
    # Config hash only. The adapter tensor payload is not hashed or loaded here.
    if file_sha256(adapter / "adapter_config.json")["sha256"] != admission["adapter_pins"]["adapter_config.json"]:
        raise ValueError("adapter configuration identity changed")
    report["adapter_config"] = validate_adapter_config(json.loads((adapter / "adapter_config.json").read_text()))
    normalized = normalize_config((merged / "config.json").read_bytes())
    config = Qwen3VLMoeConfig.from_dict(normalized)
    if (config.text_config.num_experts != 128 or config.text_config.num_hidden_layers != 48
            or config.text_config.tie_word_embeddings is not False):
        raise ValueError("normalized HF4 configuration failed semantic checks")
    report["normalized_config_sha256"] = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
    raw_index = (merged / "model.safetensors.index.json").read_bytes()
    expected_names = admitted_weight_names(raw_index)
    weight_map = json.loads(raw_index)["weight_map"]
    observed = {}
    report["shard_headers"] = {}
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("invalid shard filename")
        header, receipt = _header(merged / filename)
        if receipt["file_bytes"] != admission["prior_source"]["full_merged_shard_hashes"][filename]["bytes"]:
            raise ValueError("shard length differs from prior full-hash receipt")
        if set(observed) & set(header):
            raise ValueError("duplicate tensor key across shard headers")
        if any(weight_map.get(name) != filename for name in header):
            raise ValueError("tensor header disagrees with index shard mapping")
        observed.update(header)
        report["shard_headers"][filename] = receipt
    if set(observed) != expected_names:
        raise ValueError("incomplete header tensor inventory")
    for name, metadata in observed.items():
        if ".mlp.experts." in name:
            shape = [128, 1536, 2048] if name.endswith("gate_up_proj") else [128, 2048, 768]
            if metadata["shape"] != shape or metadata["dtype"] != "BF16":
                raise ValueError("unexpected packed expert header shape/dtype")
    adapter_header, report["adapter_header"] = _header(adapter / "adapter_model.safetensors")
    pairs = discover_lora_pairs(adapter_header)
    expected_pairs = {f"base_model.model.model.language_model.layers.{layer}.self_attn.{target}"
                      for layer in range(48) for target in ("q_proj", "k_proj", "v_proj", "o_proj")}
    if len(pairs) != 192 or {pair.module for pair in pairs} != expected_pairs:
        raise ValueError("unexpected adapter pair inventory")
    for pair in pairs:
        a, b = adapter_header[pair.a_key], adapter_header[pair.b_key]
        name = pair.module.removeprefix("base_model.model.") + ".weight"
        if (len(a["shape"]) != 2 or len(b["shape"]) != 2 or a["shape"][0] != 16 or b["shape"][1] != 16
                or observed[name]["shape"] != [b["shape"][0], a["shape"][1]]):
            raise ValueError("adapter/header attention shapes disagree")
    if reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"]) != admission["runner_reconciliation"]:
        raise ValueError("runner reconciliation changed")
    base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / admission["revision"]
    if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
        raise ValueError("processor identity changed")
    from screen_schema import cpu_backend_preflight
    report["stage"]="cpu_tokenizer_grammar_preflight";report["stage_started_monotonic"]=time.monotonic()
    def save_grammar_partial(journal):
        report["grammar_preflight"]=journal;atomic_json(progress_path,report)
    report["grammar_preflight"]=cpu_backend_preflight(base,config.text_config.vocab_size,
        save_partial=save_grammar_partial)
    import asyncio
    from screen_cpu import preflight,validate_preflight
    report["stage"]="screen_cpu_preflight";report["stage_started_monotonic"]=time.monotonic()
    def save_preflight_partial(journal):
        report["screen_preflight"]=journal;atomic_json(progress_path,report)
    report["screen_preflight"]=asyncio.run(preflight(Path(progress_path).parent/"artifacts",save_preflight_partial))
    if len(validate_preflight(report["screen_preflight"]))!=8:
        raise ValueError("fresh eight-case development render admission required")
    report["browser_close_returned"]=report["screen_preflight"]["browser_close_returned"]
    report["artifacts"]=report["screen_preflight"]["artifacts"]
    report.update({"status": "imports_passed", "stage": "completed", "stage_started_monotonic": time.monotonic(),
                   "elapsed_s": time.monotonic() - started, "adapter_pairs": len(pairs),
                   "merged_tensor_count": len(observed), "full_shard_hashing_repeated": False,
                   "prior_full_shard_receipt_sha256": admission["prior_report_sha256"],
                   "cuda_initialized": torch.cuda.is_initialized()})
    if report["cuda_initialized"]:
        raise ValueError("CPU admission initialized CUDA")
    atomic_json(progress_path, report)


def worker_entry(kind, admission, progress_path):
    """Same exact import/spawn target for CPU admission and later GPU work."""
    os.setsid()
    try:
        from screen_contract import executing_join
        executing_join(admission["source_manifest"], {"proposals/adam_screen_v2/screen_entry.py": __file__})
        if kind == "imports":
            _verify_imports(admission, progress_path)
        elif kind == "development":
            from screen_worker import run_screen_development
            run_screen_development(admission, progress_path)
        else:
            raise ValueError("unknown worker kind")
    except BaseException as exc:
        path = Path(progress_path)
        prior = {}
        try:
            if path.exists() and path.stat().st_size <= 16 * 1024 * 1024:
                prior = json.loads(path.read_text())
        except BaseException:
            prior = {"previous_progress_unreadable": True}
        prior.update({"schema": 1, "status": "failed",
                      "pid": os.getpid(), "process_group": os.getpgrp(),
                      "uncaught_child_error": {"type": type(exc).__name__, "message": str(exc)[:4000]}})
        if prior.get("stage") != "generation":
            prior.update(stage="failed", stage_started_monotonic=time.monotonic())
        atomic_json(path, prior)
