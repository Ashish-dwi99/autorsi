"""Reversible standard-LoRA edits; importing this module does not import torch.

The permitted experiment changes B to lambda * B, leaving A and the base model
unchanged. Thus the adapter contribution (alpha / r) * B @ A is scaled by
lambda, subject to the tensor dtype's arithmetic. This is not a claim that a
separately served merged checkpoint is bitwise equivalent to base + adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from typing import Any


BASE_MODEL = "browser-use/bu-30b-a3b-preview"
RANK = 16
ALPHA = 32
TARGET_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
ALLOWED_SCALES = frozenset({0.0, 0.5, 0.75, 1.0})
_KEY = re.compile(
    r"^(?P<module>base_model\.model\..+\.(?P<target>q_proj|k_proj|v_proj|o_proj))"
    r"\.lora_(?P<side>A|B)\.weight$"
)


class PatchValidationError(ValueError):
    """The artifact is outside the fixed standard-LoRA experiment."""


@dataclass(frozen=True)
class LoraPair:
    module: str
    target: str
    a_key: str
    b_key: str


def validate_scale(scale: Any) -> float:
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise PatchValidationError("scale must be one of 0, 0.5, 0.75, 1")
    value = float(scale)
    if not math.isfinite(value) or value not in ALLOWED_SCALES:
        raise PatchValidationError("scale must be one of 0, 0.5, 0.75, 1")
    return value


def validate_adapter_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact base and ordinary alpha/r LoRA interpretation.

This validates adapter serialization, not the provenance, revision, or training
quantization of the base weights. Empty optional PEFT configuration fields are
accepted; nonempty variants that alter inference or add trainable objects are
refused.
"""
    if not isinstance(config, Mapping):
        raise PatchValidationError("adapter config must be a mapping")
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise PatchValidationError("unexpected base_model_name_or_path")
    if config.get("peft_type") != "LORA":
        raise PatchValidationError("only PEFT LORA is supported")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if type(rank) is not int or rank != RANK:
        raise PatchValidationError("rank must be exactly 16")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or alpha != ALPHA:
        raise PatchValidationError("lora_alpha must be exactly 32")
    targets = config.get("target_modules")
    if not isinstance(targets, (list, tuple)) or any(type(x) is not str for x in targets):
        raise PatchValidationError("target_modules must be an explicit string list")
    if len(targets) != 4 or set(targets) != TARGET_MODULES:
        raise PatchValidationError("target_modules must contain q/k/v/o_proj exactly once")
    for key in (
        "use_dora", "use_rslora", "use_qalora", "lora_bias", "fan_in_fan_out",
        "load_in_4bit", "load_in_8bit",
    ):
        if config.get(key, False) is not False:
            raise PatchValidationError(f"unsupported flag: {key}")
    if config.get("bias", "none") != "none":
        raise PatchValidationError("only bias=none is supported")
    if config.get("task_type", "CAUSAL_LM") != "CAUSAL_LM":
        raise PatchValidationError("only task_type=CAUSAL_LM is supported")
    for key in (
        "rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication",
        "trainable_token_indices", "quantization_config", "loftq_config",
        "alora_invocation_tokens", "arrow_config", "megatron_config",
    ):
        value = config.get(key)
        if value is not None and value != {} and value != []:
            raise PatchValidationError(f"unsupported nonempty option: {key}")
    dropout = config.get("lora_dropout", 0.0)
    if (isinstance(dropout, bool) or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout)) or not 0 <= dropout < 1):
        raise PatchValidationError("invalid lora_dropout")
    normalized = dict(config)
    normalized["target_modules"] = sorted(targets)
    try:
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise PatchValidationError("adapter config is not finite JSON data") from exc
    return {
        "base_model_name_or_path": BASE_MODEL,
        "peft_type": "LORA",
        "rank": RANK,
        "alpha": ALPHA,
        "effective_alpha_over_rank": ALPHA / RANK,
        "target_modules": sorted(TARGET_MODULES),
        "normalized_config_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def discover_lora_pairs(state_dict: Mapping[str, Any]) -> tuple[LoraPair, ...]:
    """Accept serialized A/B weight pairs only, with all four targets per block.

No key is silently ignored. In-memory adapter-name suffixes, biases, saved base
weights, magnitude vectors and other extra objects require a different recipe.
"""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise PatchValidationError("nonempty adapter tensor mapping required")
    modules: dict[str, dict[str, str]] = {}
    targets_by_block: dict[str, set[str]] = {}
    for key in state_dict:
        match = _KEY.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            raise PatchValidationError(f"unsupported adapter tensor key: {key!r}")
        module, target, side = match.group("module", "target", "side")
        sides = modules.setdefault(module, {})
        if side in sides:
            raise PatchValidationError(f"duplicate LoRA side for {module}")
        sides[side] = key
        targets_by_block.setdefault(module.rsplit(".", 1)[0], set()).add(target)
    pairs = []
    for module, sides in sorted(modules.items()):
        if set(sides) != {"A", "B"}:
            raise PatchValidationError(f"missing A/B pair for {module}")
        pairs.append(LoraPair(module, module.rsplit(".", 1)[1], sides["A"], sides["B"]))
    for block, targets in targets_by_block.items():
        if targets != TARGET_MODULES:
            raise PatchValidationError(f"incomplete q/k/v/o projection inventory for {block}")
    return tuple(pairs)


def _validate_matrix(tensor: Any, key: str, torch_module: Any) -> None:
    if not torch_module.is_tensor(tensor) or not tensor.is_floating_point():
        raise PatchValidationError(f"floating-point tensor required: {key}")
    if getattr(tensor, "is_quantized", False) or str(tensor.dtype) not in {
        "torch.float32", "torch.float16", "torch.bfloat16", "float32", "float16", "bfloat16",
    }:
        raise PatchValidationError(f"ordinary FP32/FP16/BF16 adapter tensor required: {key}")
    if tensor.ndim != 2 or any(d <= 0 for d in tensor.shape):
        raise PatchValidationError(f"nonempty matrix required: {key}")
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise PatchValidationError(f"nonfinite matrix: {key}")


def scale_lora_b(
    state_dict: Mapping[str, Any],
    config: Mapping[str, Any],
    scale: float,
    *,
    torch_module: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return detached tensor copies and descriptive patch metadata.

The input mapping and tensors are never modified. Scale 1 clones without doing
arithmetic, providing the explicit adapter baseline. The injectable tensor API
lets pure tests exercise validation without importing torch or loading a model.
"""
    configuration = validate_adapter_config(config)
    factor = validate_scale(scale)
    pairs = discover_lora_pairs(state_dict)
    if torch_module is None:
        import torch as torch_module
    matrices = []
    for pair in pairs:
        a, b = state_dict[pair.a_key], state_dict[pair.b_key]
        _validate_matrix(a, pair.a_key, torch_module)
        _validate_matrix(b, pair.b_key, torch_module)
        if a.shape[0] != RANK or b.shape[1] != RANK:
            raise PatchValidationError(f"rank/shape mismatch: {pair.module}")
        if a.dtype != b.dtype or a.device != b.device:
            raise PatchValidationError(f"A/B dtype or device mismatch: {pair.module}")
        matrices.append({
            "module": pair.module, "a_key": pair.a_key, "b_key": pair.b_key,
            "a_shape": list(a.shape), "b_shape": list(b.shape), "dtype": str(a.dtype),
        })
    result: dict[str, Any] = {}
    for pair in pairs:
        result[pair.a_key] = state_dict[pair.a_key].detach().clone()
        b = state_dict[pair.b_key].detach().clone()
        result[pair.b_key] = b if factor == 1.0 else b.mul(factor)
        _validate_matrix(result[pair.b_key], pair.b_key, torch_module)
    return result, {
        "schema_version": 1,
        "recipe": "standard_lora_scale_B_only_v1",
        "scale": factor,
        "adapter_delta_multiplier": factor,
        "baseline": factor == 1.0,
        "configuration": configuration,
        "pair_count": len(pairs),
        "tensor_count": len(result),
        "matrices": matrices,
        "original_tensors_modified": False,
        "base_weights_modified": False,
        "finite_precision_note": "B scaling uses its stored floating-point dtype; rounding can occur.",
        "evaluation_requirement": "Use inference/eval mode with adapter dropout disabled for every candidate and baseline.",
        "existing_merged_checkpoint_equivalence": "NOT_ESTABLISHED",
        "evaluation_status": "NOT_EVALUATED",
    }


def file_sha256(path: str | Path, *, max_bytes: int = 256 * 1024 * 1024) -> dict[str, Any]:
    """Hash a bounded regular adapter artifact without loading tensors."""
    path = Path(path)
    if path.is_symlink():
        raise PatchValidationError("artifact symlinks are not accepted")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise PatchValidationError("artifact is not a bounded regular file")
        digest = hashlib.sha256()
        count = 0
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > max_bytes:
                raise PatchValidationError("artifact grew beyond its bound")
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (count != before.st_size or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns or after.st_ctime_ns != before.st_ctime_ns):
        raise PatchValidationError("artifact changed while hashing")
    return {"path": str(path.resolve()), "bytes": count, "sha256": digest.hexdigest()}


def build_patch_provenance(
    source_adapter_path: str | Path,
    patched_adapter_path: str | Path,
    source_config_path: str | Path,
    patch_metadata: Mapping[str, Any],
    *,
    base_revision: str | None = None,
    existing_merged_reference: str | None = None,
) -> dict[str, Any]:
    """Join real file hashes to recipe metadata, without writing or executing.

Callers must serialize to a new destination themselves. These hashes describe
the observed artifacts; they do not establish training quality, an exact base
revision, merged-model equivalence, or any permission to deploy.
"""
    source, destination, config_path = map(Path, (source_adapter_path, patched_adapter_path, source_config_path))
    if source.resolve() == destination.resolve() or os.path.samefile(source, destination):
        raise PatchValidationError("patched adapter must be a different file")
    source_record = file_sha256(source)
    output_record = file_sha256(destination)
    config_record = file_sha256(config_path, max_bytes=1024 * 1024)
    with config_path.open("rb") as stream:
        config_bytes = stream.read(1024 * 1024 + 1)
    if (len(config_bytes) > 1024 * 1024
            or hashlib.sha256(config_bytes).hexdigest() != config_record["sha256"]):
        raise PatchValidationError("adapter config changed between hashing and parsing")
    configuration = validate_adapter_config(json.loads(config_bytes))
    if patch_metadata.get("recipe") != "standard_lora_scale_B_only_v1":
        raise PatchValidationError("unexpected patch recipe")
    validate_scale(patch_metadata.get("scale"))
    if patch_metadata.get("configuration") != configuration:
        raise PatchValidationError("patch metadata does not match adapter config")
    # Detach the record from caller-owned mutable metadata and enforce JSON data.
    metadata = json.loads(json.dumps(dict(patch_metadata), allow_nan=False))
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "separate_lora_adapter_patch",
        "source_adapter": source_record,
        "patched_adapter": output_record,
        "source_config": config_record,
        "patch": metadata,
        "base_model": {"id": BASE_MODEL, "revision": base_revision},
        "existing_merged_reference": existing_merged_reference,
        "existing_merged_checkpoint_equivalence": "NOT_ESTABLISHED",
        "quality_or_latency_improvement": "NOT_ESTABLISHED",
        "originals_policy": "retain source adapter, source config, and original merged checkpoint unchanged",
    }
