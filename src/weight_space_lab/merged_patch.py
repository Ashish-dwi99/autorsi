"""Reversible attention edits relative to the existing merged BU checkpoint.

W_lambda = cast_BF16(float32(W_star) + (lambda - 1) * 2 * (B @ A)).
W_star is the loaded merged checkpoint, not a reconstructed base. In particular,
lambda=0 does not establish equivalence to the original BU base checkpoint.
Importing this module does not import torch or load a model.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

from .patch import (
    RANK, TARGET_MODULES, PatchValidationError, _validate_matrix,
    discover_lora_pairs, validate_adapter_config, validate_scale,
)


def merged_weight_key(adapter_module: str) -> str:
    prefix = "base_model.model."
    if not isinstance(adapter_module, str) or not adapter_module.startswith(prefix):
        raise PatchValidationError("unexpected serialized adapter prefix")
    return adapter_module[len(prefix):] + ".weight"


@contextmanager
def _fp32_arithmetic(torch):
    """Temporarily disable autocast and TF32, restoring caller settings."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


class MergedLoraPatcher:
    """Keep only targeted originals and adapter matrices as CPU tensor copies.

    The caller must serialize all generation and edits. No generation may run
    concurrently with apply(). A failed apply can leave a partial variant; stop
    inference and either terminate the experiment or explicitly restore with
    apply(1). No non-baseline variant is admitted after an incomplete edit.
    """

    def __init__(self, model, adapter_state, config, *, torch_module=None):
        self.configuration = validate_adapter_config(config)
        pairs = discover_lora_pairs(adapter_state)
        expected = {
            f"base_model.model.model.language_model.layers.{layer}.self_attn.{target}"
            for layer in range(48) for target in TARGET_MODULES
        }
        if len(pairs) != 192 or {pair.module for pair in pairs} != expected:
            raise PatchValidationError("exact 48-layer, q/k/v/o 192-pair adapter required")
        if getattr(model, "training", True) is not False:
            raise PatchValidationError("model must already be in eval mode")
        quantization = getattr(getattr(model, "config", None), "quantization_config", None)
        if getattr(model, "is_quantized", False) or quantization not in (None, {}):
            raise PatchValidationError("quantized merged models are outside this recipe")
        if torch_module is None:
            import torch as torch_module
        self._torch = torch_module
        self._model = model
        parameters = dict(model.named_parameters())  # References, never a whole-model clone.
        validated = []
        storage_ids = set()
        for pair in pairs:
            key = merged_weight_key(pair.module)
            if key not in parameters:
                raise PatchValidationError(f"missing exact unfused merged parameter: {key}")
            a, b, weight = adapter_state[pair.a_key], adapter_state[pair.b_key], parameters[key]
            for tensor, name in ((a, pair.a_key), (b, pair.b_key), (weight, key)):
                _validate_matrix(tensor, name, torch_module)
                if str(tensor.layout) not in {"torch.strided", "strided"}:
                    raise PatchValidationError(f"dense strided matrix required: {name}")
            if a.shape[0] != RANK or b.shape[1] != RANK or tuple(weight.shape) != (b.shape[0], a.shape[1]):
                raise PatchValidationError(f"adapter/merged dimensions disagree: {key}")
            if a.dtype != b.dtype or a.device != b.device:
                raise PatchValidationError(f"adapter A/B dtype or device mismatch: {key}")
            if str(weight.dtype) not in {"torch.bfloat16", "bfloat16"} or weight.device.type != "cuda":
                raise PatchValidationError(f"BF16 CUDA merged parameter required: {key}")
            storage_id = (str(weight.device), weight.data_ptr())
            if storage_id in storage_ids:
                raise PatchValidationError("target parameters alias the same storage")
            storage_ids.add(storage_id)
            validated.append((key, weight, a, b))
        # Finish every validation before retaining copies or modifying any weight.
        self._targets = {}
        self._originals = {}
        self._adapter = {}
        with torch_module.no_grad():
            for key, weight, a, b in validated:
                self._targets[key] = weight
                self._originals[key] = weight.detach().to(device="cpu").clone()
                self._adapter[key] = (
                    a.detach().to(device="cpu").clone(),
                    b.detach().to(device="cpu").clone(),
                )
        self._busy = False
        self._state = "original_merged"
        self._scale = 1.0

    def apply(self, scale: float) -> dict[str, Any]:
        """Rebuild a variant from CPU originals; never accumulate prior edits.

        Statistics describe the realized BF16 delta from W_star. Baseline restore
        copies bytes and checks exact tensor equality, without matrix arithmetic.
        Caller provenance must separately pin the merged and adapter artifacts.
        """
        factor = validate_scale(scale)
        if self._busy:
            raise PatchValidationError("concurrent/reentrant patch application refused")
        if self._state == "partial_or_failed" and factor != 1.0:
            raise PatchValidationError("incomplete edit: only explicit baseline restoration is permitted")
        if getattr(self._model, "training", True) is not False:
            raise PatchValidationError("model must remain in eval mode")
        torch = self._torch
        self._busy = True
        self._state = "partial_or_failed"
        entries = []
        total_changed = total_elements = 0
        total_squared_delta = 0.0
        total_max_abs = 0.0
        try:
            with _fp32_arithmetic(torch):
                for key, weight in self._targets.items():
                    original = self._originals[key].to(device=weight.device)
                    if factor == 1.0:
                        weight.copy_(original)
                        if not bool(torch.equal(weight, original)):
                            raise PatchValidationError(f"baseline restoration mismatch: {key}")
                        changed = 0
                        max_abs = squared_delta = 0.0
                    else:
                        a, b = self._adapter[key]
                        a = a.to(device=weight.device, dtype=torch.float32)
                        b = b.to(device=weight.device, dtype=torch.float32)
                        update = torch.matmul(b, a)
                        update.mul_((factor - 1.0) * 2.0)
                        update.add_(original.to(dtype=torch.float32))
                        if not bool(torch.isfinite(update).all().item()):
                            raise PatchValidationError(f"nonfinite FP32 edited parameter: {key}")
                        edited = update.to(dtype=weight.dtype)
                        if not bool(torch.isfinite(edited).all().item()):
                            raise PatchValidationError(f"nonfinite BF16 edited parameter: {key}")
                        changed = int(torch.count_nonzero(edited != original).item())
                        delta = edited.to(dtype=torch.float32) - original.to(dtype=torch.float32)
                        max_abs = float(delta.abs().max().item())
                        squared_delta = float(torch.sum(delta.square(), dtype=torch.float64).item())
                        if not math.isfinite(max_abs) or not math.isfinite(squared_delta):
                            raise PatchValidationError(f"nonfinite delta statistics: {key}")
                        weight.copy_(edited)
                        if not bool(torch.equal(weight, edited)):
                            raise PatchValidationError(f"edited parameter copy mismatch: {key}")
                    elements = int(weight.numel())
                    entries.append({
                        "weight_key": key, "shape": list(weight.shape), "elements": elements,
                        "changed_elements": changed, "max_abs_delta": max_abs,
                        "l2_delta": math.sqrt(squared_delta),
                    })
                    total_elements += elements
                    total_changed += changed
                    total_max_abs = max(total_max_abs, max_abs)
                    total_squared_delta += squared_delta
                if not math.isfinite(total_squared_delta):
                    raise PatchValidationError("aggregate squared delta is not finite")
            self._state = "original_merged" if factor == 1.0 else "edited_merged"
            self._scale = factor
        finally:
            self._busy = False
        return {
            "schema_version": 1,
            "recipe": "existing_merged_plus_fp32_scaled_adapter_delta_v1",
            "formula": "cast_BF16(float32(W_star) + (lambda - 1) * 2 * (float32(B) @ float32(A)))",
            "scale": factor, "baseline": factor == 1.0, "pair_count": len(entries),
            "target_dtype": "bfloat16", "matmul_dtype": "float32", "tf32_enabled_during_edit": False,
            "rebuilt_from_immutable_cpu_originals": True, "baseline_copy_equality_verified": factor == 1.0,
            "zero_scale_is_original_base": "NOT_ESTABLISHED",
            "configuration": self.configuration, "targets": entries,
            "totals": {"elements": total_elements, "changed_elements": total_changed,
                       "max_abs_delta": total_max_abs, "l2_delta": math.sqrt(total_squared_delta)},
            "evaluation_status": "NOT_EVALUATED",
        }


# Stable integration name used by the bounded Modal pilot.
MergedWeightEditor = MergedLoraPatcher
