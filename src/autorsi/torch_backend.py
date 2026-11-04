"""Optional Torch integration for ordinary dense, eager, deepcopy-able modules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .backends import InMemoryBackend
from .contracts import ModelState, ValidationError


class TorchBackend(InMemoryBackend):
    """Own a detached nn.Module and edit allowlisted floating-point parameters.

    No model/provider download occurs. Model copies can require several times
    the model's memory. Quantized, distributed, sharded, compiled and offloaded
    runtimes require a separate backend. Targeted tied storage is rejected.
    """
    def __init__(self, model, *, editable, max_abs_delta: float = 0.1, max_total_delta: float | None = None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchBackend requires PyTorch: install this checkout with 'pip install .[torch]'") from exc
        self._torch = torch
        super().__init__(model, editable=editable, max_abs_delta=max_abs_delta, max_total_delta=max_total_delta)
        self._model.eval()

    def _clone(self, model):
        result = super()._clone(model)
        result.eval()
        return result

    def _inference_context(self):
        return self._torch.inference_mode()

    def _validate_model(self, model):
        torch = self._torch
        if not isinstance(model, torch.nn.Module):
            raise ValidationError("TorchBackend expects a torch.nn.Module")
        if hasattr(model, "_orig_mod") or hasattr(model, "hf_device_map"):
            raise ValidationError("compiled/offloaded/sharded models need a custom backend")
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            raise ValidationError("parallel/distributed models need a custom backend")
        parameters = dict(model.named_parameters(remove_duplicate=False))
        missing = self._editable - parameters.keys()
        if missing:
            raise ValidationError(f"editable names are not parameters: {sorted(missing)}")
        storage = {}
        state = model.state_dict()
        if not state:
            raise ValidationError("model has no state_dict tensors")
        for key, value in state.items():
            if (not torch.is_tensor(value) or value.is_quantized or value.device.type == "meta"
                    or value.layout != torch.strided or value.is_complex()):
                raise ValidationError(f"only dense, real, materialized state_dict tensors are supported: {key}")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise ValidationError(f"nonfinite model state: {key}")
            identity = (str(value.device), value.untyped_storage().data_ptr())
            storage.setdefault(identity, []).append(key)
        for aliases in storage.values():
            if len(aliases) > 1 and self._editable.intersection(aliases):
                raise ValidationError(f"editable tied/aliased tensors require a custom recipe/backend: {aliases}")
        for key in self._editable:
            value = parameters[key]
            if not value.is_floating_point() or value.numel() == 0:
                raise ValidationError(f"editable parameters must be nonempty floating-point tensors: {key}")

    def _fingerprint(self, model):
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            digest.update(json.dumps([key, list(value.shape), str(value.dtype)]).encode())
            # A contiguous clone owns precisely its bytes, including BF16; no
            # NumPy dependency or pickle metadata enters the fingerprint.
            value = value.detach().to("cpu").contiguous().clone()
            digest.update(bytes(value.untyped_storage()))
        return digest.hexdigest()

    def _apply(self, model, recipe):
        torch = self._torch
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for edit in recipe.edits:
                target = parameters[edit.parameter]
                old = target.detach().to(device="cpu", dtype=torch.float64)
                proposed = old * edit.value if edit.operation == "scale" else old + edit.value
                realized = proposed.to(dtype=target.dtype)
                if not torch.isfinite(realized).all().item():
                    raise ValidationError(f"edit produced nonfinite values: {edit.parameter}")
                delta = (realized.to(torch.float64) - old).abs().max().item()
                if delta > self._max_abs_delta:
                    raise ValidationError(f"edit exceeds max_abs_delta for {edit.parameter}: {delta}")
                target.copy_(realized)

    def _validate_total_delta(self, model):
        torch = self._torch
        original = dict(self._origin.named_parameters())
        current = dict(model.named_parameters())
        for key in self._editable:
            delta = (current[key].detach().to(device="cpu", dtype=torch.float64)
                     - original[key].detach().to(device="cpu", dtype=torch.float64)).abs().max().item()
            if delta > self._max_total_delta:
                raise ValidationError(f"edit exceeds max_total_delta from original weights for {key}: {delta}")

    def export(self, state: ModelState, directory: Path):
        """Export weights only; instantiate the same architecture to reload."""
        self._check(state)
        path = directory / "weights.pt"
        with path.open("xb") as stream:
            self._torch.save({key: value.detach().cpu().clone()
                              for key, value in state.payload.state_dict().items()}, stream)
        return path
