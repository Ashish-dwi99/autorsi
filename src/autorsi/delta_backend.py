"""Memory-lean PyTorch backend: states are tensors for the editable parameters only.

TorchBackend deep-copies the whole model for every capture, candidate, evaluation
and promotion, and keeps several copies alive (serving, origin, controller anchor,
current, candidate). For a 0.5B model that is several GB. DeltaTorchBackend keeps
ONE model. A state is a mapping {parameter name: fp32 tensor} for the editable
parameters only; unchanged tensors are shared between states (never mutated in
place). Evaluating a candidate swaps those tensors into the model under the
serving lock and restores the serving tensors afterwards. Promotion is a pointer
swap. Editable tensors are held in float32 (master weights), so small edits are
not rounded away by bfloat16; run inference with autocast via compute_dtype.

Trade-off: evaluation holds the serving lock, so serving pauses while a candidate
is evaluated (the stock backend evaluates on a separate copy).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time
from types import MappingProxyType

from .backends import PromotionDeadlineError, StaleRevisionError
from .contracts import DeltaEdit, EditRecipe, Evaluator, ModelState, ValidationError, finite, metrics, name


class DeltaTorchBackend:
    def __init__(self, model, *, editable, max_abs_delta: float = 0.1, max_total_delta: float | None = None,
                 copy: bool = True, compute_dtype=None):
        import torch
        self._torch = torch
        if isinstance(editable, str) or not editable:
            raise ValidationError("editable must be a nonempty collection of exact parameter names")
        self._editable = tuple(sorted(set(editable)))
        for key in self._editable:
            name(key, "editable parameter")
        self._max_abs_delta = finite(max_abs_delta, "max_abs_delta", minimum=0)
        self._max_total_delta = finite(max_abs_delta if max_total_delta is None else max_total_delta,
                                       "max_total_delta", minimum=0)
        if not isinstance(model, torch.nn.Module):
            raise ValidationError("DeltaTorchBackend expects a torch.nn.Module")
        if copy:
            from copy import deepcopy
            model = deepcopy(model)
        params = dict(model.named_parameters(remove_duplicate=False))
        missing = set(self._editable) - params.keys()
        if missing:
            raise ValidationError(f"editable names are not parameters: {sorted(missing)}")
        seen = {}
        for key, value in model.state_dict().items():
            seen.setdefault(value.untyped_storage().data_ptr(), []).append(key)
        for aliases in seen.values():
            if len(aliases) > 1 and set(aliases) & set(self._editable):
                raise ValidationError(f"editable tied/aliased tensors are not supported: {aliases}")
        self._model = model.eval()
        for q in model.parameters():          # only editable tensors may ever receive gradients
            q.requires_grad_(False)
        self._params = {k: params[k] for k in self._editable}
        for key, p in self._params.items():
            if not p.is_floating_point() or p.numel() == 0:
                raise ValidationError(f"editable parameters must be nonempty floating-point tensors: {key}")
            p.requires_grad_(False)
            p.data = p.data.detach().to(torch.float32).contiguous()
        self._compute_dtype = compute_dtype
        self._lock = threading.RLock()
        self._local = threading.local()
        self._revision = 0
        self._owner = object()
        self._hashes: dict[int, str] = {}
        self._origin = {k: self._params[k].data for k in self._editable}
        for key, t in self._origin.items():
            if not torch.isfinite(t).all().item():
                raise ValidationError(f"nonfinite editable weights: {key}")
        self._serving = dict(self._origin)

    def _ctx(self):
        stack = ExitStack()
        stack.enter_context(self._torch.inference_mode())
        if self._compute_dtype is not None:
            stack.enter_context(self._torch.autocast("cpu", dtype=self._compute_dtype))
        return stack

    def _hash_tensor(self, t):
        key = id(t)
        h = self._hashes.get(key)
        if h is None:
            c = t.detach().cpu().contiguous()
            h = hashlib.sha256(c.reshape(-1).view(self._torch.uint8).numpy()).hexdigest()
            self._hashes[key] = h
        return h

    def _fingerprint(self, tensors):
        d = hashlib.sha256()
        for k in self._editable:
            d.update(json.dumps([k, list(tensors[k].shape)]).encode())
            d.update(self._hash_tensor(tensors[k]).encode())
        return d.hexdigest()

    def _install(self, tensors):
        for k, p in self._params.items():
            p.data = tensors[k]

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def recipe_space(self):
        return MappingProxyType({"parameters": self._editable,
                                 "operations": ("scale", "add", "add_tensor"),
                                 "max_abs_delta": self._max_abs_delta,
                                 "max_total_delta": self._max_total_delta,
                                 "bound_reference": "per-step delta from current; total delta from original"})

    @contextmanager
    def inference(self):
        with self._lock, self._ctx():
            self._install(self._serving)
            self._local.depth = getattr(self._local, "depth", 0) + 1
            try:
                yield self._model
            finally:
                self._local.depth -= 1

    def _state(self, tensors, revision):
        return ModelState(revision, self._fingerprint(tensors), tensors, self._owner)

    def _check(self, state):
        if not isinstance(state, ModelState) or state.owner is not self._owner:
            raise ValidationError("snapshot belongs to another backend")
        for k in self._editable:   # states are immutable by convention; verify the bytes
            self._hashes.pop(id(state.payload[k]), None)
        if self._fingerprint(state.payload) != state.fingerprint:
            raise ValidationError("snapshot is invalid or was mutated after capture")

    def capture(self):
        with self._lock:
            return self._state(dict(self._serving), self._revision)

    def candidate(self, state: ModelState, recipe: EditRecipe):
        torch = self._torch
        self._check(state)
        if not isinstance(recipe, EditRecipe):
            raise ValidationError("proposer must return EditRecipe or None")
        unknown = {e.parameter for e in recipe.edits} - set(self._editable)
        if unknown:
            raise ValidationError(f"parameters are not editable: {sorted(unknown)}")
        new = dict(state.payload)
        with torch.no_grad():
            for e in recipe.edits:
                old = state.payload[e.parameter]
                if isinstance(e, DeltaEdit):
                    if tuple(e.delta.shape) != tuple(old.shape):
                        raise ValidationError(f"delta shape mismatch for {e.parameter}")
                    proposed = old + e.delta.detach().to(old.device, torch.float32)
                elif e.operation == "scale":
                    proposed = old * e.value
                else:
                    proposed = old + e.value
                if not torch.isfinite(proposed).all().item():
                    raise ValidationError(f"edit produced nonfinite values: {e.parameter}")
                if (proposed - old).abs().max().item() > self._max_abs_delta * (1 + 1e-6):
                    raise ValidationError(f"edit exceeds max_abs_delta for {e.parameter}")
                if (proposed - self._origin[e.parameter]).abs().max().item() > self._max_total_delta * (1 + 1e-6):
                    raise ValidationError(f"edit exceeds max_total_delta from original weights for {e.parameter}")
                new[e.parameter] = proposed.contiguous()
        return self._state(new, state.revision)

    def evaluate(self, state: ModelState, evaluator: Evaluator):
        self._check(state)
        with self._lock:
            before = {k: self._hash_tensor(state.payload[k]) for k in self._editable}
            self._install(state.payload)
            try:
                with self._ctx():
                    result = metrics(evaluator(self._model))
            finally:
                self._install(self._serving)
            for k in self._editable:
                self._hashes.pop(id(state.payload[k]), None)
                if self._hash_tensor(state.payload[k]) != before[k]:
                    raise ValidationError("evaluator mutated model weights; evaluation must be read-only")
            return result

    def promote(self, state: ModelState, *, expected_revision: int, deadline: float | None = None):
        if getattr(self._local, "depth", 0):
            raise RuntimeError("cannot promote inside inference(); call improvement between agent tasks")
        self._check(state)
        acquired = (self._lock.acquire() if deadline is None else
                    self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())))
        if not acquired:
            raise PromotionDeadlineError("deadline reached waiting for an inference safe point")
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise PromotionDeadlineError("deadline reached before promotion; active weights unchanged")
            if self._revision != expected_revision:
                raise StaleRevisionError("serving revision changed; candidate was not promoted")
            self._serving = dict(state.payload)
            self._install(self._serving)
            self._revision += 1
            return self._revision
        finally:
            self._lock.release()

    def export(self, state: ModelState, directory: Path):
        """Write only the editable tensors (fp32). Load with model.load_state_dict(..., strict=False)."""
        self._check(state)
        path = directory / "edited_tensors.pt"
        with path.open("xb") as stream:
            self._torch.save({k: state.payload[k].detach().cpu().clone() for k in self._editable}, stream)
        return path

    def gradients(self, loss_fn, names=None):
        """Gradients of loss_fn(model) w.r.t. editable tensors at the serving weights. Weights unchanged."""
        torch = self._torch
        names = tuple(names or self._editable)
        with self._lock:
            self._install(self._serving)
            for k in names:
                self._params[k].requires_grad_(True)
                self._params[k].grad = None
            try:
                with torch.enable_grad():
                    if self._compute_dtype is not None:
                        with torch.autocast("cpu", dtype=self._compute_dtype):
                            loss = loss_fn(self._model)
                    else:
                        loss = loss_fn(self._model)
                    loss.backward()
                return {k: self._params[k].grad.detach().clone() for k in names}, float(loss.detach())
            finally:
                for k in names:
                    self._params[k].grad = None
                    self._params[k].requires_grad_(False)
