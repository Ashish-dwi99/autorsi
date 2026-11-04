"""Transactional in-memory backends and a dependency-free reference backend."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time
from types import MappingProxyType

from .contracts import EditRecipe, Evaluator, ModelState, ValidationError, finite, metrics, name


class StaleRevisionError(RuntimeError):
    """Another writer promoted weights while a candidate was being evaluated."""


class PromotionDeadlineError(RuntimeError):
    """The safe point could not be acquired before the admission deadline."""


class InMemoryBackend:
    """Subclass hooks implement cloning, editing, hashing and serialization.

    The backend owns its serving model; the caller's original is never changed.
    All inference must go through inference(), and must be read-only. Do not
    retain or use the yielded object outside that context. Native asynchronous
    inference must finish before exiting. Custom callbacks are trusted code.
    """
    def __init__(self, model, *, editable, max_abs_delta: float = 0.1, max_total_delta: float | None = None):
        if isinstance(editable, str):
            raise ValidationError("editable must be a collection of exact parameter names")
        self._editable = frozenset(editable)
        if not self._editable:
            raise ValidationError("editable must explicitly allowlist at least one parameter")
        for key in self._editable:
            name(key, "editable parameter")
        self._max_abs_delta = finite(max_abs_delta, "max_abs_delta", minimum=0)
        if self._max_abs_delta == 0:
            raise ValidationError("max_abs_delta must be positive")
        self._max_total_delta = finite(max_abs_delta if max_total_delta is None else max_total_delta,
                                       "max_total_delta", minimum=0)
        if self._max_total_delta == 0:
            raise ValidationError("max_total_delta must be positive")
        self._lock = threading.RLock()
        self._local = threading.local()
        self._revision = 0
        self._owner = object()
        self._validate_model(model)
        self._model = self._clone(model)
        self._origin = self._clone(self._model)

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def recipe_space(self):
        return MappingProxyType({"parameters": tuple(sorted(self._editable)),
                                 "operations": ("scale", "add"),
                                 "max_abs_delta": self._max_abs_delta,
                                 "max_total_delta": self._max_total_delta,
                                 "bound_reference": "per-step delta from current; total delta from original"})

    def _clone(self, model):
        return deepcopy(model)

    def _inference_context(self):
        return nullcontext()

    @contextmanager
    def inference(self):
        """Hold a serving safe point for the complete synchronous inference."""
        with self._lock, self._inference_context():
            self._local.depth = getattr(self._local, "depth", 0) + 1
            try:
                yield self._model
            finally:
                self._local.depth -= 1

    def _state(self, model, revision):
        return ModelState(revision, self._fingerprint(model), model, self._owner)

    def _check(self, state):
        if not isinstance(state, ModelState) or state.owner is not self._owner:
            raise ValidationError("snapshot belongs to another backend; capture this backend's model first")
        if self._fingerprint(state.payload) != state.fingerprint:
            raise ValidationError("snapshot is invalid or was mutated after capture")

    def capture(self):
        with self._lock:
            return self._state(self._clone(self._model), self._revision)

    def candidate(self, state: ModelState, recipe: EditRecipe):
        self._check(state)
        if not isinstance(recipe, EditRecipe):
            raise ValidationError("proposer must return EditRecipe or None")
        unknown = {edit.parameter for edit in recipe.edits} - self._editable
        if unknown:
            raise ValidationError(f"parameters are not editable: {sorted(unknown)}")
        model = self._clone(state.payload)
        self._apply(model, recipe)
        self._validate_model(model)
        self._validate_total_delta(model)
        return self._state(model, state.revision)

    def evaluate(self, state: ModelState, evaluator: Evaluator):
        self._check(state)
        model = self._clone(state.payload)
        before = self._fingerprint(model)
        with self._inference_context():
            result = metrics(evaluator(model))
        if self._fingerprint(model) != before:
            raise ValidationError("evaluator mutated model weights or persistent buffers; evaluation must be read-only")
        return result

    def promote(self, state: ModelState, *, expected_revision: int, deadline: float | None = None):
        if getattr(self._local, "depth", 0):
            raise RuntimeError("cannot promote inside inference(); call improvement between agent tasks")
        self._check(state)
        self._validate_model(state.payload)
        self._validate_total_delta(state.payload)
        replacement = self._clone(state.payload)
        acquired = (self._lock.acquire() if deadline is None else
                    self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())))
        if not acquired:
            raise PromotionDeadlineError("deadline reached waiting for an inference safe point")
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise PromotionDeadlineError("deadline reached before promotion; active weights unchanged")
            if self._revision != expected_revision:
                raise StaleRevisionError("serving revision changed; candidate was not promoted")
            self._model = replacement
            self._revision += 1
            return self._revision
        finally:
            self._lock.release()

    def _validate_model(self, model):
        raise NotImplementedError

    def _fingerprint(self, model):
        raise NotImplementedError

    def _apply(self, model, recipe):
        raise NotImplementedError

    def _validate_total_delta(self, model):
        raise NotImplementedError

    def export(self, state: ModelState, directory: Path):
        raise NotImplementedError


class ScalarBackend(InMemoryBackend):
    """Reference backend for models represented by named finite scalar weights.

    Useful for dependency-free examples and testing custom evaluators. It is not
    an LLM backend. An evaluator can run real linear arithmetic on this mapping.
    """
    def _validate_model(self, model):
        if not isinstance(model, Mapping) or not model:
            raise ValidationError("ScalarBackend expects a nonempty mapping of named scalar weights")
        for key, value in model.items():
            name(key, "weight name")
            finite(value, f"weight {key}")
        if not self._editable <= set(model):
            raise ValidationError("editable contains names absent from the model")

    def _clone(self, model):
        return dict(model)

    def _fingerprint(self, model):
        return hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _apply(self, model, recipe):
        for edit in recipe.edits:
            old = model[edit.parameter]
            new = old * edit.value if edit.operation == "scale" else old + edit.value
            finite(new, f"edited weight {edit.parameter}")
            if abs(new - old) > self._max_abs_delta:
                raise ValidationError(f"edit exceeds max_abs_delta for {edit.parameter}")
            model[edit.parameter] = new

    def export(self, state: ModelState, directory: Path):
        self._check(state)
        path = directory / "weights.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state.payload, stream, sort_keys=True, allow_nan=False)
        return path

    def _validate_total_delta(self, model):
        for key in self._editable:
            if abs(model[key] - self._origin[key]) > self._max_total_delta:
                raise ValidationError(f"edit exceeds max_total_delta from original weights for {key}")
