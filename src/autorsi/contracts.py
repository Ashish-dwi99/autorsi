"""Small, provider-independent contracts. Importing AutoRSI performs no I/O."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, ContextManager, Protocol


class ValidationError(ValueError):
    """A bounded, user-actionable framework input or evaluation error."""


def finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f"{name} must be a finite number, not a boolean")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{name} must be >= {minimum}")
    return float(value)


def name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValidationError(f"{label} must be nonempty text of at most 256 characters")
    return value


def metrics(value: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValidationError("evaluator must return a nonempty mapping of metric names to finite numbers")
    return MappingProxyType({name(k, "metric name"): finite(v, f"metric {k}") for k, v in value.items()})


@dataclass(frozen=True)
class TensorEdit:
    """One bounded scalar operation on an explicitly allowlisted tensor."""
    parameter: str
    operation: str
    value: float

    def __post_init__(self):
        name(self.parameter, "parameter")
        if self.operation not in {"scale", "add"}:
            raise ValidationError("operation must be 'scale' or 'add'; generated code is never accepted")
        finite(self.value, "edit value")


@dataclass(frozen=True)
class EditRecipe:
    edits: tuple[TensorEdit, ...]
    rationale: str = ""

    def __post_init__(self):
        object.__setattr__(self, "edits", tuple(self.edits))
        if not 1 <= len(self.edits) <= 128 or any(not isinstance(e, TensorEdit) for e in self.edits):
            raise ValidationError("recipe requires 1 to 128 TensorEdit objects")
        if len({e.parameter for e in self.edits}) != len(self.edits):
            raise ValidationError("recipe may edit each parameter only once")
        if not isinstance(self.rationale, str) or len(self.rationale) > 4000:
            raise ValidationError("rationale must be text of at most 4000 characters")

    def to_dict(self):
        return {"edits": [{"parameter": e.parameter, "operation": e.operation, "value": e.value}
                          for e in self.edits], "rationale": self.rationale}


@dataclass(frozen=True)
class Goal:
    metric: str
    direction: str = "maximize"
    target: float | None = None
    min_improvement: float = 0.0

    def __post_init__(self):
        name(self.metric, "goal metric")
        if self.direction not in {"maximize", "minimize"}:
            raise ValidationError("goal direction must be 'maximize' or 'minimize'")
        if self.target is not None:
            finite(self.target, "goal target")
        finite(self.min_improvement, "min_improvement", minimum=0)

    def gain(self, current: float, candidate: float) -> float:
        return (candidate - current) * (1 if self.direction == "maximize" else -1)

    def reached(self, scores: Mapping[str, float]) -> bool:
        if self.target is None:
            return False
        return self.gain(self.target, scores[self.metric]) >= 0


@dataclass(frozen=True)
class Guard:
    """Maximum absolute loss against the controller's original baseline."""
    metric: str
    direction: str = "maximize"
    max_regression: float = 0.0

    def __post_init__(self):
        name(self.metric, "guard metric")
        if self.direction not in {"maximize", "minimize"}:
            raise ValidationError("guard direction must be 'maximize' or 'minimize'")
        finite(self.max_regression, "max_regression", minimum=0)


@dataclass(frozen=True)
class RunLimits:
    """Limits checked between synchronous calls, not native call cancellation."""
    max_steps: int = 1
    max_evaluations: int = 20
    max_seconds: float = 60.0
    max_accepted: int = 1

    def __post_init__(self):
        for key in ("max_steps", "max_evaluations", "max_accepted"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValidationError(f"{key} must be a positive integer")
        if finite(self.max_seconds, "max_seconds", minimum=0) == 0:
            raise ValidationError("max_seconds must be positive")


@dataclass(frozen=True)
class ProposalContext:
    goal: Goal
    guards: tuple[Guard, ...]
    baseline: Mapping[str, float]
    current: Mapping[str, float]
    history: tuple[Mapping[str, Any], ...]
    recipe_space: Mapping[str, Any]


@dataclass(frozen=True)
class ModelState:
    """Opaque backend-owned snapshot. Never sent to a proposer."""
    revision: int
    fingerprint: str
    payload: Any = field(repr=False, compare=False)
    owner: Any = field(default=None, repr=False, compare=False)


Evaluator = Callable[[Any], Mapping[str, float]]


class Proposer(Protocol):
    def propose(self, context: ProposalContext) -> EditRecipe | None: ...


class Backend(Protocol):
    """Trusted integration contract; adapters must isolate all candidate work.

    promote must reject stale revisions and publish only at an inference safe
    point. Failures must leave the serving model unchanged. evaluate must not
    permit callbacks to mutate a snapshot. export writes a reloadable artifact.
    """
    @property
    def revision(self) -> int: ...
    @property
    def recipe_space(self) -> Mapping[str, Any]: ...
    def capture(self) -> ModelState: ...
    def candidate(self, state: ModelState, recipe: EditRecipe) -> ModelState: ...
    def evaluate(self, state: ModelState, evaluator: Evaluator) -> Mapping[str, float]: ...
    def promote(self, state: ModelState, *, expected_revision: int, deadline: float | None = None) -> int: ...
    def export(self, state: ModelState, directory: Path) -> Path: ...
    def inference(self) -> ContextManager[Any]: ...


@dataclass(frozen=True)
class RunResult:
    status: str
    accepted: int
    revision: int
    baseline: Mapping[str, float]
    metrics: Mapping[str, float]
    attempts: tuple[Mapping[str, Any], ...]
    evaluations: int
    checkpoint: str | None = None
    journal: str | None = None
    error: str | None = None
    serving_revision: int | None = None

    def to_dict(self):
        return {"status": self.status, "accepted": self.accepted, "revision": self.revision,
                "baseline": dict(self.baseline), "metrics": dict(self.metrics),
                "attempts": list(self.attempts), "evaluations": self.evaluations,
                "checkpoint": self.checkpoint, "journal": self.journal, "error": self.error,
                "serving_revision": self.serving_revision}
