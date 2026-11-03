"""Injected execution contracts; importing this module performs no I/O.

These interfaces do not supply a universal weight editor. Implementations are
trusted integration boundaries that must enforce their own native timeouts,
truthful resource receipts and model provenance. The engine never runs shell
commands, evaluates model-produced code or treats a proposal as an outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar
import math


T = TypeVar("T")


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")


@dataclass(frozen=True)
class ResourceQuote:
    cost_usd: float
    wall_seconds: float
    model_calls: int

    def __post_init__(self):
        _nonnegative(self.cost_usd, "quoted cost")
        _nonnegative(self.wall_seconds, "quoted wall time")
        if type(self.model_calls) is not int or self.model_calls < 0:
            raise ValueError("quoted model calls must be a nonnegative integer")


@dataclass(frozen=True)
class Usage:
    cost_usd: float | None
    wall_seconds: float | None
    model_calls: int | None

    def __post_init__(self):
        if self.cost_usd is not None:
            _nonnegative(self.cost_usd, "actual cost")
        if self.wall_seconds is not None:
            _nonnegative(self.wall_seconds, "actual wall time")
        if self.model_calls is not None and (type(self.model_calls) is not int or self.model_calls < 0):
            raise ValueError("actual model calls must be a nonnegative integer or unknown")


@dataclass(frozen=True)
class Operation(Generic[T]):
    value: T
    usage: Usage


@dataclass(frozen=True)
class Proposal:
    recipe_id: str
    parameters: Mapping[str, Any]
    rationale: str = ""

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id or not isinstance(self.parameters, Mapping):
            raise ValueError("proposal requires a recipe identifier and parameter mapping")
        if not isinstance(self.rationale, str) or len(self.rationale) > 4000:
            raise ValueError("proposal rationale must be bounded text")
        # Reuse the contract's recursive immutable JSON copy, including nested
        # mappings and sequences. No callback, tensor or mutable live reference.
        from .contracts import RecipeSpec
        object.__setattr__(self, "parameters", RecipeSpec(self.recipe_id, self.parameters).parameters)


class AdapterOperationError(RuntimeError):
    """An execution failure with explicit disposition, never an implied retry.

    disposition_known=True says the operation has stopped; it is not a success
    claim. Unknown native disposition prevents further calls, including restore.
    """
    def __init__(self, message: str, *, disposition_known: bool = False, usage: Usage | None = None):
        super().__init__(message)
        self.disposition_known = disposition_known
        self.usage = usage


class EditorAdapter(Protocol):
    provider: str
    supported_models: frozenset[str]
    def quote(self, config) -> ResourceQuote: ...
    def propose(self, config, baseline, history: tuple) -> Operation[Proposal | None]: ...


class TargetAdapter(Protocol):
    """Snapshot/restore/retain target state; accept does not authorize publishing."""
    adapter_id: str
    supported_architectures: frozenset[str]
    def quote(self, operation: str, config) -> ResourceQuote: ...
    def snapshot(self, config) -> Operation[Any]: ...
    def restore(self, config, snapshot: Any) -> Operation[None]: ...
    def accept(self, config, snapshot: Any, proposal: Proposal) -> Operation[str]: ...


class RecipeAdapter(Protocol):
    recipe_id: str
    supported_architectures: frozenset[str]
    def validate_parameters(self, parameters: Mapping[str, Any]) -> None: ...
    def quote(self, config, proposal: Proposal) -> ResourceQuote: ...
    def apply(self, config, target: TargetAdapter, snapshot: Any, proposal: Proposal) -> Operation[None]: ...


class EvaluatorAdapter(Protocol):
    evaluator_id: str
    supported_architectures: frozenset[str]
    supported_domains: frozenset[str]
    feedback_allowed: bool
    def quote(self, config) -> ResourceQuote: ...
    def evaluate(self, config, target: TargetAdapter) -> Operation[Any]: ...


@dataclass(frozen=True)
class ResolvedAdapters:
    editor: EditorAdapter
    target: TargetAdapter
    recipe: RecipeAdapter
    evaluator: EvaluatorAdapter


class AdapterRegistry:
    """Explicit application-supplied implementations, never dynamic imports."""
    def __init__(self):
        self.editors, self.targets, self.recipes, self.evaluators = {}, {}, {}, {}

    @staticmethod
    def _add(table, key, adapter):
        if not isinstance(key, str) or not key or key in table:
            raise ValueError("adapter identifier missing or already registered")
        table[key] = adapter

    def register_editor(self, adapter: EditorAdapter):
        self._add(self.editors, adapter.provider, adapter)

    def register_target(self, adapter: TargetAdapter):
        self._add(self.targets, adapter.adapter_id, adapter)

    def register_recipe(self, adapter: RecipeAdapter):
        self._add(self.recipes, adapter.recipe_id, adapter)

    def register_evaluator(self, adapter: EvaluatorAdapter):
        self._add(self.evaluators, adapter.evaluator_id, adapter)

    def resolve(self, config) -> ResolvedAdapters:
        try:
            result = ResolvedAdapters(self.editors[config.editor.provider], self.targets[config.target.adapter_id],
                                      self.recipes[config.recipe.recipe_id], self.evaluators[config.evaluator.evaluator_id])
        except KeyError as exc:
            raise ValueError("unsupported adapter identifier: " + str(exc)) from exc
        if config.editor.model not in result.editor.supported_models:
            raise ValueError("editor model is unavailable through this adapter")
        architecture = config.target.architecture
        if any(architecture not in adapter.supported_architectures for adapter in (result.target, result.recipe, result.evaluator)):
            raise ValueError("target architecture is incompatible with target, recipe or evaluator adapter")
        if config.goal.domain not in result.evaluator.supported_domains:
            raise ValueError("evaluator does not support the declared goal domain")
        if result.evaluator.feedback_allowed is not True:
            raise ValueError("sealed evaluation cannot feed an iterative editing loop")
        result.recipe.validate_parameters(config.recipe.parameters)
        return result


# Short integration alias; either name resolves the same explicit registry.
Registry = AdapterRegistry
