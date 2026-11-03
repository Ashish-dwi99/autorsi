"""Serializable, provider-free experiment contracts and metric decisions.

An editor proposes a recipe; a target adapter applies it to a pinned artifact.
These are independent model identities. Compatibility and provider availability
must be established by registered adapters, not inferred from a model name or
from this configuration. Goal text never supplies an evaluation result.

Resource usage belongs to adapters.Operation.usage, not EvaluationResult.
Missing usage is not zero; the engine must retain its outstanding reservation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import json
import math
import operator
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """The requested experiment has an invalid or ambiguous contract."""


ValidationError = ConfigurationError
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPARATORS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ConfigurationError(f"{name} must be a nonempty identifier (at most 256 characters)")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None, positive: bool = False) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ConfigurationError(f"{name} must be a finite number; booleans are not numbers")
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ConfigurationError(f"{name} is outside its permitted range")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _frozen_json(value: Any, name: str = "parameters", *, depth: int = 0) -> Any:
    if depth > 20:
        raise ConfigurationError(f"{name} exceeds the JSON nesting limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return _number(value, name)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError(f"{name} must have string keys")
        return MappingProxyType({key: _frozen_json(item, name, depth=depth + 1) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_json(item, name, depth=depth + 1) for item in value)
    raise ConfigurationError(f"{name} must contain JSON values only")


@dataclass(frozen=True)
class EditorSpec:
    """Provider/model to propose edits; credential_env names a variable only."""
    provider: str
    model: str
    endpoint: str | None = None
    credential_env: str | None = None

    def __post_init__(self):
        _identifier(self.provider, "editor.provider")
        _identifier(self.model, "editor.model")
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str):
                raise ConfigurationError("editor.endpoint must be an HTTP(S) URL or null")
            try:
                parsed = urlsplit(self.endpoint)
                parsed.port
            except ValueError as exc:
                raise ConfigurationError("editor.endpoint has an invalid host or port") from exc
            if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment or len(self.endpoint) > 2048
                    or any(character.isspace() for character in self.endpoint)):
                raise ConfigurationError("editor.endpoint must be HTTP(S) without credentials, query, or fragment")
        if self.credential_env is not None and (not isinstance(self.credential_env, str)
                or _ENVIRONMENT_NAME.fullmatch(self.credential_env) is None):
            raise ConfigurationError("editor.credential_env must name an environment variable; never store a secret")


@dataclass(frozen=True)
class TargetSpec:
    """Target artifact identity, independent of the model proposing changes."""
    adapter_id: str
    model_id: str
    architecture: str
    revision: str
    artifact_pins: Mapping[str, str]

    def __post_init__(self):
        for name in ("adapter_id", "model_id", "architecture", "revision"):
            _identifier(getattr(self, name), "target." + name)
        if not isinstance(self.artifact_pins, Mapping) or not self.artifact_pins:
            raise ConfigurationError("target.artifact_pins requires at least one exact SHA256")
        copied = {}
        for name, digest in self.artifact_pins.items():
            _identifier(name, "artifact pin name")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ConfigurationError("artifact pins must be lowercase 64-character SHA256 digests")
            copied[name] = digest
        object.__setattr__(self, "artifact_pins", MappingProxyType(copied))


ModelSpec = TargetSpec


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    parameters: Mapping[str, Any]

    def __post_init__(self):
        _identifier(self.recipe_id, "recipe.recipe_id")
        if not isinstance(self.parameters, Mapping):
            raise ConfigurationError("recipe.parameters must be an object")
        object.__setattr__(self, "parameters", _frozen_json(self.parameters))


@dataclass(frozen=True)
class EvaluatorSpec:
    evaluator_id: str
    suite_id: str
    suite_revision: str

    def __post_init__(self):
        for name in ("evaluator_id", "suite_id", "suite_revision"):
            _identifier(getattr(self, name), "evaluator." + name)


@dataclass(frozen=True)
class BudgetSpec:
    """Independent editor/target ceilings; either may be zero for offline work.

    This contract is not a provider spending cap. The engine reserves before
    dispatch and retains a quote when actual usage is unreported or uncertain.
    """
    max_candidates: int
    max_evaluations: int
    max_editor_cost_usd: float
    max_target_cost_usd: float
    max_wall_seconds: float
    max_model_calls: int

    def __post_init__(self):
        _integer(self.max_candidates, "budget.max_candidates", minimum=1)
        _integer(self.max_evaluations, "budget.max_evaluations", minimum=1)
        _integer(self.max_model_calls, "budget.max_model_calls")
        _number(self.max_editor_cost_usd, "budget.max_editor_cost_usd", minimum=0)
        _number(self.max_target_cost_usd, "budget.max_target_cost_usd", minimum=0)
        _number(self.max_wall_seconds, "budget.max_wall_seconds", positive=True)
        _number(self.max_cost_usd, "combined monetary budget", minimum=0)

    @property
    def max_cost_usd(self) -> float:
        """Reporting sum; unused funds never silently transfer between pools."""
        return self.max_editor_cost_usd + self.max_target_cost_usd


@dataclass(frozen=True)
class MetricThreshold:
    """Compare actual metric or candidate/reference ratio with value.

    Example: latency <=0.1 relative_to='baseline' asks for 10x lower latency.
    A ratio requires a strictly positive measured reference and a nonnegative
    candidate. Missing measurements or a zero denominator never imply a pass.
    """
    value: float
    comparator: str
    relative_to: str | None = None

    def __post_init__(self):
        _number(self.value, "metric target")
        if not isinstance(self.comparator, str) or self.comparator not in _COMPARATORS:
            raise ConfigurationError("comparator must be >=, <=, >, or <")
        if self.relative_to is not None:
            _identifier(self.relative_to, "metric reference")
            if self.value < 0:
                raise ConfigurationError("relative-ratio target must be nonnegative")


@dataclass(frozen=True)
class RegressionGuard:
    """Allow at most tolerance directional degradation from a named reference.

    absolute uses metric units; relative uses abs(reference)*tolerance. For
    maximize, the reference minus allowance is a floor; for minimize, reference
    plus allowance is a ceiling. A zero reference therefore permits no relative
    degradation. This rule does not divide by a reference.
    """
    reference: str = "baseline"
    tolerance: float = 0.0
    mode: str = "absolute"

    def __post_init__(self):
        _identifier(self.reference, "regression reference")
        _number(self.tolerance, "regression tolerance", minimum=0)
        if not isinstance(self.mode, str) or self.mode not in {"absolute", "relative"}:
            raise ConfigurationError("regression mode must be absolute or relative")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    unit: str
    target: MetricThreshold | None = None
    regression: RegressionGuard | None = None

    def __post_init__(self):
        _identifier(self.name, "metric.name")
        if not isinstance(self.direction, str) or self.direction not in {"maximize", "minimize"}:
            raise ConfigurationError("metric.direction must be maximize or minimize")
        if not isinstance(self.unit, str) or not 0 < len(self.unit) <= 64:
            raise ConfigurationError("metric.unit must be a nonempty unit of at most 64 characters")
        if self.target is not None and not isinstance(self.target, MetricThreshold):
            raise ConfigurationError("metric.target must be MetricThreshold or null")
        if self.regression is not None and not isinstance(self.regression, RegressionGuard):
            raise ConfigurationError("metric.regression must be RegressionGuard or null")
        if self.target is None and self.regression is None:
            raise ConfigurationError("each metric requires a target or regression guard")
        allowed = {">=", ">"} if self.direction == "maximize" else {"<=", "<"}
        if self.target is not None and self.target.comparator not in allowed:
            raise ConfigurationError("target comparator contradicts the metric direction")


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    domain: str
    capability: str
    metrics: tuple[MetricSpec, ...]

    def __post_init__(self):
        for name in ("goal_id", "domain", "capability"):
            _identifier(getattr(self, name), "goal." + name)
        if not isinstance(self.metrics, (tuple, list)) or not self.metrics:
            raise ConfigurationError("goal.metrics must be a nonempty list of metric contracts")
        if any(not isinstance(metric, MetricSpec) for metric in self.metrics):
            raise ConfigurationError("goal.metrics contains a non-metric contract")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ConfigurationError("goal metric names must be unique")
        object.__setattr__(self, "metrics", tuple(self.metrics))


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    goal: GoalSpec
    editor: EditorSpec
    target: TargetSpec
    recipe: RecipeSpec
    evaluator: EvaluatorSpec
    budget: BudgetSpec

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigurationError("unsupported configuration schema_version; expected 1")
        expected = {"goal": GoalSpec, "editor": EditorSpec, "target": TargetSpec,
                    "recipe": RecipeSpec, "evaluator": EvaluatorSpec, "budget": BudgetSpec}
        for name, cls in expected.items():
            if not isinstance(getattr(self, name), cls):
                raise ConfigurationError(f"{name} must be a {cls.__name__} contract")


@dataclass(frozen=True)
class EvaluationResult:
    """Measured evaluator output; editor assertions cannot instantiate trust."""
    evaluator_id: str
    suite_id: str
    suite_revision: str
    metrics: Mapping[str, float]
    complete: bool = True
    artifact_id: str = ""

    def __post_init__(self):
        for name in ("evaluator_id", "suite_id", "suite_revision"):
            _identifier(getattr(self, name), "result." + name)
        if not isinstance(self.metrics, Mapping):
            raise ConfigurationError("result.metrics must be a mapping")
        copied = {}
        for name, value in self.metrics.items():
            _identifier(name, "result metric name")
            copied[name] = _number(value, "measured metric")
        object.__setattr__(self, "metrics", MappingProxyType(copied))
        if type(self.complete) is not bool:
            raise ConfigurationError("result.complete must be a boolean")
        if not isinstance(self.artifact_id, str):
            raise ConfigurationError("result.artifact_id must be a string")


@dataclass(frozen=True)
class MetricCheck:
    name: str
    observed: float | None
    passed: bool
    reason: str
    target_value: float | None = None
    reference_name: str | None = None
    reference_value: float | None = None


@dataclass(frozen=True)
class GoalAssessment:
    passed: bool
    checks: tuple[MetricCheck, ...]
    failures: tuple[str, ...]


def _reference(result: EvaluationResult, references: Mapping[str, EvaluationResult], name: str, metric: str) -> float:
    reference = references.get(name)
    if not isinstance(reference, EvaluationResult) or not reference.complete:
        raise ConfigurationError("missing or incomplete reference: " + name)
    if (reference.evaluator_id, reference.suite_id, reference.suite_revision) != (
            result.evaluator_id, result.suite_id, result.suite_revision):
        raise ConfigurationError("reference uses a different evaluator or suite revision: " + name)
    if metric not in reference.metrics:
        raise ConfigurationError("reference omitted metric: " + metric)
    return reference.metrics[metric]


def assess_metrics(goal: GoalSpec, result: EvaluationResult,
                   references: Mapping[str, EvaluationResult] | None = None) -> GoalAssessment:
    """All configured metric targets and retention guards must pass together."""
    if not isinstance(goal, GoalSpec) or not isinstance(result, EvaluationResult):
        raise ConfigurationError("typed goal and evaluator result required")
    references = {} if references is None else references
    if not isinstance(references, Mapping):
        raise ConfigurationError("references must be a mapping of actual evaluator results")
    failures = []
    checks = []
    expected = {metric.name for metric in goal.metrics}
    if not result.complete:
        failures.append("evaluation is incomplete")
    missing, extra = expected - set(result.metrics), set(result.metrics) - expected
    if missing:
        failures.append("missing metrics: " + ", ".join(sorted(missing)))
    if extra:
        failures.append("unexpected metrics: " + ", ".join(sorted(extra)))
    if failures:
        return GoalAssessment(False, (), tuple(failures))
    for spec in goal.metrics:
        observed = result.metrics[spec.name]
        if spec.target is not None:
            target = spec.target
            ref_value = None
            value = observed
            try:
                if target.relative_to is not None:
                    ref_value = _reference(result, references, target.relative_to, spec.name)
                    if ref_value <= 0 or observed < 0:
                        raise ConfigurationError("ratio requires positive reference and nonnegative candidate")
                    value = observed / ref_value
                    _number(value, "computed metric ratio")
                passed = _COMPARATORS[target.comparator](value, target.value)
                reason = ("target passed" if passed else "target failed")
                if target.relative_to is not None:
                    reason += " using candidate/reference ratio"
            except (ConfigurationError, OverflowError, ZeroDivisionError) as exc:
                passed, reason = False, str(exc)
            checks.append(MetricCheck(spec.name, observed, passed, reason, target.value, target.relative_to, ref_value))
            if not passed:
                failures.append(spec.name + ": " + reason)
        if spec.regression is not None:
            guard = spec.regression
            ref_value = None
            bound = None
            try:
                ref_value = _reference(result, references, guard.reference, spec.name)
                allowance = guard.tolerance if guard.mode == "absolute" else abs(ref_value) * guard.tolerance
                _number(allowance, "computed regression allowance")
                bound = ref_value - allowance if spec.direction == "maximize" else ref_value + allowance
                _number(bound, "computed regression bound")
                passed = observed >= bound if spec.direction == "maximize" else observed <= bound
                reason = "regression guard passed" if passed else "regression guard failed"
            except (ConfigurationError, OverflowError) as exc:
                passed, reason = False, str(exc)
            checks.append(MetricCheck(spec.name, observed, passed, reason, bound, guard.reference, ref_value))
            if not passed:
                failures.append(spec.name + ": " + reason)
    return GoalAssessment(not failures, tuple(checks), tuple(failures))


def _object(data: Any, cls: type, where: str) -> dict:
    if not isinstance(data, Mapping):
        raise ConfigurationError(where + " must be an object")
    allowed = {field.name for field in fields(cls)}
    extra = set(data) - allowed
    if extra:
        raise ConfigurationError(where + " has unsupported fields: " + ", ".join(sorted(str(key) for key in extra)))
    return dict(data)


def _construct(cls: type, data: Any, where: str):
    try:
        return cls(**_object(data, cls, where))
    except TypeError as exc:
        raise ConfigurationError(where + " has missing or invalid fields") from exc


def config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    top = _object(data, ExperimentConfig, "config")
    try:
        goal = _object(top["goal"], GoalSpec, "goal")
        if not isinstance(goal.get("metrics"), (list, tuple)):
            raise ConfigurationError("goal.metrics must be a list")
        metrics = []
        for entry in goal["metrics"]:
            metric = _object(entry, MetricSpec, "metric")
            if metric.get("target") is not None:
                metric["target"] = _construct(MetricThreshold, metric["target"], "metric.target")
            if metric.get("regression") is not None:
                metric["regression"] = _construct(RegressionGuard, metric["regression"], "metric.regression")
            metrics.append(_construct(MetricSpec, metric, "metric"))
        goal["metrics"] = tuple(metrics)
        top["goal"] = _construct(GoalSpec, goal, "goal")
        for name, cls in (("editor", EditorSpec), ("target", TargetSpec), ("recipe", RecipeSpec),
                          ("evaluator", EvaluatorSpec), ("budget", BudgetSpec)):
            top[name] = _construct(cls, top[name], name)
        return _construct(ExperimentConfig, top, "config")
    except KeyError as exc:
        raise ConfigurationError("configuration is missing a required section") from exc


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ConfigurationError("only contract/JSON values are serializable")


def config_to_dict(config: ExperimentConfig) -> dict:
    if not isinstance(config, ExperimentConfig):
        raise ConfigurationError("ExperimentConfig required")
    return to_dict(config)


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Validate a fresh serializable copy; no adapter/provider/model is touched."""
    return config_from_dict(config_to_dict(config))


def loads_config(text: str | bytes) -> ExperimentConfig:
    def unique(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ConfigurationError("duplicate JSON field: " + key)
            output[key] = value
        return output

    def nonfinite(_):
        raise ConfigurationError("nonfinite JSON number")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    except ConfigurationError:
        raise
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ConfigurationError("configuration is not valid JSON") from exc
    return config_from_dict(value)


def dumps_config(config: ExperimentConfig) -> str:
    return json.dumps(config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n"
