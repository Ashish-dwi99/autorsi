"""Bounded generic orchestration over explicitly registered execution adapters.

No provider, browser, model, filesystem or shell operation is built in. Native
timeouts and truthful resource ceilings belong to the injected adapters; this
synchronous engine cannot cancel a hidden native thread. Unknown disposition
stops the experiment without issuing a potentially overlapping restore.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
import time

from .adapters import AdapterOperationError, AdapterRegistry, Operation, Proposal, ResourceQuote, Usage
from .contracts import EvaluationResult, assess_metrics, validate_config


def _plain(value):
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class RunResult:
    status: str
    succeeded: bool
    target_state: str
    baseline: EvaluationResult | None
    attempts: tuple[dict, ...]
    resources: dict
    artifact_id: str | None = None
    error: str | None = None

    def to_dict(self):
        return _plain(self)


class BudgetExhausted(RuntimeError):
    disposition_known = True


class ResourceOverrun(RuntimeError):
    disposition_known = True


class _Ledger:
    def __init__(self, budget, clock):
        self.budget, self.clock, self.started = budget, clock, clock()
        self.cost = {"editor": 0.0, "target": 0.0}
        self.wall, self.calls, self.events = 0.0, 0, []
        self.evaluations = 0

    def admit(self, requests):
        costs = dict(self.cost)
        wall, calls = max(self.wall, self.clock() - self.started), self.calls
        for bucket, quote in requests:
            if bucket not in costs or not isinstance(quote, ResourceQuote):
                raise ValueError("typed resource quote and editor/target bucket required")
            costs[bucket] += quote.cost_usd
            wall += quote.wall_seconds
            calls += quote.model_calls
        b = self.budget
        if (costs["editor"] > b.max_editor_cost_usd or costs["target"] > b.max_target_cost_usd
                or wall > b.max_wall_seconds
                or calls > b.max_model_calls):
            raise BudgetExhausted("prospective operation and rollback reserves exceed the remaining budget")

    def charge(self, bucket, label, quote, usage, measured):
        if not isinstance(usage, Usage):
            usage = Usage(None, None, None)
        charged_cost = quote.cost_usd if usage.cost_usd is None else usage.cost_usd
        charged_wall = max(measured, quote.wall_seconds if usage.wall_seconds is None else usage.wall_seconds)
        charged_calls = quote.model_calls if usage.model_calls is None else usage.model_calls
        self.cost[bucket] += charged_cost
        self.wall += charged_wall
        self.calls += charged_calls
        self.events.append({"operation": label, "bucket": bucket, "quote": _plain(quote), "usage": _plain(usage),
                            "charged_cost_usd": charged_cost, "charged_wall_seconds": charged_wall,
                            "charged_model_calls": charged_calls, "measured_call_wall_seconds": measured,
                            "unknown_usage_retains_reservation": any(value is None for value in
                                (usage.cost_usd, usage.wall_seconds, usage.model_calls))})
        return (charged_cost > quote.cost_usd + 1e-12 or charged_wall > quote.wall_seconds + 1e-9
                or charged_calls > quote.model_calls)

    def invoke(self, bucket, label, quote, callback, *, reserve=()):
        self.admit([(bucket, quote), *reserve])
        started = self.clock()
        try:
            operation = callback()
        except BaseException as exc:
            self.charge(bucket, label, quote, getattr(exc, "usage", None), max(0.0, self.clock() - started))
            if isinstance(exc, AdapterOperationError):
                raise
            raise AdapterOperationError(f"{label}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(operation, Operation) or not isinstance(operation.usage, Usage):
            self.charge(bucket, label, quote, None, max(0.0, self.clock() - started))
            raise AdapterOperationError("adapter did not return a typed operation receipt")
        overrun = self.charge(bucket, label, quote, operation.usage, max(0.0, self.clock() - started))
        if overrun:
            raise ResourceOverrun(label + " exceeded its prospective resource quote")
        return operation.value

    def result(self):
        return {"editor_cost_charged_or_reserved_usd": self.cost["editor"],
                "target_cost_charged_or_reserved_usd": self.cost["target"],
                "wall_charged_or_reserved_seconds": self.wall,
                "observed_engine_wall_seconds": max(0.0, self.clock() - self.started),
                "model_calls_charged_or_reserved": self.calls, "evaluations": self.evaluations,
                "events": self.events,
                "limits_are_adapter_enforced": "Engine admission is not physical cancellation or a billing receipt"}


def _evaluation(config, result):
    if not isinstance(result, EvaluationResult):
        raise ValueError("only the evaluator's typed result can be assessed")
    spec = config.evaluator
    if (result.evaluator_id, result.suite_id, result.suite_revision) != (spec.evaluator_id, spec.suite_id, spec.suite_revision):
        raise ValueError("evaluation suite identity differs from the frozen goal")
    if set(result.metrics) != {metric.name for metric in config.goal.metrics}:
        raise ValueError("evaluation metric names differ from the frozen goal")
    return result


def run_experiment(config, registry: AdapterRegistry, *, references=None, clock=time.monotonic) -> RunResult:
    """Search a feedback-allowed suite; accept only actual evaluator thresholds.

    Baseline counts as an evaluation. Every candidate starts from the same saved
    target; an unmet target or regression restores it before another proposal.
    Snapshot handles remain adapter-private and are not sent to the editor.
    Sealed/heldout evaluators require a separate non-search workflow.
    """
    validate_config(config)
    adapters = registry.resolve(config)
    references = dict(references or {})
    if "baseline" in references:
        raise ValueError("baseline reference is measured here, never caller-supplied")
    ledger = _Ledger(config.budget, clock)
    attempts, baseline = [], None
    target_state = "unchanged"

    def finish(status, succeeded=False, artifact_id=None, error=None):
        return RunResult(status, succeeded, target_state, baseline, tuple(attempts), ledger.result(), artifact_id, error)

    def evaluate(reserve=()):
        if ledger.evaluations >= config.budget.max_evaluations:
            raise BudgetExhausted("evaluation cap reached")
        quote = adapters.evaluator.quote(config)
        # An admitted evaluation attempt consumes the slot even on failure.
        ledger.admit([("target", quote), *reserve])
        ledger.evaluations += 1
        result = ledger.invoke("target", "evaluate", quote,
            lambda: adapters.evaluator.evaluate(config, adapters.target), reserve=reserve)
        return _evaluation(config, result)

    try:
        baseline = evaluate()
        if not baseline.complete:
            return finish("baseline_incomplete", error="No complete baseline; no edit was attempted")
        references["baseline"] = baseline
        if assess_metrics(config.goal, baseline, references).passed:
            return finish("baseline_meets_goal", True, baseline.artifact_id or None)
    except BaseException as exc:
        return finish("baseline_failed", error=f"{type(exc).__name__}: {exc}")

    for index in range(config.budget.max_candidates):
        snapshot, dirty, restore_entered = None, False, False
        attempt = {"candidate": index + 1, "state": "proposal", "proposal": None}
        attempts.append(attempt)
        try:
            if ledger.evaluations >= config.budget.max_evaluations:
                raise BudgetExhausted("no candidate evaluation remains")
            proposal = ledger.invoke("editor", "propose", adapters.editor.quote(config),
                lambda: adapters.editor.propose(config, baseline, tuple(_plain(a) for a in attempts[:-1])))
            if proposal is None:
                attempt["state"] = "editor_stopped"
                return finish("editor_stopped")
            if not isinstance(proposal, Proposal) or proposal.recipe_id != config.recipe.recipe_id:
                raise ValueError("editor proposed an unsupported or different recipe")
            adapters.recipe.validate_parameters(proposal.parameters)
            attempt["proposal"] = _plain(proposal)
            snapshot_quote = adapters.target.quote("snapshot", config)
            restore_quote = adapters.target.quote("restore", config)
            accept_quote = adapters.target.quote("accept", config)
            apply_quote = adapters.recipe.quote(config, proposal)
            eval_quote = adapters.evaluator.quote(config)
            reserves = [("target", q) for q in (apply_quote, eval_quote, restore_quote, accept_quote)]
            ledger.admit([("target", snapshot_quote), *reserves])
            snapshot = ledger.invoke("target", "snapshot", snapshot_quote,
                lambda: adapters.target.snapshot(config), reserve=reserves)
            attempt["state"] = "applying"
            dirty, target_state = True, "modified_unaccepted"
            ledger.invoke("target", "apply", apply_quote,
                lambda: adapters.recipe.apply(config, adapters.target, snapshot, proposal),
                reserve=[("target", q) for q in (eval_quote, restore_quote, accept_quote)])
            result = evaluate(reserve=[("target", q) for q in (restore_quote, accept_quote)])
            assessment = assess_metrics(config.goal, result, references)
            attempt.update({"evaluation": _plain(result), "assessment": _plain(assessment)})
            if assessment.passed:
                artifact = ledger.invoke("target", "accept", accept_quote,
                    lambda: adapters.target.accept(config, snapshot, proposal), reserve=[("target", restore_quote)])
                if not isinstance(artifact, str) or not artifact:
                    raise ValueError("accepted target artifact receipt is missing")
                dirty, target_state = False, "accepted"
                attempt["state"] = "accepted"
                return finish("accepted", True, artifact)
            restore_entered = True
            ledger.invoke("target", "restore", restore_quote, lambda: adapters.target.restore(config, snapshot))
            dirty, target_state = False, "restored"
            attempt["state"] = "rolled_back_unmet_goal"
        except BaseException as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            disposition_known = isinstance(exc, (BudgetExhausted, ResourceOverrun, ValueError)) or getattr(exc, "disposition_known", False)
            if dirty and restore_entered:
                target_state, attempt["state"] = "restore_unconfirmed", "restore_unconfirmed"
                return finish("blocked_target_state", error=attempt["error"])
            if dirty and disposition_known:
                try:
                    ledger.invoke("target", "restore_after_failure", restore_quote,
                                  lambda: adapters.target.restore(config, snapshot))
                    dirty, target_state = False, "restored"
                    attempt["state"] = "rolled_back_after_failure"
                except BaseException as restore_error:
                    attempt["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
                    target_state = "restore_unconfirmed"
                    attempt["state"] = "restore_unconfirmed"
                    return finish("blocked_target_state", error=attempt["restore_error"])
            elif dirty:
                target_state = "unknown_operation_disposition"
                attempt["state"] = "unknown_operation_disposition"
                return finish("blocked_target_state", error=attempt["error"])
            else:
                attempt["state"] = "stopped_before_edit"
            return finish("budget_exhausted" if isinstance(exc, BudgetExhausted) else "candidate_failed", error=attempt["error"])
    return finish("target_unmet")
