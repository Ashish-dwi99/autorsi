"""Evaluation-gated search; candidate failures never alter serving weights."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .contracts import (Backend, EditRecipe, Evaluator, Goal, Guard, ProposalContext,
                        Proposer, RunLimits, RunResult, ValidationError)
from .backends import PromotionDeadlineError


class _LimitReached(RuntimeError):
    pass


class _JournalFailure(RuntimeError):
    pass


class RollbackJournalError(RuntimeError):
    """Rollback committed, but its final journal write failed; revision is known."""
    def __init__(self, revision):
        self.revision = revision
        super().__init__(f"rollback succeeded at revision {revision}, but its final journal write failed")


def _append_event(journal_path, kind, **data):
    try:
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, **data}, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise _JournalFailure("journal write failed; inspect returned revision/accepted count for serving state") from exc


def _assess(goal, guards, reference, current, candidate):
    required = {goal.metric, *(guard.metric for guard in guards)}
    for label, scores in (("baseline", reference), ("current", current), ("candidate", candidate)):
        missing = required - scores.keys()
        if missing:
            raise ValidationError(f"{label} evaluation is missing required metrics: {sorted(missing)}")
    gain = goal.gain(current[goal.metric], candidate[goal.metric])
    reasons = []
    if gain <= 0 or gain < goal.min_improvement:
        reasons.append("objective did not improve by the required amount")
    reasons.extend(_guard_reasons(guards, reference, candidate))
    return reasons


def _guard_reasons(guards, reference, candidate):
    reasons = []
    for guard in guards:
        sign = 1 if guard.direction == "maximize" else -1
        change = sign * (candidate[guard.metric] - reference[guard.metric])
        if change < -guard.max_regression:
            reasons.append(f"protected metric regressed: {guard.metric}")
    return reasons


class AutoRSI:
    """Bounded proposal -> detached edit -> evaluation -> safe-point promotion.

    A controller freezes its original baseline at construction. Every guard is
    compared to that baseline across run()/step() calls, avoiding cumulative
    tolerance erosion. Evaluators and all Python adapters are trusted code.
    Goal targets stop search; intermediate strict improvements may be accepted.
    """
    def __init__(self, *, backend: Backend, evaluator: Evaluator, goal: Goal,
                 proposer: Proposer, guards: tuple[Guard, ...] | list[Guard] = (),
                 validation_evaluator: Evaluator | None = None,
                 output_dir: str | Path = "autorsi-runs"):
        if not callable(evaluator) or not isinstance(goal, Goal):
            raise ValueError("provide an evaluator callable and a Goal")
        guards = tuple(guards)
        if any(not isinstance(g, Guard) for g in guards) or len({g.metric for g in guards}) != len(guards):
            raise ValueError("guards must contain one Guard per distinct protected metric")
        if validation_evaluator is not None and not callable(validation_evaluator):
            raise ValueError("validation_evaluator must be callable")
        if not callable(getattr(proposer, "propose", None)):
            raise ValueError("proposer must implement propose(context)")
        self.backend, self.evaluator, self.goal = backend, evaluator, goal
        self.proposer, self.guards = proposer, guards
        self.validation_evaluator = validation_evaluator
        self.output_dir = Path(output_dir)
        self._anchor = backend.capture()
        self._reference = None
        self._validation_reference = None
        self._rollback = []
        self._history = []
        self._lock = threading.Lock()

    def step(self, *, limits: RunLimits | None = None):
        """One proposal, suitable for an explicit agent task-boundary callback."""
        return self.run(limits=replace(limits or RunLimits(), max_steps=1, max_accepted=1))

    def rollback(self):
        """Restore the last accepted predecessor at a safe point; revision-checked."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this controller is already running")
        try:
            if not self._rollback:
                raise ValueError("no accepted change is available to roll back")
            state, expected, journal = self._rollback[-1]
            _append_event(journal, "rollback_prepared", expected_revision=expected,
                          predecessor_fingerprint=state.fingerprint)
            revision = self.backend.promote(state, expected_revision=expected)
            self._rollback.pop()
            # Earlier rollback entries now refer to the newly restored state.
            if self._rollback:
                predecessor, _, predecessor_journal = self._rollback[-1]
                self._rollback[-1] = (predecessor, revision, predecessor_journal)
            try:
                _append_event(journal, "rollback_committed", revision=revision, fingerprint=state.fingerprint)
            except _JournalFailure as exc:
                raise RollbackJournalError(revision) from exc
            return revision
        finally:
            self._lock.release()

    def run(self, *, limits: RunLimits | None = None):
        limits = limits or RunLimits()
        if not isinstance(limits, RunLimits):
            raise ValueError("limits must be RunLimits")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("this controller is already running")
        try:
            return self._run(limits)
        finally:
            self._lock.release()

    def _run(self, limits):
        started = time.monotonic()
        attempts, accepted, evaluations = [], 0, 0
        current, scores, validation_scores = self.backend.capture(), {}, None
        checkpoint, status, error = None, "step_limit", None
        run_dir = self.output_dir / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        journal = run_dir / "journal.jsonl"

        def event(kind, **data):
            _append_event(journal, kind, **data)

        def check():
            if time.monotonic() - started >= limits.max_seconds:
                raise _LimitReached("elapsed time limit reached between calls")

        def evaluate(state, callback):
            nonlocal evaluations
            check()
            if evaluations >= limits.max_evaluations:
                raise _LimitReached("evaluation limit reached")
            evaluations += 1
            value = self.backend.evaluate(state, callback)
            check()  # A late result cannot authorize promotion.
            return value

        try:
            event("run_started", baseline_fingerprint=self._anchor.fingerprint,
                  starting_fingerprint=current.fingerprint, revision=current.revision,
                  goal=vars(self.goal), guards=[vars(g) for g in self.guards], limits=vars(limits),
                  validation_gate=self.validation_evaluator is not None)
            if self._reference is None:
                self._reference = evaluate(self._anchor, self.evaluator)
            scores = (self._reference if current.fingerprint == self._anchor.fingerprint
                      else evaluate(current, self.evaluator))
            # Validate metric presence even when no candidate is attempted.
            _assess(self.goal, self.guards, self._reference, scores, scores)
            if self.validation_evaluator is not None:
                if self._validation_reference is None:
                    self._validation_reference = evaluate(self._anchor, self.validation_evaluator)
                validation_scores = (self._validation_reference if current.fingerprint == self._anchor.fingerprint
                                     else evaluate(current, self.validation_evaluator))
                _assess(self.goal, self.guards, self._validation_reference, validation_scores, validation_scores)
            event("baseline_evaluated", baseline=dict(self._reference), current=dict(scores))
            if _guard_reasons(self.guards, self._reference, scores):
                status = "current_regression"
            elif (validation_scores is not None and
                  _guard_reasons(self.guards, self._validation_reference, validation_scores)):
                status = "current_regression"
            elif self.goal.reached(scores) and (validation_scores is None or self.goal.reached(validation_scores)):
                status = "goal_reached"
            else:
                for index in range(limits.max_steps):
                    check()
                    context = ProposalContext(self.goal, self.guards, self._reference, scores,
                                              tuple(deepcopy(self._history[-8:])), self.backend.recipe_space)
                    recipe = self.proposer.propose(context)
                    check()
                    if recipe is None:
                        status = "proposer_stopped"
                        break
                    if not isinstance(recipe, EditRecipe):
                        raise ValidationError("proposer must return EditRecipe or None")
                    attempt = {"step": index + 1, "recipe": recipe.to_dict(), "status": "proposed"}
                    attempts.append(attempt)
                    event("proposal", **attempt)
                    candidate = self.backend.candidate(current, recipe)
                    candidate_scores = evaluate(candidate, self.evaluator)
                    reasons = _assess(self.goal, self.guards, self._reference, scores, candidate_scores)
                    attempt["metrics"] = dict(candidate_scores)
                    candidate_validation = None
                    if not reasons and self.validation_evaluator is not None:
                        candidate_validation = evaluate(candidate, self.validation_evaluator)
                        validation_reasons = _assess(self.goal, self.guards, self._validation_reference,
                                                     validation_scores, candidate_validation)
                        if validation_reasons:
                            reasons.append("validation gate rejected candidate")
                        # Validation values are journaled but not sent to proposer.
                        event("validation_evaluated", step=index + 1, metrics=dict(candidate_validation),
                              reasons=validation_reasons)
                    if reasons:
                        attempt.update(status="rejected", reasons=reasons)
                        event("candidate_rejected", **attempt)
                        self._history.append(deepcopy(attempt))
                        continue
                    check()
                    destination = run_dir / f"candidate-{index + 1}"
                    destination.mkdir()
                    path = self.backend.export(candidate, destination)
                    event("candidate_prepared", step=index + 1, fingerprint=candidate.fingerprint,
                          checkpoint=str(path), metrics=dict(candidate_scores))
                    check()
                    revision = self.backend.promote(candidate, expected_revision=current.revision,
                                                    deadline=started + limits.max_seconds)
                    self._rollback.append((current, revision, journal))
                    current = replace(candidate, revision=revision)
                    scores, validation_scores = candidate_scores, candidate_validation
                    accepted += 1
                    checkpoint = str(path)
                    attempt.update(status="accepted", revision=revision, fingerprint=current.fingerprint,
                                   checkpoint=checkpoint)
                    self._history.append(deepcopy(attempt))
                    event("candidate_accepted", **attempt)
                    if self.goal.reached(scores) and (validation_scores is None or self.goal.reached(validation_scores)):
                        status = "goal_reached"
                        break
                    if accepted >= limits.max_accepted:
                        status = "accept_limit"
                        break
        except (_LimitReached, PromotionDeadlineError) as exc:
            status, error = "limit_reached", str(exc)
        except _JournalFailure as exc:
            status, error = "journal_error", str(exc)
        except Exception as exc:
            # Errors are local exception classes only: provider exceptions can
            # contain credentials, URLs or payloads. Applications may wrap their
            # callbacks to log diagnostic detail in an appropriate private sink.
            status, error = "error", f"{type(exc).__name__}: operation failed; serving weights were not changed by that operation"
            if isinstance(exc, (ValueError, RuntimeError)) and type(exc).__module__.startswith("autorsi"):
                error = str(exc)
            if attempts and attempts[-1]["status"] == "proposed":
                attempts[-1]["status"] = "failed"
        serving_revision = self.backend.revision
        if serving_revision != current.revision:
            status, error = "stale_revision", "serving model changed concurrently; metrics describe the returned revision only"
        result = RunResult(status, accepted, current.revision, dict(self._reference or {}),
                           dict(scores), tuple(attempts), evaluations, checkpoint, str(journal), error, serving_revision)
        try:
            event("run_finished", **result.to_dict())
        except _JournalFailure as exc:
            result = replace(result, status="journal_error", error=str(exc))
        return result
