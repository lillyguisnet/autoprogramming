"""Trusted controller for resource-aware Pi portfolio search.

Low-level Pi protocol handling lives in :mod:`autoprogramming.pi_rpc` and
implementation-worker isolation/bundling lives in
:mod:`autoprogramming.pi_worker`. This module owns portfolio policy, evaluation,
and finalization.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import metric as metric_mod
from .budget import BudgetLedger
from .errors import BudgetExceededError, RunnerError
from .objectives import MetricSuite, SelectionPolicy, approve_suite
from .pi_rpc import (
    ORCHESTRATOR_SYSTEM as _ORCHESTRATOR_SYSTEM,
    PiResult,
    PiRpcClient,
    PiUsage,
    json_object as _json_object,
)
from .pi_worker import (
    WORKER_SYSTEM as _WORKER_SYSTEM,
    PiWorkerRunner,
    avenue_dir as _avenue_dir,
    materialize_bundle as _materialize_bundle,
    task_document as _task_document,
    worker_env as _worker_env,
    worker_run_dir as _worker_run_dir,
)
from .portfolio import ApproachTier, AvenueSpec, AvenueStatus, Portfolio
from .resources import ResourceError, Resources


def _normalize_metric_suite_proposal(
    proposal: dict, names: tuple[str, ...]
) -> tuple[MetricSuite, list[str]]:
    """Repair stale role names after a critic rewrites ``metric_code``.

    Model output remains a proposal requiring user sign-off, but it must be a
    structurally valid proposal that cannot wedge ``prepare()``. Unknown names
    are dropped, every real metric gets one role, and any repair is recorded for
    review in ``metric_proposal.json``.
    """
    if not names:
        raise RunnerError("Pi metric proposal defined no quality metrics.")
    known = set(names)
    adjustments: list[str] = []

    def known_unique(raw) -> list[str]:
        result: list[str] = []
        for value in raw if isinstance(raw, (list, tuple)) else ():
            name = str(value)
            if name in known and name not in result:
                result.append(name)
        return result

    raw_acceptance = proposal.get("acceptance")
    acceptance = known_unique(raw_acceptance)
    if not acceptance:
        acceptance = [names[0]]
        adjustments.append(
            f"acceptance names did not match revised METRICS; defaulted to {names[0]!r}"
        )
    elif list(raw_acceptance or ()) != acceptance:
        adjustments.append("removed unknown or duplicate acceptance metric names")

    raw_diagnostic = proposal.get("diagnostic")
    diagnostic = [
        name for name in known_unique(raw_diagnostic) if name not in acceptance
    ]
    unassigned = [name for name in names if name not in acceptance and name not in diagnostic]
    if unassigned:
        diagnostic.extend(unassigned)
        adjustments.append(
            f"classified unassigned revised metrics as diagnostic: {unassigned!r}"
        )
    if list(raw_diagnostic or ()) != diagnostic and not unassigned:
        adjustments.append("removed unknown, duplicate, or overlapping diagnostic names")

    raw_preference = proposal.get("preference_order")
    preference = [
        name for name in known_unique(raw_preference) if name in acceptance
    ]
    if not preference:
        preference = list(acceptance)
        adjustments.append("defaulted preference_order to the acceptance metrics")
    elif list(raw_preference or ()) != preference:
        adjustments.append("removed non-acceptance names from preference_order")

    floors: dict[str, float] = {}
    raw_floors = proposal.get("floors")
    if isinstance(raw_floors, dict):
        for raw_name, raw_value in raw_floors.items():
            name = str(raw_name)
            if name not in acceptance:
                adjustments.append(f"dropped floor for non-acceptance metric {name!r}")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                adjustments.append(f"dropped non-numeric floor for {name!r}")
                continue
            if not math.isfinite(value):
                adjustments.append(f"dropped non-finite floor for {name!r}")
                continue
            floors[name] = value

    return MetricSuite(
        acceptance=tuple(acceptance),
        diagnostic=tuple(diagnostic),
        policy=SelectionPolicy(
            floors=floors,
            preference_order=tuple(preference),
        ),
    ), adjustments


def _execute_pi_calls(
    items,
    *,
    max_workers: int,
    ledger: BudgetLedger,
    reservation_dollars: float | None,
    invoke,
):
    """Run Pi calls with bounded in-flight dollar commitments.

    If a dollar budget exists but no per-call bound was confirmed, calls are
    serialized so only one can overshoot at a time. With a confirmed bound,
    each in-flight call reserves that headroom before launch. New calls are
    submitted only after completed usage has been settled.
    """
    pending = list(items)
    completed: list[tuple[object, object | None, Exception | None]] = []
    undispatched: list[object] = []
    dollar_limited = ledger.limits.get("dollars") is not None
    parallelism = max(1, int(max_workers))
    if dollar_limited and reservation_dollars is None:
        parallelism = 1

    def result_usage_dollars(result) -> float:
        pi_result = result[0] if isinstance(result, tuple) else result
        if not isinstance(pi_result, PiResult):
            raise RunnerError("Internal Pi call did not return a PiResult.")
        return float(pi_result.usage.dollars)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures: dict[concurrent.futures.Future, tuple[object, str | None]] = {}
        stopped = False

        def fill_slots() -> None:
            nonlocal stopped
            while pending and len(futures) < parallelism and not stopped:
                item = pending.pop(0)
                token = None
                try:
                    ledger.check()
                    if dollar_limited and reservation_dollars is not None:
                        token = ledger.reserve_dollars(
                            reservation_dollars, category="agent"
                        )
                except BudgetExceededError:
                    pending.insert(0, item)
                    # Existing reservations may be the only reason headroom is
                    # unavailable. Wait for a completion and retry with actual
                    # spend before declaring the remaining calls undispatchable.
                    if futures:
                        return
                    undispatched.extend(pending)
                    pending.clear()
                    stopped = True
                    return
                futures[pool.submit(invoke, item)] = (item, token)

        fill_slots()
        while futures:
            done, _pending_futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                item, token = futures.pop(future)
                try:
                    result = future.result()
                    actual = result_usage_dollars(result)
                    if token is None:
                        ledger.charge(dollars=actual, category="agent")
                    else:
                        ledger.settle_reservation(token, dollars=actual)
                    completed.append((item, result, None))
                except Exception as exc:
                    if token is not None:
                        ledger.release_reservation(token)
                    completed.append((item, None, exc))
            fill_slots()

    return completed, undispatched


class PiOrchestratorBackend:
    """Breadth-first optimizer driven by one Pi orchestrator and Pi workers."""

    def __init__(
        self,
        command: tuple[str, ...] = ("pi",),
        *,
        orchestrator_model: str | None = None,
        worker_model: str | None = None,
        resources: Resources | None = None,
        strict_isolation: bool = False,
        orchestrator_timeout: float = 900.0,
        worker_timeout: float = 1200.0,
    ):
        if orchestrator_timeout <= 0 or worker_timeout <= 0:
            raise ValueError("Pi timeouts must be positive.")
        self.command = tuple(command)
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.resources = resources
        self.strict_isolation = strict_isolation
        self.orchestrator_timeout = float(orchestrator_timeout)
        self.worker_timeout = float(worker_timeout)

    def run(self, harness, context: dict) -> None:
        ws = harness.workspace
        resources = self.resources or _load_resources(ws)
        resources.ensure_confirmed()
        if not resources.pi_may_receive_task_data:
            raise ResourceError(
                "This resource profile forbids task-data egress but does not mark "
                "the configured Pi models as local. Pi orchestration and workers "
                "must inspect task context and development examples. Set "
                "search.pi_local=True for an actually local Pi provider, or "
                "explicitly permit external egress; API credentials alone are "
                "never consent."
            )
        if self.strict_isolation:
            raise RunnerError(
                "strict_isolation=True requires the forthcoming OS sandbox worker "
                "adapter; cooperative isolation is available now but must not be "
                "misrepresented as a security boundary."
            )
        if not metric_mod.is_approved(ws):
            if not self._propose_metric_suite(harness, resources):
                return
        if context.get("mode") == "prepare" or context.get("prepare_only"):
            return

        portfolio_path = _orchestration_dir(ws) / "portfolio.json"
        if portfolio_path.exists():
            try:
                portfolio = Portfolio.load(portfolio_path)
                if portfolio.resources != resources:
                    raise ValueError("resource profile changed")
            except Exception as exc:
                raise RunnerError(
                    f"Cannot resume portfolio state at {portfolio_path}: {exc}. "
                    "Restore it or start a fresh workspace; silently replacing "
                    "orchestration history would lose budget and coverage state."
                ) from exc
        else:
            portfolio = self._create_portfolio(harness, resources)
            portfolio.write(portfolio_path)
        self._recover_pending_avenues(harness, portfolio, portfolio_path)
        worker = PiWorkerRunner(self.command, timeout=self.worker_timeout)
        runnable_states = [
            a for a in portfolio.avenues
            if not a.candidates
            and a.status != AvenueStatus.INFEASIBLE
            and a.spec.tier != ApproachTier.COMPOSITION
        ]
        specs = [a.spec for a in runnable_states]
        for state in runnable_states:
            state.status = AvenueStatus.RUNNING
        portfolio.write(portfolio_path)

        # Bootstrap mode can expose at most five distinct candidates to val.
        from .guards import BOOTSTRAP_MAX_VAL_CANDIDATES, is_bootstrap
        if is_bootstrap(ws) and len(specs) > BOOTSTRAP_MAX_VAL_CANDIDATES:
            deferred = specs[BOOTSTRAP_MAX_VAL_CANDIDATES:]
            specs = specs[:BOOTSTRAP_MAX_VAL_CANDIDATES]
            for spec in deferred:
                state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
                state.status = AvenueStatus.INFEASIBLE
                state.notes.append(
                    "not dispatched: bootstrap data permits at most five distinct "
                    "val candidates; gather 30+ validated examples for full breadth"
                )
            portfolio.write(portfolio_path)

        results: dict[str, tuple[PiResult, Path]] = {}
        ledger = BudgetLedger(ws.budget_json)
        completed, undispatched = _execute_pi_calls(
            specs,
            max_workers=min(
                resources.search.max_parallel_agents, max(1, len(specs))
            ),
            ledger=ledger,
            reservation_dollars=resources.search.max_dollars_per_agent_call,
            invoke=lambda spec: self._run_avenue(
                harness, spec, resources, worker
            ),
        )
        for spec in undispatched:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            state.status = AvenueStatus.PLANNED
            state.notes.append("not dispatched: agent dollar headroom is exhausted")
        for spec, item, error in completed:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            if error is not None:
                state.status = AvenueStatus.FAILED
                state.notes.append(str(error))
                continue
            result, sandbox = item
            results[spec.id] = (result, sandbox)
            state.status = AvenueStatus.READY
            if result.stderr.strip():
                state.notes.append(result.stderr.strip()[-1000:])
        portfolio.write(portfolio_path)

        for spec in specs:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            item = results.get(spec.id)
            if item is None or state.status == AvenueStatus.FAILED:
                continue
            _pi_result, sandbox = item
            solution = sandbox / "solution.py"
            if not solution.exists():
                state.status = AvenueStatus.FAILED
                state.notes.append("worker did not create solution.py")
                continue
            source = solution.read_text(encoding="utf-8")
            if not re.search(r"(?m)^def predict\s*\(", source):
                state.status = AvenueStatus.FAILED
                state.notes.append("solution.py does not define predict")
                continue
            try:
                from .candidates import next_name

                expected_name = next_name(ws)
                state.begin_candidate(expected_name)
                portfolio.write(portfolio_path)
                source = _materialize_bundle(source, sandbox, ws, spec.id)
                cand = harness.new_candidate(source=source)
                if cand.name != expected_name:
                    raise RunnerError(
                        f"Candidate journal expected {expected_name}, got {cand.name}."
                    )
                train_report = harness.eval(cand.name, split="train", per_instance=True)
                val_report = harness.eval(cand.name)
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                state.status = AvenueStatus.FAILED
                state.notes.append(f"candidate import/eval failed: {exc}")
                continue
            means = {name: float(obj["mean"]) for name, obj in val_report.objectives.items()}
            state.record_result(cand.name, means, improved=True)
            state.notes.append(
                f"train aggregate {train_report.mean:.4g}; val aggregate stored privately"
            )
            portfolio.write(portfolio_path)

        portfolio.write(portfolio_path)
        if not any(a.candidates for a in portfolio.avenues):
            print("[autoprogramming] Pi portfolio produced no evaluable implementations; inspect " + str(portfolio_path))
            return

        # Breadth is complete before exploitation. Give each successful family
        # a second, stateful Pi turn so one mediocre configuration cannot dismiss
        # an entire mechanism. Bootstrap mode deliberately skips this expansion.
        if not is_bootstrap(ws):
            # A mandatory second configuration protects breadth. After that,
            # the main Pi orchestrator allocates the final deepening round from
            # aggregate vectors only; workers never see the dashboard.
            self._deepen_avenues(harness, portfolio, resources, worker, portfolio_path)
            decision = self._round_decision(harness, portfolio, resources)
            selected = set(decision.get("deepen") or ())
            if selected:
                self._deepen_avenues(
                    harness, portfolio, resources, worker, portfolio_path,
                    selected_ids=selected,
                )
            if decision.get("compose", True):
                self._compose_frontier(harness, portfolio, resources, worker, portfolio_path)

        portfolio.write(portfolio_path)
        if not portfolio.may_finalize:
            raise RunnerError(
                "Portfolio policy refused early finalization: feasible breadth or "
                "the required exploration reserve is incomplete. Resume the run "
                f"from {portfolio_path}."
            )
        harness.finalize()

    def _recover_pending_avenues(self, harness, portfolio, portfolio_path) -> None:
        """Finish an import/evaluation journal entry after controller restart."""
        from . import scoring
        from .candidates import load_candidate

        for avenue in portfolio.avenues:
            name = avenue.pending_candidate
            if not name:
                continue
            try:
                load_candidate(harness.workspace, name)
            except Exception as exc:
                avenue.pending_candidate = None
                avenue.status = AvenueStatus.PLANNED
                avenue.notes.append(f"discarded missing pending candidate {name}: {exc}")
                portfolio.write(portfolio_path)
                continue
            try:
                scores = scoring.load_scores(harness.workspace)
                sub = scores.get("candidates", {}).get(name, {}).get("val")
                if (
                    isinstance(sub, dict)
                    and scoring.score_provenance_current(
                        harness.workspace, name, "val"
                    )
                ):
                    objectives = {
                        objective: float(stats["mean"])
                        for objective, stats in sub.get("objectives", {}).items()
                    }
                else:
                    harness.eval(name, split="train", per_instance=True)
                    val = harness.eval(name)
                    objectives = {
                        objective: float(stats["mean"])
                        for objective, stats in val.objectives.items()
                    }
                improved = True
                if avenue.candidates:
                    try:
                        improved = bool(
                            harness.compare(avenue.candidates[-1], name).improved
                        )
                    except Exception:
                        improved = True
                avenue.record_result(name, objectives, improved=improved)
                avenue.notes.append(
                    "recovered candidate evaluation journal after controller restart"
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                avenue.pending_candidate = None
                if name not in avenue.candidates:
                    avenue.candidates.append(name)
                avenue.rounds += 1
                avenue.status = AvenueStatus.FAILED
                avenue.notes.append(f"pending candidate recovery failed: {exc}")
            portfolio.write(portfolio_path)

    def _round_decision(self, harness, portfolio, resources) -> dict:
        try:
            BudgetLedger(harness.workspace.budget_json).check()
        except BudgetExceededError:
            return {"deepen": [], "compose": False, "rationale": "budget exhausted"}
        prompt = f"""Allocate the last exploitation round of this breadth-first
portfolio. You may deepen avenues with plausible headroom and request one
cross-family composition. Do not write code. Return JSON:
{{"deepen": [avenue ids], "compose": true, "rationale": "..."}}
Choose at most {max(1, resources.search.max_parallel_agents)} avenues. An avenue
with acceptable numbers is not a reason to ignore a different mechanism.
Portfolio state (aggregate vectors only):\n{json.dumps(portfolio.to_dict(), default=str)}"""
        try:
            with self._orchestrator(harness.workspace) as client:
                result = client.prompt(prompt)
            BudgetLedger(harness.workspace.budget_json).charge(
                dollars=result.usage.dollars, category="agent"
            )
            decision = _json_object(result.text)
            valid = {a.spec.id for a in portfolio.avenues}
            decision["deepen"] = [
                name for name in decision.get("deepen", []) if name in valid
            ][: resources.search.max_parallel_agents]
            return decision
        except Exception as exc:
            return {"deepen": [], "compose": True, "rationale": f"fallback: {exc}"}

    def _deepen_avenues(
        self, harness, portfolio, resources, worker, portfolio_path,
        *, selected_ids: set[str] | None = None,
    ) -> None:
        ws = harness.workspace
        active = [
            avenue for avenue in portfolio.avenues
            if avenue.candidates
            and (selected_ids is None or avenue.spec.id in selected_ids)
            and avenue.rounds < avenue.spec.max_rounds
            and avenue.status not in (AvenueStatus.FAILED, AvenueStatus.INFEASIBLE)
        ]
        if not active:
            return
        try:
            BudgetLedger(ws.budget_json).check()
        except BudgetExceededError:
            return

        def deepen(avenue):
            root = _avenue_dir(ws, avenue.spec.id)
            task = (
                "Re-open task.md and the current solution.py. This is a second "
                "engineering pass on the same assigned strategy. Find weaknesses "
                "in generalization, edge cases, parsing, failure handling, startup, "
                "and repeated-call efficiency. Improve solution.py materially "
                "without changing to a different strategy; syntax-check it."
            )
            model = self.worker_model or (
                resources.search.pi_models[0] if resources.search.pi_models else None
            )
            return worker.run(
                root, task, model=model, session_id=avenue.spec.id,
                allowed_api_providers=avenue.spec.allowed_api_providers,
            )

        ledger = BudgetLedger(ws.budget_json)
        completed, undispatched = _execute_pi_calls(
            active,
            max_workers=min(resources.search.max_parallel_agents, len(active)),
            ledger=ledger,
            reservation_dollars=resources.search.max_dollars_per_agent_call,
            invoke=deepen,
        )
        for avenue in undispatched:
            avenue.notes.append(
                "refinement not dispatched: agent dollar headroom is exhausted"
            )

        from .candidates import load_candidate, source_sha
        for avenue, result, error in completed:
            if error is not None:
                # Keep the baseline alive; a failed refinement is not a
                # failure of the already-evaluated avenue.
                avenue.notes.append(f"refinement failed: {error}")
                continue
            solution = _avenue_dir(ws, avenue.spec.id) / "solution.py"
            if not solution.exists():
                avenue.notes.append("refinement removed solution.py")
                continue
            source = solution.read_text(encoding="utf-8")
            baseline = load_candidate(ws, avenue.candidates[-1])
            # A second turn that made no code change still counts as an attempted
            # configuration, but does not spend candidate-eval budget.
            if source_sha(baseline) == hashlib.sha256(source.encode()).hexdigest():
                avenue.rounds += 1
                avenue.no_progress_rounds += 1
                avenue.notes.append("second configuration made no source change")
                if avenue.no_progress_rounds >= portfolio.policy.stagnation_rounds:
                    avenue.status = AvenueStatus.STAGNANT
                continue
            try:
                from .candidates import next_name

                expected_name = next_name(ws)
                avenue.begin_candidate(expected_name)
                portfolio.write(portfolio_path)
                source = _materialize_bundle(
                    source, solution.parent, ws, avenue.spec.id
                )
                cand = harness.new_candidate(source=source)
                if cand.name != expected_name:
                    raise RunnerError(
                        f"Candidate journal expected {expected_name}, got {cand.name}."
                    )
                harness.eval(cand.name, split="train", per_instance=True)
                val = harness.eval(cand.name)
                comparison = harness.compare(baseline.name, cand.name)
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                return
            except Exception as exc:
                avenue.notes.append(f"refined implementation failed: {exc}")
                continue
            avenue.record_result(
                cand.name,
                {name: float(obj["mean"]) for name, obj in val.objectives.items()},
                improved=bool(comparison.improved),
            )
            avenue.notes.append(str(comparison))
            portfolio.write(portfolio_path)

    def _compose_frontier(self, harness, portfolio, resources, worker, portfolio_path) -> None:
        tradeoffs = harness.tradeoffs()
        if len(tradeoffs.nondominated) < 2:
            return
        try:
            BudgetLedger(harness.workspace.budget_json).check()
        except BudgetExceededError:
            return
        from .candidates import load_candidate

        names = tradeoffs.nondominated[:3]
        spec = AvenueSpec(
            id="frontier-composition",
            tier=ApproachTier.COMPOSITION,
            title="Complementary implementation composition",
            hypothesis="The supplied mechanisms have complementary strengths that a bounded router or cascade can combine.",
            implementation_brief=(
                "Study component_*.py and build one self-contained solution.py that "
                "combines them as a router, cascade, pipeline, or ensemble. Inline "
                "everything needed; do not import component files at runtime."
            ),
            mechanism="task-specific bounded router or cascade over frontier mechanisms",
            compose_from=tuple(names),
            max_rounds=1,
        )
        root = _avenue_dir(harness.workspace, spec.id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text(_task_document(harness.schema, spec, resources))
        from .data import development_partition
        fit_rows, _probe = development_partition(list(harness.data.train))
        with (root / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for row in fit_rows:
                fh.write(json.dumps(row, default=str) + "\n")
        for i, name in enumerate(names):
            (root / f"component_{i}.py").write_text(load_candidate(harness.workspace, name).source)
        model = self.worker_model or (
            resources.search.pi_models[0] if resources.search.pi_models else None
        )
        result = worker.run(
            root,
            "Build the self-contained composition described in task.md from the supplied component files. Create and syntax-check solution.py.",
            model=model,
            session_id=spec.id,
            allowed_api_providers=tuple(resources.runtime.api_providers),
        )
        BudgetLedger(harness.workspace.budget_json).charge(
            dollars=result.usage.dollars, category="agent"
        )
        from .portfolio import AvenueState
        composed = AvenueState(spec=spec)
        portfolio.avenues.append(composed)
        solution = root / "solution.py"
        if not solution.exists():
            composed.status = AvenueStatus.FAILED
            composed.notes.append("composition worker did not create solution.py")
            return
        try:
            from .candidates import next_name

            expected_name = next_name(harness.workspace)
            composed.begin_candidate(expected_name)
            portfolio.write(portfolio_path)
            source = _materialize_bundle(
                solution.read_text(encoding="utf-8"), root, harness.workspace, spec.id
            )
            cand = harness.new_candidate(source=source)
            if cand.name != expected_name:
                raise RunnerError(
                    f"Candidate journal expected {expected_name}, got {cand.name}."
                )
            harness.eval(cand.name, split="train", per_instance=True)
            val = harness.eval(cand.name)
            composed.record_result(
                cand.name,
                {name: float(obj["mean"]) for name, obj in val.objectives.items()},
                improved=cand.name in harness.tradeoffs().nondominated,
            )
        except Exception as exc:
            composed.status = AvenueStatus.FAILED
            composed.notes.append(str(exc))
        portfolio.write(portfolio_path)

    def _orchestrator(self, ws) -> PiRpcClient:
        return PiRpcClient(
            self.command,
            cwd=ws.root,
            model=self.orchestrator_model,
            system_prompt=_ORCHESTRATOR_SYSTEM,
            timeout=self.orchestrator_timeout,
        )

    def _propose_metric_suite(self, harness, resources: Resources) -> bool:
        ws = harness.workspace
        BudgetLedger(ws.budget_json).check()
        sample = list(harness.data.train[: min(8, len(harness.data.train))])
        prompt = f"""Propose 2-4 independent quality lenses for this task. Include at
least one direct correctness lens and, where meaningful, one graded or robustness
lens. Do not use cost or latency; the controller supplies those. Return JSON:
{{"metric_code": "complete Python defining METRICS", "acceptance": [names],
  "diagnostic": [names], "preference_order": [acceptance names],
  "floors": {{}}, "rationale": "..."}}
Metric functions receive predicted and expected bare values for one output, or
dicts for multiple outputs. They must be deterministic and return numeric scores.
Every name in acceptance, diagnostic, preference_order, and floors must exactly
match a key in the returned METRICS mapping.
Schema:\n{harness.schema.describe()}\nResource policy:\n{json.dumps(resources.to_dict())}\nDevelopment examples:\n{json.dumps(sample, default=str)}"""
        with self._orchestrator(ws) as client:
            result = client.prompt(prompt)
        BudgetLedger(ws.budget_json).charge(dollars=result.usage.dollars, category="agent")
        proposal = _json_object(result.text)
        critic_feedback = ""
        critic_cost = 0.0
        try:
            BudgetLedger(ws.budget_json).check()
            critic_prompt = f"""Act as an adversarial metric critic. Find proxy
hacking, flat metrics, formatting blind spots, semantic blind spots, evaluator
self-preference, and missing robustness lenses in this proposal. Return the same
JSON schema with a corrected complete proposal, plus `critic_feedback`. Do not
remove a direct correctness lens. All role and policy names must exactly match
keys in the corrected METRICS mapping. Task schema:\n{harness.schema.describe()}\nExamples:
{json.dumps(sample, default=str)}\nProposal:\n{json.dumps(proposal)}"""
            with PiRpcClient(
                self.command,
                cwd=ws.root,
                model=self.orchestrator_model,
                system_prompt=(
                    "You are an independent evaluation-design critic. You never "
                    "implement the task. Return only requested JSON."
                ),
                timeout=self.orchestrator_timeout,
            ) as critic:
                critique = critic.prompt(critic_prompt)
            critic_cost = critique.usage.dollars
            BudgetLedger(ws.budget_json).charge(
                dollars=critic_cost, category="agent"
            )
            revised = _json_object(critique.text)
            if revised.get("metric_code"):
                proposal = revised
            critic_feedback = str(revised.get("critic_feedback") or "")
        except Exception as exc:
            critic_feedback = f"metric critic unavailable: {exc}"

        code = str(proposal.get("metric_code") or "")
        if "METRICS" not in code and "def metric" not in code:
            raise RunnerError("Pi metric proposal did not define METRICS or metric().")
        metric_mod.write_metric(ws, code)
        names = tuple(metric_mod.quality_metrics(ws))
        suite, adjustments = _normalize_metric_suite_proposal(proposal, names)
        proposed = {
            "suite": suite.to_dict(),
            "rationale": proposal.get("rationale", ""),
            "critic_feedback": critic_feedback,
            "proposal_adjustments": adjustments,
            "pi_usage_dollars": result.usage.dollars + critic_cost,
        }
        proposal_path = ws.root / "metric_proposal.json"
        proposal_path.write_text(json.dumps(proposed, indent=2) + "\n")
        if os.environ.get("AP_AUTO_APPROVE_METRIC", "").strip().lower() in ("1", "true", "yes"):
            approve_suite(ws, "auto (AP_AUTO_APPROVE_METRIC)", suite)
            return True
        print(
            "[autoprogramming] Pi proposed a metric suite and paused before search. "
            f"Review {ws.metric_py} and {proposal_path}; demonstrate it on real "
            "examples, then approve with:\n"
            "    import json\n"
            "    from autoprogramming.objectives import approve_suite, MetricSuite\n"
            f"    suite = MetricSuite.from_dict(json.load(open({str(proposal_path)!r}))['suite'])\n"
            "    approve_suite(prg.workspace, 'your name', suite)\n"
            "or use prg.approve_metric_suite(...), then resume optimize()."
        )
        return False

    def _create_portfolio(self, harness, resources: Resources) -> Portfolio:
        BudgetLedger(harness.workspace.budget_json).check()
        feasibility = resources.feasibility()
        prompt = f"""Design a diverse task-specific implementation portfolio across
EVERY feasible tier in the supplied feasibility map. Each avenue must use a
materially distinct mechanism. Do not write code. Return JSON:
{{"avenues": [{{"id":"...", "tier":1, "title":"...",
"hypothesis":"...", "implementation_brief":"...", "mechanism":"...",
"runtime_requirements":[], "allowed_api_providers":[], "max_rounds":3,
"wildcard":false}}], "exclusions": {{"tier": "reason"}}}}
Schema:\n{harness.schema.describe()}\nResources:\n{json.dumps(resources.to_dict())}\nFeasibility:\n{json.dumps(feasibility)}"""
        try:
            with self._orchestrator(harness.workspace) as client:
                result = client.prompt(prompt)
            BudgetLedger(harness.workspace.budget_json).charge(
                dollars=result.usage.dollars, category="agent"
            )
            value = _json_object(result.text)
            specs = [AvenueSpec.from_dict(v) for v in value.get("avenues", [])]
            return Portfolio.create(
                resources,
                specs,
                exclusions={int(k): str(v) for k, v in value.get("exclusions", {}).items()},
                fill_missing=True,
            )
        except Exception as exc:
            print(f"[autoprogramming] Pi portfolio plan was invalid ({exc}); filling a deterministic breadth-first portfolio.")
            return Portfolio.create(resources, [], fill_missing=True)

    def _run_avenue(self, harness, spec: AvenueSpec, resources: Resources, worker: PiWorkerRunner):
        root = _avenue_dir(harness.workspace, spec.id)
        root.mkdir(parents=True, exist_ok=True)
        # Generic filenames and prose intentionally reveal no optimizer, candidate,
        # metric, score, split, or competing-agent context.
        task_doc = _task_document(harness.schema, spec, resources)
        (root / "task.md").write_text(task_doc)
        from .data import development_partition

        rows = list(harness.data.train)
        try:
            seed = int(json.loads(harness.workspace.split_json.read_text()).get("seed", 0))
        except (OSError, ValueError):
            seed = 0
        fit_rows, probe_rows = development_partition(rows, seed=seed)
        with (root / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for row in fit_rows:
                fh.write(json.dumps(row, default=str) + "\n")
        task = (
            "Implement the function described in task.md using only the assigned "
            "strategy and permitted resources. Inspect examples.jsonl, create "
            "solution.py, test it locally, and stop when this implementation is robust."
        )
        model = self.worker_model or (resources.search.pi_models[0] if resources.search.pi_models else None)
        # Probe contents never enter the worker directory or prompt.
        _ = probe_rows
        return worker.run(
            root, task, model=model, session_id=spec.id,
            allowed_api_providers=spec.allowed_api_providers,
        ), root


def _orchestration_dir(workspace) -> Path:
    path = Path(workspace.root) / ".ap" / "controller"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_resources(workspace) -> Resources:
    path = getattr(workspace, "resources_json", Path(workspace.root) / "resources.json")
    if not Path(path).exists():
        raise RunnerError(
            "Pi orchestration requires a confirmed Resources profile. Pass "
            "resources=ap.Resources(...) to optimize(), or construct "
            "PiOrchestratorBackend(resources=...)."
        )
    return Resources.from_dict(json.loads(Path(path).read_text()))
