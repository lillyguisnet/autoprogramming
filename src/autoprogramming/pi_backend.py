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
import shutil
from pathlib import Path

from . import metric as metric_mod
from .adherence import (
    ApproachAudit,
    deterministic_audit,
    semantic_audit_prompt,
)
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
from .portfolio import (
    ApproachTier,
    AvenueSpec,
    AvenueStatus,
    Portfolio,
    ensure_avenue_contract,
)
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


def _missing_avenue_capabilities(spec: AvenueSpec, resources: Resources) -> list[str]:
    """Controller-side preflight for explicitly required build capabilities."""
    missing: list[str] = []
    candidate_providers = set(resources.search.candidate_api_providers or ())
    for capability in spec.required_capabilities:
        if capability == "package-installs" and resources.search.allow_package_installs is not True:
            missing.append("third-party package installation is not confirmed")
        elif capability == "package-installs" and shutil.which("uv") is None:
            missing.append("uv is not installed, so isolated dependencies cannot be resolved")
        elif capability == "model-downloads" and resources.search.allow_model_downloads is not True:
            missing.append("pretrained model downloads are not confirmed")
        elif capability == "fine-tuning" and not resources.search.fine_tuning:
            missing.append("fine-tuning access is not available")
        elif capability == "gpu" and resources.search.gpu is None:
            missing.append("no search-time GPU is recorded")
        elif capability == "runtime-network" and resources.runtime.network is not True:
            missing.append("runtime network access is not confirmed")
        elif capability.startswith("candidate-api:"):
            provider = capability.split(":", 1)[1]
            if provider not in candidate_providers:
                missing.append(
                    f"candidate evaluation has no confirmed {provider!r} provider access"
                )
    lowered_requirements = " ".join(spec.runtime_requirements).lower()
    if any(token in lowered_requirements for token in ("gpu", "cuda")) and resources.search.gpu is None:
        missing.append(
            "this avenue explicitly requires GPU/CUDA but no search-time GPU is recorded"
        )
    return list(dict.fromkeys(missing))


_ENVIRONMENT_ERROR_MARKERS = (
    "api key", "api_key", "apikey", "credential", "authentication", "unauthorized",
    "no module named", "modulenotfounderror", "importerror", "cuda",
    "no gpu", "out of memory", "connection error", "network is unreachable",
    "name or service not known", "model not found", "package not found",
    "no solution found", "failed to resolve", "failed to download",
    "failed to build", "uv was not found", "uv is not installed", "permission denied",
)


def _environment_blocker_text(text: str) -> list[str]:
    lowered = str(text).lower()
    if any(marker in lowered for marker in _ENVIRONMENT_ERROR_MARKERS):
        compact = " ".join(str(text).split())
        return [compact[-1200:] or "worker reported an unavailable capability"]
    return []


def _complete_environment_failure(report) -> list[str]:
    """Return setup errors when every run failed for an environmental reason."""
    expected = int(report.n_rows) * int(report.n_repeats)
    if expected <= 0 or len(report.errors) < expected:
        return []
    details = [str(error) for error in report.errors]
    joined = "\n".join(details).lower()
    if not any(marker in joined for marker in _ENVIRONMENT_ERROR_MARKERS):
        return []
    return details[:8]


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
        max_compliance_repairs: int = 2,
        max_clean_restarts: int = 1,
        semantic_adherence_review: bool = True,
    ):
        if orchestrator_timeout <= 0 or worker_timeout <= 0:
            raise ValueError("Pi timeouts must be positive.")
        if max_compliance_repairs < 0 or max_clean_restarts < 0:
            raise ValueError("Compliance repair/restart counts cannot be negative.")
        self.command = tuple(command)
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.resources = resources
        self.strict_isolation = strict_isolation
        self.orchestrator_timeout = float(orchestrator_timeout)
        self.worker_timeout = float(worker_timeout)
        self.max_compliance_repairs = int(max_compliance_repairs)
        self.max_clean_restarts = int(max_clean_restarts)
        self.semantic_adherence_review = bool(semantic_adherence_review)

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
        # Old/resumed plans and Pi-authored plans cannot weaken a tier contract by
        # omitting its machine-readable constraints.
        for state in portfolio.avenues:
            state.spec = ensure_avenue_contract(state.spec, resources)
        human_retried: set[str] = set()
        for state in portfolio.avenues:
            if state.candidates or state.status != AvenueStatus.PLANNED:
                continue
            if state.human_retry_confirmed:
                human_retried.add(state.spec.id)
                state.notes.append(
                    "preflight retried after human confirmation that the blocked "
                    "capability was fixed or should be attempted again"
                )
                state.human_retry_confirmed = False
                continue
            missing = _missing_avenue_capabilities(state.spec, resources)
            if missing:
                state.record_blocker("environment-preflight", missing)
        portfolio.write(portfolio_path)

        worker = PiWorkerRunner(self.command, timeout=self.worker_timeout)
        runnable_states = [
            a for a in portfolio.avenues
            if not a.candidates
            and a.status in (AvenueStatus.PLANNED, AvenueStatus.READY)
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
                harness, spec, resources, worker,
                human_retry_confirmed=(spec.id in human_retried),
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
            pi_result, sandbox = item
            solution = sandbox / "solution.py"
            if not solution.exists():
                blocker = _environment_blocker_text(
                    f"{pi_result.text}\n{pi_result.stderr}"
                )
                if blocker:
                    state.record_blocker("worker-reported-environment-failure", blocker)
                else:
                    state.status = AvenueStatus.FAILED
                    state.notes.append("worker did not create solution.py")
                continue
            source = solution.read_text(encoding="utf-8")
            if not re.search(r"(?m)^def predict\s*\(", source):
                blocker = _environment_blocker_text(
                    f"{pi_result.text}\n{pi_result.stderr}\n{source}"
                )
                if blocker:
                    state.record_blocker("worker-reported-environment-failure", blocker)
                else:
                    state.status = AvenueStatus.FAILED
                    state.notes.append("solution.py does not define predict")
                continue
            try:
                source = self._ensure_adherent_solution(
                    harness, state, resources, worker, sandbox,
                    initial_task=(
                        "Implement the function described in task.md using only the "
                        "non-negotiable approach contract. Inspect examples.jsonl, "
                        "create solution.py, test what the environment permits, and "
                        "never substitute another approach family."
                    ),
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                # Fail closed: an unavailable auditor cannot turn unreviewed
                # source into a valid avenue or satisfy breadth. Resume later.
                state.status = AvenueStatus.PLANNED
                state.notes.append(f"approach-adherence review failed: {exc}")
                portfolio.write(portfolio_path)
                continue
            if source is None:
                portfolio.write(portfolio_path)
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
                environment_errors = _complete_environment_failure(train_report)
                if environment_errors:
                    state.record_blocker(
                        "candidate-environment-failure", environment_errors,
                        candidate=cand.name,
                    )
                    portfolio.write(portfolio_path)
                    continue
                val_report = harness.eval(cand.name)
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                blocker = _environment_blocker_text(str(exc))
                if blocker:
                    state.record_blocker("candidate-environment-failure", blocker)
                else:
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
        if portfolio.unresolved_blockers:
            self._print_human_blockers(portfolio, portfolio_path)
        if not any(a.candidates for a in portfolio.avenues):
            if not portfolio.unresolved_blockers:
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
            if portfolio.unresolved_blockers:
                self._print_human_blockers(portfolio, portfolio_path)
                return
            raise RunnerError(
                "Portfolio policy refused early finalization: feasible breadth or "
                "the required exploration reserve is incomplete. Resume the run "
                f"from {portfolio_path}."
            )
        harness.finalize()

    def _charged_rpc_prompt(self, ws, resources, prompt: str, *, system: str) -> PiResult:
        """Run one synchronous Pi review call with ordinary budget accounting."""
        ledger = BudgetLedger(ws.budget_json)
        ledger.check()
        token = None
        if (
            ledger.limits.get("dollars") is not None
            and resources.search.max_dollars_per_agent_call is not None
        ):
            token = ledger.reserve_dollars(
                resources.search.max_dollars_per_agent_call, category="agent"
            )
        try:
            with PiRpcClient(
                self.command,
                cwd=ws.root,
                model=self.orchestrator_model,
                system_prompt=system,
                timeout=self.orchestrator_timeout,
            ) as client:
                result = client.prompt(prompt)
            if token is None:
                ledger.charge(dollars=result.usage.dollars, category="agent")
            else:
                ledger.settle_reservation(token, dollars=result.usage.dollars)
            return result
        except Exception:
            if token is not None:
                ledger.release_reservation(token)
            raise

    def _charged_worker_turn(
        self, ws, resources, worker, root, task: str, *, session_id: str,
        allowed_api_providers: tuple[str, ...],
    ) -> PiResult:
        ledger = BudgetLedger(ws.budget_json)
        ledger.check()
        token = None
        if (
            ledger.limits.get("dollars") is not None
            and resources.search.max_dollars_per_agent_call is not None
        ):
            token = ledger.reserve_dollars(
                resources.search.max_dollars_per_agent_call, category="agent"
            )
        try:
            model = self.worker_model or (
                resources.search.pi_models[0] if resources.search.pi_models else None
            )
            result = worker.run(
                root, task, model=model, session_id=session_id,
                allowed_api_providers=allowed_api_providers,
            )
            if token is None:
                ledger.charge(dollars=result.usage.dollars, category="agent")
            else:
                ledger.settle_reservation(token, dollars=result.usage.dollars)
            return result
        except Exception:
            if token is not None:
                ledger.release_reservation(token)
            raise

    def _audit_solution(self, harness, state, resources, source: str) -> ApproachAudit:
        static = deterministic_audit(state.spec, source)
        state.record_audit(static.to_dict())
        if not static.adherent or not self.semantic_adherence_review:
            return static
        result = self._charged_rpc_prompt(
            harness.workspace,
            resources,
            semantic_audit_prompt(state.spec, source),
            system=(
                "You are an independent implementation-mechanism auditor. You do "
                "not solve the function and never judge its score. Reject any code "
                "path that substitutes another approach when required infrastructure "
                "is absent. Return only the requested JSON object."
            ),
        )
        audit = ApproachAudit.from_dict(_json_object(result.text), reviewer="pi")
        state.record_audit(audit.to_dict())
        return audit

    def _ensure_adherent_solution(
        self, harness, state, resources, worker, root: Path, *, initial_task: str,
    ) -> str | None:
        """Audit, repair, and if necessary clean-restart one avenue.

        Invalid source never reaches ``new_candidate``. A poor but faithful
        implementation may score zero; a cross-tier substitute is not an
        implementation of this avenue at all.
        """
        solution = root / "solution.py"
        repairs = 0
        restarts = 0
        while solution.exists():
            source = solution.read_text(encoding="utf-8")
            audit = self._audit_solution(harness, state, resources, source)
            if audit.adherent:
                return source

            detail = "; ".join(audit.violations) or "mechanism adherence was not demonstrated"
            state.notes.append(f"rejected noncompliant worker output: {detail}")
            if repairs < self.max_compliance_repairs:
                repairs += 1
                task = f"""Your current solution.py was rejected because it violates the
non-negotiable approach contract:
- {chr(10).join(audit.violations) or 'Required mechanism was not demonstrated.'}

Required repair:
{audit.repair_instructions or 'Implement the assigned mechanism as the only answer-producing path.'}

Edit solution.py in place. Do not preserve a fallback for safety. Missing
packages, credentials, GPU, models, or network must produce a precise failure;
they never justify another approach. Re-read task.md, keep the assigned
mechanism, and syntax-check the repaired file."""
                self._charged_worker_turn(
                    harness.workspace, resources, worker, root, task,
                    session_id=state.spec.id,
                    allowed_api_providers=state.spec.allowed_api_providers,
                )
                continue

            if restarts < self.max_clean_restarts:
                restarts += 1
                state.restart_count += 1
                repairs = 0
                solution.unlink(missing_ok=True)
                artifact_root = root / "artifacts" / state.spec.id
                if artifact_root.exists():
                    shutil.rmtree(artifact_root)
                task = (
                    initial_task
                    + "\n\nThis is a clean restart because a previous engineer "
                    "substituted another mechanism. Start from task.md; do not "
                    "recreate or retain any cross-family fallback."
                )
                self._charged_worker_turn(
                    harness.workspace, resources, worker, root, task,
                    session_id=f"{state.spec.id}-compliance-restart-{restarts}",
                    allowed_api_providers=state.spec.allowed_api_providers,
                )
                continue

            state.status = AvenueStatus.NONCOMPLIANT
            state.notes.append(
                "avenue exhausted compliance repairs/restarts; no source was "
                "imported or evaluated"
            )
            return None
        state.status = AvenueStatus.NONCOMPLIANT
        state.notes.append("worker did not leave solution.py after compliance repair")
        return None

    @staticmethod
    def _print_human_blockers(portfolio, portfolio_path) -> None:
        print(
            "[autoprogramming] one or more assigned approaches are completely "
            "blocked by environment/setup failures. They have NOT been replaced "
            "with fallback approaches and will not be discarded without human "
            "confirmation:"
        )
        for avenue in portfolio.unresolved_blockers:
            details = "; ".join((avenue.blocker or {}).get("details", []))
            print(f"  - {avenue.spec.id} ({avenue.spec.title}): {details}")
        print(
            "Fix the capability, then confirm a retry; or explicitly confirm that "
            "the approach is unavailable:\n"
            "    import autoprogramming as ap\n"
            "    prg = ap.attach(<workspace>)\n"
            "    prg.resolve_blocker('<avenue-id>', 'retry', confirmed_by='user')\n"
            "    # or, only after the user agrees:\n"
            "    prg.resolve_blocker('<avenue-id>', 'exclude', confirmed_by='user')\n"
            f"Then resume optimize(). State: {portfolio_path}"
        )

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
            and avenue.status not in (
                AvenueStatus.FAILED, AvenueStatus.INFEASIBLE,
                AvenueStatus.BLOCKED, AvenueStatus.NONCOMPLIANT,
            )
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
                "engineering pass that must push the SAME non-negotiable mechanism, "
                "not merely solve the function. Find weaknesses in generalization, "
                "edge cases, parsing, failure handling, startup, and repeated-call "
                "efficiency. Improve solution.py materially. Never add a fallback "
                "from another family for missing packages, credentials, GPU, model, "
                "or network; fail clearly instead. Syntax-check it."
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
            try:
                source = self._ensure_adherent_solution(
                    harness, avenue, resources, worker, solution.parent,
                    initial_task=(
                        "Re-implement the function in task.md as a materially "
                        "improved configuration of the exact same non-negotiable "
                        "approach. Never substitute another family."
                    ),
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                return
            except Exception as exc:
                avenue.notes.append(f"refinement adherence review failed: {exc}")
                continue
            if source is None:
                portfolio.write(portfolio_path)
                continue
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
                train = harness.eval(cand.name, split="train", per_instance=True)
                environment_errors = _complete_environment_failure(train)
                if environment_errors:
                    avenue.record_blocker(
                        "candidate-environment-failure", environment_errors,
                        candidate=cand.name,
                    )
                    portfolio.write(portfolio_path)
                    continue
                val = harness.eval(cand.name)
                comparison = harness.compare(baseline.name, cand.name)
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                return
            except Exception as exc:
                blocker = _environment_blocker_text(str(exc))
                if blocker:
                    avenue.record_blocker("candidate-environment-failure", blocker)
                else:
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
        spec = ensure_avenue_contract(AvenueSpec(
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
        ), resources)
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
            allowed_api_providers=tuple(
                provider for provider in resources.runtime.api_providers
                if provider in set(resources.search.candidate_api_providers or ())
            ),
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
            source = self._ensure_adherent_solution(
                harness, composed, resources, worker, root,
                initial_task=(
                    "Build the explicit composition in task.md from the supplied "
                    "components. Only this composition contract permits cross-family "
                    "routing; do not silently replace all components with a new family."
                ),
            )
            if source is None:
                portfolio.write(portfolio_path)
                return
            from .candidates import next_name

            expected_name = next_name(harness.workspace)
            composed.begin_candidate(expected_name)
            portfolio.write(portfolio_path)
            source = _materialize_bundle(
                source, root, harness.workspace, spec.id
            )
            cand = harness.new_candidate(source=source)
            if cand.name != expected_name:
                raise RunnerError(
                    f"Candidate journal expected {expected_name}, got {cand.name}."
                )
            train = harness.eval(cand.name, split="train", per_instance=True)
            environment_errors = _complete_environment_failure(train)
            if environment_errors:
                composed.record_blocker(
                    "candidate-environment-failure", environment_errors,
                    candidate=cand.name,
                )
                portfolio.write(portfolio_path)
                return
            val = harness.eval(cand.name)
            composed.record_result(
                cand.name,
                {name: float(obj["mean"]) for name, obj in val.objectives.items()},
                improved=cand.name in harness.tradeoffs().nondominated,
            )
        except Exception as exc:
            blocker = _environment_blocker_text(str(exc))
            if blocker:
                composed.record_blocker("candidate-environment-failure", blocker)
            else:
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
"runtime_requirements":[], "allowed_api_providers":[],
"required_capabilities":[], "required_mechanisms":["non-negotiable evidence"],
"forbidden_substitutions":["specific cross-family fallback"], "max_rounds":3,
"wildcard":false}}], "exclusions": {{"tier": "reason"}}}}
Every avenue is a hard mechanism experiment: missing packages, credentials, GPU,
models, or network may block it but must never justify another implementation
family. Specify what must be present and what substitutions are forbidden.
Schema:\n{harness.schema.describe()}\nResources:\n{json.dumps(resources.to_dict())}\nFeasibility:\n{json.dumps(feasibility)}"""
        try:
            with self._orchestrator(harness.workspace) as client:
                result = client.prompt(prompt)
            BudgetLedger(harness.workspace.budget_json).charge(
                dollars=result.usage.dollars, category="agent"
            )
            value = _json_object(result.text)
            specs = [
                ensure_avenue_contract(AvenueSpec.from_dict(v), resources)
                for v in value.get("avenues", [])
            ]
            return Portfolio.create(
                resources,
                specs,
                exclusions={int(k): str(v) for k, v in value.get("exclusions", {}).items()},
                fill_missing=True,
            )
        except Exception as exc:
            print(f"[autoprogramming] Pi portfolio plan was invalid ({exc}); filling a deterministic breadth-first portfolio.")
            return Portfolio.create(resources, [], fill_missing=True)

    def _run_avenue(
        self, harness, spec: AvenueSpec, resources: Resources,
        worker: PiWorkerRunner, *, human_retry_confirmed: bool = False,
    ):
        root = _avenue_dir(harness.workspace, spec.id)
        root.mkdir(parents=True, exist_ok=True)
        # Generic filenames and prose intentionally reveal no optimizer, candidate,
        # metric, score, split, or competing-agent context.
        task_doc = _task_document(harness.schema, spec, resources)
        if human_retry_confirmed:
            task_doc += (
                "\n## Human-confirmed retry\nA human reviewed the previous setup "
                "blocker and confirmed that the capability was fixed or that this "
                "approach must be attempted again. Treat that confirmation as newer "
                "than any stale detected-resource field above. Preserve the assigned "
                "mechanism; do not introduce a fallback.\n"
            )
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
            "Implement the function described in task.md through exactly its "
            "non-negotiable approach contract. Inspect examples.jsonl, create "
            "solution.py, and push that mechanism as far as possible. A working "
            "answer from another implementation family is invalid. If infrastructure "
            "is absent, preserve the mechanism and fail clearly rather than adding "
            "a fallback. Test what the environment permits and syntax-check the file."
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
