"""Resource-aware, breadth-first portfolios of implementation avenues.

The language model may suggest avenues, but this module is the deterministic
policy layer: it validates coverage, prevents early convergence, and records
why an approach family was excluded.  The orchestrator cannot waive these
rules merely because one local idea already has an acceptable score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum

from .resources import Resources


class ApproachTier(IntEnum):
    GENERALIST_AGENT = 1
    MODEL_GRAPH = 2
    SINGLE_MODEL_CALL = 3
    FINETUNED_MODEL = 4
    SPECIALIZED_DEEP_MODEL = 5
    CLASSICAL_ML = 6
    CODE_AND_RULES = 7
    COMPOSITION = 8


TIER_LABELS = {
    ApproachTier.GENERALIST_AGENT: "runtime generalist or coding agent",
    ApproachTier.MODEL_GRAPH: "graph of model calls",
    ApproachTier.SINGLE_MODEL_CALL: "single model call",
    ApproachTier.FINETUNED_MODEL: "fine-tuned language model",
    ApproachTier.SPECIALIZED_DEEP_MODEL: "specialized pretrained/deep model",
    ApproachTier.CLASSICAL_ML: "classical machine learning",
    ApproachTier.CODE_AND_RULES: "algorithms, features, and rules",
    ApproachTier.COMPOSITION: "cross-tier cascade, router, ensemble, or pipeline",
}


class AvenueStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    READY = "ready"
    EVALUATED = "evaluated"
    STAGNANT = "stagnant"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class AvenueSpec:
    """One independent hypothesis an implementation-only worker will pursue."""

    id: str
    tier: ApproachTier
    title: str
    hypothesis: str
    implementation_brief: str
    mechanism: str
    runtime_requirements: tuple[str, ...] = ()
    allowed_api_providers: tuple[str, ...] = ()
    max_rounds: int = 3
    wildcard: bool = False
    compose_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", ApproachTier(int(self.tier)))
        object.__setattr__(self, "runtime_requirements", tuple(self.runtime_requirements))
        object.__setattr__(self, "allowed_api_providers", tuple(self.allowed_api_providers))
        object.__setattr__(self, "compose_from", tuple(self.compose_from))
        if not self.id or not self.id.replace("-", "_").isidentifier():
            raise ValueError(f"Invalid avenue id {self.id!r}.")
        if not all((self.title.strip(), self.hypothesis.strip(), self.mechanism.strip())):
            raise ValueError(f"Avenue {self.id!r} needs a title, hypothesis, and mechanism.")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")

    @property
    def fingerprint(self) -> str:
        normalized = " ".join(self.mechanism.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tier"] = int(self.tier)
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "AvenueSpec":
        accepted = {
            key: value[key]
            for key in (
                "id", "tier", "title", "hypothesis", "implementation_brief",
                "mechanism", "runtime_requirements", "allowed_api_providers",
                "max_rounds", "wildcard", "compose_from",
            )
            if key in value
        }
        return cls(**accepted)


@dataclass
class AvenueState:
    spec: AvenueSpec
    status: AvenueStatus = AvenueStatus.PLANNED
    rounds: int = 0
    no_progress_rounds: int = 0
    candidates: list[str] = field(default_factory=list)
    pending_candidate: str | None = None
    last_objectives: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def begin_candidate(self, candidate: str) -> None:
        """Journal an imported candidate before evaluation starts."""
        self.pending_candidate = str(candidate)
        self.status = AvenueStatus.READY

    def record_result(self, candidate: str, objectives: dict[str, float], improved: bool) -> None:
        self.rounds += 1
        if candidate not in self.candidates:
            self.candidates.append(candidate)
        if self.pending_candidate == candidate:
            self.pending_candidate = None
        self.last_objectives = dict(objectives)
        self.status = AvenueStatus.EVALUATED
        self.no_progress_rounds = 0 if improved else self.no_progress_rounds + 1
        if self.no_progress_rounds >= 2 or self.rounds >= self.spec.max_rounds:
            self.status = AvenueStatus.STAGNANT

    @classmethod
    def from_dict(cls, value: dict) -> "AvenueState":
        return cls(
            spec=AvenueSpec.from_dict(value["spec"]),
            status=AvenueStatus(value.get("status", AvenueStatus.PLANNED.value)),
            rounds=int(value.get("rounds", 0)),
            no_progress_rounds=int(value.get("no_progress_rounds", 0)),
            candidates=list(value.get("candidates", [])),
            pending_candidate=(
                str(value["pending_candidate"])
                if value.get("pending_candidate") is not None
                else None
            ),
            last_objectives={
                str(k): float(v) for k, v in value.get("last_objectives", {}).items()
            },
            notes=list(value.get("notes", [])),
        )

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "rounds": self.rounds,
            "no_progress_rounds": self.no_progress_rounds,
            "candidates": list(self.candidates),
            "pending_candidate": self.pending_candidate,
            "last_objectives": dict(self.last_objectives),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class PortfolioPolicy:
    """Controller-enforced allocation and stopping policy."""

    breadth_fraction: float = 0.4
    deepening_fraction: float = 0.4
    composition_fraction: float = 0.2
    min_configs_before_abandon: int = 2
    stagnation_rounds: int = 2
    require_wildcard: bool = True

    def __post_init__(self) -> None:
        total = self.breadth_fraction + self.deepening_fraction + self.composition_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Portfolio budget fractions must sum to 1.0, got {total}.")
        if min(
            self.breadth_fraction,
            self.deepening_fraction,
            self.composition_fraction,
        ) < 0:
            raise ValueError("Portfolio budget fractions cannot be negative.")
        if self.min_configs_before_abandon < 1 or self.stagnation_rounds < 1:
            raise ValueError("Portfolio stopping counts must be positive.")


@dataclass
class Portfolio:
    resources: Resources
    avenues: list[AvenueState]
    exclusions: dict[int, str]
    policy: PortfolioPolicy = field(default_factory=PortfolioPolicy)

    @classmethod
    def create(
        cls,
        resources: Resources,
        specs: list[AvenueSpec],
        *,
        exclusions: dict[int, str] | None = None,
        policy: PortfolioPolicy | None = None,
        fill_missing: bool = True,
    ) -> "Portfolio":
        exclusions = {int(k): str(v) for k, v in (exclusions or {}).items()}
        specs = list(specs)
        effective_policy = policy or PortfolioPolicy()
        if fill_missing:
            present = {int(s.tier) for s in specs}
            for tier, fact in resources.feasibility().items():
                if fact["feasible"] and tier not in present and tier != 8:
                    specs.append(default_avenue(ApproachTier(tier), resources))
                elif not fact["feasible"]:
                    exclusions.setdefault(tier, str(fact["reason"]))
            if effective_policy.require_wildcard and not any(s.wildcard for s in specs):
                specs.append(AvenueSpec(
                    id="wildcard",
                    tier=ApproachTier.CODE_AND_RULES,
                    title="Non-obvious wildcard",
                    hypothesis=(
                        "A task-specific mechanism outside the obvious baselines "
                        "may expose a missed quality/cost region."
                    ),
                    implementation_brief=(
                        "Invent one non-obvious algorithmic or hybrid mechanism "
                        "that is materially different from the other assigned avenues."
                    ),
                    mechanism="task-specific wildcard mechanism unlike the planned families",
                    wildcard=True,
                ))
        result = cls(
            resources=resources,
            avenues=[AvenueState(spec=s) for s in specs],
            exclusions=exclusions,
            policy=effective_policy,
        )
        result.validate()
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "Portfolio":
        result = cls(
            resources=Resources.from_dict(value["resources"]),
            avenues=[AvenueState.from_dict(v) for v in value.get("avenues", [])],
            exclusions={int(k): str(v) for k, v in value.get("exclusions", {}).items()},
            policy=PortfolioPolicy(**value.get("policy", {})),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path) -> "Portfolio":
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        ids = [a.spec.id for a in self.avenues]
        if len(ids) != len(set(ids)):
            raise ValueError("Portfolio avenue ids must be unique.")
        fingerprints: dict[tuple[int, str], str] = {}
        for avenue in self.avenues:
            key = (int(avenue.spec.tier), avenue.spec.fingerprint)
            if key in fingerprints:
                raise ValueError(
                    f"Avenues {fingerprints[key]!r} and {avenue.spec.id!r} repeat "
                    "the same mechanism within one tier."
                )
            fingerprints[key] = avenue.spec.id

        feasibility = self.resources.feasibility()
        represented = {int(a.spec.tier) for a in self.avenues}
        missing = [
            tier for tier, info in feasibility.items()
            if tier <= 7 and info["feasible"] and tier not in represented
        ]
        if missing:
            raise ValueError(f"Portfolio omitted feasible approach tiers: {missing}.")
        unexplained = [
            tier for tier, info in feasibility.items()
            if tier <= 7 and not info["feasible"] and tier not in self.exclusions
        ]
        if unexplained:
            raise ValueError(f"Portfolio exclusions need reasons for tiers: {unexplained}.")

    @property
    def breadth_complete(self) -> bool:
        if any(avenue.pending_candidate for avenue in self.avenues):
            return False
        represented = {
            int(a.spec.tier)
            for a in self.avenues
            if a.status not in (AvenueStatus.PLANNED, AvenueStatus.RUNNING)
        }
        return all(
            not info["feasible"] or tier > 7 or tier in represented
            for tier, info in self.resources.feasibility().items()
        )

    @property
    def may_finalize(self) -> bool:
        if any(avenue.pending_candidate for avenue in self.avenues):
            return False
        if not self.breadth_complete:
            return False
        if self.policy.require_wildcard:
            wildcards = [a for a in self.avenues if a.spec.wildcard]
            if not wildcards or not any(
                a.status not in (AvenueStatus.PLANNED, AvenueStatus.RUNNING)
                for a in wildcards
            ):
                return False
        return any(a.status in (AvenueStatus.EVALUATED, AvenueStatus.STAGNANT) for a in self.avenues)

    def to_dict(self) -> dict:
        return {
            "resources": self.resources.to_dict(),
            "policy": asdict(self.policy),
            "exclusions": {str(k): v for k, v in sorted(self.exclusions.items())},
            "avenues": [a.to_dict() for a in self.avenues],
            "breadth_complete": self.breadth_complete,
            "may_finalize": self.may_finalize,
        }

    def write(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


_DEFAULTS = {
    ApproachTier.GENERALIST_AGENT: (
        "runtime-agent", "Runtime reasoning agent",
        "A generalist tool-using agent can solve ambiguous cases dynamically.",
        "Implement predict using one runtime coding/reasoning agent with a tightly scoped task prompt.",
        "one tool-using runtime agent with explicit termination and output parsing",
    ),
    ApproachTier.MODEL_GRAPH: (
        "model-graph", "Decomposed model graph",
        "Planning, solving, and checking in separate calls may outperform one monolithic prompt.",
        "Implement a small graph of model calls with explicit intermediate contracts and a final verifier.",
        "planner-solver-verifier graph with bounded calls",
    ),
    ApproachTier.SINGLE_MODEL_CALL: (
        "single-model", "Single model call",
        "A strong task-specific prompt may provide the best quality/cost balance.",
        "Implement one model call with robust formatting, parsing, retries, and honest cost reporting.",
        "single current model call with task-specific prompt and parser",
    ),
    ApproachTier.FINETUNED_MODEL: (
        "finetuned-model", "Fine-tuned model",
        "Task examples may support a compact specialized fine-tune.",
        "Build or configure a fine-tuned model and provide a lazy runtime wrapper.",
        "supervised fine-tune with held-out build validation",
    ),
    ApproachTier.SPECIALIZED_DEEP_MODEL: (
        "specialized-model", "Specialized pretrained model",
        "A task-specific pretrained architecture may solve the task locally and cheaply.",
        "Research a current specialized model, verify it loads, and implement lazy local inference.",
        "current task-specialized pretrained model used as a pipeline stage",
    ),
    ApproachTier.CLASSICAL_ML: (
        "classical-ml", "Classical machine learning",
        "Compact learned features may capture the task without a generative model.",
        "Train a classical model on the provided examples; persist artifacts and implement inference.",
        "cross-validated classical estimator with task-appropriate sparse or engineered features",
    ),
    ApproachTier.CODE_AND_RULES: (
        "code-rules", "Algorithms and rules",
        "Direct algorithms and carefully generalized rules may be fastest and most reliable.",
        "Implement a dependency-light algorithmic baseline; generalize patterns rather than memorizing examples.",
        "stdlib algorithm and generalized feature/rule system",
    ),
    ApproachTier.COMPOSITION: (
        "composition", "Cross-tier composition",
        "Complementary implementations may form a better cascade or router than either alone.",
        "Build a bounded cascade, router, pipeline, or ensemble from complementary mechanisms.",
        "confidence-aware router or cascade across complementary implementation families",
    ),
}


def default_avenue(tier: ApproachTier, resources: Resources) -> AvenueSpec:
    avenue_id, title, hypothesis, brief, mechanism = _DEFAULTS[tier]
    providers = resources.runtime.api_providers if tier in (
        ApproachTier.GENERALIST_AGENT,
        ApproachTier.MODEL_GRAPH,
        ApproachTier.SINGLE_MODEL_CALL,
    ) else ()
    return AvenueSpec(
        id=avenue_id,
        tier=tier,
        title=title,
        hypothesis=hypothesis,
        implementation_brief=brief,
        mechanism=mechanism,
        allowed_api_providers=providers,
    )
