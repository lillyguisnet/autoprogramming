"""Honest numbers: seeded bootstrap statistics and the evaluate/compare loop.

Every aggregate carries a bootstrap confidence interval, stochastic
candidates are scored across repeats with the variance reported, and
comparisons are paired per row — "improved" means the interval excludes
zero, not that the point estimate went up.
"""

from __future__ import annotations

import importlib
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev, pvariance
from typing import TYPE_CHECKING, Sequence

from . import metric
from .budget import BudgetLedger
from .errors import DataDisciplineError, MemorizationWarning

if TYPE_CHECKING:
    from .workspace import Workspace

DEFAULT_REPEATS = 3


def _sibling(name: str):
    """Import a sibling module at call time (unit tests stub it in sys.modules)."""
    return importlib.import_module(f"autoprogramming.{name}")


@dataclass
class EvalReport:
    """One candidate's evaluation on one split, with uncertainty attached.

    ``per_row`` is populated only for ``split="train"`` with
    ``per_instance=True`` — per-row val scores are never returned to the
    caller, only aggregates (the harness persists them internally).
    """

    candidate: str
    split: str
    mean: float
    std: float
    ci95: tuple[float, float]
    n_rows: int
    n_repeats: int
    repeat_variance: float
    per_row: dict[str, float] | None
    per_field: dict[str, float] | None
    errors: list[str]
    flags: list[str]

    def __str__(self) -> str:
        lines = [
            f"{self.candidate} on {self.split}: {self.mean:.3f} ± {self.std:.3f} "
            f"(n={self.n_repeats} repeats), 95% CI "
            f"[{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]"
        ]
        lines.extend(self.flags)
        return "\n".join(lines)


@dataclass
class CompareReport:
    """A paired comparison of two candidates on stored per-row scores.

    ``a`` is the baseline, ``b`` the challenger; ``diff_mean`` is the mean of
    ``b - a`` per row. ``improved`` is True only when the 95% CI of that
    difference lies entirely above zero — the CI excludes zero in the
    challenger's favor.
    """

    a: str
    b: str
    split: str
    diff_mean: float
    ci95: tuple[float, float]
    improved: bool

    def __str__(self) -> str:
        verdict = (
            "improved — the CI excludes 0 in the challenger's favor"
            if self.improved
            else "not improved — the CI does not exclude 0; the difference may be noise"
        )
        return (
            f"{self.b} vs {self.a} on {self.split}: diff {self.diff_mean:+.3f}, "
            f"95% CI [{self.ci95[0]:.3f}, {self.ci95[1]:.3f}] — {verdict}"
        )


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Pure and seeded.

    Degenerate inputs degenerate honestly: an empty list gives ``(0.0, 0.0)``,
    a single value or a constant list gives ``(v, v)``.
    """
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1 or min(vals) == max(vals):
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI of the mean paired difference ``b - a``. Pure and seeded.

    Resampling the per-row differences is equivalent to resampling the pairs,
    which keeps row-level correlation intact.
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired bootstrap needs equal-length paired samples, got "
            f"{len(a)} and {len(b)}."
        )
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed, alpha=alpha)


def load_scores(workspace) -> dict:
    """Read scores.json, clearing it first if the metric changed underneath it.

    Scores under different metrics are never comparable, so a stored
    ``metric_sha`` that no longer matches the current metric.py archives the
    old scores and returns a fresh skeleton (same rule as
    ``metric.write_metric`` — whoever notices first clears).
    """
    current = metric.metric_sha(workspace)
    path = Path(workspace.scores_json)
    if not path.exists():
        return metric.scores_skeleton(current)
    scores = json.loads(path.read_text())
    if scores.get("metric_sha") != current:
        if scores.get("candidates"):
            metric.invalidate_scores(workspace, current)
            return metric.scores_skeleton(current)
        scores["metric_sha"] = current
    scores.setdefault("candidates", {})
    scores.setdefault("val_scored", [])
    scores.setdefault("flags", {})
    return scores


def save_scores(workspace, scores: dict) -> None:
    """Write scores.json."""
    Path(workspace.scores_json).write_text(json.dumps(scores, indent=2) + "\n")


def evaluate(
    workspace,
    candidate_name: str,
    split: str = "val",
    per_instance: bool = False,
    n_repeats: int | None = None,
) -> EvalReport:
    """Score one candidate on one split and persist the result.

    Order is load-bearing: data-discipline guards run before anything else,
    metric approval before any candidate is loaded, val selection pressure is
    registered before any spend, and the budget is checked before every row —
    a mid-eval BudgetExceededError propagates with nothing persisted for the
    partial eval. Per-row score maps are persisted for both train and val, but
    ``per_row`` is only returned for train with ``per_instance=True``.
    """
    guards = _sibling("guards")
    guards.assert_eval_allowed(split, per_instance)

    metric.ensure_approved(workspace)
    metric_fn = metric.load_metric(workspace)
    approval = json.loads(Path(workspace.metric_approval).read_text())
    weights = approval.get("weights")

    candidates_mod = _sibling("candidates")
    data_mod = _sibling("data")
    candidate = candidates_mod.load_candidate(workspace, candidate_name)
    rows = data_mod.load_split(workspace, split)

    if split == "val":
        # A metric change must be noticed (archive + reset, on disk) BEFORE
        # the val registration is written: guards appends to scores.json as it
        # stands, and a stale metric_sha there would make the later
        # load_scores() archive the registration along with the old scores —
        # silently dropping this candidate from val_scored.
        load_scores(workspace)
        guards.register_val_candidate(workspace, candidate.name)

    if n_repeats is not None:
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be at least 1, got {n_repeats!r}.")
        reps = n_repeats
    else:
        reps = 1 if candidate.deterministic else DEFAULT_REPEATS

    runner_mod = _sibling("runner")
    ledger = BudgetLedger(workspace.budget_json)
    schema = workspace.schema

    row_means: dict[str, float] = {}
    row_variances: list[float] = []
    field_scores: dict[str, list[float]] = {}
    run_errors: list[str] = []

    for i, row in enumerate(rows):
        row_id = f"row_{i}"
        ledger.check()
        inputs = schema.coerce_inputs(row)
        expected = schema.coerce_expected(row)
        repeat_scores: list[float] = []
        for rep in range(1, reps + 1):
            run = runner_mod.run_candidate(workspace, candidate, inputs)
            ledger.charge(eval_calls=1, dollars=run.cost_dollars or 0.0)
            if run.ok:
                score, per_field = metric.score_pair(
                    metric_fn, schema, run.outputs, expected, weights
                )
                if per_field:
                    for fname, fscore in per_field.items():
                        field_scores.setdefault(fname, []).append(fscore)
            else:
                score = 0.0
                first = (run.error or "candidate failed").strip().splitlines()[0]
                run_errors.append(f"{row_id} repeat {rep}: {first}")
            repeat_scores.append(score)
        row_means[row_id] = fmean(repeat_scores)
        row_variances.append(pvariance(repeat_scores))

    means = list(row_means.values())
    mean = fmean(means) if means else 0.0
    std = pstdev(means) if means else 0.0
    ci95 = bootstrap_ci(means)
    repeat_variance = fmean(row_variances) if row_variances else 0.0
    per_field_agg = (
        {name: fmean(vals) for name, vals in field_scores.items()}
        if field_scores
        else None
    )

    scores = load_scores(workspace)
    entry = scores["candidates"].setdefault(candidate.name, {})
    entry[split] = {
        "rows": dict(row_means),
        "mean": mean,
        "std": std,
        "ci95": [ci95[0], ci95[1]],
        "n_repeats": reps,
        "repeat_variance": repeat_variance,
        "per_field": per_field_agg,
    }

    flags: list[str] = []
    if "train" in entry and "val" in entry:
        train_rows = rows if split == "train" else data_mod.load_split(workspace, "train")
        flags = list(
            guards.memorization_check(
                candidate.source,
                entry["train"]["mean"],
                entry["val"]["mean"],
                train_rows,
                schema,
            )
        )
        if flags:
            scores["flags"][candidate.name] = flags
        else:
            scores["flags"].pop(candidate.name, None)
    save_scores(workspace, scores)
    if flags:
        warnings.warn("; ".join(flags), MemorizationWarning, stacklevel=2)

    return EvalReport(
        candidate=candidate.name,
        split=split,
        mean=mean,
        std=std,
        ci95=ci95,
        n_rows=len(rows),
        n_repeats=reps,
        repeat_variance=repeat_variance,
        per_row=dict(row_means) if (split == "train" and per_instance) else None,
        per_field=per_field_agg,
        errors=run_errors,
        flags=flags,
    )


def _row_index(row_id: str) -> int:
    try:
        return int(row_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def compare(workspace, a: str, b: str, split: str = "val") -> CompareReport:
    """Compare two candidates using per-row scores already stored in scores.json.

    Rows are paired by row id, so the difference is a paired statistic —
    row-to-row difficulty cancels out. Both candidates must have been
    evaluated on the split first; comparisons never trigger new spend.
    """
    scores = load_scores(workspace)
    stored = scores.get("candidates", {})
    row_maps: dict[str, dict] = {}
    for name in (a, b):
        rows = stored.get(name, {}).get(split, {}).get("rows")
        if not rows:
            raise DataDisciplineError(
                f"Cannot compare {a!r} vs {b!r} on {split!r}: {name!r} has no "
                f"stored per-row {split} scores. Comparisons pair the per-row "
                f"scores recorded by evaluate() so the difference is measured on "
                f"identical rows; run prg.eval({name!r}, split={split!r}) first."
            )
        row_maps[name] = rows
    common = sorted(set(row_maps[a]) & set(row_maps[b]), key=_row_index)
    if not common:
        raise DataDisciplineError(
            f"Cannot compare {a!r} vs {b!r} on {split!r}: their stored per-row "
            f"scores share no row ids, so no paired difference exists. Re-evaluate "
            f"both candidates on the current {split} split."
        )
    a_vals = [float(row_maps[a][r]) for r in common]
    b_vals = [float(row_maps[b][r]) for r in common]
    diff_mean = fmean(bv - av for av, bv in zip(a_vals, b_vals))
    ci95 = paired_bootstrap_diff(a_vals, b_vals)
    return CompareReport(
        a=a, b=b, split=split, diff_mean=diff_mean, ci95=ci95, improved=ci95[0] > 0
    )
