"""The evaluation metric: proposed, demonstrated, approved — never silently trusted.

The entire search optimizes whatever the workspace's ``metric.py`` says, so a
wrong metric produces a confidently-scored wrong program. This module owns the
metric file's lifecycle: writing it, hashing it, recording user sign-off, and
invalidating recorded scores when the metric changes — scores under different
metrics are never comparable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .errors import MetricChangedError, MetricNotApprovedError, SchemaError

if TYPE_CHECKING:
    from .schema import Schema
    from .workspace import Workspace

AUTO_APPROVE_ENV = "AP_AUTO_APPROVE_METRIC"
_TRUTHY = ("1", "true", "yes")

_METRIC_CACHE: dict[tuple[str, str], Callable] = {}


_STAMP_PREFIX = "# metric.py — approved by"


def _without_stamp(code: str) -> str:
    """The metric code with any approval-stamp first line removed.

    The stamp is sign-off metadata, not metric semantics: approving a metric
    (or re-approving it on a later date) must never look like the metric
    itself changed, or the very act of blessing a metric would invalidate
    the scores recorded under it.
    """
    lines = code.splitlines()
    if lines and lines[0].startswith(_STAMP_PREFIX):
        lines = lines[1:]
    return "\n".join(lines)


def metric_sha(workspace) -> str | None:
    """sha256 hex digest of metric.py's code, or None if absent.

    The approval stamp line is excluded from the hash — see
    :func:`_without_stamp` — so the sha changes exactly when the scoring
    behavior can change.
    """
    path = Path(workspace.metric_py)
    if not path.exists():
        return None
    code = _without_stamp(path.read_text(encoding="utf-8"))
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def scores_skeleton(sha: str | None) -> dict:
    """The empty scores.json structure, tied to the given metric sha."""
    return {"metric_sha": sha, "candidates": {}, "val_scored": [], "flags": {}}


def invalidate_scores(workspace, new_sha: str | None) -> None:
    """Archive scores.json to scores.archive/<k>.json and reset it to a skeleton.

    Called when the metric changes — by :func:`write_metric` here and by
    ``scoring.load_scores`` (whoever notices the change first clears). Scores
    recorded under different metrics are never comparable, so keeping them
    live would let a metric edit silently rewrite history; archiving preserves
    the old numbers for forensics while making clear they no longer count.
    """
    path = Path(workspace.scores_json)
    old: dict | None = None
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = None
    if old and old.get("candidates"):
        archive_dir = Path(workspace.root) / "scores.archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        k = 0
        while (archive_dir / f"{k}.json").exists():
            k += 1
        dest = archive_dir / f"{k}.json"
        dest.write_text(json.dumps(old, indent=2) + "\n")
        msg = (
            f"metric.py changed: {len(old['candidates'])} candidate score set(s) "
            f"recorded under the previous metric were archived to {dest} and "
            f"scores.json was reset. Scores under different metrics are never "
            f"comparable — re-approve the new metric, then re-evaluate candidates "
            f"under it."
        )
        print(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    path.write_text(json.dumps(scores_skeleton(new_sha), indent=2) + "\n")


def write_metric(workspace, code: str) -> None:
    """Write metric.py; if its code changes, archive and clear recorded scores.

    A changed metric also voids any prior approval implicitly: the approval
    record pins the old sha, so :func:`ensure_approved` will refuse until the
    new metric is signed off again. Proposing code identical to what is
    already on disk (approval stamp aside) is a no-op that keeps the stamped
    file, the approval, and every recorded score — re-proposing the same
    metric in a fresh session must not wipe anything.
    """
    path = Path(workspace.metric_py)
    if path.exists() and _without_stamp(path.read_text(encoding="utf-8")) == _without_stamp(code):
        return
    old = metric_sha(workspace)
    path.write_text(code)
    new = metric_sha(workspace)
    if old is not None and old != new:
        invalidate_scores(workspace, new)


def _read_approval(workspace) -> dict | None:
    path = Path(workspace.metric_approval)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def is_approved(workspace) -> bool:
    """True when an approval record exists and matches the current metric.py sha."""
    sha = metric_sha(workspace)
    if sha is None:
        return False
    approval = _read_approval(workspace)
    return approval is not None and approval.get("sha") == sha


def ensure_approved(workspace) -> None:
    """Refuse to proceed unless the current metric.py carries a valid sign-off.

    This is the only place the ``AP_AUTO_APPROVE_METRIC`` environment variable
    is consulted: a truthy value ("1"/"true"/"yes") auto-approves an
    unapproved metric for unattended runs. A metric whose approval sha no
    longer matches is never auto-approved — it changed after sign-off.
    """
    sha = metric_sha(workspace)
    if sha is None:
        raise MetricNotApprovedError(
            "Refused to score: this workspace has no metric.py. The entire search "
            "optimizes whatever metric.py says, so scoring without a metric would "
            "optimize an undefined objective. Propose one with "
            "harness.propose_metric(code, examples), or write it with "
            "autoprogramming.metric.write_metric(ws, code) and approve it."
        )
    approval = _read_approval(workspace)
    if approval is not None:
        if approval.get("sha") == sha:
            return
        raise MetricChangedError(
            f"metric.py changed after it was approved (approved sha "
            f"{str(approval.get('sha'))[:8]}…, on disk {sha[:8]}…). Scores under "
            f"different metrics are never comparable, so the old sign-off is void. "
            f"Demonstrate the new metric on real examples and re-approve it via "
            f"harness.propose_metric(...) or autoprogramming.metric.approve(ws, "
            f"'your name')."
        )
    if os.environ.get(AUTO_APPROVE_ENV, "").strip().lower() in _TRUTHY:
        approve(workspace, f"auto ({AUTO_APPROVE_ENV})")
        return
    raise MetricNotApprovedError(
        "metric.py exists but has not been approved. The entire search optimizes "
        "whatever metric.py says — a wrong metric produces a confidently-scored "
        "wrong program — so scoring is refused until someone signs off. Show the "
        "metric scoring real examples (harness.propose_metric(code, examples) "
        "runs that conversation), or approve directly with "
        "autoprogramming.metric.approve(ws, 'your name'). Unattended runs may "
        f"set {AUTO_APPROVE_ENV}=1 to skip the sign-off deliberately."
    )


def approve(workspace, approved_by: str, weights: dict | None = None) -> None:
    """Record sign-off for the current metric.py.

    Stamps the file's first line with ``# metric.py — approved by {who} on
    {date}`` (replacing any previous stamp) and writes metric_approval.json
    pinning the resulting sha, so any later edit to the file is detectable.
    ``weights`` records the per-field aggregate weighting for multi-output
    programs — that weighting is part of the sign-off.
    """
    path = Path(workspace.metric_py)
    if not path.exists():
        raise MetricNotApprovedError(
            "There is no metric.py to approve in this workspace. Approval records "
            "a sign-off on a concrete metric file, so an approval without one "
            "would be meaningless — write the metric first (write_metric or "
            "harness.propose_metric), then approve it."
        )
    now = datetime.now(timezone.utc)
    header = f"{_STAMP_PREFIX} {approved_by} on {now:%Y-%m-%d}"
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith(_STAMP_PREFIX):
        lines[0] = header
    else:
        lines.insert(0, header)
    path.write_text("\n".join(lines) + "\n")
    record = {
        "sha": metric_sha(workspace),
        "approved_by": approved_by,
        "approved_at": now.isoformat(),
        "weights": weights,
    }
    Path(workspace.metric_approval).write_text(json.dumps(record, indent=2) + "\n")


def load_metric(workspace) -> Callable:
    """Import the workspace's metric.py by path and return its metric() callable.

    Cached by (path, sha), so editing the file yields a fresh import. Loading
    does not require approval — :func:`demonstrate` uses it during the
    sign-off conversation; scoring goes through :func:`ensure_approved` first.
    """
    path = Path(workspace.metric_py)
    sha = metric_sha(workspace)
    if sha is None:
        raise MetricNotApprovedError(
            "Cannot load a metric: this workspace has no metric.py. Write one "
            "with autoprogramming.metric.write_metric(ws, code) or propose one "
            "via harness.propose_metric(code, examples)."
        )
    key = (str(path.resolve()), sha)
    cached = _METRIC_CACHE.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(f"_ap_metric_{sha[:16]}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SchemaError(
            f"metric.py failed to import ({exc!r}). The metric must be a "
            f"self-contained module defining metric(predicted, expected); fix "
            f"the error, then re-approve the file."
        ) from exc
    fn = getattr(module, "metric", None)
    if not callable(fn):
        raise SchemaError(
            "metric.py does not define a callable named metric. The scoring "
            "contract is metric(predicted, expected) returning a float (or a "
            "per-field dict for multi-output programs); define that function "
            "and re-approve the file."
        )
    _METRIC_CACHE[key] = fn
    return fn


def demonstrate(workspace, examples: list[tuple]) -> list[dict]:
    """Score (predicted, expected) pairs with the CURRENT metric.py.

    Works whether or not the metric is approved — this is what feeds the
    sign-off conversation, where the user checks the scores against their
    intuition of "good" before anything optimizes them.
    """
    fn = load_metric(workspace)
    return [
        {"predicted": predicted, "expected": expected, "score": fn(predicted, expected)}
        for predicted, expected in examples
    ]


def score_pair(
    metric_fn: Callable,
    schema,
    predicted: dict,
    expected: dict,
    weights: dict | None = None,
) -> tuple[float, dict | None]:
    """Score one prediction against one expected row.

    Single-output programs call ``metric_fn(predicted_value, expected_value)``
    and expect a number; per-field scores are None. Multi-output programs call
    ``metric_fn(predicted_dict, expected_dict)`` and expect a dict keyed by
    output names; the aggregate is the weighted mean (``weights`` from the
    approval record, defaulting to 1.0 per field). Returns
    ``(aggregate, per_field_dict_or_None)``.
    """
    names = schema.output_names
    if len(names) == 1:
        name = names[0]
        raw = metric_fn(predicted[name], expected[name])
        try:
            return float(raw), None
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"metric() returned {raw!r} for single-output program "
                f"{schema.name!r}; the metric contract for a single output is one "
                f"numeric score, so aggregation stays well-defined. Return a float."
            ) from exc
    raw = metric_fn(dict(predicted), dict(expected))
    if not isinstance(raw, dict):
        raise SchemaError(
            f"metric() returned {type(raw).__name__} for multi-output program "
            f"{schema.name!r}; the metric contract for multiple outputs is a dict "
            f"keyed by output names {list(names)} so every field is scored "
            f"explicitly. Return e.g. {{{names[0]!r}: 0.9, ...}}."
        )
    missing = [n for n in names if n not in raw]
    if missing:
        raise SchemaError(
            f"metric() result is missing scores for {missing!r}; the metric "
            f"contract for multi-output programs is one score per output name "
            f"({list(names)}) so no field silently drops out of the aggregate. "
            f"Add the missing fields."
        )
    per_field = {n: float(raw[n]) for n in names}
    w = {n: float((weights or {}).get(n, 1.0)) for n in names}
    total = sum(w.values())
    if total <= 0:
        raise SchemaError(
            "The approved metric weights sum to zero, so the aggregate score "
            "would be undefined; give at least one output a positive weight "
            "(weights are part of the metric sign-off)."
        )
    aggregate = sum(per_field[n] * w[n] for n in names) / total
    return aggregate, per_field
