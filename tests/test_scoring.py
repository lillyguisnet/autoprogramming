"""Tests for autoprogramming.scoring — bootstrap stats, evaluate order, compare.

Sibling modules that other groups own (guards, data, candidates, runner) are
stubbed into sys.modules per-test, so this file passes standalone.
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming import metric, scoring
from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.errors import (
    BudgetExceededError,
    DataDisciplineError,
    MemorizationWarning,
    MetricNotApprovedError,
)
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


class Answer(str):
    """Direct answer to the question, one sentence."""


class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""


def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""


SHOUT_SCHEMA = Schema.from_function(shout)
QA_SCHEMA = Schema.from_function(qa)

EXACT = 'def metric(predicted, expected):\n    return 1.0 if predicted == expected else 0.0\n'
MULTI_METRIC = (
    "def metric(predicted, expected):\n"
    "    answer = 1.0 if predicted['Answer'] == expected['Answer'] else 0.0\n"
    "    conf = 1.0 - abs(float(predicted['Confidence']) - float(expected['Confidence']))\n"
    "    return {'Answer': answer, 'Confidence': conf}\n"
)
CAND_SOURCE = 'def predict(text):\n    return text.upper() + "!"\n'

SPLITS = {
    "train": [
        {"text": "hi", "Loud": "HI!"},
        {"text": "yo", "Loud": "YO!"},
        {"text": "bad", "Loud": "WRONG"},
    ],
    "val": [
        {"text": "ok", "Loud": "OK!"},
        {"text": "go", "Loud": "GO!"},
    ],
}


class FakeWorkspace:
    """Duck-typed stand-in for workspace.Workspace: paths plus a schema."""

    def __init__(self, root: Path, schema: Schema):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.schema = schema
        self.metric_py = root / "metric.py"
        self.metric_approval = root / "metric_approval.json"
        self.scores_json = root / "scores.json"
        self.budget_json = root / "budget.json"


def good_run(inputs):
    return SimpleNamespace(
        ok=True, outputs={"Loud": inputs["text"].upper() + "!"}, error=None,
        cost_dollars=None,
    )


def install_fakes(monkeypatch, *, splits=SPLITS, candidate=None, run_fn=good_run,
                  memo_flags=None, calls=None):
    """Stub guards/data/candidates/runner in sys.modules for one test."""
    guards = types.ModuleType("autoprogramming.guards")

    def assert_eval_allowed(split, per_instance):
        if split == "test":
            raise DataDisciplineError("test belongs to finalize()")
        if split not in ("train", "val"):
            raise DataDisciplineError(f"unknown split {split!r}")
        if split == "val" and per_instance:
            raise DataDisciplineError("per-row val scores are refused")

    def register_val_candidate(ws, name):
        s = scoring.load_scores(ws)
        if name not in s["val_scored"]:
            s["val_scored"].append(name)
            scoring.save_scores(ws, s)

    guards.assert_eval_allowed = assert_eval_allowed
    guards.register_val_candidate = register_val_candidate
    guards.memorization_check = (
        lambda source, train_mean, val_mean, train_rows, schema: list(memo_flags or [])
    )

    data = types.ModuleType("autoprogramming.data")
    data.load_split = lambda ws, split: [dict(r) for r in splits[split]]

    cands = types.ModuleType("autoprogramming.candidates")
    cands.load_candidate = lambda ws, name: candidate

    runner = types.ModuleType("autoprogramming.runner")

    def run_candidate(ws, cand, inputs, timeout=120.0):
        if calls is not None:
            calls.append(dict(inputs))
        return run_fn(inputs)

    runner.run_candidate = run_candidate

    for name, mod in (("guards", guards), ("data", data),
                      ("candidates", cands), ("runner", runner)):
        monkeypatch.setitem(sys.modules, f"autoprogramming.{name}", mod)


def make_candidate(deterministic=True, name="candidate_0"):
    return SimpleNamespace(name=name, source=CAND_SOURCE, deterministic=deterministic)


@pytest.fixture(autouse=True)
def _no_auto_approve(monkeypatch):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)


@pytest.fixture
def ws(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))
    return workspace


# ------------------------------------------------------------- bootstrap_ci


def test_bootstrap_ci_empty():
    assert scoring.bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_single_value():
    assert scoring.bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_constant_list_is_degenerate():
    assert scoring.bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_ci_seeded_and_bounded():
    vals = [0.1, 0.4, 0.35, 0.9, 0.6, 0.2]
    lo1, hi1 = scoring.bootstrap_ci(vals, seed=1)
    lo2, hi2 = scoring.bootstrap_ci(vals, seed=1)
    assert (lo1, hi1) == (lo2, hi2)
    assert scoring.bootstrap_ci(vals, seed=2) != (lo1, hi1)
    mean = sum(vals) / len(vals)
    assert min(vals) <= lo1 <= mean <= hi1 <= max(vals)
    assert lo1 < hi1


def test_bootstrap_ci_two_point_hand_check():
    lo, hi = scoring.bootstrap_ci([0.0, 1.0])
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------- paired_bootstrap_diff


def test_paired_diff_constant_shift_is_exact():
    assert scoring.paired_bootstrap_diff([1.0, 2.0, 3.0], [1.5, 2.5, 3.5]) == (0.5, 0.5)


def test_paired_diff_identical_is_zero():
    assert scoring.paired_bootstrap_diff([0.3, 0.6], [0.3, 0.6]) == (0.0, 0.0)


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        scoring.paired_bootstrap_diff([1.0], [1.0, 2.0])


# ---------------------------------------------------------- scores.json I/O


def test_load_scores_skeleton_without_file(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    assert scoring.load_scores(workspace) == {
        "metric_sha": None, "candidates": {}, "val_scored": [], "flags": {},
    }


def test_save_and_load_roundtrip(ws):
    scores = scoring.load_scores(ws)
    scores["candidates"]["candidate_0"] = {"val": {"rows": {"row_0": 1.0}, "mean": 1.0}}
    scoring.save_scores(ws, scores)
    assert scoring.load_scores(ws) == scores


def test_load_scores_archives_when_metric_changed_underneath(ws):
    scores = scoring.load_scores(ws)
    scores["candidates"]["candidate_0"] = {"val": {"rows": {"row_0": 1.0}, "mean": 1.0}}
    scoring.save_scores(ws, scores)

    ws.metric_py.write_text("def metric(p, e):\n    return 0.5\n")
    with pytest.warns(UserWarning, match="never comparable"):
        fresh = scoring.load_scores(ws)

    assert fresh["candidates"] == {}
    assert fresh["metric_sha"] == metric.metric_sha(ws)
    archived = ws.root / "scores.archive" / "0.json"
    assert "candidate_0" in json.loads(archived.read_text())["candidates"]


# ------------------------------------------------------------------ evaluate


def test_evaluate_train_per_instance(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="train", per_instance=True)

    assert rep.candidate == "candidate_0"
    assert rep.split == "train"
    assert rep.per_row == {"row_0": 1.0, "row_1": 1.0, "row_2": 0.0}
    assert rep.mean == pytest.approx(2 / 3)
    assert rep.std == pytest.approx((2 / 9) ** 0.5)
    assert rep.n_rows == 3
    assert rep.n_repeats == 1
    assert rep.repeat_variance == 0.0
    assert rep.errors == []
    assert 0.0 <= rep.ci95[0] <= rep.mean <= rep.ci95[1] <= 1.0

    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["train"]["rows"] == rep.per_row


def test_evaluate_train_without_per_instance_hides_per_row(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="train")
    assert rep.per_row is None
    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["train"]["rows"] == {
        "row_0": 1.0, "row_1": 1.0, "row_2": 0.0,
    }


def test_evaluate_val_returns_aggregate_but_persists_rows(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="val")

    assert rep.per_row is None
    assert rep.mean == 1.0
    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["val"]["rows"] == {"row_0": 1.0, "row_1": 1.0}
    assert stored["val_scored"] == ["candidate_0"]


def test_guards_run_before_metric_approval(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "bare_ap", SHOUT_SCHEMA)
    install_fakes(monkeypatch, candidate=make_candidate())
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="test")
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="val", per_instance=True)
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="bogus")


def test_metric_approval_checked_before_any_run(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(), calls=calls)
    with pytest.raises(MetricNotApprovedError):
        scoring.evaluate(workspace, "candidate_0", split="val")
    assert calls == []
    assert "candidate_0" not in scoring.load_scores(workspace)["candidates"]


def test_auto_approve_env_flows_through_evaluate(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    install_fakes(monkeypatch, candidate=make_candidate())

    rep = scoring.evaluate(workspace, "candidate_0", split="val")
    assert rep.mean == 1.0
    record = json.loads(workspace.metric_approval.read_text())
    assert record["approved_by"] == "auto (AP_AUTO_APPROVE_METRIC)"


def test_repeats_deterministic_one_stochastic_three(ws, monkeypatch):
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=True), calls=calls)
    scoring.evaluate(ws, "candidate_0", split="train")
    assert len(calls) == 3

    calls.clear()
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=False), calls=calls)
    rep = scoring.evaluate(ws, "candidate_0", split="val")
    assert len(calls) == 6
    assert rep.n_repeats == scoring.DEFAULT_REPEATS == 3


def test_explicit_n_repeats_overrides(ws, monkeypatch):
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=True), calls=calls)
    rep = scoring.evaluate(ws, "candidate_0", split="val", n_repeats=2)
    assert len(calls) == 4
    assert rep.n_repeats == 2


def test_budget_checked_before_each_row_and_abort_persists_nothing(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=2))

    def costly(inputs):
        run = good_run(inputs)
        run.cost_dollars = 0.5
        return run

    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=costly)
    with pytest.raises(BudgetExceededError):
        scoring.evaluate(workspace, "candidate_0", split="train")

    assert "candidate_0" not in scoring.load_scores(workspace)["candidates"]
    ledger = BudgetLedger(workspace.budget_json)
    assert ledger.spent["eval_calls"] == 2
    assert ledger.spent["dollars"] == pytest.approx(1.0)


def test_failed_run_scores_zero_and_records_error(ws, monkeypatch):
    def sometimes_fail(inputs):
        if inputs["text"] == "yo":
            return SimpleNamespace(
                ok=False, outputs=None, cost_dollars=None,
                error="Traceback (most recent call last):\n  ...\nValueError: boom",
            )
        return good_run(inputs)

    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=sometimes_fail)
    rep = scoring.evaluate(ws, "candidate_0", split="train", per_instance=True)
    assert rep.per_row == {"row_0": 1.0, "row_1": 0.0, "row_2": 0.0}
    assert rep.errors == ["row_1 repeat 1: Traceback (most recent call last):"]


def test_repeat_variance_across_stochastic_repeats(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))

    state = {"n": 0}

    def flaky(inputs):
        state["n"] += 1
        out = inputs["text"].upper() + "!" if state["n"] == 1 else "XX"
        return SimpleNamespace(ok=True, outputs={"Loud": out}, error=None, cost_dollars=None)

    splits = {"train": [{"text": "hi", "Loud": "HI!"}], "val": []}
    install_fakes(monkeypatch, splits=splits, run_fn=flaky,
                  candidate=make_candidate(deterministic=False))
    rep = scoring.evaluate(workspace, "candidate_0", split="train")

    assert rep.n_repeats == 3
    assert rep.mean == pytest.approx(1 / 3)
    assert rep.repeat_variance == pytest.approx(2 / 9)
    assert rep.ci95 == (pytest.approx(1 / 3), pytest.approx(1 / 3))


def test_memorization_flags_stored_and_warned(ws, monkeypatch):
    flags = ["memorizer: train 1.00 vs val 0.20 — verbatim train outputs in source"]
    install_fakes(monkeypatch, candidate=make_candidate(), memo_flags=flags)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MemorizationWarning)
        train_rep = scoring.evaluate(ws, "candidate_0", split="train")
    assert train_rep.flags == []

    with pytest.warns(MemorizationWarning):
        val_rep = scoring.evaluate(ws, "candidate_0", split="val")
    assert val_rep.flags == flags
    stored = json.loads(ws.scores_json.read_text())
    assert stored["flags"]["candidate_0"] == flags
    assert str(val_rep).splitlines()[1] == flags[0]


def test_multi_output_per_field_and_weights(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "qa_ap", QA_SCHEMA)
    metric.write_metric(workspace, MULTI_METRIC)
    metric.approve(workspace, "tester", weights={"Answer": 3.0, "Confidence": 1.0})
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))

    splits = {
        "train": [],
        "val": [{"question": "capital of France?", "Answer": "Paris", "Confidence": "0.9"}],
    }

    def qa_run(inputs):
        return SimpleNamespace(
            ok=True, outputs={"Answer": "Paris", "Confidence": 0.5},
            error=None, cost_dollars=None,
        )

    cand = SimpleNamespace(name="candidate_0", source="def predict(question): ...",
                           deterministic=True)
    install_fakes(monkeypatch, splits=splits, candidate=cand, run_fn=qa_run)
    rep = scoring.evaluate(workspace, "candidate_0", split="val")

    assert rep.mean == pytest.approx(0.9)
    assert rep.per_field["Answer"] == pytest.approx(1.0)
    assert rep.per_field["Confidence"] == pytest.approx(0.6)
    stored = json.loads(workspace.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["val"]["per_field"]["Confidence"] == pytest.approx(0.6)


def test_evaluate_rejects_zero_repeats(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    with pytest.raises(ValueError):
        scoring.evaluate(ws, "candidate_0", split="val", n_repeats=0)


# ------------------------------------------------------------------- compare


def _store_rows(workspace, name, split, rows):
    scores = scoring.load_scores(workspace)
    scores["candidates"].setdefault(name, {})[split] = {
        "rows": rows,
        "mean": sum(rows.values()) / len(rows),
        "std": 0.0, "ci95": [0.0, 1.0], "n_repeats": 1,
        "repeat_variance": 0.0, "per_field": None,
    }
    scoring.save_scores(workspace, scores)


def test_compare_improved_on_constant_shift(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5, "row_1": 0.25, "row_2": 0.75})
    _store_rows(workspace, "candidate_1", "val", {"row_0": 1.0, "row_1": 0.75, "row_2": 1.25})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert rep.a == "candidate_0" and rep.b == "candidate_1"
    assert rep.diff_mean == pytest.approx(0.5)
    assert rep.ci95 == (0.5, 0.5)
    assert rep.improved
    assert "improved" in str(rep)


def test_compare_not_improved_when_ci_includes_zero(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5, "row_1": 0.5, "row_2": 0.5, "row_3": 0.5})
    _store_rows(workspace, "candidate_1", "val", {"row_0": 0.9, "row_1": 0.1, "row_2": 0.9, "row_3": 0.1})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert not rep.improved
    assert rep.ci95[0] <= 0.0
    assert "not improved" in str(rep)


def test_compare_pairs_by_row_id_intersection(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.0, "row_1": 1.0})
    _store_rows(workspace, "candidate_1", "val", {"row_1": 1.2, "row_0": 0.3, "row_2": 9.9})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert rep.diff_mean == pytest.approx(0.25)
    assert rep.improved


def test_compare_requires_stored_scores(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5})
    with pytest.raises(DataDisciplineError, match="eval"):
        scoring.compare(workspace, "candidate_0", "candidate_1")
    with pytest.raises(DataDisciplineError):
        scoring.compare(workspace, "candidate_0", "candidate_0", split="train")


# ------------------------------------------------------------------ reports


def test_eval_report_str_matches_readme_shape():
    rep = scoring.EvalReport(
        candidate="candidate_1", split="val", mean=0.914, std=0.031,
        ci95=(0.860, 0.950), n_rows=10, n_repeats=3, repeat_variance=0.001,
        per_row=None, per_field=None, errors=[], flags=[],
    )
    assert str(rep) == (
        "candidate_1 on val: 0.914 ± 0.031 (n=3 repeats), 95% CI [0.860, 0.950]"
    )


def test_compare_report_str_names_both_candidates():
    rep = scoring.CompareReport(
        a="candidate_0", b="candidate_1", split="val",
        diff_mean=0.08, ci95=(0.02, 0.14), improved=True,
    )
    text = str(rep)
    assert "candidate_1" in text and "candidate_0" in text
    assert "+0.080" in text and "CI" in text


def test_metric_edit_and_reapproval_keeps_val_registration(ws, monkeypatch):
    """The MetricChangedError recovery flow (edit + re-approve) must not
    silently drop a candidate from val_scored on its next val eval.

    Uses the REAL guards module so registration goes through the same
    scores.json writes production uses; only data/candidates/runner are faked.
    """
    import importlib

    real_guards = importlib.import_module("autoprogramming.guards")
    install_fakes(monkeypatch, candidate=make_candidate())
    monkeypatch.setitem(sys.modules, "autoprogramming.guards", real_guards)
    (ws.root / "data").mkdir(exist_ok=True)
    ws.split_json = ws.root / "data" / "split.json"
    ws.split_json.write_text(json.dumps({
        "seed": 0,
        "ratios": [0.6, 0.2, 0.2],
        "counts": {"train": 3, "val": 50, "test": 2},
        "data_sha": "0" * 64,
        "bootstrap": False,
    }))

    scoring.evaluate(ws, "candidate_0", split="val")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]

    # The prescribed recovery flow: edit metric.py in place, re-approve it.
    ws.metric_py.write_text(
        ws.metric_py.read_text() + "# tightened after review\n"
    )
    metric.approve(ws, "tester")

    with pytest.warns(UserWarning, match="never comparable"):
        report = scoring.evaluate(ws, "candidate_0", split="val")
    assert report.mean == 1.0

    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]
    assert stored["metric_sha"] == metric.metric_sha(ws)
    assert "candidate_0" in stored["candidates"]
    assert (ws.root / "scores.archive" / "0.json").exists()
