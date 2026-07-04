"""Tests for harness.py — the agent-side `prg` handle.

harness.py imports data/candidates/workspace/runner/metric/scoring, which are
owned by sibling groups and may not exist yet while groups work concurrently;
these tests skip cleanly until the full stack is present and run for real at
integration time. All candidates are stdlib-only (no PEP 723 deps), so the
runner's uv-free fast path is used — no network, no uv.
"""

from __future__ import annotations

import io
import json

import pytest

candidates_mod = pytest.importorskip("autoprogramming.candidates")
data_mod = pytest.importorskip("autoprogramming.data")
metric_mod = pytest.importorskip("autoprogramming.metric")
scoring_mod = pytest.importorskip("autoprogramming.scoring")
workspace_mod = pytest.importorskip("autoprogramming.workspace")
harness = pytest.importorskip(
    "autoprogramming.harness",
    reason="sibling modules not written yet",
    exc_type=ImportError,
)

from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.errors import (
    DataDisciplineError,
    FinalizedError,
    MetricNotApprovedError,
    NotOptimizedError,
)
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


GOOD_CANDIDATE = """\
# /// script
# [tool.ap]
# deterministic = true
# ///
def predict(text):
    return text.upper() + "!"
"""

BAD_CANDIDATE = """\
# /// script
# [tool.ap]
# deterministic = true
# ///
def predict(text):
    return text
"""

METRIC_SRC = "def metric(predicted, expected):\n    return 1.0 if predicted == expected else 0.0\n"


def make_rows(n, start=0, expected_fn=None):
    fn = expected_fn or (lambda t: t.upper() + "!")
    return [
        {"text": f"sample sentence {i}", "Loud": fn(f"sample sentence {i}")}
        for i in range(start, start + n)
    ]


def make_workspace(tmp_path, n_train=6, n_val=4, n_test=3, bootstrap=False,
                   test_expected_fn=None, eval_calls=10_000):
    schema = Schema.from_function(shout)
    splits = {
        "train": make_rows(n_train, 0),
        "val": make_rows(n_val, 100),
        "test": make_rows(n_test, 200, expected_fn=test_expected_fn),
    }
    all_rows = splits["train"] + splits["val"] + splits["test"]
    ws = workspace_mod.Workspace.create(
        tmp_path / "shout_ap", schema, splits,
        seed=0, ratios=(0.6, 0.2, 0.2),
        data_sha=data_mod.data_sha(all_rows), bootstrap=bootstrap,
    )
    BudgetLedger.start(ws.budget_json, Budget(eval_calls=eval_calls))
    return ws


def approve_metric(ws):
    metric_mod.write_metric(ws, METRIC_SRC)
    metric_mod.approve(ws, "tester")


# ------------------------------------------------------------ attach & props


def test_attach_and_properties(tmp_path):
    ws = make_workspace(tmp_path)
    h = harness.attach(str(ws.root))
    assert isinstance(h, harness.AgentHarness)
    assert h.schema.name == "shout"
    assert h.budget["eval_calls"] == 10_000
    assert len(h.data.train) == 6
    assert not hasattr(h.data, "test")


def test_new_candidate_delegates(tmp_path):
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    cand = h.new_candidate(source=GOOD_CANDIDATE)
    assert cand.name == "candidate_0"
    assert cand.path.exists()


# -------------------------------------------------------------- traced runs


def test_run_traces_a_train_row(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    tr = h.run("candidate_0", split="train", row=0)
    assert tr.result.ok
    assert tr.score == 1.0
    assert tr.expected == {"Loud": "SAMPLE SENTENCE 0!"}
    text = str(tr)
    assert "expected:" in text
    assert "score:" in text
    assert BudgetLedger(ws.budget_json).spent["eval_calls"] == 1


def test_run_without_approved_metric_scores_none(tmp_path):
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    tr = h.run("candidate_0", split="train", row=1)
    assert tr.result.ok
    assert tr.score is None
    assert "metric not approved" in str(tr)


def test_run_refuses_val_and_test_before_any_spend(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    for split in ("val", "test"):
        with pytest.raises(DataDisciplineError) as exc:
            h.run("candidate_0", split=split)
        assert "train" in str(exc.value)
    assert BudgetLedger(ws.budget_json).spent["eval_calls"] == 0


# --------------------------------------------------------------------- eval


def test_eval_delegates_and_guards(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    report = h.eval("candidate_0", n_repeats=1)
    assert report.split == "val"
    assert report.mean == 1.0
    with pytest.raises(DataDisciplineError):
        h.eval("candidate_0", split="test")
    with pytest.raises(DataDisciplineError):
        h.eval("candidate_0", split="val", per_instance=True)


def test_compare_delegates(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.new_candidate(source=BAD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    h.eval("candidate_1", n_repeats=1)
    cr = h.compare("candidate_1", "candidate_0")
    assert cr.a == "candidate_1"
    assert cr.b == "candidate_0"
    assert cr.improved


# ----------------------------------------------------------------- frontier


def test_frontier_from_real_train_evals(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.new_candidate(source=BAD_CANDIDATE)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.eval("candidate_0", split="train", per_instance=True, n_repeats=1)
    h.eval("candidate_1", split="train", per_instance=True, n_repeats=1)
    fr = h.frontier()
    assert fr.missing == ["candidate_2"]
    assert fr.nondominated == ["candidate_0"]
    assert fr.rows["row_0"]["score"] == 1.0
    assert fr.rows["row_0"]["candidates"] == ["candidate_0"]
    assert "candidate_0" in str(fr)


def test_frontier_true_pareto_dominance(tmp_path):
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    scores = scoring_mod.load_scores(ws)
    scores["candidates"] = {
        "candidate_0": {"train": {"rows": {"row_0": 1.0, "row_1": 0.0}}},
        "candidate_1": {"train": {"rows": {"row_0": 0.0, "row_1": 1.0}}},
        "candidate_2": {"train": {"rows": {"row_0": 0.9, "row_1": 0.0}}},
        "candidate_3": {"train": {"rows": {"row_0": 1.0, "row_1": 0.0}}},
    }
    scoring_mod.save_scores(ws, scores)
    fr = h.frontier()
    assert fr.nondominated == ["candidate_0", "candidate_1", "candidate_3"]
    assert fr.rows["row_0"] == {"score": 1.0, "candidates": ["candidate_0", "candidate_3"]}
    assert fr.rows["row_1"] == {"score": 1.0, "candidates": ["candidate_1"]}
    assert fr.missing == []


def test_frontier_empty(tmp_path):
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    fr = h.frontier()
    assert fr.rows == {}
    assert fr.nondominated == []
    assert "per-row train scores" in str(fr)


# ----------------------------------------------------------------- finalize


def test_finalize_happy_path(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.new_candidate(source=BAD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    h.eval("candidate_1", n_repeats=1)

    rep = h.finalize(top_k=2)
    assert rep.activated == "candidate_0"
    assert rep.entries[0]["candidate"] == "candidate_0"
    assert rep.entries[0]["test_mean"] == 1.0
    assert rep.entries[0]["demoted"] is False
    assert rep.entries[0]["note"] == "val was 1.00 — healthy gap"
    assert rep.val_reliability == "ok"

    text = str(rep)
    assert text.splitlines()[0] == "test scores (evaluated once):"
    assert "  candidate_0: 1.00   (val was 1.00 — healthy gap)" in text
    assert "activated: candidate_0" in text

    active = ws.active
    assert active["active"] == "candidate_0"
    assert active["finalized"] is True
    assert active["test_score"] == 1.0
    assert ws.final_report.exists()
    stored = json.loads(ws.final_report.read_text())
    assert stored["activated"] == "candidate_0"


def test_finalize_refuses_second_run(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    h.finalize()
    with pytest.raises(FinalizedError) as exc:
        h.finalize()
    msg = str(exc.value)
    assert "once" in msg
    assert "final_report.json" in msg


def test_finalize_demotes_overfit_and_says_so_loudly(tmp_path):
    ws = make_workspace(
        tmp_path, test_expected_fn=lambda t: "a totally different expectation"
    )
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)

    rep = h.finalize(top_k=1)
    entry = rep.entries[0]
    assert entry["demoted"] is True
    assert entry["test_mean"] == 0.0
    assert entry["gap"] == 1.0
    assert "overfit to val, demoted" in entry["note"]
    assert rep.activated == "candidate_0"
    assert "overfit to val, demoted" in str(rep)


def test_finalize_requires_val_scores(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    with pytest.raises(NotOptimizedError) as exc:
        h.finalize()
    assert "val" in str(exc.value)


def test_finalize_refuses_when_all_candidates_flagged(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    scores = scoring_mod.load_scores(ws)
    scores.setdefault("flags", {})["candidate_0"] = ["memorizer: injected for test"]
    scoring_mod.save_scores(ws, scores)
    with pytest.raises(NotOptimizedError) as exc:
        h.finalize()
    assert "memorizer" in str(exc.value)


def test_finalize_skips_flagged_but_keeps_clean(tmp_path):
    ws = make_workspace(tmp_path)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.new_candidate(source=BAD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    h.eval("candidate_1", n_repeats=1)
    scores = scoring_mod.load_scores(ws)
    scores.setdefault("flags", {})["candidate_0"] = ["memorizer: injected for test"]
    scoring_mod.save_scores(ws, scores)
    rep = h.finalize()
    assert rep.activated == "candidate_1"
    assert [e["candidate"] for e in rep.entries] == ["candidate_1"]


def test_finalize_charges_budget_without_check(tmp_path):
    ws = make_workspace(tmp_path, n_val=1, n_test=2, eval_calls=1)
    approve_metric(ws)
    h = harness.AgentHarness(ws)
    h.new_candidate(source=GOOD_CANDIDATE)
    h.eval("candidate_0", n_repeats=1)
    assert BudgetLedger(ws.budget_json).exhausted() == "eval_calls"
    rep = h.finalize(top_k=1)
    assert rep.activated == "candidate_0"
    assert BudgetLedger(ws.budget_json).spent["eval_calls"] == 3


# ----------------------------------------------------------- propose_metric


def test_propose_metric_auto_approve(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    result = h.propose_metric(METRIC_SRC, [("A!", "A!"), ("A!", "B!")])
    assert result is True
    assert metric_mod.is_approved(ws)


def test_propose_metric_refuses_without_anyone_to_ask(tmp_path, monkeypatch):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO())
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    with pytest.raises(MetricNotApprovedError) as exc:
        h.propose_metric(METRIC_SRC, [("A!", "A!")])
    msg = str(exc.value)
    assert "approve" in msg
    assert "AP_AUTO_APPROVE_METRIC" in msg
    assert not metric_mod.is_approved(ws)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_propose_metric_interactive_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)
    monkeypatch.setattr("sys.stdin", FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    result = h.propose_metric(METRIC_SRC, [("A!", "A!")], note="exact match")
    assert result is True
    assert metric_mod.is_approved(ws)
    out = capsys.readouterr().out
    assert "expected:" in out


def test_propose_metric_interactive_feedback(tmp_path, monkeypatch):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)
    monkeypatch.setattr("sys.stdin", FakeTTY())
    monkeypatch.setattr("builtins.input", lambda prompt="": "synonyms should score higher")
    ws = make_workspace(tmp_path)
    h = harness.AgentHarness(ws)
    result = h.propose_metric(METRIC_SRC, [("A!", "A!")])
    assert result == "synonyms should score higher"
    assert not metric_mod.is_approved(ws)


# -------------------------------------------------------- report formatting


def test_final_report_str_mirrors_readme():
    entries = [
        {"candidate": "candidate_1", "val_mean": 0.91, "test_mean": 0.89,
         "gap": 0.02, "demoted": False, "note": "val was 0.91 — healthy gap"},
        {"candidate": "candidate_4", "val_mean": 0.92, "test_mean": 0.84,
         "gap": 0.08, "demoted": True, "note": "val was 0.92 — overfit to val, demoted"},
    ]
    rep = harness.FinalReport(entries=entries, activated="candidate_1", val_reliability="ok")
    lines = str(rep).splitlines()
    assert lines[0] == "test scores (evaluated once):"
    assert lines[1] == "  candidate_1: 0.89   (val was 0.91 — healthy gap)"
    assert lines[2] == "  candidate_4: 0.84   (val was 0.92 — overfit to val, demoted)"
    assert lines[-1] == "activated: candidate_1"


def test_final_report_str_flags_unreliable_val():
    entries = [
        {"candidate": "candidate_0", "val_mean": 0.9, "test_mean": 0.88,
         "gap": 0.02, "demoted": False, "note": "val was 0.90 — healthy gap"},
    ]
    rep = harness.FinalReport(entries=entries, activated="candidate_0",
                              val_reliability="unreliable")
    assert "must not be quoted" in str(rep)


def test_final_report_round_trips_to_dict():
    rep = harness.FinalReport(entries=[], activated=None, val_reliability="ok")
    d = rep.to_dict()
    assert d == {"entries": [], "activated": None, "val_reliability": "ok"}
    assert json.dumps(d)
