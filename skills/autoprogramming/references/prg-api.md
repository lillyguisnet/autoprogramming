# `prg` — the agent-side API for driving an optimization workspace

Two names, one workspace, two trust levels: the user's `Program` can only run
the active candidate; `prg` (an `AgentHarness`) can create and score
candidates — always through harness-enforced guards. Every rule below is
enforced by the library (violations raise, with the reasoning in the message);
none of it depends on your good manners.

## Attach

```py
import autoprogramming as ap
prg = ap.attach("translate_ap")     # any workspace path; returns an AgentHarness
```

Raises `WorkspaceError` when the directory does not exist, has no
`active.json`, or has no `schema.py`.

## Inspect

```py
prg.schema        # Schema (immutable): .describe() prints inputs/outputs + docstrings,
                  # .input_names, .output_names, .expected_columns
prg.workspace     # the Workspace object (.root, .candidates_dir, ...)
prg.budget        # dict of remaining headroom: {"dollars": ..., "eval_calls": ..., "minutes": ...}
                  # None where no limit was set; floored at 0
prg.data          # SplitView(train=<n>, val=<n>)
```

### `prg.data` semantics

- `prg.data.train` — fully readable `Rows`: `len()`, iteration, indexing and
  slicing (returns dict copies), `.columns`, `.sample(k, seed=0)`
  (deterministic). Reflect here, freely.
- `prg.data.val` — `GuardedRows`: **`len()` only**. Iterating or indexing
  raises `DataDisciplineError`. Val exists for selection; you never see why a
  val row scored low, only aggregates, so you cannot edit candidates to fix
  specific val examples.
- There is **no `prg.data.test`** attribute at all, and no
  `prg.eval(split="test")`. Test belongs to `finalize()` — evaluated exactly
  once, at the end. `data/test.csv` is written chmod 400.

## Metric first: `propose_metric`

No scoring happens until the workspace's `metric.py` is approved.

```py
prg.propose_metric(code: str, examples: list[tuple], note: str = "") -> bool | str
```

- `code` — full source of `metric.py`. Contract:
  - single-output program: `def metric(predicted, expected) -> float`
    (called with the bare output values);
  - multi-output program: `def metric(predicted: dict, expected: dict) -> dict`
    keyed by output type names — one score per output, no field may be missing.
- `examples` — `(predicted, expected)` pairs (bare values for single-output,
  dicts for multi-output) scored with the candidate metric and shown as a
  demonstration table. Pick 3+ that show its judgment: an exact match, a near
  miss (synonym/rounding), a clear failure.
- Returns `True` once approved (interactive terminal sign-off, or the
  `AP_AUTO_APPROVE_METRIC` env var — see below). Returns the user's feedback
  **string** when they push back: revise the code and propose again. Raises
  `MetricNotApprovedError` when stdin is not a terminal and auto-approve is
  off — in that case surface the demonstration to the user and record their
  sign-off with:

```py
import autoprogramming as apm
apm.metric.approve(prg.workspace, "user name", weights=None)
# weights (multi-output only): {"Answer": 2.0, "Confidence": 1.0} — the per-field
# aggregate weighting, part of the sign-off; must sum positive. Aggregate score
# is the weighted mean of the per-field scores.
```

Facts that matter:

- Approval stamps line 1 of `metric.py`
  (`# metric.py — approved by <who> on <date>`) and pins the file's sha in
  `metric_approval.json`. The stamp line is excluded from the sha, so
  re-approving never looks like a metric change.
- **Editing `metric.py` after scores exist archives `scores.json`** to
  `scores.archive/<k>.json` and resets it — scores under different metrics are
  never comparable — and voids the approval (`MetricChangedError` on the next
  scoring attempt until re-approved). Re-proposing byte-identical code is a
  no-op that keeps everything.
- `AP_AUTO_APPROVE_METRIC=1` (or `true`/`yes`) auto-approves an unapproved
  metric for unattended runs. It never re-approves a metric that changed after
  sign-off. Do not set it to dodge the sign-off conversation with a present user.

## Create candidates: `new_candidate`

```py
prg.new_candidate(source: str | None = None, from_: str | None = None) -> Candidate
```

Exactly one of `source` (fresh file contents) or `from_` (name of an existing
candidate to copy) — both or neither is a `CandidateError`. Writes
`candidates/candidate_<max+1>.py` (starting at `candidate_0`); metadata is
validated before the file lands. Candidates are plain files — after creating
via `from_`, edit the new file directly like any Python file.

### Candidate file conventions (PEP 723 single-file scripts)

Each candidate defines `predict(<input parameter names>)` returning:
- the bare output value (single-output program), or
- a tuple in schema order (multi-output), or a dict keyed by output names.

Values are coerced to the builtin bases of the schema types on the way out.

```py
# candidates/candidate_0.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "translate-ap"]
#
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
#
# [tool.ap]
# cost_per_call = 0.0002
# ///
from openai import OpenAI
from translate_ap.schema import French

_client = None

def predict(english: str) -> French:
    global _client
    if _client is None:
        _client = OpenAI()
    ...
```

- **No work at import time.** Clients and models load lazily inside
  `predict()`; importing the package must never need an API key or network.
- **Dependencies** go in the `# /// script` block (at most one per file, valid
  TOML). A stdlib-only candidate needs no block at all — and runs on the fast
  path without uv. Dep-bearing candidates run under `uv run --no-project` in
  their own ephemeral environment, so conflicting deps across candidates
  coexist; `uv` must be on PATH or the run raises `RunnerError`.
- **The self-reference** (`"translate-ap"` plus its `[tool.uv.sources]` entry)
  lets the candidate import `<pkg>.schema` / `<pkg>.paths` when run standalone
  (`uv run candidates/candidate_0.py`). At harness eval time it is stripped
  and the package is injected via `sys.path` instead, so evals stay hermetic.
- **`[tool.ap]` keys** (all optional):
  - `deterministic = true` — the candidate makes no stochastic calls; it is
    scored with 1 repeat instead of 3, saving budget. Default false.
  - `cost_per_call = <float>` — flat dollar cost charged per run when the run
    does not report its own cost.
  - `fetch = ["huggingface:...", ...]` — declared artifact download steps
    (recorded candidate metadata for heavyweight models).
- **`AP_COST_DOLLARS`** — a candidate that spends money per call should set
  this module global after each `predict()` (e.g.
  `AP_COST_DOLLARS = tokens * rate`); the runner reads it after the call and
  charges the ledger with it, overriding `cost_per_call`.
- **Artifacts** (weights, pickles, lookup tables) belong in the workspace's
  `artifacts/` directory, resolved as
  `from <package>.paths import artifacts` (a `pathlib.Path`).
- Each run executes in a subprocess with the env var `AP_WORKSPACE` set to the
  workspace root, a default timeout of 120 s (a timeout kills the whole
  process group and scores 0), and results reported through a file — stdout
  cannot forge an outcome. A `predict()` exception is a failed run scoring
  0.0, never a crash of the harness.

## Score: `eval`

```py
prg.eval(candidate, split="val", per_instance=False, n_repeats=None) -> EvalReport
```

- `split="val"` (default): selection. Aggregate only — `per_instance=True` on
  val raises `DataDisciplineError`. Each distinct candidate scored on val is
  registered as selection pressure (see guards below).
- `split="train"`: reflection. `per_instance=True` returns `per_row`
  (`{"row_0": score, ...}`) so you can find the worst rows.
- `split="test"`: always refused.
- `n_repeats`: default 1 for `deterministic = true` candidates, else 3; the
  score per row is the mean over repeats. Must be >= 1 when given.
- The budget is checked before every row; a mid-eval `BudgetExceededError`
  propagates with nothing persisted for the partial eval. Every run charges
  `eval_calls` and dollars to the ledger.
- Metric must be approved or this raises before any spend.

`EvalReport` fields: `candidate`, `split`, `mean`, `std`,
`ci95: (lo, hi)` (seeded bootstrap over per-row means), `n_rows`, `n_repeats`,
`repeat_variance`, `per_row` (train + `per_instance=True` only, else `None`),
`per_field` (multi-output aggregates or `None`), `errors` (failed-run
summaries; failed runs score 0.0), `flags` (memorization flags, see below).

## Trace: `run`

```py
prg.run(candidate, split="train", row=0) -> TracedRun
```

The reflection primitive: one full traced run of a single **train** row —
inputs, outputs or traceback, stdout/stderr, timing, cost, expected values,
and the metric score (`None` if the metric is not yet approved; 0.0 for a
failed run). Any split other than `"train"` raises `DataDisciplineError`; an
out-of-range row raises `IndexError`. Charges 1 eval call.

## Compare: `compare`

```py
prg.compare(a, b, split="val") -> CompareReport
```

Paired comparison over per-row scores **already stored** by `eval()` — never
triggers new spend. `a` is the baseline, `b` the challenger; `diff_mean` is
the mean of `b - a` per row; `ci95` is a seeded paired-bootstrap CI;
`improved` is `True` only when the CI lies entirely above zero. Raises
`DataDisciplineError` when either candidate has no stored per-row scores for
that split, or their stored rows share no ids — eval both first. (Per-row val
scores are persisted internally for exactly this pairing; they are never
returned to you.)

**"Improved" means the CI excludes zero.** A point estimate going up is not a
win; do not keep a mutation `compare()` cannot distinguish from noise.

## Survey: `frontier`

```py
prg.frontier() -> FrontierReport
```

True Pareto view over stored per-row **train** scores: `rows` maps each row id
to its best score and the candidates achieving it; `nondominated` lists
candidates no other candidate dominates (>= on every shared row, > on one);
`missing` lists candidates with no per-row train scores yet (eval them with
`split="train", per_instance=True` to include them). Complementary candidates
that win different rows are prime material for a merged/pipeline candidate.

## Finish: `finalize`

```py
prg.finalize(top_k=2) -> FinalReport
```

The **only** code path that reads the test split, run once per workspace:

1. Requires an approved metric and at least one val-scored candidate
   (else `NotOptimizedError`). Memorization-flagged candidates are excluded;
   if every val-scored candidate is flagged, finalize refuses.
2. Takes the `top_k` eligible candidates by val mean and evaluates each on
   every test row (1 repeat if deterministic, else 3). Charges the budget but
   never checks it — the report card happens even on an exhausted budget.
3. Demotes any finalist whose val-to-test drop exceeds
   `max(0.05, half the width of its val CI)` — that is overfitting to val,
   reported loudly, never hidden.
4. Activates the best test scorer among the non-demoted (best test overall if
   all were demoted, with a warning note), writes `final_report.json`, and
   rewrites `active.json` + `pyproject.toml` so the shipped package runs the
   winner with exactly its runtime deps.

`FinalReport`: `entries` (per finalist: `candidate`, `val_mean`, `test_mean`,
`gap`, `demoted`, `note`), `activated`, and `val_reliability` — `"ok"`,
`"warn"` (val absorbed many selection decisions; noisy), or `"unreliable"`
(val scores must not be quoted; the test column is the only report card).

Calling `finalize()` a second time raises `FinalizedError` — a second pass
would let the report card steer the search and turn test into another val
set. Do not skip finalize either: an un-finalized workspace ships nothing.

## Guards you will hit (by design)

| Action | Result |
|---|---|
| `eval(split="test")`, or any unknown split | `DataDisciplineError` |
| `eval(split="val", per_instance=True)` | `DataDisciplineError` |
| `run()` on a val/test row | `DataDisciplineError` |
| Iterating/indexing `prg.data.val` | `DataDisciplineError` |
| Scoring with unapproved / changed metric | `MetricNotApprovedError` / `MetricChangedError` |
| 6th distinct candidate on val in bootstrap mode (< 30 examples) | `BootstrapModeError`, raised before any spend |
| Any charge once a limit is hit | `BudgetExceededError` (finalize still runs) |
| Second `finalize()` | `FinalizedError` |

Soft signals (warnings, not errors):

- **Val selection pressure** — once more distinct candidates than val rows
  have been compared, a `ValReliabilityWarning` fires (`warn`); past 3x the
  val size, val scores are declared `unreliable` and will not be reported as
  final. Prefer few, genuinely different candidates over many small tweaks.
- **Memorization** — once a candidate has both train and val aggregates, it is
  flagged (`MemorizationWarning`, recorded in `scores.json` flags and
  excluded from selection) when train mean exceeds val mean by more than 0.2
  with train mean above 0.5, or when at least `max(3, 10% of train rows)`
  distinct train string outputs (length >= 8) appear verbatim in the candidate
  source. A regex/rules candidate may legitimately win — but only on data it
  never saw.

## The loop

1. **Metric sign-off first** — `propose_metric` before any eval; changing it
   later wipes every recorded score.
2. **Seed** a complete working candidate; `prg.eval("candidate_0")` for a val
   baseline.
3. **Reflect on train**: `prg.eval(name, split="train", per_instance=True)` to
   find the worst rows; `prg.run(name, split="train", row=i)` to read their
   traces; form a hypothesis.
4. **Mutate**: `prg.new_candidate(from_=best)`, edit the file, eval. Keep it
   only if `prg.compare(best, new).improved` is `True`.
5. **Watch the budget**: check `prg.budget` between iterations and stop
   cleanly before it runs dry. Stop early when improvements stop separating
   from noise. Explore genuinely different approaches (LLM prompt, rules,
   classical ML, local model, pipeline) — `frontier()` shows which rows each
   approach wins.
6. **`prg.finalize()`** — one-time test eval, activation, sealed report.
