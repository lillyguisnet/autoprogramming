---
name: autoprogramming
description: Build and optimize a typed Python program or prompt from examples or labeled input/output pairs with the autoprogramming library — DSPy/GEPA-style prompt optimization generalized to whole implementations. Use when the user wants to turn example pairs into an optimized program, optimize or evolve a prompt against a metric, distill a program into a cheaper model, or improve a deployed program from production logs. Covers defining @ap.program schemas, gathering data, proposing and getting sign-off on the evaluation metric, setting an explicit ap.Budget, running optimize()/distill(), driving the candidate loop yourself via ap.attach, and reading results honestly (confidence intervals, overfit demotion, train/val/test discipline).
---

# AutoProgramming — conversational front-end

You are the front-end for the `autoprogramming` library inside the user's own
project. The user describes a task ("I want a translator", "classify these
tickets"); your job is to turn the conversation into four concrete artifacts —
a typed `@ap.program` definition, the data, a user-approved metric, and an
explicit `ap.Budget` — and then call `optimize()` exactly like any library
user would. A coding-agent search then evolves complete candidate
implementations (plain `.py` files: LLM calls, regex, scikit-learn, local
models, pipelines) under a strict train/val/test discipline, and the output is
a portable Python package with zero runtime dependence on the optimizer.

## Install

```sh
uv add "git+https://github.com/lillyguisnet/autoprogramming.git"
# or: pip install "git+https://github.com/lillyguisnet/autoprogramming.git"
```

- Requires Python >= 3.12.
- `uv` must be on PATH at evaluation time: candidates that declare third-party
  dependencies run under `uv run --no-project` in their own ephemeral
  environments (conflicting deps coexist). Stdlib-only candidates run without uv.

## The conversation: five questions

Ask these, in order, before writing any code. Everything else the library and
the optimizer figure out.

1. **What goes in, what comes out?** "English in, French out" is enough.
   Becomes the `@ap.program` definition (below).
2. **Do you have examples?** 30+ pairs unlocks full optimization; 5–29 runs in
   bootstrap mode (baselines only, at most 5 candidates compared — the
   library enforces this); fewer than 5 is refused outright. If none: offer to
   generate synthetic pairs the user validates.
3. **What does "good" mean?** Exact match? Same meaning? Latency? Cost per
   call? This shapes the metric you will propose.
4. **Sign off on the metric.** THE critical step — the entire search optimizes
   whatever the metric says, so a wrong metric produces a confidently-scored
   wrong program. Never skip; never approve on the user's behalf. Demonstrate
   the proposed metric on real examples and iterate until the scores match the
   user's intuition:

   ```
   I propose chrF for scoring. Watch it score three examples:
     expected:  "Où est la gare ?"
     predicted: "Où se trouve la gare ?"  -> 0.81  (synonym, mild penalty)
     predicted: "Où est la gare ?"        -> 1.00  (exact)
     predicted: "La gare est fermée."     -> 0.34  (wrong meaning)
   Does 0.81 for the synonymous phrasing match your intuition of "good"?
   If synonyms should score ~1.0, I'll use embedding similarity instead.
   ```

   For multi-output programs the per-field aggregate weighting is also part of
   the sign-off.
5. **Any constraints?** Data allowed to leave the machine? Max cost per call?
   Offline only? Which packages may be installed? These narrow the candidate
   search space.

## Define the program

Types are subclasses of builtins; docstrings become descriptions the optimizer
uses. Schema errors (missing annotations, defaults, `*args`, generics/unions)
surface at decoration time.

```py
import autoprogramming as ap

class French(str):
    """Natural, idiomatic French. Formal register unless the input is casual."""

@ap.program
def translate(english: str) -> French:
    """Translate English text to French."""
```

Multiple outputs are a tuple of DISTINCT types — output names come from their
types, so two outputs of the same type is a `SchemaError`:

```py
class Answer(str):
    """Direct answer to the question, one sentence."""

class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""

@ap.program
def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""
```

Data rows must cover every input (by parameter name) and every expected output
(by output type name): here `question`, `Answer`, `Confidence`. Accepted
sources: a list of dicts, a DataFrame (anything with `.to_dict("records")`),
or a path to a `.csv`/`.jsonl` file. Extra columns are dropped with a warning;
missing columns are a `SchemaError`.

## Budget — explicit, no default

`optimize()` and `distill()` refuse to run without one, because evaluation
spends real money and time (LLM candidates cost money to *score*, not just to
mutate). Ask the user what they are willing to spend:

```py
ap.Budget(dollars=20)                    # or eval_calls=2000, or minutes=30
ap.Budget(dollars=20, minutes=60)        # combinable; first limit hit stops the run
```

## Run optimize()

```py
report = translate.optimize(data=pairs, budget=ap.Budget(dollars=20))
```

What happens mechanically:

- The data is normalized and **split exactly once** into train/val/test
  (default ratios 0.6/0.2/0.2, deterministic by `seed=0`). The split is pinned
  by a content hash (`data_sha`) and is never redone for that workspace.
- Fewer than 30 rows puts the workspace in **bootstrap mode**; fewer than 5 is
  refused.
- A workspace directory is created — default `<program name>_ap/` (e.g.
  `translate_ap/`) — which is itself a valid, installable Python package:
  `pyproject.toml`, `__init__.py` (runs the active candidate), `schema.py`
  (immutable), `active.json`, `data/` (the three split CSVs; `test.csv` is
  chmod 400 and harness-controlled), `candidates/` (one PEP 723 `.py` file
  each), `artifacts/` (weights, pickles), and `scores.json`. `metric.py` is
  NOT among the created files — it appears only after the metric is proposed
  and approved (the sign-off step below); until then any scoring attempt
  raises `MetricNotApprovedError`.
- A backend then launches an optimizer coding agent against the workspace
  (the `claude` CLI when it is on PATH). **When no agent backend is
  available**, `optimize()` prints that the workspace is ready for a manual
  session with the `ap.attach` snippet — the workspace plus this skill is
  everything needed to drive the loop yourself. Read
  [references/prg-api.md](references/prg-api.md) for the complete agent-side
  `prg` API before doing that.
- The metric must be proposed, demonstrated on real examples, and approved
  before any scoring happens. Changing the metric later archives and clears
  all recorded scores — scores under different metrics are never comparable.
- The optimizer loop is: reflect on **train** failures (full traces allowed),
  mutate candidate files, select on **val** (aggregate scores only), repeat
  within budget. **test** is untouchable until `finalize()` evaluates it once,
  at the end, on the top candidates, and activates the winner.
- Returns a `FinalReport` when candidates were scored; `None` when the
  workspace is awaiting a manual session.

Afterwards the program is a normal function and a normal package:

```py
translate("Hello, how are you?")   # => French('Bonjour, comment allez-vous ?')
translate.save("translate_ap")     # relocate the workspace (no-op if already there)
# and: pip install ./translate_ap  — deps of the active candidate install automatically
```

## Read results honestly

- **The test column is the report card.** `FinalReport` lists, per finalist,
  `val_mean`, `test_mean`, and a note. Test was evaluated exactly once; quote
  the test number, never the val number.
- **Overfit demotion.** A finalist whose test score drops far below its val
  score (gap beyond the val CI half-width, min 0.05) is marked
  "overfit to val, demoted". If every finalist is demoted, the report says so
  and activates the best test score anyway — tell the user to treat the
  numbers with suspicion.
- **Val reliability.** When val absorbed many selection decisions relative to
  its size, the report carries `val_reliability` of `"warn"` (val scores are
  noisy) or `"unreliable"` (val scores must not be quoted at all).
- **Every eval carries uncertainty.** `EvalReport` shows mean, std, a 95%
  bootstrap CI, and repeat variance (stochastic candidates are scored with 3
  repeats by default). "Improved" means a paired-comparison CI **excludes
  zero** — not that the point estimate went up. Report wins to the user in
  those terms.

## Production lifecycle

```py
translate.enable_logging()
# every call appends one JSONL entry to <workspace>/logs/<UTC date>.jsonl:
# {"inputs": {...}, "outputs": {...}, "candidate": "...", "n_repeat": 1, "timestamp": "..."}
```

Two different improvement paths — do not mix them up:

- **Distill** (make it cheaper, same behavior). Raw unreviewed logs are the
  RIGHT data here on purpose: the goal is imitation, so the program's own
  outputs are the training target. Budget is still required. The result lands
  in a new workspace (default `<name>_distilled_ap`); the original is untouched.

  ```py
  translate.distill(model="gpt-4.1-nano", data="logs", budget=ap.Budget(dollars=5))
  ```

- **Re-optimize** (make it better). Logs alone cannot do this — optimizing
  toward your own outputs reinforces your own errors, so
  `optimize(data="logs")` is refused by design. A human must add the
  correction signal first:

  ```py
  translate.review_logs()   # interactive: accept / correct / reject sampled entries
  translate.optimize(data="logs:reviewed", budget=ap.Budget(dollars=10))
  ```

  Only accepted/corrected entries become training data. Because the original
  workspace's split is fixed forever, re-optimization goes to a **sibling
  workspace** (`<name>_reopt_ap`, then `_reopt2_ap`, ...) automatically — and
  the call re-binds the program to that sibling. To resume the run later,
  first re-bind the ORIGINAL workspace (the one holding the reviewed logs),
  then repeat the call; it finds the sibling that already holds exactly that
  reviewed data and resumes it instead of creating another directory:

  ```py
  translate.use("translate_ap")   # the workspace with the reviewed logs
  translate.optimize(data="logs:reviewed", budget=ap.Budget(dollars=10))
  ```

  Repeating the call WITHOUT re-binding raises a misleading
  `DataDisciplineError` ("no reviewed log entries exist yet") — the program
  is then reading logs from the empty sibling, not from the original
  workspace. Re-bind with `.use()` as above; do not re-review logs.

## Refusals to expect — explain them, don't fight them

Every refusal in this library is deliberate and self-explaining. When one
fires, relay the reason to the user and take the suggested path; do not look
for workarounds.

| Refusal | Why | What to do |
|---|---|---|
| `optimize()`/`distill()` without a budget (`BudgetError`) | Evaluation spends real money; there is no default | Ask the user for `dollars`/`eval_calls`/`minutes` |
| Fewer than 5 rows (`DataDisciplineError`) | Nothing to select on, even in bootstrap mode | Gather more examples, or generate synthetic ones the user validates |
| `data_sha` mismatch when re-optimizing an existing workspace | The data was split once; re-splitting leaks val/test into train | Pass the original data, or point at a new workspace path |
| `optimize(data="logs")` (`DataDisciplineError`) | Unreviewed logs only echo the current program | `review_logs()` first, then `data="logs:reviewed"`; or `distill()` if the goal is compression |
| Scoring before metric sign-off (`MetricNotApprovedError`) | The search optimizes whatever the metric says | Run the demonstration + sign-off conversation with the user |
| Metric edited after approval (`MetricChangedError`) | Old sign-off is void; old scores are archived, not comparable | Re-demonstrate and re-approve |
| Bootstrap cap: 6th candidate on val (`BootstrapModeError`) | Differences on a tiny val set are one row of noise | Pick the best baseline already scored, or get to 30+ examples |
| Reading val rows, tracing val/test, eval on test, per-row val scores (`DataDisciplineError`) | Reflection is train-only; selection is aggregate-only; test belongs to `finalize()` | Reflect on train; read val as aggregates; finalize at the end |
| Second `finalize()` (`FinalizedError`) | Test is evaluated exactly once | Read `final_report.json`; start a fresh workspace to keep improving |

## Driving the loop yourself

When you (rather than a launched optimizer agent) create candidates, score
them, and finalize — via `prg = ap.attach("<workspace>")` — read
[references/prg-api.md](references/prg-api.md) first. It documents every
`prg` method signature, the candidate PEP 723 file conventions, and the
reflect-on-train / select-on-val / finalize loop.
