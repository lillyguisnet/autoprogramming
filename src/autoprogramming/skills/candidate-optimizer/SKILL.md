---
name: candidate-optimizer
description: Optimize candidate implementations inside an autoprogramming workspace. Use when the working directory (or a nearby one, typically named like translate_ap or another *_ap package) contains active.json, schema.py, candidates/, and scores.json, or when asked to optimize, improve, evaluate, or add candidates for an autoprogramming program. Covers attaching via ap.attach, surveying the approach ladder and seeding a diverse portfolio across cost tiers, composing pretrained models as pipeline stages, multi-metric sign-off with a primary, the quality/cost tradeoff frontier, the train/val/test data discipline, PEP 723 candidate file conventions, and honest budget accounting.
---

# Candidate optimizer for autoprogramming workspaces

You are optimizing one typed program by writing, evaluating, and evolving
complete candidate implementations — plain Python files — under a strict data
discipline and an explicit budget. Your job is not to polish one idea; it is to
survey a spectrum of approaches, baseline a diverse portfolio, then deepen the
ones the data rewards — watching quality AND cost.

## Find the workspace and attach

The workspace root is the directory containing `active.json`, `schema.py`, and
`candidates/` — usually your current working directory. (When this skill has
been copied into a workspace, this file itself sits below that root, under
`.agents/skills/candidate-optimizer/` or `.claude/skills/candidate-optimizer/`;
if the cwd is somewhere else, walk up from this file to find the root.) The
workspace is itself a valid, installable Python package. Work from the root
and attach:

    import autoprogramming as ap
    prg = ap.attach("<workspace root>")

`prg` is your handle. Everything else is file operations on `candidates/*.py`.

Read per-workspace facts live from the workspace, not from memory:

- `prg.schema` — the program's inputs, outputs, and docstrings. schema.py is
  immutable: you satisfy it, you never edit it.
- `prg.data` — `prg.data.train` rows are readable; `prg.data.val` supports
  len() only.
- `prg.budget` — remaining dollars / eval_calls / minutes.
- `data/split.json` — split sizes and the bootstrap flag. `"bootstrap": true`
  means fewer than ~30 examples: build and compare baseline candidates only.
  Fine-grained mutation loops are refused because score differences at that
  data size are one row of noise, and at most 5 distinct candidates may be
  scored on val. Offer to generate synthetic examples for the user to
  validate.

## The prg API

    prg.schema                                   # inputs/outputs & docstrings
    prg.data.train                               # readable rows: iterate, sample, inspect
    prg.data.val                                 # len() only — val rows are never readable
    prg.budget                                   # remaining dollars / eval_calls / minutes
    prg.propose_metric(code, examples, primary=..)  # metric sign-off — FIRST, before any eval
    prg.new_candidate(source=...)                # or prg.new_candidate(from_="candidate_0")
    prg.eval("candidate_1")                      # val: every objective scored from one run
    prg.eval("candidate_1", split="train", per_instance=True)   # per-row — train only
    prg.run("candidate_1", split="train", row=17)               # full trace — train only
    prg.compare("candidate_0", "candidate_1")    # paired bootstrap diff on val (primary)
    prg.compare("candidate_0", "candidate_1", objective="cost_dollars")  # any objective
    prg.tradeoffs()                              # the quality/cost Pareto frontier
    prg.frontier()                               # Pareto frontier over train rows
    prg.finalize(top_k=2)                        # one-time test eval + activation — LAST

## Survey the approach ladder before you commit

A candidate is any Python that satisfies the schema, so the search space spans a
whole cost/capability spectrum. From most expensive/capable to cheapest:

1. Generalist harness — a coding agent or reasoning model doing the task live.
2. A graph of several model calls (plan, act, critique, retrieve).
3. A single model call with an optimized prompt.
4. A finetuned small model.
5. A specialized deep net (a task-specific pretrained model).
6. Classical ML (a fitted scikit-learn head, gradient boosting, nearest neighbor).
7. Hand-written features, rules, regex, or lookup logic.

Rules to follow, not options:

- **Breadth before depth.** Do NOT seed one candidate and mutate it forever —
  that single-step trap wastes budget refining a local idea while the winning
  family goes untried. First seed a PORTFOLIO of genuinely distinct candidates
  spanning several tiers (e.g. a model call, a classical head, a rules baseline,
  a pretrained-model pipeline) and baseline all of them. THEN deepen the ones the
  data rewards.
- **Compose across tiers.** The best solution is usually a compound system, not a
  single tier. Use a heavy pretrained model as ONE STAGE, not only end to end.
  Patterns worth naming and trying: pipeline/decomposition (segment, then
  measure; extract, then classify); cascade (a cheap model handles the easy bulk,
  an expensive one handles the hard tail); ensemble/vote; router (pick a path per
  input); learned-feature + classical-head (embed with a pretrained model, fit a
  small classifier on top).
- **Do not dismiss a family on one try.** A tier or model family is ruled out only
  after a fair attempt — the right variant, sane pre/post-processing, and at least
  two configs. A single failed config is not a dead family. Record WHY a path was
  dropped as a scored result (`prg.eval` / `prg.compare` output), not a hunch.
- **Pick the cheapest tier that can plausibly clear the bar**, then climb only when
  the data shows that tier's ceiling. Cost is an objective, not an afterthought
  (see below): a rules candidate at 0.95 beats a model call at 0.90 when the CI
  separates and it never memorized.

## Use current tools, not remembered ones

Do NOT trust training memory for "the latest" model, checkpoint, or library — it
is stale by construction. Before you fetch a model or pick a package, CHECK what
is current: the model hub, the package index, or a quick web search, and prefer
the current best-for-cost. The classic failure is reaching for a version you
remember (an old SAM when a newer, smaller, faster SAM exists; a superseded
checkpoint) instead of the one that ships today. Verify a model actually exists
and LOADS before you build a candidate around it — a candidate wired to a model
that no longer resolves is a wasted iteration.

## Step 0: the metric set and primary — metric sign-off before ANY scoring

The entire search optimizes whatever metric.py says; a wrong metric produces a
confidently-scored wrong program. Metrics are a set of LENSES, and using several
is normal and cheap — every candidate is scored on all of them from ONE run, so
extra quality metrics cost ~nothing.

1. Write the metric source. Either a single `def metric(predicted, expected)`
   (single-output returns a float; multi-output returns a dict keyed by output
   names), OR a `METRICS = {"<name>": fn, ...}` dict of several named quality
   metrics. Names must be non-empty, unique, and must not collide with the cost
   objectives `cost_dollars` / `latency_s`.
2. Pick 3+ real (predicted, expected) pairs that show judgment: an exact match, a
   near miss (synonym / rounding), and a clear failure.
3. Call `prg.propose_metric(code, examples, primary="<name>")`. The demo table
   shows one column per quality metric with the primary marked, so the user signs
   off on the whole set at once. Three outcomes:
   - True — approved; proceed.
   - A string — user feedback; revise and propose again.
   - MetricNotApprovedError — stdin is not a terminal and AP_AUTO_APPROVE_METRIC
     is unset, so nobody could sign off. The metric WAS written; only sign-off is
     missing. Surface the demonstration table from the error to the user, and on
     their approval record it:

         import autoprogramming as apm
         apm.metric.approve(prg.workspace, "<approver>", primary="<name>")
         # multi-output: also pass weights={"Answer": 2.0, "Confidence": 1.0}

     AP_AUTO_APPROVE_METRIC=1 auto-approves for fully unattended runs. Never use
     either path to dodge the sign-off conversation when a user is reachable.

The **primary** is the headline quality metric: it drives val ranking, the
default activation, and the top-line number you report. Primary and weights are
config, not code — changing which metric is primary or re-weighting does NOT
invalidate any scores. Editing a metric's CODE re-scores candidates from cached
outputs, not by re-running them; only a candidate whose own code changed needs a
re-run.

Diagnose a stuck loop through the lens set: a flat metric (hard exact-match)
gives a whole family a zero gradient and hides real progress. Add a graded metric
to SEE intra-family movement, and make it primary if the strict one is the wrong
compass — but never silently optimize a different thing than you report. The
primary is what the headline number means.

## Cost is an objective

`cost_dollars` and `latency_s` are measured automatically off every run (lower is
better) — no metric.py involvement. Every `prg.eval` reports them alongside the
quality metrics, so you always know what a candidate costs. The goal is the best
quality PER cost, not perfection at any price.

- During the loop, read `prg.tradeoffs()` to see the quality/cost Pareto frontier
  over evaluated candidates — the set where no other candidate beats it on every
  objective. Candidates off the frontier are strictly dominated; drop them.
- Use `prg.compare(a, b, objective="cost_dollars")` (or `latency_s`) to test a
  cost/latency difference the same paired-bootstrap way you test quality. It is
  direction-aware: for these minimized objectives "improved" means the challenger
  is reliably CHEAPER or FASTER (lower is better), not higher.
- Present the USER the frontier, not just the single most-accurate candidate:
  "here is the cheap-good one, the mid one, and the expensive-best one." Let them
  choose the point on the curve. `prg.finalize()` activates a sensible default
  (the best-primary candidate on the frontier) and prints the cheaper/faster
  frontier alternatives with the exact one-liner to switch to each.

## Data discipline (harness-enforced; violations raise, they don't warn)

- Reflect on TRAIN failures only. prg.run() and per-instance scores are refused
  on val and test. You never learn why a val row scored low — only the aggregate —
  so you cannot edit candidates to fix specific val rows.
- VAL is for selection only, and selection pressure is capped. Every eval scores
  the identical val set; the harness counts how many distinct candidates val has
  judged, and past its capacity val scores lose meaning and are not reported as
  final. Prefer few, genuinely different candidates over many small tweaks.
- TEST is off-limits to you entirely. prg.finalize() — the harness, not you —
  evaluates it once, at the end, on the top candidates, then activates the winner.
- "Improved" means the paired-bootstrap CI excludes zero, not that the point
  estimate went up. Use prg.compare() before believing a win.
- Memorizers are flagged and excluded from selection: train score far above val,
  or verbatim training outputs pasted into candidate source. A lookup table over
  train inputs cannot win; a candidate only wins on data it never saw.

## Candidate conventions: PEP 723 single-file scripts in candidates/

- Each candidate is candidates/candidate_<n>.py defining `predict(<input names>)`
  that returns the output value (or a tuple in schema order for multi-output
  programs). Create files with prg.new_candidate; edit them like any Python file.
- Dependencies go in a PEP 723 block at the top of the file:

      # /// script
      # requires-python = ">=3.11"
      # dependencies = ["openai>=1.0"]
      # ///

  stdlib-only candidates need no block at all. Candidates with conflicting
  dependencies can coexist — each runs in its own environment.
- No work at import time: clients and models load lazily inside predict(), so
  importing the package never needs an API key or network access.
- A candidate that makes no stochastic calls should declare `[tool.ap]` with
  `deterministic = true` in its block — it is then scored with 1 repeat instead of
  3, which saves budget.
- A candidate that spends money per call should set the module global
  AP_COST_DOLLARS after each predict() so the ledger and the cost objective stay
  honest (or declare a flat `cost_per_call` in `[tool.ap]`).
- Model weights, pickles, and lookup tables belong in artifacts/, resolved via
  `from <package>.paths import artifacts`. Pretrained-model candidates may declare
  a `fetch = ["huggingface:..."]` download step in `[tool.ap]`.
- Explore genuinely different approaches AND compositions of them — the metric
  does not care how a candidate works, only that it wins honestly.

## The loop

0. **Metric set + primary sign-off** — `prg.propose_metric(code, examples,
   primary=...)` before any eval. Add a graded lens if a strict one is flat.
1. **Survey the ladder, seed a diverse portfolio.** Check current tools, then
   create several genuinely distinct candidates across tiers (a model call, a
   classical head, a rules baseline, a pretrained-model-as-a-stage pipeline) —
   breadth before depth.
2. **Baseline the portfolio on val**, then read `prg.tradeoffs()` to see where
   each lands on the quality/cost frontier. Drop dominated dead ends; record why.
3. **Deepen the promising tiers.** Compose (cascade, ensemble, router,
   learned-feature + classical-head), tune, and try a second config before
   dismissing a family. Reflect on train: `prg.eval(name, split="train",
   per_instance=True)` to find the worst rows, `prg.run(name, split="train",
   row=i)` to read their traces, `prg.frontier()` to see which candidate wins
   which rows. Keep a mutation only when `prg.compare(best, new)` says the CI
   excludes zero.
4. **Watch the frontier, not one number.** Check `prg.budget` between iterations;
   stop when the quality/cost frontier stops advancing (a new candidate no longer
   joins or displaces the frontier), not merely when a single metric plateaus.
5. **prg.finalize()** — one-time test eval, activation of the best-primary
   frontier default, sealed report. Present the tradeoff frontier to the user and
   the one-liner to switch to a cheaper/faster alternative. Do not skip finalize:
   an un-finalized workspace ships nothing.

## Budget honesty

Every evaluation call is charged against the budget — model-calling candidates
cost money to SCORE, not just to write. Re-scoring from cached outputs (after a
metric edit) is free and never charges the budget; only actual runs do. Check
prg.budget between iterations and stop cleanly before it runs dry; there is no
free retry once the ledger hits zero.
