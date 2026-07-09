---
name: candidate-optimizer
description: Optimize candidate implementations inside an autoprogramming workspace. Use when the working directory (or a nearby one, typically named like translate_ap or another *_ap package) contains active.json, schema.py, candidates/, and scores.json, or when asked to optimize, improve, evaluate, or add candidates for an autoprogramming program. Covers attaching via ap.attach, metric sign-off before any scoring, the train/val/test data discipline, PEP 723 candidate file conventions, the reflect-mutate-select loop, and honest budget accounting.
---

# Candidate optimizer for autoprogramming workspaces

You are optimizing one typed program by writing, evaluating, and evolving
complete candidate implementations — plain Python files — under a strict data
discipline and an explicit budget.

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

Per-workspace facts are deliberately not written down in this file — read
them live from the workspace:

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
    prg.propose_metric(code, examples, note="")  # metric sign-off — FIRST, before any eval
    prg.new_candidate(source=...)                # or prg.new_candidate(from_="candidate_0")
    prg.eval("candidate_1")                      # val score: aggregate + 95% CI only
    prg.eval("candidate_1", split="train", per_instance=True)   # per-row — train only
    prg.run("candidate_1", split="train", row=17)               # full trace — train only
    prg.compare("candidate_0", "candidate_1")    # paired bootstrap diff on val
    prg.frontier()                               # Pareto frontier over train rows
    prg.finalize(top_k=2)                        # one-time test eval + activation — LAST

## Step 0: metric sign-off, before ANY scoring

The entire search optimizes whatever metric.py says; a wrong metric produces a
confidently-scored wrong program. The metric is proposed, demonstrated, and
approved — never silently trusted:

1. Write the metric source: `def metric(predicted, expected) -> float` for a
   single-output program, or dict -> dict keyed by output names for a
   multi-output program.
2. Pick 3+ real (predicted, expected) pairs that show its judgment: an exact
   match, a near miss (synonym / rounding), and a clear failure.
3. Call prg.propose_metric(code, examples). Three outcomes:
   - True — approved; proceed.
   - A string — user feedback; revise the metric and propose again.
   - MetricNotApprovedError — stdin is not a terminal and the
     AP_AUTO_APPROVE_METRIC env var is unset, so nobody could sign off
     (the normal case when you run non-interactively). The metric WAS
     written to metric.py; only the sign-off is missing. Surface the
     demonstration table from the error message to the user, and once they
     approve, record it:

         import autoprogramming as apm
         apm.metric.approve(prg.workspace, "<approver name>")
         # multi-output: pass weights={"Answer": 2.0, "Confidence": 1.0}

     For fully unattended runs, AP_AUTO_APPROVE_METRIC=1 in the environment
     auto-approves an unapproved metric. Never use either path to dodge the
     sign-off conversation when a user is reachable.

Changing the metric later invalidates every recorded score, so get this right
before spending budget.

## Data discipline (harness-enforced; violations raise, they don't warn)

- Reflect on TRAIN failures only. prg.run() and per-instance scores are
  refused on val and test. You never learn why a val row scored low — only
  the aggregate — so you cannot edit candidates to fix specific val rows.
- VAL is for selection only, and selection pressure is capped. Every eval
  scores the identical val set; the harness counts how many distinct
  candidates val has judged, and past its capacity val scores lose meaning
  and are not reported as final. Do not burn val evals on trivial variants.
- TEST is off-limits to you entirely. prg.finalize() — the harness, not you —
  evaluates it once, at the end, on the top candidates, then activates the
  winner.
- "Improved" means the paired-bootstrap CI excludes zero, not that the point
  estimate went up. Use prg.compare() before believing a win.
- Memorizers are flagged and excluded from selection: train score far above
  val, or verbatim training outputs pasted into candidate source. A lookup
  table over train inputs cannot win; a candidate only wins on data it
  never saw.

## Candidate conventions: PEP 723 single-file scripts in candidates/

- Each candidate is candidates/candidate_<n>.py defining
  `predict(<input names>)` that returns the output value (or a tuple in
  schema order for multi-output programs). Create files with
  prg.new_candidate; edit them like any Python file.
- Dependencies go in a PEP 723 block at the top of the file:

      # /// script
      # requires-python = ">=3.11"
      # dependencies = ["openai>=1.0"]
      # ///

  stdlib-only candidates need no block at all. Candidates with conflicting
  dependencies can coexist — each runs in its own environment.
- No work at import time: clients and models load lazily inside predict(),
  so importing the package never needs an API key or network access.
- A candidate that makes no stochastic calls should declare
  `[tool.ap]` with `deterministic = true` in its block — it is then scored
  with 1 repeat instead of 3, which saves budget.
- A candidate that spends money per call should set the module global
  AP_COST_DOLLARS after each predict() so the ledger stays honest.
- Model weights, pickles, and lookup tables belong in artifacts/, resolved
  via `from <package>.paths import artifacts`.
- Explore genuinely different approaches — an LLM call, regex rules,
  classical ML, a local model, a pipeline of all four. The metric does not
  care how a candidate works, only that it wins honestly.

## The loop

1. Seed a complete, working candidate; eval it on val for a baseline.
2. prg.eval(name, split="train", per_instance=True) to find the worst train
   rows; prg.run() them; read the traces; form a hypothesis.
3. prg.new_candidate(from_=best), edit, eval. Keep it only if prg.compare()
   says the CI excludes zero.
4. Repeat while prg.budget affords another iteration. Stop early when
   improvements stop separating from noise — a spent budget with no clear
   winner is worse than an early stop.
5. Finish with prg.finalize() so test is evaluated once and a winner is
   activated. Do not skip this: an un-finalized workspace ships nothing.

## Budget honesty

Every evaluation call is charged against the budget — LLM candidates cost
money to SCORE, not just to write. Check prg.budget between iterations and
stop cleanly before it runs dry; there is no free retry after the ledger
hits zero.
