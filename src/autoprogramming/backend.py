"""How the coding agent gets launched against a workspace (or how you attach yourself)."""

from __future__ import annotations

import json
import shutil
import string
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .errors import AutoProgrammingError, RunnerError

if TYPE_CHECKING:
    from .harness import AgentHarness


OPTIMIZER_PROMPT = """\
You are the optimizer agent for an autoprogramming workspace. Your job is to
find the best implementation of one typed program by writing, evaluating, and
evolving complete candidate implementations — plain Python files — under a
strict data discipline and an explicit budget.

Workspace (work from here; it is itself a valid, installable Python package):
  $workspace

Program schema (immutable — you satisfy it; you never edit schema.py):
$schema

Data: $data_sizes.
$bootstrap_note
Budget remaining: $budget
Every evaluation call is charged against this budget — LLM candidates cost
money to SCORE, not just to write. Check prg.budget between iterations and
stop cleanly before it runs dry.
Run context: $context

== Attach ==

    import autoprogramming as ap
    prg = ap.attach("$workspace")

`prg` is your handle. Everything else is file operations on candidates/*.py.

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

== Step 0: metric sign-off, before ANY scoring ==

The entire search optimizes whatever metric.py says; a wrong metric produces a
confidently-scored wrong program. The metric is proposed, demonstrated, and
approved — never silently trusted:

1. Write the metric source: `def metric(predicted, expected) -> float` for a
   single-output program, or dict -> dict keyed by output names for a
   multi-output program.
2. Pick 3+ real (predicted, expected) pairs that show its judgment: an exact
   match, a near miss (synonym / rounding), and a clear failure.
3. Call prg.propose_metric(code, examples). True means approved — proceed.
   A string is user feedback — revise the metric and propose again.

Changing the metric later invalidates every recorded score, so get this right
before spending budget.

== Data discipline (harness-enforced; violations raise, they don't warn) ==

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

== Candidate conventions: PEP 723 single-file scripts in candidates/ ==

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

== The loop ==

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
"""


def build_prompt(harness: "AgentHarness", context: dict) -> str:
    """Fill OPTIMIZER_PROMPT with one workspace's facts (sizes, budget, mode)."""
    ws = harness.workspace
    root = getattr(ws, "root", ws)
    try:
        budget = json.dumps(harness.budget)
    except (AutoProgrammingError, TypeError, ValueError):
        budget = "not recorded yet"
    sizes = "unknown (data/split.json not found)"
    bootstrap = False
    split_json = getattr(ws, "split_json", Path(root) / "data" / "split.json")
    try:
        info = json.loads(Path(split_json).read_text())
        counts = info.get("counts", {})
        sizes = (
            f"{counts.get('train', '?')} train / {counts.get('val', '?')} val / "
            f"{counts.get('test', '?')} test rows"
        )
        bootstrap = bool(info.get("bootstrap", False))
    except (OSError, ValueError):
        pass
    if bootstrap:
        bootstrap_note = (
            "BOOTSTRAP MODE: fewer than 30 examples. Build and compare baseline "
            "candidates only — fine-grained mutation loops are refused because "
            "score differences at this data size are one row of noise, and at "
            "most 5 distinct candidates may be scored on val. Offer to generate "
            "synthetic examples for the user to validate."
        )
    else:
        bootstrap_note = "Full optimization mode."
    return string.Template(OPTIMIZER_PROMPT).substitute(
        workspace=str(root),
        schema=harness.schema.describe(),
        data_sizes=sizes,
        bootstrap_note=bootstrap_note,
        budget=budget,
        context=json.dumps(context, sort_keys=True),
    )


@runtime_checkable
class AgentBackend(Protocol):
    """Anything that can drive one optimization run against a harness."""

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Drive the loop against ``harness``; return when done."""
        ...


class NoOpBackend:
    """A backend that launches nothing and tells you how to drive ``prg`` yourself."""

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Print the workspace path, the attach snippet, and the prg API."""
        root = getattr(harness.workspace, "root", harness.workspace)
        print(
            "\n".join(
                [
                    "[autoprogramming] no agent backend ran (NoOpBackend).",
                    f"Workspace ready at: {root}",
                    "Attach and optimize manually:",
                    "",
                    "    import autoprogramming as ap",
                    f'    prg = ap.attach("{root}")',
                    "",
                    "    prg.schema                                  # inputs/outputs & docstrings",
                    "    prg.propose_metric(code, examples)          # metric sign-off comes first",
                    "    prg.new_candidate(source=...)               # candidates/candidate_<n>.py",
                    '    prg.eval("candidate_0")                     # val: aggregate + CI only',
                    '    prg.eval("candidate_0", split="train", per_instance=True)',
                    '    prg.run("candidate_0", split="train", row=0)  # traces: train rows only',
                    '    prg.compare("candidate_0", "candidate_1")   # improved iff CI excludes 0',
                    "    prg.frontier()                              # Pareto frontier over train",
                    "    prg.budget                                  # remaining dollars/calls/minutes",
                    "    prg.finalize()                              # one-time test eval + activate",
                ]
            )
        )


class ClaudeCodeBackend:
    """Launches the Claude Code CLI with OPTIMIZER_PROMPT, cwd'd to the workspace."""

    def __init__(self, command: tuple[str, ...] = ("claude",)):
        self.command = tuple(command)

    def run(self, harness: "AgentHarness", context: dict) -> None:
        """Build the prompt and run ``<command> -p <prompt>`` in the workspace."""
        if shutil.which(self.command[0]) is None:
            raise RunnerError(
                f"Agent binary {self.command[0]!r} was not found on PATH, so the "
                f"optimization agent cannot be launched — optimize() needs a "
                f"coding agent to drive the loop. Install the Claude Code CLI, "
                f"or pass backend=NoOpBackend() and drive the loop yourself via "
                f"ap.attach(<workspace>)."
            )
        prompt = build_prompt(harness, context)
        root = getattr(harness.workspace, "root", harness.workspace)
        subprocess.run([*self.command, "-p", prompt], cwd=str(root), check=False)


def default_backend() -> AgentBackend:
    """ClaudeCodeBackend when the ``claude`` CLI is on PATH, else NoOpBackend."""
    if shutil.which("claude"):
        return ClaudeCodeBackend()
    return NoOpBackend()
