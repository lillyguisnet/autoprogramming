# Orchestrated portfolio search

Status: initial implementation, July 2026.

## Decision

The optimizer agent is a strategy orchestrator. It does not write candidate
implementations. A trusted Python controller dispatches independent Pi workers,
one implementation avenue per isolated context, evaluates their solution
bundles, and returns only aggregate objective vectors to the orchestrator.

This replaces prompt-only encouragement to "try diverse approaches" with a
controller-enforced breadth policy.

## Roles and trust boundaries

| Role | Sees | Does not see |
|---|---|---|
| User | resource proposal, metric demonstrations, final report | hidden rows |
| Python controller | all run state, expected outputs, metrics, budget | n/a |
| Pi orchestrator | schema, confirmed resources, portfolio state, aggregate vectors | hidden rows; implementation tools |
| Pi mechanism auditor | one avenue contract and its proposed source/dependency metadata | examples, metrics, scores, other candidates, val/test |
| Pi implementation worker | generic task brief, development examples, its own files and assigned mechanism | optimizer identity, metrics, scores, other workers, parent workspace, val/test |
| Candidate runtime | one input at a time, schema/runtime artifacts | expected val/test outputs |

Resource-confirmed workspaces store val and test outside the coding-agent
workspace. Pi workers additionally load only a cooperative root-guard extension.
That guard prevents accidental traversal but is **not** an OS security boundary.
`strict_isolation=True` therefore refuses until an OS sandbox adapter is
configured; the project must never describe path filtering as adversarial
sandboxing.

## Resources

`Resources` separates three contracts:

1. `SearchResources`: build hardware, Pi models, parallelism, package/model
   download permission, fine-tuning access.
2. `RuntimeResources`: the CPU/GPU/network/API/cost/latency envelope of the
   shipped package.
3. `DataPolicy`: whether task data may leave the machine and to which domains.

Hardware may be detected. Egress, network, downloads, package installation, and
candidate-evaluation API access are never inferred as consent. Runtime API
permission and usable search-time provider access are separate facts. If egress
is forbidden, Pi must be explicitly
confirmed local because worker tool results include task context and examples in
model requests. Profiles contain provider names and capability limits, never
credentials.

A dollar-limited run serializes Pi calls when no per-call agent bound is known.
If `SearchResources.max_dollars_per_agent_call` is confirmed, the controller
reserves that much headroom for each in-flight call, settles actual reported
usage on completion, and submits additional calls only when committed headroom
allows it. This prevents several ordinary parallel calls from independently
consuming the same remaining dollars. A provider call that exceeds its declared
bound can still overshoot by that single call, after which no new call launches.

## Approach ladder and portfolio gates

The controller tracks eight tiers:

1. runtime generalist/coding agent
2. graph of model calls
3. single model call
4. fine-tuned model
5. specialized pretrained/deep model
6. classical machine learning
7. algorithms, features, and rules
8. cross-tier composition

Every feasible tier must be attempted or carry an explicit infeasibility reason.
Each avenue is a hard mechanism experiment: workers may not replace a blocked
API/deep/classical/rules mechanism with another family merely to return a valid
answer. Dependencies are resolved from PEP 723 rather than inferred from the
worker's current environment. Before candidate import, deterministic source
checks and an independent Pi mechanism audit reject cross-tier fallbacks; the
controller requests bounded in-session repairs and then clean-session restarts.
Rejected source is never evaluated under that avenue.

Default budget allocation is 40% breadth, 40% deepening, and 20%
composition/wildcards. A family receives two materially different engineering
passes before it is abandoned unless it hard-fails. After breadth, the main Pi
orchestrator allocates remaining deepening from aggregate objective vectors.

Workers use persistent Pi session IDs per avenue, but their only durable task
context is their isolated directory and their own prior implementation.

## Metric roles

A metric suite classifies every quality lens as one of:

- **acceptance**: user-approved definition of task success; final selection may
  use it;
- **diagnostic**: orchestrator-selected search lens; useful for plateaus and
  blind spots but unable to silently choose the winner;
- **operational**: harness-owned cost and latency objectives.

There is no required weighted scalar for suite-aware search. Acceptance floors
filter invalid operating points; Pareto dominance preserves quality/cost
tradeoffs; a lexicographic preference policy is committed before test is opened.
The legacy `primary` field remains only as a compatibility headline for reports
and old workspaces.

## Data flow

The public train split is deterministically partitioned for Pi work:

- `dev_fit`: written to an avenue's generic `examples.jsonl`;
- controller probe: withheld from implementation workers;
- `val`: controller-only selection;
- `test`: controller-only, once at finalization.

Resource-confirmed runs place val/test under the controller-private data root
(`$AP_PRIVATE_DATA_DIR` or `~/.cache/autoprogramming/private`). They are not
written under the agent workspace.

## Pi integration

The implementation is split by responsibility: `pi_rpc.py` owns JSONL/RPC
framing and usage collection, `pi_worker.py` owns implementation task bundles,
environment scrubbing, and worker process launch, while `pi_backend.py` is the
trusted portfolio controller.

Python uses Pi's documented process APIs:

- strategy-only `--mode rpc` calls with no built-in tools or discovered project
  resources, including independent source-vs-mechanism adherence reviews;
- parallel `--mode json --print` implementation workers with skills, context,
  prompts, themes, and user extensions disabled;
- an explicitly loaded root-guard extension;
- per-avenue session IDs and session directories;
- assistant usage/cost collected from Pi message events and charged to the same
  dollar ledger as candidate evaluation.

The Python controller validates all JSON plans, fills missing feasible tiers,
serializes candidate import/evaluation, propagates abort/failure state, and owns
finalization. `PiOrchestratorBackend` can pin orchestrator/worker model patterns
(including Pi thinking suffixes such as `:low`) and separate process timeouts, so
unattended runs need not inherit an unexpectedly expensive or slow global model
configuration. Terminal provider errors are rejected even when Pi itself exits
with status zero.

### Live integration validation

On 2026-07-12 the complete staged path was exercised against Pi 0.80.6 and a
real `openai-codex/gpt-5.4-mini:low` model: metric proposal and independent
critique, user-side suite approval, portfolio planning, four persistent-session
workers behind the root guard, private val/test evaluation, budget attribution,
finalization, activation, and a production call all completed. The smoke run
used 30 candidate evaluations and $0.0425 of reported agent usage. This is a
compatibility smoke test, not a security claim; strict OS isolation remains the
limitation below.

## Score integrity

Every new split score records the candidate source SHA that produced it.
Comparison, tradeoff calculation, and finalization reject stale source. The
schema is also pinned at workspace creation. Unknown candidate cost is represented
as unknown/conservatively infinite for Pareto selection, never as zero dollars.
Candidate import is journaled in portfolio state before source/artifact creation;
on resume the controller finishes an already-scored or partially evaluated
candidate rather than dispatching a duplicate. Orphan artifact namespaces from
a crash before candidate creation are replaced only when no candidate file can
reference them.

## Human confirmation for blocked approaches

A preflight failure or an all-run environmental failure marks an avenue
`blocked`, which does not satisfy portfolio breadth. The controller pauses and
shows the missing capability instead of accepting fallback code or silently
excluding the family. A human may fix/provision it and choose `retry`, or may
explicitly confirm `exclude`, via `prg.resolve_blocker(...)`. The decision and
approver are persisted in portfolio state. A retry bypasses one stale preflight
snapshot so newly provisioned hardware/access can be exercised.

## User flow

```python
resources = ap.Resources(
    search=ap.SearchResources(
        max_parallel_agents=4,
        pi_local=True,  # or permit external egress for a remote Pi provider
        candidate_api_providers=("openai",),  # usable during candidate evaluation
        allow_package_installs=True,
        allow_model_downloads=True,
    ),
    runtime=ap.RuntimeResources(
        network=True,
        api_providers=("openai",),
        max_dollars_per_call=0.002,
    ),
    data=ap.DataPolicy(external_egress=False),
    confirmed=True,
)

prepared = translate.prepare(
    pairs,
    resources=resources,
    budget=ap.Budget(dollars=2),
)
prepared.show_metric_suite()
prepared.demonstrate_metrics([...])
prepared.approve_metrics("user")
report = prepared.optimize(ap.Budget(dollars=20))
```

`prepare()` fixes the data split and asks Pi for a multi-lens proposal but does
not dispatch implementation workers. Search resumes only after sign-off.

## Remaining hardening

The controller-private directory and Pi root guard prevent ordinary accidental
leakage. Adversarial same-user isolation still requires an OS-level worker and
candidate sandbox, so strict isolation currently refuses. Candidate evaluation
already reuses one process per candidate/split: lazy clients and models persist,
`latency_s` is warm repeated-call latency when warm samples exist, and
`cold_start_s` is reported separately.
