# Changelog

## 0.2.0 — 2026-07-12

- Added a resource-aware Pi portfolio backend with one strategy orchestrator and
  parallel, implementation-only avenue workers.
- Added explicit search, deployment, and data-egress resource contracts.
- Added acceptance/diagnostic metric suites, acceptance floors, and
  precommitted Pareto selection policies.
- Added breadth, second-configuration, stagnation, deepening, and composition
  policy state.
- Added the staged `Program.prepare()` / `PreparedRun` workflow.
- Added controller-private val/test storage for resource-confirmed runs and a
  deterministic dev-fit/controller-probe partition.
- Pinned schemas, split scores, and declared artifact bundles by content hash.
- Unknown candidate cost is no longer represented as free.
- Added Pi RPC/JSON usage accounting, worker context isolation flags, persistent
  avenue sessions, and a cooperative root-guard extension.
- Added in-flight agent dollar reservations; dollar-limited Pi work serializes
  when no confirmed per-call bound is available.
- Added pre-import candidate journaling and pending-evaluation recovery to avoid
  duplicate candidates across controller crashes.
- Added persistent candidate processes so lazy clients/models survive across
  rows, with warm latency and cold start reported separately.
- Split Pi RPC, worker isolation/bundling, and trusted portfolio control into
  focused modules.
- Hardened Pi integration against zero-exit provider failures and stale metric
  role names after critic rewrites; exposed separate worker/orchestrator timeouts.
- Validated the complete staged workflow against Pi 0.80.6 with real orchestrator,
  critic, and implementation-worker model calls.
- Added CI, an MIT license, architecture documentation, and adversarial tests.

### Known limitation

The worker root guard and controller-private files prevent ordinary accidental
leakage but are not an adversarial same-user OS sandbox. Strict isolation
currently refuses rather than silently weakening that guarantee.

## 0.1.0 — 2026-07-12

- Initial complete library API, Agent Skills, guarded train/val/test workflow,
  portable workspaces, metric approval, output caching, and multi-objective
  quality/cost/latency reporting.
