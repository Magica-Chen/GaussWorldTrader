# Four-agent validation and release record

This record maps the implementation to
[the version 1.1 specification](GaussWorldTrader_Four_Agent_Implementation_Plan_v1_1.md).
The delivery baseline is the `four-agents` branch. Tests use synthetic accounts, an
injected clock and broker doubles. The explicit implementation-plan request includes
the selected offline tests, synthetic fixture and isolated CI. The repository's other
local tests remain gitignored. Reproduce the selected suite on a fresh checkout.

## Evidence levels

| Evidence | What it establishes |
| --- | --- |
| Schema and unit checks | Contract validation and isolated deterministic invariants |
| Synthetic two-session replay | Causal role handovers, no-trade cases, persistence and scope separation under supplied fixtures |
| Broker-double lifecycle tests | Local timeout, partial-fill, cancellation and restart logic against the encoded broker contract |
| Explicit online paper contract tests | Intended account/SDK order mechanics and actual endpoint permissions |
| Recorded forward observation and holdout evaluation | Operational reliability and scoped research/economic evidence over time |

Software tests establish their named invariants. Paper and live operational gates remain
separate from fixture success. No actual broker-writing test or real-money arming forms
part of this implementation session.

## Reproduction

The implementation environment reports Python 3.13.5, alpaca-py 0.42.0, Pydantic 2.11.7
and SQLAlchemy 2.0.43. The project supports Python 3.12 and later. Offline tooling was
installed under `/tmp/gauss-test-deps` for this workspace.

```bash
PYTHONPATH=/tmp/gauss-test-deps:$PWD python -m pytest -o addopts='' -q tests/test_execution_safety.py tests/test_gauss_existing_adapters.py tests/test_live_session_safety.py tests/test_four_agent*.py tests/test_runtime_operational.py tests/test_momentum_strategy.py tests/runtime tests/ui
```

The original local test directory contains online account and notification scripts.
Run only explicitly selected offline tests during implementation; notification tests
that send email/Slack require separate user authorization.

The integrated replay exercises both profiles with explicit below-minimum, feasible,
cash-poor and capacity-limited accounts. All four roles complete, feasible cases reach
shadow intents, late safety news invalidates a plan, and broker writes and asserted
fills remain zero. Separate broker doubles exercise actual order-state transitions.
The offline SDK tests cover signed debit/credit multi-leg requests, actual option
permissions and buying power, out-of-scope exposure, and the legacy `PositionManager`
ownership guard using captured requests and synthetic SDK responses.

The startup compatibility audit added coverage for the real adapter's order-list request
and full reconciliation path. `Sort` comes from `alpaca.common.enums`. Account permissions
are validated directly from the REST response because alpaca-py 0.42 still requires
`dtbp_check` and `pdt_check`, fields [removed from the API on July 6, 2026](https://docs.alpaca.markets/us/changelog/2026-07-06-pdt-db49dba).
Missing or malformed permission flags remain entry blockers, and closing-only restrictions
are enforced. Read-only checks against the configured paper account passed for account,
positions, orders and activities; no broker writes or runtime-state changes were performed.

Wheel and sdist builds, installation outside the checkout, 96 nested-module imports
with socket access blocked, and installed CLI/bot help and configuration checks passed.
The workspace verification helper was `/tmp/verify_gauss_distribution.py`; its detailed
results were written to `/tmp/gauss-final-distribution-4_503ykb/`. These temporary paths
are implementation-session evidence. For a fresh installation, build with
`python -m build`, install the wheel in a new environment, change out of the source
checkout, and run the installed `trading-cli session validate-config` and
`trading-cli list-strategies` commands.

Real Streamlit rendering and a persisted replay client were exercised with AppTest.
The inspected browser captures show the [session view](images/gauss_session_offline.png)
and [operator controls](images/gauss_session_controls_offline.png). Opening these views
did not start the session runtime or submit orders.

The [T01–T60 acceptance matrix](FOUR_AGENT_ACCEPTANCE_MATRIX.md) names individual test
functions and their evidence scope. A named synthetic test establishes its
encoded invariant; online contract and forward-observation gates remain separate.

The combined offline suite passed **216 tests** after the startup compatibility fixes, including the real Chromium
scenario-switch check. CI runs the client-state check and skips the optional browser
check when Playwright/Chromium is unavailable. Hosted CI has not run in this local session.

## Work-package traceability

The implementation groups cohesive service components under `src/runtime/` rather than
creating every proposed package path individually.

| Package | Principal implementation/evidence | Release interpretation |
| --- | --- | --- |
| WP00 | Existing entry-point regressions, package discovery/import checks, selected tracked offline tests and fixture | Isolated CI is included under the explicit implementation-plan request |
| WP01 | Legacy execution corrections plus mode/account-enforcing risk and gateway modules | Every sizing and broker-writing path requires focused checks |
| WP02 | Versioned models, calendar, SQLite records/projections/outbox/jobs/leases | Restart and conflict behavior exercised with synthetic state |
| WP03 | Profile clock, typed evidence, capabilities and news versions | Current broker state remains separate from signal latency |
| WP03A | Account feasibility, explicit scenario scopes and cost/capacity calculations | Capital is always supplied or observed |
| WP04 | PostGauss/CloseGauss roles and injected immutable research inputs | Empty candidates and insufficient evidence are valid outcomes |
| WP05 | PreGauss validation and immutable plan eligibility events | Account/profile/news changes require reassessment |
| WP06 | LiveGauss shadow integration and independent supervision | Shadow/replay gateway rejects broker writes |
| WP07 | Gateway/order ledger/reconciliation | Online paper contract acceptance requires explicit setup |
| WP08 | Contract/group validation, valuation and residual exposure | Options entries remain gated by readiness and operator policy |
| WP09 | CLI, separate dashboard, continuous bot, backup/restore, operations guide | Control commands authenticate and remain durable |
| WP10 | `evaluation.py` scoped metric comparison and causal evaluation methodology | Longitudinal strategy and economic conclusions require collected evidence |

## Evaluation protocol

Keep operational correctness, research selection value and economic viability as three
separate outcomes. Each sample supplies account ID, actual/hypothetical scope, explicit
capital, data profile, strategy, selection method, period, evidence kind and chronological
partition. Compare fixed-watchlist, CloseGauss and PreGauss selection with identical
execution/risk assumptions. Preserve all attempted strategies and parameter trials.

`EvaluationSample` and `compare_evaluations` validate explicit scoped observations.
Reports retain every matrix cell and reject overlapping development/validation/holdout
periods. Decision-only samples have no trading P&L. Actual-fill gross P&L subtracts
declared fees once; broker equity changes remove external cashflows and already include
fees. Operating costs use unique allocation IDs and are reported separately before the
all-in result. Spreads and slippage embodied in actual fills are never deducted again.

Section 18.3 metrics include wall/as-of timestamps, profile delay and additional lag,
quote age, record/source coverage, backlog and recovery, candidates and alternatives,
validation reversals, cancel/replace rates, fill latency, spread/slippage benchmarks,
fees, exposure, realised/unrealised P&L, cashflow-adjusted drawdown, minimum feasible size,
capacity and operating cost per candidate/group. Missing observations or denominators
are reported as `null`; supplied zero counts remain zero. Decision-only evidence
cannot assert fills, an equity history or profits. Nested development windows cannot
hide holdout overlap. Synthetic metrics establish arithmetic and scope validation
without claiming economic performance.

Snapshot reports preserve immutable supplied bars/news and frozen account references.
Point-in-time fundamentals absent from the supplied evidence are reported unavailable.
The implementation does not backdate present-day fundamentals into a historical report.

Record source event/interval-end time, receipt/availability, signal as-of, decision,
submission and fill times. A delayed trigger can lead to a decision only when available;
its simulated fill must occur after that decision. Missing current executable option
quotes limits a test to decision quality or disclosed simulation assumptions.

Use explicit account scenarios around instrument minimum-size boundaries, with separate
cases for depleted cash, permissions, reservations and capacity. Keep permissions and
data entitlement fixed for capital-only comparisons. A scenario with larger capital
must recompute quantities, units, costs, portfolio limits and liquidity capacity.

## Release gates requiring deployment evidence

| Gate | Required record before enabling |
| --- | --- |
| Paper entries | Approved strategy/profile/paper contract; account identity; fresh data/account readiness; reconciled restart/unknown-order tests |
| Long-option paper entries | Genuine required quote feed, verified contracts/multipliers, account options permission and confirmed group lifecycle tests |
| Defined-risk spread entries | Combined-order restrictions, ratios, group exits, assignment and residual-stock handling exercised for the actual deployment |
| Live entries | Explicit operator arming bound to account/environment, strategy, risk/data policy and deployment version; scoped validation and operational review |
| Free-profile live entries | Initial policy keeps these disabled; any future exception needs separately validated delayed-signal execution/exit approval |
| Economic claims | Chronological holdout/forward records, realistic fill/cost evidence and account/profile-specific uncertainty |

The initial release has no automatic promotion, subscription purchase or agent-controlled
risk-limit increase. Every enabled account/data/strategy combination needs its own
recorded authorization and applicable operational evidence.
