# GaussWorldTrader: Four-Agent Implementation Plan

**PostGauss · CloseGauss · PreGauss · LiveGauss**

| Document control | Value |
| --- | --- |
| Version | 1.1 |
| Prepared | 6 September 2026 |
| Revised | 7 September 2026 |
| Data operating profiles | `FREE_DELAYED` and `SUBSCRIBED_REALTIME` |
| Account scope | Capital-adaptive; no assumed account balance or universal balance tier |
| Repository | `Magica-Chen/GaussWorldTrader` |
| Reviewed branch | `master` |
| Reviewed commit | `31374551bae6fd34a0fe56fe11d208f4ff04fbb4` |
| Status | Proposed implementation specification; not implemented |
| Suggested repository location | `docs/FOUR_AGENT_IMPLEMENTATION_PLAN.md` |
| First delivery target | A complete, persistent four-agent workflow in shadow mode, with no broker order submissions |

This plan extends the existing platform rather than replacing it. Repository observations are grounded in the reviewed commit and identified by references such as [R01]. Broker-specific requirements are linked to official documentation such as [A01]. All new modules, interfaces, configuration fields, operating thresholds and acceptance criteria below are **proposed engineering requirements**, not existing features or evidence of trading profitability.

The review was static. The application was not executed, the account was not inspected, and no live or paper orders were placed. Recheck the repository baseline and current broker capabilities before implementation. This document does not authorise enabling real-money trading.

### Revision 1.1 — data modes and account-adaptive suitability

This revision makes two requirements apply throughout the implementation: **free operation uses a deliberately 15-minute-delayed market-information view; subscribed operation uses entitled real-time market information**, and **strategy/instrument selection must be evaluated against each account rather than a fixed starting balance**. The same four roles run under both data profiles, with explicit differences in timing, strategy eligibility and execution readiness.

The 15-minute delay is an information cutoff, not a promise of exact delivery latency or a reason to delay broker order/position updates. Indicative options quotes must not be relabelled as genuine delayed OPRA quotes. Account suitability means the best-supported feasible choice under declared constraints, including **no trade**; it is not a claim of a universally optimal or guaranteed-profitable strategy.

The architecture and repository baseline from version 1.0 are retained. Configuration, contracts, agent duties, work packages, tests and rollout gates are updated consistently below.

## Contents

1. [Objective, scope and design decisions](#1-objective-scope-and-design-decisions)
2. [Repository baseline and required corrections](#2-repository-baseline-and-required-corrections)
3. [Target architecture and authority](#3-target-architecture-and-authority)
4. [Responsibilities of the four agents](#4-responsibilities-of-the-four-agents)
5. [Session scheduling and service lifecycle](#5-session-scheduling-and-service-lifecycle)
6. [Shared market-data service](#6-shared-market-data-service)
7. [News and event processing](#7-news-and-event-processing)
8. [Data contracts and research handovers](#8-data-contracts-and-research-handovers)
9. [Plan, order and position state machines](#9-plan-order-and-position-state-machines)
10. [Persistence, delivery and concurrency](#10-persistence-delivery-and-concurrency)
11. [Risk gate and execution integration](#11-risk-gate-and-execution-integration)
12. [Options-specific readiness requirements](#12-options-specific-readiness-requirements)
13. [Repository changes and interfaces](#13-repository-changes-and-interfaces)
14. [Configuration and operating budgets](#14-configuration-and-operating-budgets)
15. [CLI and dashboard changes](#15-cli-and-dashboard-changes)
16. [Implementation work packages](#16-implementation-work-packages)
17. [Test and acceptance matrix](#17-test-and-acceptance-matrix)
18. [Research validation and performance evaluation](#18-research-validation-and-performance-evaluation)
19. [Deployment, recovery and rollback](#19-deployment-recovery-and-rollback)
20. [End-to-end acceptance scenario](#20-end-to-end-acceptance-scenario)
21. [Release checklist and implementation order](#21-release-checklist-and-implementation-order)
22. [Sources and evidence boundary](#22-sources-and-evidence-boundary)

## 1. Objective, scope and design decisions

### 1.1 Objective

Add a session-aware research and execution workflow:

**PostGauss observes and reviews → CloseGauss investigates and proposes → PreGauss challenges and validates → LiveGauss evaluates and acts.**

PostGauss and PreGauss continuously monitor through a shared collector. CloseGauss analyses a frozen, timestamped evidence set. LiveGauss monitors approved candidates and all actual exposure during the permitted trading session. News can invalidate a plan at any stage, including after entry. Market-information timestamps follow the selected data profile; operational broker state remains current. Account suitability is recalculated at research, validation and execution boundaries.

The four roles are not four independent trading authorities. Only a single controlled execution service can submit orders. A deterministic risk gate has veto authority over every new exposure request.

### 1.2 User requirements carried into the design

| Requirement | Implementation interpretation |
| --- | --- |
| Support different account amounts | Read actual equity, available cash, usable buying power, holdings, reservations and permissions; never assume a universal starting balance |
| Select the most suitable feasible approach | Each role uses a shared account-suitability assessment; compare approved strategies, instruments and quantities against the account and data profile, including a no-trade outcome |
| Evaluate alternative capital amounts | Provide explicit hypothetical account scenarios for research; keep their results separate from the actual execution account |
| Selective intraday trading | Entry-count, exposure and holding-period limits come from an operator-approved mandate; agents may recommend fewer entries, never increase limits autonomously |
| Continuous observation | Continuously collect and evaluate available information; do not equate this with continuous LLM calls or always-current prices |
| Free market-data case | Use `FREE_DELAYED`: a 900-second market-information lag, plus measured transport/completeness lag; delayed stock SIP polling is the primary consolidated path |
| Subscribed market-data case | Use `SUBSCRIBED_REALTIME`: entitled real-time stock SIP and, where enabled, option OPRA data with measured freshness checks |
| Options access differs by account | Discover actual permissions and feed capabilities independently; approval does not establish affordability or suitability |
| Pre-/post-market monitoring | Run both roles continuously on their profile's available information; initial extended-hours activity remains observation, not execution |
| Reuse GaussWorldTrader | Preserve the strategy registry, analytical capabilities, execution adapters and user interfaces |

A **position group** is one managed strategy position: one stock position, one long option, or a supported multi-leg structure. A spread is one group containing several contracts, not several unrelated trades.

A larger account must not automatically receive more leverage, more trades, more complex options or a paid subscription. A smaller account must not automatically be steered towards cheap, illiquid or highly leveraged contracts. Feasibility and suitability are computed from actual constraints and evidence, not account-size stereotypes.

### 1.3 Delivery scope

The first usable release must implement all four roles in shadow mode. Paper execution follows after risk, persistence and reconciliation tests pass. Stocks provide the first end-to-end execution path. Options support is an explicit subsequent gate, not silently assumed ready because submission methods already exist.

Retain existing crypto and other workflows, but do not enrol them automatically in this US stock/options session controller. Preserve the wheel and other existing strategies as research capabilities; do not automatically approve them for an intraday mandate merely because an account can afford them.

Exclude from the initial release: uncovered option selling, autonomous strategy-code deployment, autonomous risk-limit increases, same-day-expiry trading, extended-hours entries, distributed microservices and mandatory new data subscriptions.

### 1.4 Non-negotiable invariants

1. A research output cannot place an order or change risk policy.
2. Every automated entry references an approved strategy version and an eligible session-plan version.
3. Missing or invalid data block new exposure; they do not disable existing-position supervision.
4. The system never equates an order submission with a fill, a cancellation request with a cancellation, or an exit request with a flat account.
5. Risk-reducing actions are not blocked by entry quotas or expired entry plans, but still require valid order mechanics and reconciled quantities.
6. Research inputs, approvals, revisions and execution decisions are durable and auditable.
7. An LLM failure, budget limit or research timeout cannot interrupt the independent risk/position-management path.
8. Existing exposure and unresolved orders remain supervised outside normal entry hours.
9. A market-information profile and an execution mode are independent: subscribing never arms trading, and choosing paper mode never removes data latency.
10. Delayed observations cannot be presented as current executable quotes or used to invent fills at an earlier price.
11. Broker positions, fills, order states, account restrictions and operational alerts are never intentionally delayed by the market-data profile.
12. Every account-specific plan references a versioned suitability assessment; actual order sizing uses a fresh account snapshot and deterministic risk checks.
13. Hypothetical capital scenarios cannot change actual balances, reservations or order quantities.
14. Account-size changes or model recommendations cannot silently increase approved risk limits.

## 2. Repository baseline and required corrections

### 2.1 Components to preserve and reuse

| Existing path | Observed capability | Integration decision |
| --- | --- | --- |
| `src/strategy/base.py` | `SignalSnapshot`, `ActionPlan`, `MarketDataContext` and strategy interfaces | Retain; introduce a separate session-plan contract above `ActionPlan` [R01] |
| `src/strategy/registry.py` | Factories for existing stock, option and multi-agent strategies | Reuse factories; add a separate approval registry [R02] |
| `src/agent/multi_agent/` | Technical, fundamental, sentiment, risk and decision analysis | Reuse analytical components, especially inside CloseGauss [R03] |
| `src/strategy/multi_agent_strategy.py` | Calls the analysis orchestrator from strategy evaluation | Keep legacy behaviour; do not use unrestricted fresh committee decisions as LiveGauss's entry authority [R04] |
| `src/data/alpaca_provider.py` | Historical data, quotes, account coupling and stream factories | Split market-data access from trading-account access; add explicit feed policy [R05] |
| `src/data/news_provider.py` | Alpaca/Finnhub retrieval, normalisation and deduplication | Extend into durable, version-aware event processing [R06] |
| `src/trade/live/` | Live signal loops and shared streaming for homogeneous engine types | Reuse adapters; move lifecycle ownership to the session runtime [R07] |
| `src/trade/engine/execution.py` | Quantity selection and order submission through asset engines | Harden and place behind a portfolio-wide risk gate [R08] |
| `src/ui/dashboard.py` | Streamlit views, market/news stream queues and account integration | Reuse views; the UI must not own the persistent trading service [R09] |
| `src/settings.py`, `pyproject.toml` | Configuration, database URL, Pydantic/SQLAlchemy dependencies and test tooling | Extend rather than add a competing configuration stack [R10], [R10A] |

The current technical/fundamental/sentiment agents describe **analytical specialisms**. The four Gauss agents describe **session responsibilities**. Keep both levels; do not rename one set into the other.

### 2.2 Corrections to track explicitly

These findings concern the inspected implementations. They do not establish that every possible caller exhibits the same behaviour. Write focused regression tests before changing the relevant code.

| ID | Observed issue or gap | Required change and release gate |
| --- | --- | --- |
| F01 | Live context paths contain USD 100,000 portfolio fallbacks [R08], [R11] | Missing/non-finite account values must reject new entries; required before any broker-connected execution |
| F02 | Generic sizing computes `portfolio_value * risk_pct / current_price`; it is not stop-distance sizing. Its affordability calculation has no option-multiplier term [R08] | Separate capital allocation, planned loss and option structural loss; introduce instrument-aware sizing |
| F03 | The quantity-override branch returns an adjusted quantity without passing through the normal affordability calculation [R08] | Validate every requested/overridden quantity at the final risk gate; overrides never bypass capital limits |
| F04 | Stock/option session methods use weekday checks and fixed hours [R12], [R13] | Use a shared calendar and instrument-session policy, including holidays and early closes |
| F05 | Free-tier historical stock requests are forced to IEX by one tier Boolean [R05] | Introduce explicit delayed/real-time profiles and evidence clocks; do not silently substitute IEX or infer option entitlements from stock access [A01] |
| F06 | Options state aggregates positions and treats zero summed signed quantity as flat [R13] | Model contract legs and position groups; opposite legs must not cancel operational exposure |
| F07 | Options monitoring subscribes to underlying trades and inherits generic price-based monitoring [R11], [R13] | Separate underlying signal prices from contract-premium valuation, exits and close symbols |
| F08 | Research-facing news records collapse timestamps; analytical agents fetch their own evidence [R03A], [R06] | Preserve article versions and receipt time; inject immutable evidence into CloseGauss |
| F09 | `fast` mode's sentiment-labelled component uses `MomentumStrategy` [R03] | Label it a price proxy; run a separate news-risk service in both fast and LLM modes |
| F10 | The live runner only shares streams across compatible engine classes; dashboard streams have separate lifecycle state [R07], [R09] | Centralise per-endpoint collection and internal fan-out; do not create one broker stream per agent |
| F11 | `.gitignore` excludes `tests/` [R14] | Track tests and sanitised fixtures; retain exclusions for secrets, runtime data and outputs |
| F12 | Packaging explicitly enumerates packages [R10] | Include all new packages and verify existing nested packages in a built-wheel import test |
| F13 | Automatic order pricing adds/subtracts a fixed increment, and metadata is propagated into execution [R08] | Revalidate final prices, permitted ticks, debit/credit bounds and typed execution fields after conversion |

Treat F06–F07 as blockers for unattended options trading. Existing multi-leg submission support is reusable but is not a complete position-lifecycle guarantee.

## 3. Target architecture and authority

### 3.1 Architecture

```text
Alpaca historical/streaming adapters + permitted news sources
                              |
                FeedPolicy + EvidenceClock + EventService
                              |
                 Durable events + immutable snapshots
                              |
        +---------------------+---------------------+
        |                     |                     |
    PostGauss             CloseGauss             PreGauss
 observe/review           batch research         revalidate
        |                     |                     |
        +---------- candidates and session plans ---+
                              |
                          LiveGauss
                evaluate profile-compatible rules
                              |
                    Deterministic RiskGate
                              |
                 ExecutionGateway / order ledger
                              |
                Existing Alpaca execution adapters
                              |
                            Alpaca

Shared across all four roles: AccountProfile + AccountSuitabilityService
Broker order/account updates -> Reconciler -> positions/reservations/risk
Dashboard and CLI <-> authenticated control interface <-> session runtime
```

### 3.2 Deployment shape

Use a single-host modular application initially. AccountSuitabilityService is a shared calculation and reporting component, **not a fifth autonomous trading agent**. All records remain scoped to account and environment; the first runtime still owns only one execution account.

Deployment components:

- One supervised runtime owns broker connections, the event store, the scheduler, the risk gate and order submission.
- Bounded research workers execute expensive model calls and analysis outside the execution loop. Workers receive serialised evidence and an allowlisted environment, not broker credentials.
- The dashboard is a separate UI process. Closing a browser must not stop monitoring or execution.

Use SQLite with migrations for the first local store and reuse the existing SQLAlchemy dependency. Keep database access behind a repository interface so PostgreSQL remains a later option, not a first-release requirement. Avoid Redis, Kafka, Kubernetes and an additional agent framework unless a measured requirement justifies them.

### 3.3 Authority boundaries

| Component | May do | Must not do |
| --- | --- | --- |
| PostGauss | Read events/account snapshots; create reviews and candidates | Submit orders; approve a new strategy |
| CloseGauss | Read frozen evidence; create conditional plans and experiments | Read newer evidence silently; auto-deploy generated code |
| PreGauss | Revalidate eligible plans; request documented revisions | Increase risk budgets; bypass missing evidence |
| LiveGauss | Evaluate eligible plans; request entries/exits | Mint strategy approvals; override risk vetoes |
| AccountSuitabilityService | Compute feasibility, rank supported alternatives and explain account/profile fit | Submit orders, fabricate capital, approve strategies or change risk policy |
| RiskGate | Approve/reject bounded order intents; reserve capacity | Rely on an LLM's permission to enforce a limit |
| ExecutionGateway | Submit, query, cancel and replace authorised intents | Accept arbitrary agent-generated broker payloads |
| Reconciler | Update actual order/position state; flag discrepancies | Pretend missing responses mean no exposure |
| Operator | Arm an explicitly configured mode; approve strategies; issue audited controls | Change limits invisibly or via untrusted news text |

Only gateway/collection code receives the relevant broker secrets. Do not assume separate read-only Alpaca credentials exist. Enforce read-only access through internal interfaces even when the underlying account key is capable of trading. Existing constructors that instantiate trading clients from global settings must be kept out of research workers.

When the session runtime owns an account, legacy CLI/dashboard automated entry paths must route through the same gate or refuse to run. Acquire an exclusive account/environment lease to prevent accidental second runtimes. Uncoordinated trading from another machine remains an operational risk; detect external changes through reconciliation rather than claiming the local lock prevents them.

## 4. Responsibilities of the four agents

### 4.1 Shared operating contract

Each role receives `DataContext`, `AccountProfile`, the approved mandate and an injected clock. Market observations are constrained by `signal_as_of`, not by the role's name. In `FREE_DELAYED`, continuous monitoring means repeated updates to a moving delayed view. In `SUBSCRIBED_REALTIME`, it means updates from the entitled current feed. CloseGauss remains snapshot-based in both cases.

The shared suitability service separates **hard feasibility** from **evidence-based ranking**. Each report states what fits, what does not fit and what remains uncertain. A candidate can be technically attractive but unsuitable for the actual account or data latency. Every stage may return no trade. Section 11.6 specifies the calculation and report contract.

### 4.2 PostGauss — continuous post-market observation and review

**Inputs:** current broker fills/orders/activities and account snapshots; completed-session observations available under the data profile; approved strategies; versioned news/events.

**Workflow:**

1. Reconcile actual positions and outstanding orders immediately after the permitted trading window ends. Do not wait for delayed market data to become available.
2. Separate execution faults, data gaps and rule violations from ordinary trading losses. Adjust performance attribution for deposits, withdrawals and other non-trading cashflows.
3. Complete the market scan when the required data watermark covers the session. Record any difference between current account state and delayed market valuation.
4. Continue lightweight post-market observation: delayed historical polling in the free profile; entitled streaming in the subscribed profile.
5. Screen candidates using approved quantitative rules. Calculate preliminary account feasibility before spending on deeper research.
6. Review capital utilisation, liquidity capacity, exposure concentration and operating costs against the account's mandate. Flag unsuitable instrument/size combinations without changing policy.
7. Publish candidate versions and a `SuitabilityReview` for CloseGauss. Material later events remain in the append-only event log.

**Outputs:** `SessionReview`, `CandidateSet`, preliminary `AccountSuitabilityReport`, incidents and data-quality exceptions.

**Limits:** no orders, strategy promotion or forced candidate count. An LLM report failure must not affect broker reconciliation. Delayed marks are labelled, not substituted for current liquidation values.

### 4.3 CloseGauss — reproducible, account-aware batch research

**Inputs:** a frozen evidence snapshot, actual or explicitly hypothetical account-profile snapshot, approved strategy versions, prior validation results, data profile and finite research budget.

**Workflow:**

1. Freeze source versions, market/receipt cutoffs, account-profile ID, data-policy ID and cost assumptions.
2. Calculate which strategy/instrument combinations are feasible given capital, permissions, mandate, data latency, liquidity and operational readiness.
3. Reuse technical, fundamental and sentiment analysts against injected evidence; investigate counter-evidence and upcoming events.
4. Compare approved alternatives for the same thesis, such as permitted shares, a long option, a supported defined-risk structure or no trade. Include an alternative only when its own validation and data requirements pass.
5. Rank feasible choices using the declared objective and uncertainty, not raw expected dollar profit or LLM confidence alone.
6. Produce account-specific conditional plans with sizing envelopes, required evidence, invalidation conditions and suitability reasons. Final quantities remain the risk gate's responsibility.
7. When requested, repeat the assessment over operator-supplied capital scenarios; compare feasibility boundaries, constraints, cost burden and capacity. Clearly label all such results hypothetical.
8. Route new strategies, parameter changes and unsupported structures to the experimental queue rather than the live planning queue.

**Outputs:** `ResearchReport`, `AccountSuitabilityReport`, optional `CapitalScenarioComparison`, versioned `SessionPlan` records, rejection records and `StrategyExperiment` proposals.

**Limits:** no orders or policy changes; no unrestricted reads after the snapshot cutoff. Account-specific plans cannot be reused for another account without reassessment. More capital does not authorise a previously unapproved strategy.

### 4.4 PreGauss — continuous validation of information and account fit

**Inputs:** target-session plans, new events, profile-eligible pre-market observations, coverage/health state, event calendar, current broker account state and existing reservations.

**Workflow:**

1. Select the correct next trading session and reject expired or mismatched plans.
2. Monitor continuously with the active profile; reassess periodically and on material events. In the free case, report the latest observed market time rather than describe the opening conditions as current.
3. Refresh account feasibility after P&L, deposits/withdrawals, pending orders, assignments or permission changes. Never treat a capital increase as permission to relax limits.
4. Re-evaluate available price ranges, event restrictions, strategy latency tolerance and operating-cost assumptions.
5. Mark plans eligible, deferred or rejected with explicit data and suitability reasons. A delayed-data strategy must have its own approved evaluation contract.
6. Issue a versioned reassessment when the appropriate instrument or sizing envelope changes. Do not silently rewrite the existing plan or an open position's exit policy.
7. Publish readiness separately for research, paper mechanics and live execution. Fresh executable option checks, when required, remain deferred to a verified options session and suitable feed.

**Outputs:** `ValidationRecord`, refreshed `AccountSuitabilityReport`, plan events, subscription priorities and readiness report.

**Limits:** no orders, risk-budget increases or last-minute shortcuts. Missing current quotes are not waived because the account is larger or the research thesis is strong.

### 4.5 LiveGauss — ongoing profile-aware decisions and current account supervision

**Inputs:** eligible plans, market evidence with explicit as-of times, news-risk events, actual broker positions/orders, current account state and risk policy.

**Workflow:**

1. Confirm plan, account, strategy, data profile and execution-mode compatibility.
2. Evaluate triggers using completed observations permitted by the profile. A free-profile trigger at wall time `t` is based on the delayed information view, not an assertion about the market at `t`.
3. Recalculate feasible quantity and incremental exposure using current broker state, reservations and the intent's permitted price bounds. Reject the trade when minimum tradable size does not fit.
4. Require genuine current quotes for strategies whose execution contract demands them. Delayed/indicative data cannot satisfy that requirement.
5. Produce a typed order intent and request deterministic risk approval. No profile or subscription status automatically authorises broker writes.
6. Manage actual fills and exits from broker-confirmed quantities. Keep operational updates current even in the free profile; use an explicit degraded-supervision policy when current market prices are unavailable.
7. Handle invalidation, partial fills and cancel/fill races. Stop new entries and start planned closing attempts using actual session time, never a delayed clock.
8. Continue reconciliation and alerts until all exposure and unresolved orders are accounted for.

**Outputs:** decisions, suitability/risk rejections, authorised intents where permitted, actual position events and execution-quality records tagged with profile and account snapshot.

**Limits:** no forced trade, autonomous leverage increase or LLM dependency for protective action. Delayed-mode shadow and paper-mechanics results must not be presented as real-time strategy results.

## 5. Session scheduling and service lifecycle

### 5.1 Use a calendar service

Introduce `SessionCalendar` and an injected clock. Persist UTC timestamps; use `America/New_York` to interpret US exchange sessions and `Europe/London` for optional display. The Alpaca calendar returns session-specific opens/closes, including early closures [A02]. Do not derive holidays by weekday checks.

Distinguish the regular-session calendar from instrument-specific trading restrictions. A stock calendar alone must not authorise every option contract or extended-hours order. Use a conservative configured execution window inside the verified permitted window.

### 5.2 Proposed operating schedule

The scheduler uses **actual wall-clock session time**. Market availability affects which evidence a role can analyse; it must not shift broker closing times or emergency controls.

| Activity | `FREE_DELAYED` | `SUBSCRIBED_REALTIME` |
| --- | --- | --- |
| Market collection | Continuous bounded polling up to the delayed cutoff | Continuous entitled streaming plus historical recovery |
| Broker reconciliation | Immediate/current; not intentionally delayed | Immediate/current |
| PostGauss observation | Rolling delayed market view; current operational/event alerts | Rolling current market view; current operational/event alerts |
| Completed-session scan | Start only when the watermark covers the close, normally after close + 15 minutes + configured buffer/completion checks | Start when the session data pass completeness checks; no intentional 15-minute hold |
| CloseGauss | When the required snapshot is complete; a configured research offset is a scheduling preference, not a substitute for readiness | Same snapshot requirement, potentially available earlier |
| PreGauss | Proposed start open − 90 minutes; review every 10 minutes plus material-event checks; display delayed as-of time | Same review windows with current entitled observations |
| Pre-open readiness | Approximately open − 5 minutes, with unresolved current-price requirements explicitly deferred | Approximately open − 5 minutes, still conditional on execution-time checks |
| LiveGauss entries | Only strategies/modes permitted for delayed inputs; enforce the actual entry cutoff | Only approved strategies with current feed/quote checks; enforce the same actual cutoff |
| Planned closing attempts | Configured offset before the real permitted close; never delayed by 15 minutes | Same |
| Reconciliation/alerts | On events, periodically and after reconnects/closing attempts, including outside entry hours | Same |

Entry offsets, research offsets and holding periods are mandate settings, not universal account-size rules. Persist the computed schedule. Clamp invalid intervals on shortened sessions. If a delayed breakout arrives after its actual entry expiry, reject it; do not move the expiry to preserve the trade. Later market corrections create new evidence versions.

CloseGauss may use the same final historical session dataset in both profiles once it is available. Do not delay an already historical dataset for a further 15 minutes after receiving it.

### 5.3 Independent lifecycle states

Track each role as `IDLE`, `RUNNING`, `DEGRADED`, `FAILED` or `COMPLETED`. Separately track the runtime as `STARTING`, `RECONCILING`, `READY`, `ENTRY_PAUSED`, `MANAGE_ONLY` or `STOPPING`.

PostGauss and CloseGauss can overlap. `ENTRY_PAUSED` does not mean all processes stop. A cancelled research task does not cancel risk-monitoring tasks. Isolate task failures and bound worker CPU, memory and request concurrency.

For missed jobs after restart, use durable job IDs and input hashes. Rerun missing research if useful, but never execute an overdue entry simply because its scheduled job is now being replayed.

## 6. Shared market-data service

### 6.1 Two explicit operating profiles

Implement `FREE_DELAYED` and `SUBSCRIBED_REALTIME` as first-class profiles shared by all four roles. Do not let each role choose a different implicit meaning of "current".

| Property | `FREE_DELAYED` | `SUBSCRIBED_REALTIME` |
| --- | --- | --- |
| Market-information contract | Market observations become eligible no earlier than 900 seconds after their effective event time | No intentional market-information delay; actual transport and processing lag still measured |
| Primary stock path | Historical SIP polling with an eligible end cutoff | Entitled SIP streaming; REST for history/recovery |
| Data quality | Label source, coverage, actual age and completeness | Same; a paid connection can still be stale or incomplete |
| Options | Use only verified historical/indicative capabilities for research; preserve indicative provenance | Use entitled OPRA for current quote-dependent execution |
| Research/live roles | All four roles run on a moving delayed information view | All four roles run on the entitled real-time view; CloseGauss still freezes snapshots |
| Strategy choice | Requires explicit approval for the delayed evidence contract | Requires approval for its current-data contract |
| Execution permission | Determined separately by execution mode, strategy and risk policy | Determined separately; subscription never arms orders |

**Verified provider distinction:** Alpaca permits stock historical SIP requests with an `end` at least 15 minutes old; latest SIP endpoints require the relevant subscription [A01]. Basic also includes real-time IEX, while the paid Trading API plan includes consolidated stock/OPRA access [A03]. **Product decision:** this rebuild deliberately uses delayed consolidated stock evidence for its free profile, rather than mixing live IEX into that profile's signals. Legacy IEX functionality can remain separately labelled; it is not a silent fallback or a third default Gauss profile.

**Options qualification:** Alpaca describes indicative quotes as derived rather than actual OPRA quotes, and indicative trades as derived and 15-minute delayed [A10]. Therefore, "free = 15-minute-delayed" defines the system's evidence policy; it does **not** establish that every free options quote is a genuine delayed consolidated quote. Probe the exact historical endpoints and entitlements. Unavailable genuine quote history stays unavailable; never fabricate it or relabel indicative data.

Do not assume payment provides every historical dataset or redistribution right. Record endpoint-specific capabilities and verify access without buying a subscription automatically.

### 6.2 Evidence clock and the 15-minute cutoff

Introduce `EvidenceClock` alongside the actual runtime clock. Persist at least:

- `wall_time`: actual decision/runtime time.
- `effective_event_time`: trade/quote event time; **bar interval end** for completed bars, not its start label.
- `received_at`: when the system actually received the record.
- `available_at`: when the record/version became usable by this system.
- `signal_as_of`: latest market time permitted for the decision's evidence set.
- `configured_delay_seconds`, `observed_age_seconds`, `data_profile` and `quality_class`.

For a polling request under the free profile:

```text
safe_request_end = wall_time - 900 seconds - entitlement_boundary_buffer
signal_as_of <= safe_request_end
record is eligible only if:
    effective_event_time <= signal_as_of
    available_at <= wall_time
    received_at <= wall_time
    required interval is complete and quality rules pass
```

This implements **at least 15 minutes** of market-information delay. Polling, gaps and source corrections may make the observation older. Do not claim an exact 900-second delivery guarantee. Do not add 900 seconds to receipt time for records that are already sufficiently old; that would unintentionally create a double delay. Evaluate completed bars using their end times to avoid leaking the final minutes of an interval.

Under the subscribed profile, there is no intentional 900-second cutoff, but source/receipt times, completeness and instrument-specific freshness still apply. In both cases a watermark is dataset-specific; do not infer completeness for a missing symbol from another symbol's latest event.

Keep **strategy evidence** separate from **current operational state**. Never delay broker fills, balances, positions, permission changes, cancellations, session deadlines or operational alerts. Display delayed analytical marks separately from current broker-reported equity, with both timestamps.

### 6.3 Capabilities and profile selection

Replace global `is_pro_tier` inference with `DataCapabilitySnapshot`: account/environment identity, feed/endpoint, access outcome, quality class, coverage window, limits, checked time and expiry. Check stock, option, historical and news capabilities independently. A timeout is `UNKNOWN/UNAVAILABLE`, not proof of a free entitlement.

Configured profile is an operator choice. A subscribed account may deliberately run a delayed experiment. Conversely, setting `SUBSCRIBED_REALTIME` does not grant SIP/OPRA access. Report the requested and effective capability set; never silently pretend they match.

On lost required entitlement or a feed failure, pause dependent new entries. Do not automatically move a real-time strategy to delayed data. A deliberate mode change creates a new data-policy version, clears incompatible signal caches and requires plan/strategy revalidation. Existing position ownership and broker reconciliation continue.

### 6.4 Collection, transport and fan-out

Use one managed collector per required endpoint within verified connection limits. The free stock path uses incremental historical polling with overlap/deduplication, pagination, shared rate limiting and the safe cutoff. The subscribed path uses streaming plus recovery. Both adapters emit the same typed event envelope without disguising their timing or quality.

Prioritise holdings and outstanding orders, then approved candidates, benchmarks and discovery symbols. Never silently evict a held contract for a candidate. Subscribe only to required option contracts. Callback work is validation, timestamping and queueing, not LLM research.

Use bounded queues and durable handling for critical events. Under overload, record gaps and pause dependent entries. Broker events and critical news must not be dropped or blocked behind a historical scan. Batch broad research scans and prevent parallel retry storms.

### 6.5 Latency-aware strategy and execution policy

Keep data profile independent of `replay | shadow | paper | live`.

| Execution mode | Free delayed case | Subscribed real-time case |
| --- | --- | --- |
| Replay | Release evidence at its simulated availability time; model fills only after the actual decision/submission time | Same causal rules, using recorded real-time availability |
| Shadow | Full four-role workflow, delayed triggers and labelled hypothetical intents | Full workflow with current observations; still no broker writes |
| Paper | Permit only a strategy explicitly approved for delayed inputs and paper mechanics; broker orders/fills occur at current wall time | Permit approved paper strategies with required current quotes and account checks |
| Live | **Initial release: new entries disabled for this profile.** A later delayed-input live strategy needs a separate approval proving its order/exit controls do not depend on unavailable current quotes | Eligible only after live arming, strategy validation and all risk/data gates; not automatically enabled |

This initial free-profile live restriction is a project rollout policy, not an assertion that the broker prohibits orders from free accounts. Both profiles are fully implemented for monitoring, research and causal evaluation. Do not weaken a current-quote requirement just to make delayed paper orders pass; select a compatible test strategy or remain in shadow mode.

A strategy record specifies permitted profiles, maximum signal latency, required evidence quality, minimum observation history and execution-quote requirements. A longer holding period alone does not prove delay tolerance; validate it. A genuine 15-minute-delayed quote is still not a current execution quote. Indicative options never qualify as genuine OPRA by ageing them.

### 6.6 Freshness and quality checks

Separate **intentional delay**, **additional delivery lag** and **execution-quote freshness**. A healthy delayed stream must not fail merely because it exceeds a real-time two-second threshold, and it must not pass a current-execution check merely because it meets its own delayed threshold.

Validate non-finite values, spread sides, contract IDs, clock skew, supported sessions and cross-leg timestamp skew. Preserve source revisions rather than overwriting the historical decision input. No new observation may be due when an exchange is closed; that is different from a feed outage or an unchanged current market.

Store adjustment basis and keep raw execution prices separate from adjusted research series. Filter session data by actual calendar boundaries; a successful HTTP response or a daily bar alone is not proof of a complete intended session.

### 6.7 Required user-visible status

Every chart, candidate, plan and evaluation report must show `FREE_DELAYED` or `SUBSCRIBED_REALTIME`, actual market as-of time, observed lag, feed/quality class and execution readiness. Use "Delayed observation" rather than "Live price" in the free profile. Keep current broker position/order status visually distinct. An unsupported quote feed, unverified entitlement or stale snapshot is an explicit blocker, not an empty success.

## 7. News and event processing

### 7.1 Shared pipeline

Implement one pipeline:

**Receive → preserve source version → deduplicate → classify relevance → verify where required → emit event → update affected plans.**

Alpaca news messages provide article identifiers, creation/update timestamps and associated symbols [A04]. Preserve these fields and add local receipt time. Keep the original record or a permitted content reference alongside the derived annotation.

Deduplicate within and across providers without losing provenance or corrections. Use provider/article/version identifiers plus content hashes; an updated story is not a duplicate to discard. Several stories about the same event may form one event cluster, but their source records remain distinct.

### 7.2 Coverage and source use

| Event family | Initial use |
| --- | --- |
| Company announcements and filings | Identify catalysts and contradictions; verify important claims against primary disclosures when accessible |
| Earnings and other scheduled company events | Apply approved event restrictions and planned holding-window checks |
| Scheduled macro releases | Maintain a separate calendar; ticker filtering alone is insufficient |
| Sector-wide developments | Map relevance to affected candidates without assuming every sector constituent has identical exposure |
| Rumours, conflicting reports and unclear timestamps | Mark uncertainty; defer event-dependent entries rather than invent certainty |
| Corrections and retractions | Reassess affected plans and preserve the earlier decision record |

Create a pluggable `EventCalendarProvider`. The initial interface may ingest a verified, versioned operator-maintained calendar; automated primary-source adapters can follow. Missing a provider or calendar is a visible capability gap, not proof that no events are scheduled. This plan does not assume Alpaca news alone is a complete event calendar.

### 7.3 Agent-specific treatment

PostGauss identifies research leads and distinguishes the announcement from its observed price reaction. CloseGauss analyses implications and counter-evidence at a fixed cutoff. PreGauss checks what changed since that cutoff. LiveGauss applies predefined suspension and position-risk rules when material news arrives.

Initially, use news mainly for **eligibility, invalidation and risk**, rather than direct headline-to-order sentiment trading. A supportive headline does not remove the requirement for the approved price trigger and executable quotes.

Use urgency tiers. Critical verified events and explicit invalidations are processed immediately. Lower-priority stories can be grouped into short review batches. Never debounce critical broker/risk events behind an LLM summarisation queue.

### 7.4 Security and evidence rules

Treat article bodies, filings and web pages as untrusted data. They cannot supply executable instructions, alter system prompts, select broker endpoints or override risk policy. Restrict model outputs to validated schemas with enumerated actions and source IDs.

Research workers use allowlisted read-only tools. Exclude broker keys and unrelated user/account information from model prompts. Respect source retention/licensing terms: where full-text storage is not permitted, retain permitted metadata, references, hashes and derived annotations rather than unauthorised copies.

An LLM confidence score is an uncalibrated analytical output unless separately validated. It is not an estimated win probability or a risk-budget multiplier.

### 7.5 News timing under the two profiles

News has its own publication, update and receipt times; a stock-data subscription does not establish news latency or completeness. Collect entitled news when it arrives in both profiles, without an artificial 15-minute delay to operational alerts.

For the primary market/news **signal snapshot** in `FREE_DELAYED`, use article versions whose publication/update time is no later than `signal_as_of` and whose receipt/availability time is no later than the snapshot's wall-time cutoff. The aligned view must not describe an old price as the market reaction to a newer headline. Scheduled future events can still be known in advance: eligibility depends on when the schedule was published, not the event's future occurrence time.

A separately labelled current-news safety channel may veto or suspend a plan immediately on actually received material information. It cannot use a positive fresh headline to authorise an entry beyond the delayed strategy's approved evidence contract. PostGauss may flag that event as a new research lead, explicitly stating that the corresponding market reaction is not yet observable in the delayed view.

Under `SUBSCRIBED_REALTIME`, market and news observations can still arrive asynchronously. Record actual availability and avoid claiming synchronous knowledge. Capture safety-channel events and their receipt times in replay as well; do not retrospectively grant earlier access.

## 8. Data contracts and research handovers

### 8.1 Contract conventions

Use versioned Pydantic models at service boundaries, consistent with the repository's declared dependency [R10]. Reject unexpected fields for order/risk contracts. Use immutable serialised snapshots; a frozen dataclass containing a mutable dictionary is not sufficient isolation.

All persisted records carry `schema_version`, unique ID, creation time, account/environment scope where relevant, and provenance. Use UTC-aware datetimes. Use decimal-safe representations for money and prices; analytical arrays may use floating point with explicit finite-value checks.

### 8.2 Required records

| Contract | Required content |
| --- | --- |
| `TradingSession` | Session ID/date, calendar version, regular open/close, permitted execution bounds, observation windows, state |
| `DataCapabilitySnapshot` | Account/environment, endpoint/feed access, quality class, coverage, quotas, probe results and expiry |
| `DataContext` | Profile/policy version, wall time, signal as-of, configured/observed delay, per-source watermarks, completeness and execution-readiness state |
| `AccountProfile` | Actual/hypothetical marker, equity/cash/currency, usable buying power, positions/reservations, permissions, mandate, observation time and version |
| `AccountSuitabilityReport` | Account/profile/snapshot IDs, feasible/rejected alternatives, binding constraints, estimated sizing envelopes, objective, costs, uncertainty and reassessment triggers |
| `CapitalScenarioComparison` | Explicit scenario IDs, comparable assumptions, feasibility boundaries, strategy rankings, capacity/cost sensitivity and no-trade cases |
| `MarketEvent` | Source, event type, symbols, event/interval-end time, receipt/availability times, profile, feed, payload reference/hash and quality flags |
| `NewsEvent` | Provider/article/version IDs, source, publication/update/receipt times, event category, symbols, permitted content reference and verification state |
| `ResearchSnapshot` | Snapshot ID, target session, market/receipt cutoffs, data-context and account-profile IDs, immutable evidence manifest, feeds, completeness flags, strategy/config versions |
| `Candidate` | Underlying, originating snapshot/events, quantitative screen results, strategy/profile compatibility, preliminary suitability reference, rationale, exclusions and priority |
| `SessionPlan` | Plan/version IDs, candidate, target session, strategy/version, evidence/data-profile/account/suitability references, entry/invalidation rules, instrument constraints, sizing envelope, exit/risk-policy references and validity window |
| `ValidationRecord` | Exact plan version, validator version, validation time, signal as-of, account/suitability versions, event watermark, outcome, reasons and evidence IDs |
| `OrderIntent` | Intent ID, plan/version or managed-position reference, explicit open/close purpose, instrument legs, quantity request, price bounds, expiry and reason |
| `RiskDecision` | Intent ID, policy/data-profile versions, approved quantity/price envelope, fresh account snapshot, suitability reference, checks, reservations, decision time and expiry |
| `OrderRecord` | Stable client order ID, broker ID, intent mapping, submission state, cumulative fills, replaces/replaced-by links and reconciliation state |
| `PositionLeg` | Contract/asset ID, symbol, group ID, signed quantity, price units, multiplier, cost basis and broker reconciliation timestamp |
| `PositionGroup` | Strategy instance, originating plan, legs, exit policy, realised/unrealised P&L, reserved capital and operational state |
| `AgentRun` | Role, job ID, snapshot/input hash, account/data-profile IDs, prompt/model version where relevant, timing, token/cost accounting, output references and failure state |
| `SessionReview` | Reconciliation result, decisions/trades/rejections, execution metrics, incidents, costs and research follow-ups |

### 8.3 SessionPlan details

Separate the **entry price bound** from the forecast target and take-profit level. Define the conversion to existing `ActionPlan.target_price` explicitly; do not overload an optimistic forecast target as an executable order price.

Plans must specify the allowed holding horizon, exact strategy version, relevant data requirements and how invalidation is detected. The LLM may propose a permitted value, but deterministic validation constrains it to the approved strategy schema and operator policy.

Use a small rule vocabulary, such as `price_crosses`, `completed_bar_condition`, `spread_within_limit`, `no_blocking_event` and `time_in_window`. Rules reference approved indicators and operators. Do not execute arbitrary Python, SQL or `eval()` expressions from a plan.

A plan can target shares or a permitted option selector. An overnight option selector is provisional; record concrete contracts and quotes in the subsequent execution intent after the required execution checks.

Each plan is account-specific and profile-specific. Record `account_profile_id`, `suitability_report_id`, `data_profile`, `data_policy_version`, `signal_as_of` and required execution-quote quality. A reusable thesis or template is not an account-approved plan. Estimated capital needs and sizing envelopes are advisory; the gateway uses only a fresh risk-approved quantity. Hypothetical scenario plans carry `execution_eligible=false` and cannot be converted directly to actual order intents.

### 8.4 Illustrative serialised plan

This is a proposed schema example for a **synthetic test fixture**, not an actual market recommendation. `TEST_STOCK` must never resolve to a live order. The dates and prices belong to the test calendar.

```json
{
  "schema_version": 2,
  "plan_id": "fixture-plan-001",
  "version": 1,
  "target_session_id": "US_EQUITIES:2026-09-08",
  "created_at": "2026-09-07T22:00:00Z",
  "snapshot_id": "fixture-snapshot-001",
  "candidate_id": "fixture-candidate-001",
  "underlying": "TEST_STOCK",
  "strategy_id": "approved_breakout_retest",
  "strategy_version": "1.0.0",
  "direction": "long",
  "instrument_selector": {
    "asset_type": "stock",
    "allow_fractional": false
  },
  "entry_rules": [
    {
      "rule_type": "completed_bar_condition",
      "indicator": "close",
      "operator": "crosses_above",
      "value": "100.00",
      "timeframe": "5Min"
    },
    {
      "rule_type": "no_blocking_event"
    }
  ],
  "entry_price_bounds": {
    "max_buy_price": "100.10"
  },
  "invalidation_rules": [
    {
      "rule_type": "completed_bar_condition",
      "indicator": "close",
      "operator": "below",
      "value": "99.00",
      "timeframe": "5Min"
    }
  ],
  "exit_policy_id": "fixture-intraday-exit-v1",
  "risk_policy_id": "mandate-shadow-v1",
  "valid_from": "2026-09-08T13:35:00Z",
  "entry_expires_at": "2026-09-08T19:30:00Z",
  "evidence_ids": [
    "fixture-evidence-001"
  ],
  "limitations": [
    "Synthetic acceptance-test data only"
  ],
  "account_profile_id": "fixture-account-actual-001",
  "suitability_report_id": "fixture-suitability-001",
  "data_profile": "SUBSCRIBED_REALTIME",
  "data_policy_version": "fixture-realtime-v1",
  "signal_as_of": "2026-09-07T20:00:00Z",
  "execution_requirements": {
    "quote_quality": "genuine_current",
    "required_feed": "sip"
  }
}
```

For a delayed counterpart, generate a new plan/version with `FREE_DELAYED`, a 900-second signal-delay contract, compatible strategy approval and the appropriate evidence cutoff. Do not merely edit the displayed profile label or pretend an indicative quote meets the example's execution requirement.

Approval state is a separate durable projection of validation events, not a mutable field that the plan's author can self-approve. The example contains no quantity or broker credentials. Final sizing belongs to RiskGate.

### 8.5 Snapshot and point-in-time requirements

For a replay decision at wall time `t`, evidence must have an eligible source time under the selected profile and a receipt/availability time no later than `t`. In the free profile, apply the 900-second cutoff to the effective event or completed-bar end time. Keep broker state and safety-channel receipt times on the operational clock. Preserve later article/bar corrections as new versions. Snapshot manifests should identify the exact versions supplied to analysts.

Current fundamentals returned today cannot be treated as historical fundamentals merely by attaching an earlier `current_date`. When a point-in-time dataset is unavailable, label the experiment accordingly and exclude that path from historical performance claims.

CloseGauss must use a `SnapshotDataReader`, not an unrestricted provider. Refactor the current analysts' self-fetching paths to accept injected evidence. Leave the legacy analysis interface available for interactive use, but distinguish its provenance from reproducible research [R03A], [R06].

## 9. Plan, order and position state machines

### 9.1 Plan eligibility

```text
DRAFT -> RESEARCH_COMPLETE -> PENDING_VALIDATION -> ELIGIBLE
              |                    |                |
              +-> REJECTED         +-> DEFERRED      +-> ENTRY_EXPIRED
                                                    +-> REVIEW_REQUIRED
                                                    +-> INVALIDATED
                                                    +-> SUPERSEDED
```

A successful new validation can return a deferred/review-required plan to `ELIGIBLE` only through the authorised transition rules. A rejected or invalidated thesis requires an explicit new version or replacement plan, not an invisible reset.

Evaluate the current plan state again immediately before submission. Bind risk approvals to a plan version, account-state and suitability versions, data-policy version, event watermark and short validity interval. If a local relevant event arrives before submission, invalidate the approval. No design can eliminate events arriving after submission; handle those through cancellation, reconciliation and position policy.

Permit one initial entry per plan version in the first release. Re-entry needs an explicitly approved template rule and a new intent; it still consumes the session entry budget.

A data-profile change or material account-capability change moves affected eligible plans to `REVIEW_REQUIRED`. A compatible reassessment may restore eligibility; a different strategy/instrument requires a new plan version. Reduced affordability blocks new exposure without deleting actual holdings. Pausing the system does not freeze market time or extend plan validity.

### 9.2 Order intent and order lifecycle

```text
PROPOSED -> RISK_REJECTED
        -> RISK_APPROVED/RESERVED -> SUBMITTING
                                      |
                         ACKNOWLEDGED / UNKNOWN
                                      |
                  OPEN / PARTIALLY_FILLED / CANCEL_PENDING
                                      |
                  FILLED / CANCELLED / REJECTED / EXPIRED
```

Map broker-native states explicitly; this diagram is a local abstraction, not a claim that these names exactly match the broker API. Store raw broker states as well.

A submission timeout results in `UNKNOWN`, not an automatic new order. Query by the stable client order identifier and reconcile before any retry. Do not assume that repeating a POST with the same identifier guarantees broker-side idempotence. Replacement orders get their own IDs and lineage; they must not create a second independent risk reservation.

### 9.3 Position lifecycle

```text
PENDING_ENTRY -> PARTIALLY_OPEN -> OPEN -> EXIT_PENDING -> CLOSED
                       |            |          |
                       +------------+----------+-> RECONCILIATION_REQUIRED
```

Broker fills create exposure even if the entry plan has since expired. Preserve the position's own exit-policy version. Entry-plan invalidation cannot delete an actual holding.

A group is closed only when all its legs are reconciled to zero and no unresolved order can reopen exposure. A zero sum of signed quantities is never a flatness test.

### 9.4 Amendments and open positions

Store immutable plan revisions and append-only approval events. Use optimistic version checks for updates. A changed entry plan does not silently widen an open position's stop or extend its holding period. Risk-increasing changes require a separate, audited operator authorisation; the initial release should simply prohibit them.

Manual broker trades and assignments may introduce unmanaged exposure. Record and alert on it, block conflicting entries, and require an explicit adoption/exit policy. Do not liquidate an unfamiliar position solely because it is absent from the current watchlist.

## 10. Persistence, delivery and concurrency

### 10.1 Initial storage layout

Use local SQLite with WAL mode, short transactions, a busy timeout and explicit migrations. Keep the active database on the supervised host, not a shared network filesystem. Use a serialised transactional write path for account-critical updates.

Suggested logical tables:

| Group | Tables |
| --- | --- |
| Session/research | `sessions`, `agent_runs`, `snapshots`, `candidates`, `plans`, `plan_events`, `validations`, `strategy_approvals`, `suitability_reports`, `capital_scenarios` |
| Market/events | `market_events`, `news_versions`, `event_annotations`, `feed_health`, `data_capabilities`, `data_policy_versions`, `consumer_offsets` |
| Trading | `order_intents`, `risk_decisions`, `risk_reservations`, `orders`, `fills`, `position_groups`, `position_legs`, `account_snapshots`, `account_profiles` |
| Operations | `jobs`, `outbox`, `operator_commands`, `incidents`, `cost_ledger`, `schema_migrations` |

Some small tables may be combined during implementation, but preserve the logical boundaries. Store large permitted research payloads as content-addressed files with database manifests rather than bloating every plan row. Back up the database and referenced files consistently.

### 10.2 Transaction boundaries

Within one transaction, persist a plan-state change and its outbox event. Similarly, persist an order intent, risk reservation and submission work item together before contacting the broker.

Use at-least-once event delivery with idempotent consumers and durable offsets. Unique keys prevent applying the same article version, job result or fill twice. Do not claim exactly-once delivery across a local database and the broker.

For concurrent triggers, acquire the account-level decision lock, refresh the relevant risk projection, reserve capacity, then release the transaction before slow network calls. Keep reservations through pending/unknown states. Release unused capacity only when broker confirmation or a definitive pre-submission rejection makes that safe.

### 10.3 Single-account risk consistency

Every account/environment has isolated profiles, suitability reports, risk limits and reservations. Hypothetical scenarios use a non-executable scope and cannot write actual-account projections.

Two candidates must not independently spend the same cash or claim the last available position slot. Reserve capital, open-risk capacity and entry capacity for pending intents. Convert reservations to actual exposure as fills arrive.

Reconciliation uses both broker state and local intent history. Where the broker's reported buying power already reflects a known order, do not subtract its reservation a second time without a documented conservative reconciliation rule. Prefer a clearly conservative availability calculation to an optimistic one; record why capacity is unavailable.

### 10.4 Recovery semantics

At startup, load unresolved local intents, query current broker orders/positions/activities and rebuild projections. Acknowledge gaps and restore subscriptions for holdings before enabling new entries. If local state is corrupted or reconciliation remains ambiguous, use `MANAGE_ONLY` with an operator alert.

Keep complete audit history even when a plan is superseded. Deleting a watchlist row, restarting the UI or rotating a report file must not remove trading state.

## 11. Risk gate and execution integration

### 11.1 Keep the existing execution interface below the new gate

The target path is:

**Eligible account/profile-specific SessionPlan → profile-compatible rule evaluation → refreshed feasibility → typed OrderIntent → current RiskDecision → final price validation → existing ActionPlan/ExecutionDecision adapter → broker order.**

Use an adapter rather than breaking all existing strategies. Strip unknown metadata at the boundary. Fields such as `override_qty`, `legs`, `order_class` and price limits must come from validated contracts, not unrestricted LLM dictionaries.

The same gateway owns normal entries, replacements and exits. Exit intents may bypass entry-only restrictions, but never side/quantity/contract validation. An explicit sell-to-close with zero holdings must not become a sell-to-open operation.

### 11.2 Entry checks

| Check | Required behaviour |
| --- | --- |
| Plan and strategy | Correct session/version; approved strategy; current eligibility; satisfied trigger |
| Account state | Confirmed environment/account; valid balances; permitted asset/strategy; no blocking broker restriction |
| Data quality | Correct data profile and approved signal latency; current quote checks remain distinct from delayed-evidence freshness |
| Suitability | Valid account-specific report; strategy/instrument still feasible; actual quantity recalculated against current capital and reservations |
| Instrument | Tradable identifier, valid quantity units, tick rules, contract and expiry restrictions |
| Capital | Sufficient conservative buying power/cash after relevant pending reservations |
| Exposure | Account/group limits, duplicate position conflicts and correlated exposure rules |
| Loss policy | Per-trade planned loss, structural option loss and daily threshold not exceeded |
| Events | No blocking verified event; required event coverage available |
| Timing | Inside the approved entry window and before plan expiry |
| Operations | Runtime healthy enough for entry, audit persistence working, execution not paused |

A daily loss threshold is a stop-new-entries rule, not a guaranteed cap on losses. Define P&L as realised plus consistently marked unrealised P&L and fees, adjusted for external deposits/withdrawals. Persist the reference equity and calculation version for each session.

### 11.3 Separate allocation from loss sizing

Do not reuse `risk_pct` ambiguously. Introduce explicit fields for capital allocation, planned trade loss and options structural loss. Budgets derive from the actual account and approved policy; there is no universal simulated-equity fallback. A research sizing envelope is never a reservation or permission to trade.

For a long stock with a valid lower stop, a proposed sizing bound is:

```text
planned_loss_per_share = entry_price - stop_price + execution_cost_buffer
quantity <= floor(planned_loss_budget / planned_loss_per_share)
quantity <= floor(capital_allocation_budget / entry_price)
quantity <= affordable_quantity_after_reservations
```

Apply the smallest valid bound and instrument quantity rules. This is planned stop-based risk, not a hard maximum loss; gaps and execution failures can exceed it.

For a standard long option or debit structure:

```text
structural_debit_per_group = approved_net_debit * verified_multiplier
quantity <= floor(structural_loss_budget / (structural_debit_per_group + fees_buffer))
```

Validate the structure before using that formula. Do not apply it to uncovered short options, adjusted contracts or unsupported combinations. A displayed premium is not a total contract cost. If one contract exceeds the budget, reject it; do not raise limits automatically or search for a cheaper contract outside the approved selector.

### 11.4 Final-price and order controls

Validate the actual final order payload after tick rounding or price adjustment. The existing fixed-increment policy must not push a buy above the approved maximum or a sale below the minimum [R08]. Separate normal entry pricing from the pre-approved emergency exit procedure.

Use bounded limit-order attempts initially. Record time-to-fill, unfilled cancellations, partial fills and the quote observed at submission. Midpoint pricing is a proposed limit, not an assumed fill.

For each intent, cap replacements and repeated attempts separately from the daily entry count. Reserve an entry slot before submission. Consume the daily entry count on the first fill that creates exposure; release the slot on a confirmed terminal zero-fill outcome. Count failed attempts in an independent operational cap. Exits do not consume entry slots.

Alpaca supports client-assigned order identifiers and order-status retrieval [A05]. Use stable intent-derived identifiers within the verified API length/format rules. Reconcile uncertain submissions before retrying.

### 11.5 Operator controls

Provide distinct controls: `pause entries`, `resume entries`, `cancel pending entries`, `manage only`, and `request flatten`.

`Pause entries` is the default kill-switch action; it preserves monitoring and position management. `Request flatten` requires explicit confirmation, runs through the controlled gateway and reports remaining exposure. It is not a promise of immediate or guaranteed liquidation.

Resuming entries requires a readiness recheck. Tightening a limit can take effect immediately; loosening limits requires an explicit operator action and a new policy version. No agent can perform that action.

### 11.6 Account-adaptive strategy and instrument suitability

Implement `AccountSuitabilityService` as shared deterministic feasibility code plus an optional explanatory/ranking layer. The four agents use it; it is not a new broker-writing agent.

**Account inputs.** Use fresh, versioned broker equity, cash, currency, usable buying power, account restrictions, options permissions, existing positions, pending commitments and measured concentration. Combine these with the operator's permitted holding horizon, asset/strategy allowlist, risk limits, leverage policy, cost budget and active data profile. Unknown required fields produce an explicit blocked/uncertain result. A larger headline equity value does not establish spare cash, supported margin or options permission.

**Assessment process:**

1. Build a feasible set from separately approved strategies; exclude unsupported assets, permissions, holding horizons and data requirements.
2. Calculate each minimum tradable unit's capital requirement, planned loss, structural/stressed exposure, estimated fees/slippage and liquidity capacity. Keep these quantities distinct.
3. Compare with actual deployable capital and pending reservations. For options, use verified contract units and supported structure economics; for shares, use permitted fractional/whole-share increments.
4. Determine whether a positive valid quantity exists. Return `NO_FEASIBLE_SIZE` when none exists, rather than widening risk limits or searching outside the approved instrument selector.
5. Assess incremental portfolio concentration, scenario losses, shared risk factors, cash buffers, assignment exposure and the cost of exiting at the proposed size.
6. Rank only feasible choices using the declared objective: for example, validated net expectancy subject to drawdown/tail-risk and capacity constraints. Show assumptions, uncertainty and binding constraints. Do not invent probabilities, weights or profit forecasts when evidence is missing.
7. Compare the leading choice with no trade. Return `INSUFFICIENT_EVIDENCE`, `COST_NOT_JUSTIFIED` or `DATA_PROFILE_INCOMPATIBLE` where appropriate.
8. Recompute before a new order and after material account/data changes. The risk gate independently verifies all hard limits; a suitability score cannot override it.

**Capital feasibility boundaries.** As a diagnostic for a supported minimum position, let `R_min` be its policy-relevant loss amount, `C_min` its capital requirement, `r` the approved loss fraction and `a` the approved allocation fraction. When these quantities are defined and positive:

```text
illustrative_minimum_equity_bound = max(R_min / r, C_min / a)
```

This is only a bound for those two constraints, not a broker eligibility threshold or sufficient suitability test. Available cash, reservations, permissions, liquidity, costs and portfolio exposure can still reject the trade. For a share stop, `R_min` is planned loss, not a guaranteed maximum. Do not apply a debit-loss formula to an unsupported option structure.

**Different capital amounts.** Accept a user-supplied scenario file containing explicit hypothetical balances and associated account constraints. No production balance bands are hard-coded. Evaluate both capital-only sensitivity with other assumptions held constant and clearly labelled scenarios with changed permissions/liquidity. Include below/at/above minimum-feasibility boundaries and beyond estimated execution capacity. Increasing capital can make a structure feasible, but does not automatically make it the preferred choice or justify proportional scaling.

**Required suitability report:**

| Report element | Content |
| --- | --- |
| Identity | Actual/hypothetical account, snapshot time, data profile, mandate and strategy versions |
| Alternatives | Approved candidate/strategy/instrument combinations plus no trade |
| Hard checks | Permissions, data latency/quality, capital, quantity units, exposure and lifecycle support |
| Feasible sizing | Preliminary quantity range and binding limit; no order authority |
| Cost burden | Trading costs plus declared allocation of data, model and hosting costs; normalised to capital where meaningful |
| Ranking | Objective, evidence, uncertainty and why the best-supported feasible choice outranks alternatives |
| Rejections | Structured reasons and what would have to change; no suggestion to bypass risk policy |
| Refresh conditions | Account/profile/event/quote changes requiring a new assessment |

An LLM may explain the comparison and propose research hypotheses. It cannot choose a fictitious balance, amend approved risk fractions, grant permissions or turn a hypothetical scenario into execution. Existing positions retain their management policy when a fresh assessment recommends no new trade.

## 12. Options-specific readiness requirements

### 12.1 Contract and group modelling

Replace aggregate option quantities with `PositionLeg` records keyed by actual contract ID/symbol and linked to a `PositionGroup`. Keep underlying-symbol signals separate from option-symbol execution and valuation.

Filter positions by group/contract rather than pooling every account option into each underlying engine. A vertical with `+1` and `−1` contracts remains an open group. Multiple strategies sharing a contract require a defined allocation/reconciliation method; prohibit that overlap initially to keep ownership unambiguous.

Fetch and validate multiplier, deliverable, expiry and contract status. Initially reject non-standard/adjusted contracts instead of inferring their economics from the symbol. Broker contract records include contract size and dated open-interest information [A06]; open interest must not be represented as live liquidity.

### 12.2 Quote-based monitoring

Subscribe to selected contracts and all held legs. Keep the stock stream for underlying triggers and the option quote stream for premium/execution checks. Alpaca documents separate indicative/OPRA option feeds and quote events [A07]. Set the intended feed explicitly rather than relying on an SDK default.

For a conservative hypothetical immediate-liquidation mark, use the bid for a long leg and the ask for a short leg, multiplied by the signed quantity and verified multiplier. Add realised cashflows and fees consistently. This is a risk mark, not a guaranteed obtainable fill; displayed size, quote age and market conditions still matter.

Do not sum asynchronous leg quotes and label the result an executable complex-order quote. Check timestamp skew across legs, report synthetic marks explicitly and apply conservative execution assumptions. Treat unreliable marks as uncertainty that blocks new exposure, not zero P&L.

### 12.3 Submission and lifecycle gates

Before enabling options, require:

1. Contract/group state and cost-basis tests, including zero-net-quantity spreads.
2. Premium-unit and multiplier-aware sizing, including overrides and final prices.
3. Actual account permissions and options buying-power checks; do not infer option-spread permissions solely from stock shorting flags.
4. Explicit open/close intent for each leg and validated leg ratios.
5. Supported combined-order submission for spreads; no ad hoc legging in the first release.
6. Whole-contract quantities and verified time-in-force/order-type restrictions.
7. Partial-fill, cancel/replace, late-fill and rejected-exit reconciliation tests.
8. Contract-specific expiry restrictions and an early closing buffer.
9. Exercise/assignment detection through account activities and positions.
10. Continued supervision if an option becomes an underlying stock position or a residual leg remains.

Alpaca's documented option validations include whole-number quantities, no notional field, no equity-style extended-hours flag, and restrictions on stop orders for multi-leg orders. Its documentation also states that assignment events require REST polling rather than only WebSocket monitoring [A06]. Build contract tests against the installed SDK and the intended paper environment rather than assuming all combinations are supported.

### 12.4 Initial options policy

Keep `options_enabled=false` for the first shadow/paper-stock release. Then enable tested long calls/puts before more complex structures. Enable defined-risk spreads only after group-level entry and exit tests pass. Keep uncovered shorts, wheel automation, calendars, ratios and expiry-day positions outside the initial session strategy allowlist.

The options selection process must be account-adaptive: compare permitted structures by minimum unit cost, structural/stressed risk, cash/collateral needs, current exposure and data quality. Capital alone does not approve a structure, and larger accounts remain subject to the same lifecycle readiness gates. In `FREE_DELAYED`, label historical/indicative research correctly; it cannot satisfy the OPRA execution gate.

A theoretical defined-risk payoff assumes the intended structure is maintained. Assignment, failed exits or residual positions may create different exposures. The bot must supervise the actual broker holdings, not only the intended strategy payoff.

For intraday option research, evaluate the distribution of the option's exit value over the intended holding period, including spreads and volatility changes. A forecast of terminal intrinsic value or the underlying's direction alone is not an intraday execution model. Do not import expiry-payoff examples as validated intraday strategy rules.

## 13. Repository changes and interfaces

### 13.1 Proposed additions

The following paths are proposed, not existing. Keep small related components together initially; split them only when their complexity warrants it.

```text
src/
  session/
    __init__.py
    models.py               # Versioned contracts and enumerations
    calendar.py             # Trading sessions, actual deadlines and injected clock
    evidence_clock.py       # Profile-specific as-of and record-release rules
    controller.py           # Role scheduling and service lifecycle
    store.py                # Persistence interfaces and transactional operations
    jobs.py                 # Durable jobs, leases and bounded worker execution
    approvals.py            # Strategy approval and plan transition policy
    evaluator.py            # Approved entry/invalidation rule evaluation
    post_gauss.py
    close_gauss.py
    pre_gauss.py
    live_gauss.py
    migrations/             # Versioned schema changes
  data/
    feed_policy.py          # Two profiles, per-endpoint entitlements and quality policy
    delayed_adapter.py      # Incremental historical polling with the 900-second cutoff
    realtime_adapter.py     # Entitled streaming and gap recovery
    event_service.py        # Shared collection, fan-out and recovery
    event_calendar.py       # Scheduled-event interface and initial adapter
    snapshot_reader.py      # Immutable research evidence access
  account/
    suitability.py          # Feasibility, account-aware comparison and sizing envelopes
    scenarios.py            # Explicit non-executable hypothetical account scenarios
  trade/
    risk_gate.py            # Portfolio-wide checks and reservations
    execution_gateway.py    # Single controlled broker-writing interface
    reconciler.py           # Orders, fills, activities and actual exposure
    position_groups.py      # Contract/group state and valuation
  ui/
    session_views.py        # Four-agent overview and controlled commands

tests/
  unit/session/
  unit/risk/
  unit/data_profiles/
  unit/account_suitability/
  unit/options/
  integration/
  replay/
  fixtures/                 # Synthetic or sanitised data only

docs/
  FOUR_AGENT_IMPLEMENTATION_PLAN.md
  FOUR_AGENT_OPERATIONS.md
  FOUR_AGENT_VALIDATION.md

config/
  gauss.free-delayed.example.toml
  gauss.subscribed-realtime.example.toml
  account_scenarios.example.toml  # User-supplied capital cases; no production balance default
```

### 13.2 Existing-file changes

| Existing file/module | Planned change |
| --- | --- |
| `src/settings.py` | Add explicit data profiles, account/suitability settings and scenario validation; remove implicit Gauss simulation-balance defaults; preserve legacy keys without silent safety conflicts |
| `src/strategy/base.py` | Preserve existing interfaces; define explicit execution-adapter semantics and typed metadata whitelist |
| `src/strategy/registry.py` | Expose supported data profiles, latency/quality contracts, instrument units and capital/risk requirements; keep availability distinct from approval |
| `src/agent/multi_agent/agents.py` | Support injected snapshot evidence and source references |
| `src/agent/multi_agent/orchestrator.py` | Accept role budgets, profile-constrained evidence and suitability inputs; do not make it the session controller |
| `src/strategy/multi_agent_strategy.py` | Preserve legacy path; avoid nested event-loop execution in the new runtime |
| `src/data/alpaca_provider.py` | Implement verified delayed historical and subscribed streaming adapters; separate trading/account access; preserve real/indicative provenance and independent option entitlements |
| `src/data/news_provider.py` | Preserve versions/timestamps; reuse common normalisation in historical and streaming paths |
| `src/trade/live/live_trading_base.py` | Accept approved-plan evaluation and shared services; isolate entry generation from position management |
| `src/trade/live/live_trading_stock.py` | Delegate calendar checks and subscriptions; prohibit independent entries under session ownership |
| `src/trade/live/live_trading_option.py` | Replace aggregate monitoring with contract/group-aware behaviour before enablement |
| `src/trade/live/live_runner.py` | Allow service-owned feed dispatch; isolate failures without abandoning holdings |
| `src/trade/engine/execution.py` | Instrument-aware sizing, explicit intents, final payload checks and no override bypass |
| `src/ui/dashboard.py` | Read session state and submit authenticated commands; stop owning duplicate session feeds |
| `main_cli.py`, `live_script.py` | Add session commands; enforce account ownership and mode safeguards |
| `.env.example`, `README.md` | Document the two data profiles, separate execution modes, account-adaptive suitability, no assumed balance, source constraints and live-arming controls |
| `.gitignore`, `pyproject.toml`, `requirements.txt` | Track tests; retain runtime exclusions; include new packages and consistent dependencies |
| `.github/workflows/` | Add isolated automated tests and wheel/import checks; do not configure live trading jobs |

### 13.3 Internal interface contracts

| Interface | Proposed operations and restrictions |
| --- | --- |
| `SessionCalendar` | `get_session(date)`, `next_session(after)`, `execution_window(instrument, session)` |
| `EvidenceClock` | `signal_as_of(wall_time, profile)`, `is_eligible(record, context)`; use interval-end time for bars |
| `DataPolicy` | Resolve profile/endpoint capabilities; reject unauthorised fallback; classify freshness versus intentional delay |
| `EventService` | `subscribe(filter)`, `latest_observation(instrument, context)`, `current_execution_quote(instrument)`, `health()`; unavailable current quotes return a blocker, never a delayed substitute |
| `AccountProfileProvider` | Load current actual broker state or an explicitly isolated scenario; no fabricated/default capital |
| `AccountSuitabilityService` | `assess(account, context, candidates, policy)`, `compare_scenarios(...)`; return constraints, sizing envelopes and supported rankings, never orders |
| `SnapshotDataReader` | `bars(...)`, `news(...)`, `fundamentals(...)` constrained to a manifest; rejects out-of-snapshot reads |
| `PlanStore` | Append plan/version, append validation, transition with expected version, query eligible plans |
| `PlanEvaluator` | Evaluate approved rules against profile-eligible evidence and current plan status; deterministic, with recorded wall/as-of times |
| `RiskGate` | Validate intent and reserve capacity atomically; return approval/rejection with reasons |
| `ExecutionGateway` | Submit/cancel/replace/query authorised intents; persist results and uncertain states |
| `Reconciler` | Reconcile orders, positions and activities; return discrepancies and readiness state |
| `AgentRunner` | Run one role job with deadline/budget/input hash; persist result or failure |

Use native async calls where appropriate. Do not invoke `run_until_complete()` inside an already-running event loop. Wrap unavoidable blocking SDK operations in bounded adapters; keep research work outside risk-sensitive callbacks.

## 14. Configuration and operating budgets

### 14.1 Configuration contract

Extend the existing TOML/settings loader. All fields below are **proposed interfaces**, not currently supported commands. Validate unknown safety keys and incompatible combinations. Supply two named example configurations, sharing the same schema and differing in the data profile; do not maintain separate strategy implementations for free versus paid access.

```toml
[gauss]
enabled = true
mode = "shadow"                       # replay | shadow | paper | live
account_environment = "paper"
live_trading_enabled = false
account_id_allowlist = []              # Explicitly populated before live arming
calendar_timezone = "America/New_York"
display_timezone = "Europe/London"

[gauss.storage]
database_url = "sqlite:///runtime/gauss_session.db"
payload_directory = "runtime/evidence"

[gauss.account]
profile_source = "broker_snapshot"    # broker_snapshot | explicit_scenario
snapshot_max_age_seconds = 30
require_currency_match = true
# No initial-equity default. Offline/scenario runs must supply an explicit profile.

[gauss.data]
profile = "FREE_DELAYED"               # FREE_DELAYED | SUBSCRIBED_REALTIME
allow_silent_feed_fallback = false
require_endpoint_entitlement_checks = true

[gauss.data.free_delayed]
market_delay_seconds = 900
entitlement_boundary_buffer_seconds = 5
poll_interval_seconds = 60
max_additional_lag_seconds = 120
stock_history_feed = "sip"
options_policy = "verified_history_or_indicative_research"
allow_current_iex_in_signals = false

[gauss.data.subscribed_realtime]
stock_feed = "sip"
option_feed = "opra"
intentional_market_delay_seconds = 0
stock_execution_quote_max_age_seconds = 5
option_execution_quote_max_age_seconds = 2

[gauss.news]
signal_alignment = "profile_market_cutoff"
current_safety_alerts_enabled = true
news_required_for_event_strategies = true

[gauss.universe]
scan_universe_limit = 200
active_candidate_limit = 5
reserve_candidate_limit = 5
strategy_allowlist = []                # References separately approved versions

[gauss.schedule]
close_research_offset_minutes = 45     # Still requires a complete snapshot
pre_start_minutes_before_open = 90
pre_review_interval_minutes = 10
entry_start_minutes_after_open = 5     # Actual session clock, not evidence clock
entry_stop_minutes_before_close = 30
close_attempt_minutes_before_close = 15
signal_timeframe = "5Min"              # Strategy must be approved for its data profile

[gauss.suitability]
required_before_plan_publication = true
required_before_pre_validation = true
required_before_entry = true
allow_no_trade = true
allow_capital_band_shortcuts = false
ranking_objective = "validated_net_expectancy_subject_to_risk"
include_operating_costs = true
include_liquidity_capacity = true

[gauss.evaluation]
capital_scenario_file = "config/account_scenarios.toml"
require_explicit_scenario_capital = true
scenario_outputs_execution_eligible = false

[gauss.risk]
policy_id = "operator-mandate-v1"
max_new_entry_groups_per_session = 3
max_open_position_groups = 5           # Policy ceiling, not a target allocation
max_entry_order_attempts_per_session = 6
max_replacements_per_intent = 2
max_capital_allocation_pct = "0.10"
planned_loss_per_trade_pct = "0.01"
max_option_structural_loss_pct = "0.01"
daily_loss_pause_pct = "0.02"
allow_leverage = false
allow_uncovered_options = false
allow_overnight_positions = false
options_enabled = false
allow_expiry_day_entries = false
allow_free_delayed_live_entries = false

[gauss.research]
paid_model_calls_enabled = false
max_parallel_jobs = 2
max_job_seconds = 300
session_llm_budget_equity_fraction = "0.0001"
session_llm_absolute_cap_usd = "5.00"
post_budget_share = "0.10"
close_budget_share = "0.60"
pre_budget_share = "0.20"
live_budget_share = "0.10"
```

To configure the subscribed profile, select `profile = "SUBSCRIBED_REALTIME"` in `[gauss.data]` and verify the required endpoint entitlements. This is a configuration choice, **not** a subscription purchase, a live-trading switch or a strategy approval. Retain the profile-specific delay/quality metadata in every report.

The example's numerical settings are illustrative operator ceilings and scheduling choices, not universal recommendations or values inferred from account amount. The suitability service can recommend fewer positions, smaller allocations or no trade; it cannot raise these ceilings. Data freshness values must be tested, not loosened to make a trade eligible. The real-time quote thresholds must not be applied to classify delayed research data as an outage.

The empty strategy allowlist prevents entries. Missing explicit replay/scenario capital fails scenario validation; missing actual broker capital blocks new entries. There is no synthetic balance fallback in runtime operation. The scenario file is required only for comparative/offline scenario runs, not for an actual-account run that does not request them.

`allow_free_delayed_live_entries=false` is the initial rollout gate from Section 6.5. Merely changing it does not authorise a delayed strategy: separate strategy, data, exit-control and operator approvals remain required. A genuine current-quote requirement is never removed by a flag.

`allow_overnight_positions=false` is an operating objective and entry restriction, not a promise that every exit fills. Residual exposure stays supervised. Existing general `TradingLimits` and Gauss limits must not conflict silently; use documented precedence and the stricter safety bound unless explicitly resolved.

### 14.2 Account scenarios and profile validation

Comparative scenario records must supply their own identity, equity, cash, currency, buying-power assumptions, current holdings/reservations, permissions, mandate and data profile. State which inputs are held constant when changing capital. Never derive permissions, margin or subscription status from equity alone. A capital-only experiment can clone a permitted scenario and vary its funds; an account-configuration experiment must disclose every changed assumption.

Provide a validator that rejects missing, negative/non-finite or inconsistent amounts; undefined required currency conversion; unsupported permissions; and attempts to mark a hypothetical scenario executable. Require at least two explicit scenarios for a comparative report. Tests can generate below/at/above instrument feasibility boundaries rather than embedding production account-size tiers.

Validate combinations of execution and data modes before starting: an unentitled subscribed profile, a current-quote strategy under the free profile, or an actual-account run using hypothetical capital must produce a clear error/readiness blocker.

### 14.3 Budget enforcement and account economics

Persist research budgets by account, session and role. Derive an allowed model-spend envelope from the operator's absolute cap and equity-normalised cap, taking the stricter bound. Reserve estimated call cost before starting; account for actual use, bounded output tokens and retries afterwards. Unknown pricing or unavailable account budget inputs block paid calls rather than disable accounting. Optional scenario-research expenditure may use a separate explicitly approved research budget; it must not pretend to be funded by hypothetical balances.

Budget exhaustion stops optional analysis, not collection, reconciliation or risk management. Cache reuse requires matching evidence, profile, mandate and relevant account assumptions. Market-only analytical summaries may be shared where their inputs genuinely match; account-specific suitability and sizing must be reassessed.

For each actual account or scenario, report trading fees/slippage assumptions, allocated data/model/hosting costs and cost as a fraction of capital where meaningful. Avoid charging the same shared subscription repeatedly within one account's report. Specify the allocation method in cross-account research; do not infer market-data sharing rights from technical access.

An agent may recommend evaluating a data subscription or a simpler strategy, with assumptions and uncertainty. It cannot buy a subscription or guarantee that trading returns will cover it. Staying in free delayed observation/research is a valid operating choice for any account amount.

## 15. CLI and dashboard changes

### 15.1 Proposed CLI

These commands are an interface specification for future implementation, not commands available in the reviewed version.

```bash
python main_cli.py session validate-config --config config.toml
python main_cli.py session doctor --config config.toml
python main_cli.py session run --mode shadow --data-profile FREE_DELAYED --config config.toml
python main_cli.py session run --mode shadow --data-profile SUBSCRIBED_REALTIME --config config.toml
python main_cli.py session assess-account --config config.toml
python main_cli.py session compare-capital --scenarios config/account_scenarios.toml --data-profile FREE_DELAYED
python main_cli.py session compare-capital --scenarios config/account_scenarios.toml --data-profile SUBSCRIBED_REALTIME
python main_cli.py session status
python main_cli.py session plans --session latest
python main_cli.py session pause-entries --reason "operator review"
python main_cli.py session manage-only --reason "data incident"
python main_cli.py session reconcile
python main_cli.py session report --session latest --format markdown
python main_cli.py session replay --fixture tests/fixtures/gauss_session_day.json
```

`doctor` verifies configuration, both clock semantics, storage, account identity, permissions, requested/effective data profile, endpoint access, strategy/data compatibility and suitability readiness. It reports the actual market as-of time and blockers. `assess-account` produces an account-specific feasibility report without submitting orders; `compare-capital` emits explicitly hypothetical results that cannot enter the execution queue. It must not place an order. `status` reports the last successful reconciliation and current coverage limitations, not just whether a process is running.

Define live arming as a separate explicit operator action bound to account/environment, risk-policy version, data-policy version, permitted strategy profiles and deployment version. No CLI default or inherited `.env` URL may silently switch shadow/paper execution to live.

### 15.2 Dashboard

Add a **Gauss Session** view with four role cards, the current session timeline, approved plans, rejection reasons, feed freshness and account exposure. Include a plan detail view showing evidence, revisions, validation history and linked orders/fills.

Display separate health states for price data, news, broker stream, reconciliation and model workers. Show a prominent data-profile badge, market as-of time, nominal/actual lag and genuine/indicative coverage labels. Distinguish intentional 15-minute latency from an outage and from current broker state. Show both trading P&L and allocated operating costs without double counting spreads already included in fills.

Add an **Account Suitability** panel showing actual equity/deployable capital, permissions, selected approach, feasible sizing envelope, binding limits, rejected alternatives and costs. Add a separate hypothetical capital-comparison view with execution disabled; never display hypothetical balances as the actual account.

Provide audited pause/manage-only/cancel-entry commands. Flatten and live-arming controls require explicit confirmation. UI authentication and the local control channel must prevent unauthorised commands. A browser refresh must not create a second runtime, duplicate streams or duplicate orders.

## 16. Implementation work packages

Use dependency-ordered, reviewable changes. The labels below are proposed work items/PR names; they are not created GitHub issues. Each package includes tests and documentation, not only implementation files. There are 12 packages, including WP03A for account-adaptive suitability. Both data profiles are release requirements, not optional follow-on implementations.

### WP00 — Baseline and regression harness

**Dependencies:** none.

Capture the reviewed commit, Python/SDK versions and existing entry-point behaviour. Remove the `tests/` ignore rule, add synthetic fixtures and introduce isolated CI. Establish representative existing stock/crypto/analysis smoke tests. Inspect dependency consistency and build a wheel to identify missing packages.

**Done when:** tests are tracked, no test requires live credentials, and baseline failures are documented rather than concealed. Covers T01, T02 and T35.

### WP01 — Execution safety prerequisites

**Dependencies:** WP00.

Address F01–F03 and F13 in the entry path: invalid account values, allocation versus loss sizing, quantity overrides, explicit close intents and final-price bounds. Quarantine the legacy options monitoring path from the new runtime until repaired. Introduce mode-enforcing gateway interfaces and account ownership checks. Separate data profile from execution mode; forbid hypothetical capital in actual-account execution.

**Done when:** shadow mode cannot submit orders, invalid balances cannot create entries, and close-only intents cannot open exposure. Covers T03, T04, T13, T15 and T31.

### WP02 — Contracts, store and session calendar

**Dependencies:** WP00.

Implement versioned models, migrations, immutable snapshots, append-only plan events, transactional outbox, injected clock and calendar windows. Define account/environment scope, job identity, plan transitions and reservation interfaces. Add `DataContext`, `AccountProfile`, suitability-report references and separate actual/evidence clocks.

**Done when:** a synthetic session and its revisions survive restart; holidays/early closes and conflicting state updates are handled deterministically. Covers T05, T06, T09, T10 and T30.

### WP03 — Shared collection, feed policy and news

**Dependencies:** WP02.

Extract collectors from UI/engine ownership. Implement `FREE_DELAYED` historical polling, `SUBSCRIBED_REALTIME` streaming, the 900-second evidence cutoff, entitlement checks, actual-vs-additional lag, no double delay and no silent IEX/indicative substitution. Add pagination, quotas, gap recovery, news versions and profile-aligned versus current-safety event channels. Provide `SnapshotDataReader`, recording/replay and holdings priority.

**Done when:** both profiles pass timestamp/coverage tests, current broker events are not delayed, dependent entries reject gaps and research cannot read outside its snapshot. Covers T07, T08, T19–T22, T27 and T36–T45.

### WP03A — Account profiles and suitability evaluation

**Dependencies:** WP01–WP03.

Implement actual/scenario profile loading, deterministic feasibility, instrument-aware minimum sizing and capacity checks, cost accounting and declared ranking objectives. Provide `AccountSuitabilityReport` and `CapitalScenarioComparison`, including rejection reasons and no-trade outcomes. Keep scenarios isolated and remove any implicit default starting balance from the new runtime and examples. Add refresh rules for account, quote, policy and data-profile changes.

**Done when:** the same candidate can be accepted, resized or rejected under different explicitly supplied account conditions with auditable reasons; no report can alter broker equity or bypass RiskGate. Covers T04, T14, T32 and T46–T58.

### WP04 — PostGauss and CloseGauss

**Dependencies:** WP02–WP03 and WP03A.

Implement reconciliation-backed reviews, profile-timed candidate screening, account-feasibility filters, snapshot jobs, counter-evidence analysis and conditional plan generation. CloseGauss compares approved instruments/strategies for the actual account and optional capital scenarios, with evidence/uncertainty rather than a balance-band lookup. Adapt existing analytical agents to injected evidence. Add bounded model calls and a separate experimental-strategy queue.

**Done when:** both roles produce durable artefacts, can return an empty candidate list, and cannot place orders or promote unapproved strategies. Covers T11, T22–T24, T34, T46–T49, T54 and T58.

### WP05 — PreGauss and plan eligibility

**Dependencies:** WP04.

Implement rolling validation against profile-eligible market evidence, current safety events, endpoint capabilities and actual account changes. Refresh suitability after balance, buying-power, permission, reservation or data-profile changes; preserve version lineage. Add conditional approval, deferral, rejection and suspension transitions. Prevent last-minute candidates from bypassing the same contract. Implement late-event invalidation and final pre-open readiness records.

**Done when:** a valid plan can be suspended by new evidence and cannot enter until the required revalidation succeeds. Covers T09–T12, T20, T44, T50 and T55.

### WP06 — LiveGauss shadow integration

**Dependencies:** WP01 and WP05.

Connect approved-plan evaluation to profile-constrained completed bars/quotes. Revalidate actual-account feasibility before each intent and keep operational deadlines on the current clock. Add portfolio-wide checks, persistent reservations, simulated intents and position-policy evaluation. Wire all four roles to the controller and run the complete recorded-session scenario.

**Done when:** the full cycle works without broker writes, every decision has a plan/evidence lineage, and a missing research job does not disable supervision. Covers T03, T12, T14, T23–T25, T36–T39, T43 and T46–T53.

### WP07 — Controlled paper execution and reconciliation

**Dependencies:** WP06.

Connect approved intents to paper order adapters, client IDs, broker updates and REST reconciliation. Implement unknown submission states, partial fills, cancel/fill races, restarts, closing attempts and daily-risk accounting. Poll required activities as well as positions/orders. Compare delayed-input paper decisions with current broker fills without backdating execution or weakening unmet quote requirements.

**Done when:** no duplicate position is created in crash/timeout tests, risk is reserved consistently and paper fills reconstruct actual state. Covers T13–T18, T25, T26, T32–T33, T38–T39, T43 and T52.

### WP08 — Options contract/group readiness

**Dependencies:** WP07.

Fix F06–F07, add contract metadata, actual option quote monitoring, unit-safe sizing and group-level valuation/exit handling. Validate permissions, per-account minimum-size feasibility, feed quality and supported order combinations. Indicative research support does not satisfy genuine OPRA execution readiness. Exercise assignment/residual-leg scenarios in fixtures; then perform explicitly configured paper tests.

**Done when:** all options blockers in Section 12 pass. Enable long-option paper operation first; spreads require their own acceptance results. Covers T04, T15–T18, T28–T29, T42, T49 and T56.

### WP09 — Operator interface and release hardening

**Dependencies:** WP06; paper/option controls depend on WP07/WP08 respectively.

Add the session CLI/dashboard, authentication, notifications, metrics, backup/restore and operating runbook. Validate packaged imports and legacy compatibility. Include controls for pauses, ownership conflicts, cost caps and account-mode mismatches. Expose market as-of/latency, actual account fit and non-executable scenario comparisons; include two profile example configurations and validated migrations.

**Done when:** an operator can inspect and stop new entries, diagnose incidents, restore state and understand residual exposure without reading application internals. Covers T01, T02, T23, T26, T31–T35 and T55–T60.

### WP10 — Evaluation and explicit live-readiness decision

**Dependencies:** WP07 and WP09; WP08 is mandatory for any options enablement.

Run operational forward observation, causal replay and strategy comparison over both data profiles and explicitly defined account-capital scenarios. Review minimum-size constraints, liquidity capacity, cost burden, permission/data incompatibilities, uncertainty and execution quality. Do not extrapolate one account/profile result to all others or promote real-time strategies by merely changing a delay flag. Resolve incidents and document approvals. A live release is optional and requires explicit operator authorisation; profitability must not be inferred from software completeness.

**Done when:** the validation report states exactly what has passed, what remains uncertain and which modes/instruments are authorised. There is no automatic live promotion. Report T36–T60 outcomes alongside the original regression suite.

## 17. Test and acceptance matrix

Run unit and replay tests without external credentials. Keep online paper contract tests explicitly marked and disabled by default. Broker-native event fixtures must reflect the pinned SDK/API contract, with synthetic account identifiers and no secrets.

### 17.1 Core regression and operational tests

| ID | Scenario | Required result |
| --- | --- | --- |
| T01 | Existing CLI/strategy workflows with session mode disabled | Representative legacy behaviour preserved |
| T02 | Build/install wheel in clean environment | New and reused nested packages import successfully |
| T03 | Shadow/replay execution attempts reach gateway | No broker order-writing call occurs |
| T04 | Missing/non-finite account data, zero buying power and multiplier/override sizing | No fabricated balance; every entry quantity passes unit and capital checks |
| T05 | Weekend, holiday, early close and UK/US daylight-saving mismatch | Correct session IDs/windows; no wall-clock trading shortcut |
| T06 | Missing/stale calendar or restart after missed schedule | No accidental entries or blind overdue-job execution |
| T07 | Free-profile delayed historical SIP request versus forbidden latest SIP request | Explicit cutoff/feed policy; no silent IEX substitution or false real-time label |
| T08 | Stock SIP and option OPRA entitlements differ; news probe fails | Independent capabilities and accurate failure reasons |
| T09 | Plan for wrong session, expired entry, superseded version | Entry rejected; existing position still supervised |
| T10 | Concurrent plan revisions/validations | Stale writer rejected; one authoritative state projection |
| T11 | New or modified strategy generated by CloseGauss | Experiment only; no automatic approval |
| T12 | Late news before/after risk approval or order submission | Approval revoked where possible; cancel/fill race reconciled |
| T13 | Timeout after broker accepts order; process crashes before local acknowledgement | Query/reconcile same intent; no blind duplicate submission |
| T14 | Two candidates trigger concurrently with one position slot | At most one new group authorised; capital reserved atomically |
| T15 | Close intent with no position; quantity override; final tick-rounded limit | No accidental opening, affordability bypass or price-bound breach |
| T16 | Partial fill followed by cancel request and late fill | Actual exposure and reservations match broker-confirmed totals |
| T17 | Duplicate/out-of-order order and fill events | No double counting; cumulative state remains consistent |
| T18 | Exit rejected, remains open or partially completes | Position not marked closed; monitoring and alerts continue |
| T19 | Stale/crossed/invalid quote, symbol-level silence or stream gap | Required entries blocked; degraded state visible |
| T20 | News disconnect, incomplete event calendar or uncertainty | Coverage gap shown; event-dependent approval not fabricated |
| T21 | Same story on two sources; correction with same article ID | Logical deduplication plus preserved source versions |
| T22 | Later article/fundamental/bar revision appears during historical replay | Future evidence cannot alter an earlier decision |
| T23 | LLM timeout, schema failure, prompt injection or budget exhaustion | No execution authority gained; position/risk tasks continue |
| T24 | No candidate passes filters | Empty valid output; no forced trade |
| T25 | Entry quota/daily-loss pause reached | New entries stop; valid risk-reducing actions remain available |
| T26 | Restart with open positions, unmanaged broker trade or corrupted local state | Reconcile first; quarantine ambiguity and preserve alerts |
| T27 | Subscription/queue capacity exhausted | Holdings prioritised; gaps surfaced; no silent critical-event loss |
| T28 | Long and short option legs sum to zero; multiple underlyings held | Correct groups remain open; no cross-underlying aggregation |
| T29 | Option multiplier, stale leg, non-standard contract, assignment or residual stock | Correct units; unsupported entry rejected; new exposure reconciled |
| T30 | Database write/outbox failure and redelivery | Transactional state; idempotent consumers; entry safety preserved |
| T31 | Paper/live account mismatch, second runtime, dashboard command without authority | Hard rejection and audit event |
| T32 | Deposits/withdrawals and fees around daily-loss calculation | No false trading-profit reset; consistent session P&L |
| T33 | Session closing attempt fails or broker is unavailable | No claim of flatness; durable incident and continued supervision |
| T34 | Research output contains unknown rule/operator/metadata or arbitrary code | Schema/policy rejection, not execution |
| T35 | Backup/restore, UI reconnect and notification delivery failure | Correct state restored; no duplicate runtime; alternative alert path visible |

### 17.2 Data-profile and account-suitability acceptance tests

| ID | Scenario | Required result |
| --- | --- | --- |
| T36 | At a fixed wall time, free-profile records straddle the 900-second cutoff | Only eligible effective event times are visible; buffer and availability times honoured |
| T37 | A bar starts before the cutoff but ends after it | Entire completed interval required; no final-minute look-ahead |
| T38 | A delayed trigger is recognised after its original market time | Record the current decision time; no fill at the earlier observed quote |
| T39 | Fill, cancellation, assignment or account restriction arrives in free mode | Process immediately on the operational clock; no artificial 15-minute hold |
| T40 | A source record already arrives 15+ minutes late | No second 15-minute wait from receipt time; actual availability still enforced |
| T41 | Subscribed profile has no entitlement, or its feed is stale | Explicit readiness rejection; no paid-label assumption or silent downgrade |
| T42 | Indicative option quote is old enough for the delayed cutoff | Remains indicative; never passes genuine OPRA execution checks |
| T43 | Delayed observation is healthy; current quote or actual entry deadline is not satisfied | Research stays healthy; execution independently rejects missing quote or expired window |
| T44 | Data subscription/profile changes with pending plans and open positions | Version/revalidate plans; preserve positions, reservations and current supervision |
| T45 | Current headline arrives after free profile's signal cutoff | Aligned signal excludes it; safety channel may veto with recorded receipt time; no fabricated price reaction |
| T46 | Multiple explicit capital scenarios assess the same candidate | Account-specific feasible sizes/rejections with declared constant assumptions; no universal starting balance |
| T47 | Replay/scenario capital is missing or actual equity is unavailable | Clear validation/entry blocker; no synthetic balance fallback |
| T48 | Equal equity but different cash, permissions or existing commitments | Different feasible sets where justified; equity alone never determines suitability |
| T49 | Minimum option contract/position falls below, at and above a policy-derived feasibility boundary | Correct unit-safe sizing/rejection; no automatic risk-limit increase |
| T50 | Capital, permissions or reservations change after CloseGauss | PreGauss and final RiskGate reassess; stale sizing cannot pass |
| T51 | Proposed size grows beyond liquidity capacity in a larger scenario | Cap/reject quantity; do not scale a backtest linearly or assume more leverage |
| T52 | Two account-specific plans compete for current cash/slots | Current atomic reservations dominate older suitability reports; no double allocation |
| T53 | All attractive strategies are unaffordable or profile-incompatible | Explicit no-trade outcome with binding reasons, not a forced cheap/high-risk substitute |
| T54 | Model chooses the nominal highest profit with unsupported probability/cost inputs | Mark insufficient evidence; unsupported optimism cannot displace hard feasibility/risk limits |
| T55 | Hypothetical scenario or another account's plan reaches actual gateway | Reject identity/scope mismatch before reservation or broker call |
| T56 | Capital grows while options permissions, lifecycle support or paid feeds remain absent | No automatic strategy/asset/data approval |
| T57 | Shared subscription/model cost is allocated to account scenarios | Declared allocation and normalised burden; no duplicate charging or automatic purchase |
| T58 | Portfolio/data profile changes but analytical cache key matches only the symbol | Reassess account fit; account-dependent outputs cannot be reused as current approvals |
| T59 | UI switches between actual account and hypothetical capital comparison | Clear scenario/profile/as-of labels; no mutation of execution account |
| T60 | Full cycle runs in both profiles across the capital/constraint matrix | All four roles preserve causal lineage, suitability checks and policy; no unintended broker writes |

Use recorded sequences for race conditions and a fake clock for all scheduling tests. Test decision branches and invariants, not just a global coverage percentage. Every F-series correction must link to a failing regression test and its fix in the implementation PR.

## 18. Research validation and performance evaluation

### 18.1 Separate three questions

**Operational correctness:** does the service follow plans, handle failures and reconcile exposure?

**Selection value:** does research improve account- and data-profile-compatible candidate/strategy selection over a simple baseline?

**Economic viability:** do results remain useful after spreads, failed fills, fees, data, model and hosting costs?

Passing the first question does not establish the other two. Evaluate all three separately by data profile and declared account scenario; neither a large account nor a paid feed establishes economic viability. Alpaca states that paper trading omits effects including latency-related slippage, order queue position and regulatory fees [A08]. Treat paper execution as an operational environment, not proof of live profitability.

### 18.2 Compare the complete pipeline

Hold execution/risk rules constant and compare a fixed-watchlist baseline, CloseGauss-selected candidates, and candidates further filtered by PreGauss. Include the actual historical selection universe and information availability, not only today's surviving or attractive symbols.

Use chronological development/validation/holdout windows and recorded forward observation. Record all strategy/parameter experiments; do not select the best of many trials and present its result as an untouched test. Model-generated confidence must not substitute for measured calibration.

Run a structured comparison over **data profile × account scenario × approved strategy × evaluation period**, with controlled assumptions and an untouched chronological holdout. Do not select the best account/profile combination retrospectively and report it as an independent validation.

For intraday options, missing historical executable quotes limits what a historical test can establish. State that limitation, use conservative fill assumptions where appropriate, and record real quote observations for prospective evaluation. Do not fabricate a precise options backtest from stock bars alone.

### 18.3 Required metrics

| Area | Metrics |
| --- | --- |
| Operations | Uptime, wall/as-of times, intentional delay, additional lag, current quote age, coverage, backlog, reconciliation gaps and recovery outcomes by profile |
| Research | Candidates and alternatives assessed, suitability rejections, no-trade sessions, validation reversals, source coverage and cost by account/profile |
| Execution | Fill rate, time-to-fill, cancel/replace rate, spread at decision, slippage to recorded benchmark and fees |
| Risk | Peak exposure, realised/unrealised loss, drawdown, residual-position incidents and limit breaches |
| Economics | Gross/net P&L, drawdown/return normalised to capital, operating-cost burden, minimum feasible size, capacity sensitivity and cost per candidate/group |

Use actual-fill P&L when available. Spreads and slippage already reflected in those fills must not be deducted a second time. Report operating expenses separately and then show the all-in result.

Retain decision reason codes and rejected opportunities so the contribution of news filters and research selection can be evaluated. Four agents agreeing is not four independent statistical observations.

### 18.4 Causal delayed-versus-real-time comparison

For a free-profile replay, advance a wall clock, release only records eligible under the evidence clock and generate decisions at that wall time. Any simulated order begins after the decision and submission latency. Use subsequent entitled/recorded quotes or a disclosed execution model for fills; never assume execution at the 15-minute-old signal quote. If adequate fill evidence is missing, report decision-only results or simulation uncertainty rather than precise executable performance.

Apply the same principle to paper results: the paper broker operates at current time even when the input signal is delayed. Record signal observation time, decision time, submission time and fill time separately. Current paper fills do not prove that the strategy had current signal data.

Compare each profile with a latency-compatible baseline and its own trading/operating costs. Test whether an already-approved strategy retains usefulness with delay; do not infer that it does because its nominal holding period exceeds 15 minutes.

### 18.5 Capital-aware validation

Use explicit hypothetical capital amounts supplied for the experiment, not built-in production tiers. At a minimum, test instrument feasibility boundaries, differing cash/buying-power constraints, capacity-limited larger sizing and no-trade cases. Hold strategy assumptions constant for pure capital sensitivity; disclose changed permissions, mandates or feeds in broader comparisons.

Do not linearly rescale a backtest's P&L to larger funds: recompute whole/fractional units, contract counts, fees, pending capital, portfolio constraints and liquidity-limited execution. Rank feasible alternatives against no trade, and show uncertainty rather than manufacture an "optimal" strategy for every amount.

Keep a distinction between operating the software correctly for many balances and having validated profitable trading for those balances. Each approved deployment specifies its tested account/data/strategy scope.

## 19. Deployment, recovery and rollback

### 19.1 Startup sequence

Validate configuration, execution mode and requested data profile; acquire account/environment ownership; check storage/migrations; load actual/evidence clocks and calendar; verify endpoint capabilities without trading; reconcile current broker state; build the actual AccountProfile; restore holdings monitoring; start independent risk supervision; assess suitability; then enable role jobs and, only if separately authorised, entries.

Credentials belong in protected environment/secret configuration, never in source, reports or fixtures. Use a supervised host with restart policy, clock synchronisation, persistent storage and an alert channel. Prefer one deployment rather than several personal machines that may all trade the same account.

### 19.2 Failure behaviour

| Failure | New entries | Existing exposure |
| --- | --- | --- |
| CloseGauss/PreGauss unavailable | No new approval or no entry for affected plans | Continue established management policy |
| Market feed stale beyond its profile contract | Block dependent entries; distinguish extra lag from intended delay | Use approved recovery/degraded-supervision path; do not represent old marks as current |
| Required subscription/entitlement lost | Pause affected entries; no automatic delayed fallback | Preserve holdings/order state; apply documented price-coverage incident response |
| Account data invalid or suitability outdated | Reconcile/reassess before any new entry | Maintain confirmed positions; do not fabricate balances or delete exposure |
| News coverage unavailable | Block strategies requiring it under policy | Apply recorded degradation rules; continue price/order supervision |
| Broker stream disconnects | Pause pending readiness check | Reconcile through REST and restore stream |
| Broker REST unavailable | No fresh entries or blind retries | Preserve unknown states; monitor/alert and reconcile when possible |
| Database cannot persist critical state | Stop new entries | Invoke documented audited emergency path; alert; never imply supervision is guaranteed if the host is failing |
| Operator pause | Block new entries | Continue monitoring and permitted exits |
| Host/process failure | No functioning local execution | Supervisor recovery and external alerting; broker-resident protections only where explicitly supported and tested |

Local software stops cannot operate while the host is down. Use supported broker-resident protections where suitable, with precise reconciliation of their orders; do not claim universal bracket/stop support for every option structure. Document the manual emergency broker-access procedure.

### 19.3 Shutdown and rollback

Pause entries first, cancel pending entry orders according to policy, reconcile and identify every remaining position. A normal stop must either leave an explicitly designated management service running or report that supervision is ending and require operator acknowledgement. Do not silently terminate with unresolved exposure.

For rollback, preserve the database and active-position policy versions. Only run an older build against a compatible schema. Prefer rolling forward a corrective migration over deleting state. Keep the new workflow behind a feature flag, but disabling that flag must not orphan positions created by it.

Back up database and evidence manifests; exercise restore in a test environment. Never validate restore by accidentally connecting to the live account with enabled execution.

## 20. End-to-end acceptance scenario

### 20.1 Fixture dimensions

Use a synthetic two-session market/news/broker fixture with an explicit test calendar. Run it under both data profiles and several explicitly supplied account profiles. Test capital conditions relative to the minimum position requirement and to estimated liquidity capacity; do not define a universal production account-size band.

Include an actual-account test scope and separate non-executable hypothetical scopes. Profile and account differences must appear in outputs, not just in test names.

### 20.2 Common four-role cycle

**Session A close:** PostGauss immediately reconciles a completed trade and current broker state. It waits for profile-appropriate session data completeness before finalising the market scan. It identifies candidates and preliminary account-feasibility constraints, including cost burden and reserved capital.

**Research:** CloseGauss reads immutable snapshot S1 with account/profile references. It compares approved strategies/instruments and produces plans P1 and P2 only where the account and profile support them. A new strategy goes to experiments. A scenario below minimum valid position size receives no trade; another may receive a feasible bounded proposal. A larger scenario is still constrained by liquidity and the mandate.

**Late event:** A material announcement arrives after S1's cutoff. Preserve the original report. Under the free profile, the current safety channel can suspend P1 even before a corresponding delayed market reaction is visible. It must not claim to have observed that reaction.

**Session B pre-market:** PreGauss rechecks the market view and current account state. A changed cash balance or pending commitment can make a previously feasible plan unsuitable. P1 is rejected or requires revision; P2 remains conditionally eligible only under its own data/strategy contract.

**Session B trading window:** In the subscribed branch, evaluate P2 with current required observations. In the free branch, evaluate it only when its delayed trigger becomes eligible. Record actual decision times. A plan that expires before the delayed trigger arrives is rejected. Missing current quotes block any strategy that requires them.

**Account comparison:** A positive valid order size is recomputed from actual current capital, not copied from the hypothetical report. The actual gateway rejects a scenario plan even if its numbers appear feasible. No research output raises risk limits or buys a subscription.

**Shadow branch:** Persist all hypothetical intents and reasons without any broker-writing call. Treat zero orders as a valid complete session.

**Permitted paper branch:** Use a strategy approved for its data profile and paper mechanics. The broker accepts an order, but the response times out. Reconciliation finds the same intent/client ID after restart; no duplicate is sent. Partial fill creates actual managed exposure, even if remaining quantity is cancelled later. Broker activity is processed immediately in both profiles.

**Exit:** Entry quota exhaustion does not block the authorised exit. An initial rejection leaves the group open and raises an alert; confirmed fills/reconciliation close it. Closing attempts use real session deadlines, not the delayed evidence clock.

**Review:** PostGauss links evidence versions, account suitability, profile timing, risk reservations, orders and fills. It reports operational faults and costs without inventing historical fills or treating account cashflows as trading profit.

### 20.3 Options and subscription-transition branches

An additional options fixture holds one long and one short leg whose signed quantities sum to zero. The group remains open, is monitored by contract and survives an assignment/residual-stock event. Indicative quotes cannot satisfy its OPRA execution checks in either profile.

During another fixture, remove a required real-time entitlement. New affected entries pause; the system neither silently switches to delayed signals nor abandons holdings. A deliberate profile change produces a new data-policy version and requires renewed plan/suitability validation.

**Acceptance:** all four roles complete across the matrix, with explicit no-trade outcomes where appropriate, causal market timing, current broker supervision and no assumed capital amount.

## 21. Release checklist and implementation order

### 21.1 First deliverable: four-agent shadow workflow

- [ ] Baseline captured; tests tracked; new packages included in distribution.
- [ ] Calendar, immutable snapshots, versioned plans and durable event delivery implemented.
- [ ] Both `FREE_DELAYED` and `SUBSCRIBED_REALTIME` implemented with separate wall/evidence clocks.
- [ ] Free-profile 900-second cutoff, no double delay and completed-bar end-time tests pass.
- [ ] Actual account profiles and isolated capital scenarios replace any implicit starting balance.
- [ ] Every role records account suitability and data-profile lineage; explicit no-trade results work.
- [ ] PostGauss, CloseGauss, PreGauss and LiveGauss complete the synthetic session scenario.
- [ ] News and data gaps remain visible; unsupported evidence cannot authorise an entry.
- [ ] No broker order-writing call is possible in shadow/replay mode.
- [ ] Operator can inspect plans, reasons, costs, health and state after restart.

### 21.2 Additional gates for paper execution

- [ ] Account/mode/profile validation, final-price checks and all sizing paths pass tests.
- [ ] Delayed inputs are never backdated into paper fills; current broker state is not delayed.
- [ ] Multiple capital/constraint scenarios pass minimum-size, reservation and capacity tests.
- [ ] Portfolio reservations prevent concurrent overspending and duplicate groups.
- [ ] Unknown submissions, partial fills, replacements and cancel/fill races reconcile correctly.
- [ ] Missing research, quota exhaustion and entry expiry do not disable valid exits.
- [ ] Closing failures, unmanaged positions and outages produce durable alerts.
- [ ] Operating and restore procedures have been exercised without live orders.

### 21.3 Additional gates for options or real-money operation

- [ ] Contract/group monitoring replaces the legacy aggregate options path.
- [ ] Required option feed and actual permissions are verified for the deployment account.
- [ ] Multipliers, close intents, multi-leg restrictions and assignment/residual exposures pass tests.
- [ ] Strategy-specific validation and realistic execution-cost evaluation are documented.
- [ ] Operator explicitly approves the account, execution/data modes, instruments, strategies and risk-policy version.
- [ ] Paid access does not arm live trading; larger equity does not grant leverage or strategy approval.
- [ ] Initial free-profile live-entry restriction is enforced; any later exception has a separately validated delayed-input strategy and exit-control approval.
- [ ] Remaining limitations are recorded; no automatic live promotion is configured.

**Implementation order:** WP00 → WP01/WP02 → WP03 → WP03A → WP04 → WP05 → WP06 → WP07 → WP08 as required → WP09 → WP10. WP09's read-only interface can begin after WP06; its execution controls wait for the corresponding execution gates.

The first milestone is the whole four-role decision cycle under both data profiles and explicit account scenarios, with durable state and zero broker writes. The next milestone is controlled paper execution. Live options are a separately reviewed capability, not the default completion state.

## 22. Sources and evidence boundary

### 22.1 Repository sources

All repository links below are pinned to `31374551bae6fd34a0fe56fe11d208f4ff04fbb4`, confirmed as the `master` head during the original preparation. Revision 1.1 updates the supplied plan against that recorded baseline; it does not claim a new repository inspection or deployment. Read the actual code when implementing: line numbers and interfaces will change after subsequent commits.

| Reference | Source |
| --- | --- |
| [R01] | Strategy contracts and base classes |
| [R02] | Strategy registry |
| [R03] | Multi-agent orchestrator; also see [R03A] for analyst implementations |
| [R04] | Multi-agent strategy wrapper |
| [R05] | Alpaca data provider |
| [R06] | News provider |
| [R07] | Shared live runner |
| [R08] | Execution engine, including context, sizing, overrides and order pricing |
| [R09] | Dashboard and stream ownership |
| [R10] | Packaging and dependencies; also see [R10A] for settings |
| [R11] | Base live engine and position monitoring |
| [R12] | Live stock module and session logic |
| [R13] | Live option module and aggregate position state |
| [R14] | Ignore rules, including `tests/` |

[R01]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/strategy/base.py
[R02]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/strategy/registry.py
[R03]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/agent/multi_agent/orchestrator.py
[R03A]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/agent/multi_agent/agents.py
[R04]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/strategy/multi_agent_strategy.py
[R05]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/data/alpaca_provider.py
[R06]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/data/news_provider.py
[R07]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/trade/live/live_runner.py
[R08]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/trade/engine/execution.py
[R09]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/ui/dashboard.py
[R10]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/pyproject.toml
[R10A]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/settings.py
[R11]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/trade/live/live_trading_base.py
[R12]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/trade/live/live_trading_stock.py
[R13]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/src/trade/live/live_trading_option.py
[R14]: https://github.com/Magica-Chen/GaussWorldTrader/blob/31374551bae6fd34a0fe56fe11d208f4ff04fbb4/.gitignore

### 22.2 Official Alpaca documentation

References [A01], [A03], [A07] and [A10] were checked for revision 1.1 on 7 September 2026. Other references are retained from the original plan. These sources establish provider constraints, not any particular account entitlement or strategy profitability. The two profile names, clock policies, rollout gates and account-suitability system are proposed product requirements rather than provider terminology. Revalidate against the installed SDK and intended environment before release.

| Reference | Source and relevance |
| --- | --- |
| [A01] | Market Data FAQ: historical SIP cutoff versus latest endpoints |
| [A02] | Calendar API: session dates and early closures |
| [A03] | Market Data API: Basic versus paid feed coverage and published quotas |
| [A04] | Real-time news: article IDs, timestamps and symbols |
| [A05] | Placing orders: client IDs and order management |
| [A06] | Options trading: contract records, order restrictions and assignment monitoring |
| [A07] | Real-time option data: feeds and quote channels |
| [A08] | Paper trading: simulation limitations |
| [A09] | Trading WebSocket: order/fill event handling |
| [A10] | Historical option data: indicative versus genuine OPRA data and derived trade delay |

[A01]: https://docs.alpaca.markets/us/docs/market-data-faq
[A02]: https://alpaca.markets/sdks/python/api_reference/trading/calendar.html
[A03]: https://docs.alpaca.markets/us/docs/about-market-data-api
[A04]: https://docs.alpaca.markets/us/docs/streaming-real-time-news
[A05]: https://docs.alpaca.markets/us/docs/orders-at-alpaca
[A06]: https://docs.alpaca.markets/us/docs/options-trading
[A07]: https://docs.alpaca.markets/us/docs/real-time-option-data
[A08]: https://docs.alpaca.markets/us/docs/paper-trading
[A09]: https://docs.alpaca.markets/us/docs/websocket-streaming
[A10]: https://docs.alpaca.markets/us/docs/historical-option-data

### 22.3 What this plan does not establish

It does not establish current account entitlements, strategy profitability, complete historical options quote availability, exact 15-minute delivery for every dataset, a universally optimal strategy for each account amount, runtime test results or a successful deployment. It does not treat strategy examples or payoff formulas as approved trading rules. The specified architecture and controls are proposed changes; the repository findings and broker documentation are their stated starting evidence.
