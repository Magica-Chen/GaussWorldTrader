# Four-agent operations

The session service owns stock/options scheduling, evidence, plans, account supervision
and operator commands. `gauss_bot.py` runs the service continuously. The CLI and the
Gauss Session dashboard inspect its durable SQLite state. Opening the dashboard does
not start a service.

The supplied operating default is **shadow execution with `FREE_DELAYED` data**.
Broker-connected account information uses the configured account's actual balances.
Offline runs require an explicit synthetic account fixture. Stock and options research
share the session controller; options entries remain subject to their separate policy,
permission, contract and quote gates.

## Configuration and startup

Use the existing TOML settings loader's `[gauss]` section and the profile examples in
`examples/`. Runtime state defaults to `results/gauss/session.sqlite3`; set
`GAUSS_DATABASE_PATH` consistently in the service, CLI and dashboard when changing it.
Provide `GAUSS_ACCOUNT_ID` for the intended account and keep credentials in the
environment or local `.env`. Set `GAUSS_CONTROL_TOKEN` to a private operator token for
commands. Keep the database, backups and token accessible only to the operator account.

```bash
python main_cli.py session validate-config --config examples/gauss.free-delayed.example.toml
python main_cli.py session doctor --config examples/gauss.free-delayed.example.toml
python gauss_bot.py --config examples/gauss.free-delayed.example.toml
```

Startup validates configuration, opens the durable store, resolves the calendar,
acquires account/environment ownership and reconciles broker state. Read readiness
blockers before enabling any entries. Unknown balances, endpoint capability or calendar
coverage remain visible blockers. Strategy registry availability and operator approval
are separate; an empty approval set produces valid no-trade sessions.

The two data profiles use the same role implementations:

| Profile | Strategy market view | Operational account/order view |
| --- | --- | --- |
| `FREE_DELAYED` | Historical consolidated SIP observations at least 900 seconds old, with an entitlement buffer and measured additional lag | Current received broker updates |
| `SUBSCRIBED_REALTIME` | Entitled current SIP; genuine OPRA required by option execution contracts | Current received broker updates |

A subscribed label does not establish entitlement. Indicative options remain labelled
indicative at every age. Changing profile invalidates incompatible plan readiness and
requires reassessment. Closing deadlines always follow the actual exchange session.

## Routine operation

For an end-of-day or holiday research report, run:

```bash
python gauss_bot.py --config examples/gauss.free-delayed.example.toml --once
```

This command scans once, writes a Markdown report plus JSON and CSV evidence under
`results/gauss/research/`, and exits. It does not start supervision, acquire an account
lease, submit orders, or require strategy execution approvals. Existing positions do
not prevent it from exiting. `--output json` returns the structured report.

With `symbols = []`, it discovers active tradable US equities from Alpaca. It excludes
OTC listings and names identifying leveraged/inverse products, warrants, preferred
shares or rights, then screens completed daily bars for price and liquidity. It analyzes
the most liquid `scan_universe_limit` names with at least 60 bars using the registered
momentum, trend-following and mean-reversion strategies in the example's allowlist.
The report gives actual coverage, conditional next-session entry/stop/target scenarios,
news links and remaining event checks. Weekends and holidays use the latest completed
session and the next scheduled opening. Symbols can be supplied to narrow the universe.

`gauss_bot.py --once` now means one-shot research. The lower-level
`main_cli.py session run --once` retains its single supervision-cycle semantics and
may continue managing residual exposure. Omitting `--once` still starts the continuous
service, whose execution gates and approvals remain required. Research allowlisting
does not assert validated profitability or approve automated execution.

```bash
python main_cli.py session status
python main_cli.py session plans
python main_cli.py session assess-account
python main_cli.py session report --format markdown
python main_cli.py session compare-capital --scenarios examples/account_scenarios.example.toml
python main_cli.py session run-task --task post --reason "review completed session"
```

The role sequence is PostGauss → CloseGauss → PreGauss → LiveGauss. PostGauss reviews
the completed session and feasible candidates. CloseGauss uses immutable evidence to
research conditional plans. PreGauss rechecks those plans against available evidence,
current account facts and safety events. LiveGauss evaluates approved triggers while
the service continues reconciling real exposure.

Check market as-of time, observed/additional lag, price and news coverage, last successful
reconciliation, outstanding orders, residual positions, agent failures and allocated
costs. A quiet session with no eligible plan is a complete operating outcome. A stale
reconciliation timestamp is a supervision issue even when the delayed market feed is
healthy.

Period-specific CLI tasks invoke the same service rules as scheduled execution. Running
a role explicitly does not extend an expired entry window, fabricate a complete session
or make a hypothetical account executable. The continuous bot should run under one
host supervisor with restart policy and a persistent working directory. A second
service targeting the same account/environment must fail ownership acquisition.

## Research and options policy

Alpaca's calendar provides exchange sessions. Earnings and scheduled economic releases
use the existing `FINNHUB_API_KEY` and `FRED_API_KEY` automatically in both one-shot
research and the continuous service. Finnhub supplies earnings dates; FRED supplies
release dates with future/no-data releases included and paginated. Responses are
cached for 15 minutes and cover today through seven days ahead. Per-source coverage
and retrieval times are retained. A missing or failed source produces partial coverage
and does not satisfy the runtime's event-calendar readiness gate.

FRED dates have no intraday release time. Reports label them as date-only. Earnings
block entries in the affected symbol for the calendar date; major CPI, PPI, employment,
GDP, personal income/outlays and retail-sales releases block all entries that date.
Other FRED releases are informational. Set `[gauss.news].event_calendar_path` (or
`GAUSS_EVENT_CALENDAR`) to a verified operator calendar to specify narrower timed
windows. FRED release coverage does not include every speech or unscheduled event.
FRED's FOMC-labelled dataset updates are informational, not inferred policy meetings;
use the Federal Reserve meeting calendar for scheduled policy decisions.
Finnhub's separately entitled economic-calendar endpoint is not required.

Snapshot research receives immutable supplied evidence and a frozen account profile.
Missing point-in-time fundamentals are reported unavailable. Deterministic research is
the default. Optional paid annotations use a separate isolated process, a strict result
schema, no broker credentials or trading tools, and bounded input/output, retries and
wall time. Experimental strategy output enters the experiment queue without approval.

To configure paid research, supply an explicit model, enable
`[gauss.research].paid_model_calls_enabled`, and set matching `model` and `pricing_id`
values in that table. First inspect a `ModelPricing` JSON/TOML document with
`session approve-model-pricing --pricing PATH --preview`; it requires ID, model,
input/output USD per million tokens, `verified_by`, source reference and a timezone-aware
expiry. Register the verified document with the same command, an explicit operator,
reason and matching `--confirm-account`. Set `OPENAI_API_KEY` only in protected local
configuration. Unknown/expired pricing or missing credentials blocks paid work.

Spend reservations use the smaller of the account-equity budget and absolute session
cap, then enforce each role's share. Frozen analysis inputs remain unchanged when fresh
actual funds are checked for a paid call. Actual usage settles each reservation once;
a timeout preserves an uncertain charge until resolved. Reports separate settled,
uncertain and still-reserved costs. A hypothetical account cannot finance a model call.

Initial option entries support verified standard long options and approved debit
verticals with genuine current leg quotes and confirmed account buying power and
permissions. The initial option adapter uses a conservative $0.10 limit-price grid,
valid across penny and non-penny increments; exit prices round within their declared
quote bounds. Actual paper validation must establish execution mechanics and fill
quality for the deployment. A zero net contract count does not establish a flat spread.
Exercise/assignment stock and unsupported residual legs remain visible and require an
explicit adoption or exit decision.

## Operator controls

Commands require an operator identity, reason and the configured control token. They
are queued durably for the running service; inspect the command audit for the execution
result. Successful queueing is distinct from completed cancellation or liquidation.

| Control | Effect |
| --- | --- |
| Pause entries | Stops new exposure and preserves position supervision |
| Resume entries | Requests a readiness recheck before further entries |
| Manage only | Keeps the service dedicated to existing exposure |
| Cancel pending entries | Requests broker cancellation; holds uncertain reservations until reconciliation |
| Request flatten | Requires explicit account confirmation and requests valid close orders |
| Reconcile | Refreshes broker orders, positions and required activities |

An unknown submission keeps its stable intent/client ID and reservation. Query the
broker before considering another submission. Partial fills create managed exposure
immediately. A rejected exit leaves exposure open. A cancelled order may still have a
late fill; compare broker-confirmed cumulative quantities before releasing capacity.

Single-leg replacement keeps the original risk reservation and immutable quantity,
uses a stable child client ID and records parent/child lineage. Each attempt must pass
current plan, price, account and risk checks within the configured replacement limit.
An unknown result requires reconciliation before any further attempt. Multi-leg
replacement is disabled until the deployment proves its combined-order mechanics.
Cancelling a parent does not skip an unresolved replacement child.

Live arming requires the intended account/environment plus the displayed strategy
approval/version/profile set and reviewed risk, data-policy and deployment versions.
An approval added after arming cannot inherit the earlier authorization. Read the
release record before issuing `session arm-live`; starting the bot or purchasing data
never creates an arming decision.

## Incident response

| Incident | Operator response |
| --- | --- |
| Price/news gap or lost entitlement | Pause affected entries, inspect coverage and restore the required feed; reassess plans after recovery |
| Missing/stale account or broker REST failure | Keep uncertain orders/positions recorded; restore connectivity and reconcile before resuming |
| Research/model failure or exhausted budget | Inspect failed job and budget records; preserve independent reconciliation and management |
| Unexpected position, assignment or residual option leg | Inspect each actual contract/stock holding, record an adoption/exit decision and keep conflicting entries blocked |
| Failed close or approaching session deadline | Inspect pending close state and remaining quantities; use the documented emergency broker-access procedure if service coverage is insufficient |
| Database/host failure | Stop new entry attempts, preserve storage, inspect broker state through an independent operator connection and recover the service |

For emergency broker access, sign into the operator's configured broker application,
verify account/environment, review open orders and actual positions, and record any
manual actions for subsequent reconciliation. Local software supervision stops when
the host stops. Broker-resident protections apply only to order structures explicitly
supported and tested for that deployment.

## Backup and restore

Critical incidents commit to the database before best-effort stderr delivery. CLI and
dashboard readers retain incident visibility when console delivery fails. Configure
the host supervisor to retain stderr and monitor the durable health/incident state.
No outbound email/Slack notification was sent during implementation or validation.

`src.runtime.operations.backup_state` uses SQLite's online backup API, includes committed
WAL state and creates a SHA-256 manifest. When research payload files are used, pass the
evidence directory so they accompany the database. Content-addressed payloads must
remain immutable during backup. A backup without a complete manifest is incomplete.

```python
from src.runtime.operations import backup_state, verify_backup, restore_state

backup_state("results/gauss/session.sqlite3", "results/gauss-backup-20260908")
verify_backup("results/gauss-backup-20260908")
restore_state("results/gauss-backup-20260908", "results/gauss-restore-test")
```

Restore verifies hashes and database integrity and writes to a new directory. It
rejects an existing destination. Exercise recovery offline, inspect restored plans,
unknown orders and holdings, and verify deployment/schema compatibility. After the
original service has relinquished ownership, reconcile the selected restored state
against the actual broker before permitting entries. Restoring a database never arms
execution or creates a broker connection.

## Shutdown and rollback

The bot shows a readable status summary in a terminal: execution mode, entry blockers,
health, agent states and session times. It prints the full summary when these change,
with a short heartbeat between changes. Use `--output text` to force this display or
`--output json` for complete structured records. Redirected output defaults to JSON.

Press **Ctrl+C** to request shutdown after the current cycle completes. The service
pauses entries, cancels entry orders and reconciles. If exposure remains or reconciliation
fails, it reports the blocked shutdown and continues management. After that notice,
press **Ctrl+C again** to explicitly acknowledge ending supervision with remaining
exposure. This ends the bot; it does not liquidate positions or cancel every broker order.
Repeated SIGTERM signals do not acknowledge exposure. The existing
`--acknowledge-unmanaged-exposure` flag provides that acknowledgement in advance.

An already running process uses the code it loaded at startup. For an older process
that will not exit after Ctrl+C, locate its PID with `pgrep -af '[p]ython gauss_bot.py'`
and, if ending supervision immediately is intended, run `kill -KILL <PID>`.
This bypasses cleanup and leaves broker positions/orders in place. Restarting loads
the updated shutdown controls and reconciles broker state.

An abrupt stop can leave a database ownership lease for up to 120 seconds after its
last renewal. A blocked startup reports the lease expiry time and exits; retry after
that time if the previous process has stopped. It does not change the account's
published state or claim to supervise stored positions. Do not delete the database
or bypass the lease to restart.

Pause entries, request cancellation of pending entries as appropriate, reconcile and
inspect every remaining position or unknown order. Preserve a designated management
service when exposure remains. A stop that ends supervision requires explicit operator
acknowledgement and reports unresolved exposure.

Preserve the database and active exit-policy versions during rollback. Use an older
build only with a compatible schema; keep the recoverable backup and evidence manifest.
Disabling the new workflow must not remove the management responsibility for positions
it created. Paper, options and real-money promotion require their recorded release
gates in [FOUR_AGENT_VALIDATION.md](FOUR_AGENT_VALIDATION.md).
