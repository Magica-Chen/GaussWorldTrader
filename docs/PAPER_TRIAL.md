# Paper trial deployment

The September 26, 2026 deployment uses the Alpaca paper account and the existing
ChatGPT subscription through Codex CLI. Real-money execution is disabled. The user
authorized automatic paper orders, discretion over existing holdings, and Slack
notifications. Credentials and the operator token remain in the private `.env`.

The user service is installed, enabled and running with lingering enabled. Startup
verification confirmed all eight symbols have all 390 completed minutes for September
25, all seven setup commands were accepted, and the Slack status message was delivered.
The authenticated resume completed; the separate retirement gate still blocks entries.
No broker orders have been submitted while the equity market is closed. The full
offline suite passed (one browser fixture skipped); a further 49 targeted tests covered
the history-pagination fix discovered during startup. Local verification is recorded
in `results/gauss/deployment-verification.json`.

## Trial contract

Account balances and position quantities remain in the private runtime ledger.
The trial uses cash without leverage. Existing holdings selected for retirement
are adopted for closure at the next regular open, September 28 at 13:30 UTC /
14:30 London.
Fresh quotes, confirmed holdings and an open market are required for these exits.
New research waits for broker confirmation that the retirement holdings and their
working orders are cleared. A scheduled exit is not a confirmed sale.

The fixed universe is SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA and TLT. Registered momentum,
trend-following and Bollinger/RSI mean-reversion strategies use their recorded default
parameters. Three explicit `paper_trial` approvals pin symbols, source versions,
parameters, account, data profile and expiry. They expire October 9, 2026 at 20:30 UTC.
The process continues monitoring afterward; further entries need renewed approval.

Limits, applied against actual account equity:

- 2% maximum capital per position and 10% total portfolio exposure.
- 0.1% planned stop loss per trade; gaps or execution can exceed that estimate.
- 1% daily loss pauses new entries, including losses incurred before process startup.
- At most three new position groups and six entry attempts per session.
- No leverage, short entries, options or planned overnight positions.

Entries use a fresh ask rounded up to the instrument tick, within a predeclared 1%
ceiling above the prior closing signal price. An out-of-bound quote defers the entry.
Profit exits retain limits; urgent stock exits use paper market orders. Partial fills,
unknown submissions, cancellations and replacements retain their reservations until
broker reconciliation resolves them.

Delayed SIP bars supply signals with a 905-second minimum release delay. Current IEX
quotes support execution only. Finnhub earnings and FRED release calendars supply
event checks; date-only major releases conservatively block the day. Each of the four
stages requires supporting Codex review before a new entry can proceed. Deterministic
price, account and risk checks remain independent.

## Evidence and limits

The historical study used 937,209 unique SIP minute bars across 145 completed sessions
from March 2 through September 25. Chronological development, validation and holdout
partitions used fixed parameters and transaction-cost scenarios. Current defaults
did not establish positive expectancy: momentum generated no trades, while the other
strategies produced sparse, mixed or negative outcomes.

The study used the previous 0.1% entry ceiling and historical SIP bars as an execution
proxy, without reconstructing historical IEX queues, AI reviews or news vetoes. The
new 1% ceiling and current-ask entry policy are explicitly unvalidated forward-trial
rules. Historical results are linked as evidence of uncertainty, not positive
validation. Trial expectancy remains `null`; ranking uses fractional planned loss,
not an invented expected return. Trial approvals cannot authorize live trading.

Private local artifacts:

- `results/gauss/paper-deployment.toml`: pinned account and runtime configuration.
- `results/gauss/paper-trial-approvals.json`: reviewable trial authorizations.
- `results/gauss/paper-deployment-commands.json`: queued command identities.
- `results/gauss/paper-codex.sqlite3`: durable runtime, decisions and notification ledger.
- `results/gauss/validation/alpaca_sip_20260302_20260925/manifest.json`: source provenance.
- `results/gauss/validation/alpaca_sip_20260302_20260925/fixed_default_session_simulation.json`:
  reproducible historical outcomes and their limitations.

## Quota, notifications and controls

Slack receives decisions, acknowledged submissions, partial/final fills and incidents.
At 5% or less remaining in any Codex allowance window, the service sends an alert and
latches monitoring-only mode. It stops analysis and all new broker actions, including
exits and cancellations, as requested. Existing broker orders can still fill;
reconciliation and Slack delivery continue. Missing or stale quota also blocks actions.
A fresh allowance above 5% and an authenticated resume are required after a threshold stop.

The private `.env` points the dashboard and default session client to this active
paper ledger. Starting the dashboard remains a separate, optional UI action.
Run these commands from the repository directory using the configured environment:

```bash
conda activate myalpacatrading
python main_cli.py session status --config results/gauss/paper-deployment.toml
python main_cli.py session pause-entries --config results/gauss/paper-deployment.toml --reason "operator pause"
python main_cli.py session resume-entries --config results/gauss/paper-deployment.toml --reason "reviewed current readiness"
systemctl --user status gauss-paper.service
journalctl --user -u gauss-paper.service -n 30 --no-pager
```

The user service restarts after an unexpected process exit. Normal stop requests retain
supervision while unresolved exposure remains, so `systemctl stop` can wait until that
exposure is explicitly handled. The account lease also prevents a second trader from
owning the same paper account. See [operations](FOUR_AGENT_OPERATIONS.md) for full controls.

## Token monitor

The monitor reads the existing ledger without starting models or changing trading.
The dashboard's Gauss Session page displays the same totals on refresh.

```bash
python main_cli.py session usage --watch
python main_cli.py session usage --json
python main_cli.py session usage --session US_EQUITIES:2026-09-28
```

Input/output totals cover recorded calls by this bot and account, grouped by stage
and session. Failed or unresolved calls with missing counts remain explicitly unknown.
These totals do not include unrelated Codex conversations or setup checks in other
ledgers. The account-wide allowance percentage comes from the runtime's quota checks,
not a token-to-percentage estimate. Values older than 120 seconds are labelled stale.
Ctrl+C closes the terminal monitor without affecting the trader.

## Start and stop from the terminal

```bash
cd ~/GaussWorldTrader
conda activate myalpacatrading

# Start the installed background service (safe if already running).
systemctl --user start gauss-paper.service
systemctl --user status gauss-paper.service --no-pager
python main_cli.py session status

# Resume entries after a prior pause, once current readiness has been checked.
python main_cli.py session resume-entries --reason "Reviewed current readiness"

# Pause only new entries; monitoring and holding management continue.
python main_cli.py session pause-entries --reason "Operator pause"

# Request a full, graceful stop without blocking the terminal.
systemctl --user stop --no-block gauss-paper.service
journalctl --user -u gauss-paper.service -n 20 --no-pager
```

A start does not erase a persisted entry pause. A graceful stop requests cancellation
of pending entries where permitted, then reconciles. If positions or unresolved orders
remain, the service keeps supervising them and reports `management_continues`; it does
not automatically liquidate the account. Stopping under the Codex quota lock does not
override the prohibition on broker actions.

To explicitly end supervision despite remaining holdings, first request the graceful
stop above and wait for `management_continues` in the log. Only then acknowledge:

```bash
systemctl --user kill --kill-whom=main --signal=SIGINT gauss-paper.service
systemctl --user status gauss-paper.service --no-pager
```

That acknowledgement ends monitoring; it does not sell remaining holdings or guarantee
cancellation of all broker orders. Confirm the unit becomes `inactive`. Start it again
with `systemctl --user start gauss-paper.service`, inspect status, and resume entries
when appropriate. The authenticated application CLI also accepts `session stop
--reason "..." --confirm-account ACCOUNT_ID`; its account ID comes from `session status`.
