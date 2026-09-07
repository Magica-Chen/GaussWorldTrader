# Four-agent acceptance matrix

This matrix maps specification T01–T60 to concrete local tests. `Offline` means the
named synthetic, SDK-double or broker-double case passed. Actual broker
contracts and longitudinal results require the gates in
[FOUR_AGENT_VALIDATION.md](FOUR_AGENT_VALIDATION.md). The explicit implementation-plan
request includes the selected offline tests, synthetic fixture and isolated CI. Other
local tests remain gitignored.

Function names below are exact; file keys keep the table readable.

| Key | Local file |
| --- | --- |
| A | [test_four_agent_acceptance.py](../tests/test_four_agent_acceptance.py) |
| G | [test_four_agent_gateway.py](../tests/test_four_agent_gateway.py) |
| R | [test_runtime_operational.py](../tests/test_runtime_operational.py) |
| P | [test_four_agent_replay.py](../tests/test_four_agent_replay.py) |
| M | [test_four_agent_replay_matrix.py](../tests/test_four_agent_replay_matrix.py) |
| H | [test_four_agent_research.py](../tests/test_four_agent_research.py) |
| E | [test_four_agent_operations.py](../tests/test_four_agent_operations.py) |
| O | [test_four_agent_options.py](../tests/test_four_agent_options.py) |
| S | [test_four_agent_sdk.py](../tests/test_four_agent_sdk.py) |
| X | [test_execution_safety.py](../tests/test_execution_safety.py) |
| D | [test_gauss_existing_adapters.py](../tests/test_gauss_existing_adapters.py) |
| C | [test_session_cli.py](../tests/runtime/test_session_cli.py) |
| U | [test_session_streamlit.py](../tests/ui/test_session_streamlit.py) |
| B | [test_four_agent_bot.py](../tests/test_four_agent_bot.py) |
| F | [test_four_agent_alerts.py](../tests/test_four_agent_alerts.py) |

| ID | Concrete test evidence | Result and remaining scope |
| --- | --- | --- |
| T01 | C `test_help_has_session_commands_without_legacy_imports`; selected momentum, execution and adapter regressions | Offline representative legacy workflows |
| T02 | Wheel/sdist build, isolated installation, 96 network-blocked nested-module imports and installed CLI checks recorded in the validation report | Packaging checks passed; clean deployment dependencies remain an installation requirement |
| T03 | G `test_t03_shadow_gateway_never_calls_broker`; M `test_t60_two_session_account_profile_matrix` | Offline zero broker writes |
| T04 | A `test_t04_invalid_balances_reject_without_fallback`; X `test_option_premium_uses_verified_contract_multiplier`, `test_nonfinite_or_negative_override_cannot_enter` | Offline invalid/zero funds, units and overrides |
| T05 | A `test_t05_early_close_and_holiday_calendar_uses_recorded_sessions`; R `test_calendar_preserves_exchange_open_across_uk_us_dst_mismatch` | Offline explicit sessions and four DST dates |
| T06 | A `test_t06_exhausted_calendar_blocks_entry`; R `test_missing_calendar_blocks_entries_but_keeps_current_holding_reconciliation`; G `test_t13_restart_rejects_reserved_work_that_never_reached_submitting` | Offline calendar blocking, current holdings and missed-entry recovery |
| T07 | D `test_existing_stock_provider_uses_delayed_sip_without_second_history_delay` | Offline request feed/cutoff; actual endpoint permission needs an online read-only probe |
| T08 | R `test_subscribed_entitlement_loss_retains_holdings_without_profile_fallback`, `test_request_budget_preserves_news_capacity_without_waiting` | Offline independent SIP/OPRA outcomes; deployment news endpoint needs its own probe |
| T09 | P expired-trigger test; G `test_t09_wrong_session_plan_rejected_while_existing_group_remains_open`, `test_t09_superseded_plan_revokes_pending_authority_and_preserves_holdings`; R profile-switch test | Offline expiry, past/future target sessions, superseded pending authority and incompatible profiles; existing exposure remains open |
| T10 | A `test_t10_stale_plan_writer_fails_without_appending_event` | Offline optimistic concurrency and audit atomicity |
| T11 | H `test_t11_experimental_output_is_queued_without_strategy_approval` | Offline generated strategy remains experimental |
| T12 | G `test_t12_news_between_risk_and_submission_revokes_approval`, `test_t12_late_news_after_submission_cancels_and_reconciles_fill_through_service` | Offline service pipeline covers veto, cancellation and late fill |
| T13 | G `test_t13_timeout_reconciles_same_client_without_duplicate_submission`, `test_t13_unknown_replacement_retains_single_reservation_and_stable_child`, `test_t13_restart_rejects_reserved_work_that_never_reached_submitting` | Offline unknown acceptance/replacement and pre-submit crash recovery |
| T14 | G `test_t14_concurrent_candidates_cannot_claim_same_position_slot` | Offline competing atomic reservations |
| T15 | G `test_t15_quantity_override_cannot_exceed_safe_size`, `test_t15_forged_risk_decision_and_modified_intent_never_reach_broker`; X `test_explicit_sell_to_close_never_opens_short`, `test_final_rounded_price_used_for_affordability` | Offline quantity, authority, close and rounded-price bounds |
| T16 | G `test_t16_partial_fill_remains_exposure_after_cancel`, `test_t16_replacement_child_fill_counts_once_and_releases_only_pending_cash`, late-news service test above | Offline partial fills and cancellation races |
| T17 | G `test_t17_late_old_fill_with_rejected_replacement_retains_managed_exposure`, `test_t16_replacement_child_fill_counts_once_and_releases_only_pending_cash` | Offline out-of-order terminal family state and fill deduplication |
| T18 | G `test_t18_broker_rejected_exit_preserves_open_group_and_confirmed_quantity`; A unknown-order flatness test; B residual-exposure management test | Offline rejected close retains actual open quantity, requires reassessment before retry and preserves management |
| T19 | O `test_indicative_or_skewed_quotes_never_pass_execution`; R `test_gap_recovery_requires_every_missing_minute_and_a_fresh_genuine_quote` | Offline quote quality and complete recovery evidence |
| T20 | R `test_verified_event_calendar_versions_are_stable_and_expire`; R `test_news_disconnect_blocks_dependent_plan_then_recovers_without_losing_holdings` | Offline calendar expiry and news disconnect/recovery pass; dependent plans defer while current holdings remain reconciled |
| T21 | D `test_news_corrections_and_cross_provider_provenance_survive_merge` | Offline logical story deduplication and retained versions |
| T22 | A `test_t22_snapshot_preserves_original_version_after_correction`; H `test_t22_worker_cannot_cite_future_or_outside_snapshot` | Offline bar/news revision isolation; absent point-in-time fundamentals remain unavailable |
| T23 | H `test_t23_subprocess_worker_has_no_broker_environment_and_completes`, `test_t23_timeout_keeps_potential_paid_charge_reserved`, disabled-paid and unknown-pricing tests; R `test_paid_research_worker_does_not_block_broker_cycle` | Offline isolated environment, paid budget and independent operations |
| T24 | P `test_two_session_profile_account_matrix` below-minimum/cash-poor cases; A `test_t53_insufficient_evidence_returns_no_trade` | Offline complete no-trade output |
| T25 | G `test_t25_entry_pause_limit_rejects_new_group_but_allows_confirmed_close`; X `test_confirmed_exit_survives_invalid_account` | Offline daily-loss threshold and entry quota block fresh exposure while current confirmed closes remain authorised |
| T26 | G `test_t26_closed_group_does_not_adopt_later_external_same_symbol_position`; R `test_unknown_newer_schema_is_not_mutated`; E `test_restore_refuses_corruption_and_existing_destination` | Offline restart, unmanaged exposure and incompatible/corrupt state |
| T27 | R `test_request_budget_preserves_news_capacity_without_waiting`, `test_large_universe_rotates_batched_history_and_persists_quotes_before_quota_failure`, `test_held_stock_and_option_quotes_survive_discovery_and_history_quota_exhaustion`, `test_confirmed_holding_instrument_lookup_bypasses_only_discovery_quota`; F queue-overflow and subscription-exhaustion tests | Offline bounded request capacity, batched history, holding quote priority, overflow visibility and retained holdings |
| T28 | A `test_t28_opposite_option_legs_are_open_even_with_zero_net_contracts`; O `test_zero_net_contracts_liquidation_is_signed_premium_value`; `test_live_session_safety.py` underlying-isolation tests | Offline per-leg exposure and underlying isolation |
| T29 | S `test_adjusted_root_is_not_approved_from_multiplier_alone`; O contract/quote tests; G `test_t29_assignment_or_exercise_reconciles_residual_stock` | Offline exercise/assignment preserves and quarantines residual long/short stock; actual broker lifecycle remains gated |
| T30 | A `test_t30_outbox_consumer_effect_and_offset_rollback_together`; R `test_immutable_record_and_transactional_outbox_rollback`; G `test_t30_reconciliation_lookup_happens_outside_account_transaction` | Offline state/outbox atomicity and broker I/O outside transaction |
| T31 | R `test_host_ownership_independent_database_and_legacy_lookup`, `test_runtime_verifies_command_signature_not_client_claim`; C `test_command_requires_token`; S `test_legacy_position_api_cannot_bypass_another_database_runtime` | Offline cross-database ownership and authenticated control |
| T32 | E `test_equity_pnl_removes_external_deposit_and_does_not_charge_fees_twice`, `test_drawdown_removes_deposit_before_measuring_trading_loss` | Offline cashflow-adjusted evaluation; runtime daily-loss activity path also needs forward reconciliation evidence |
| T33 | B `test_cycle_failure_does_not_abandon_exposure`, `test_residual_exposure_keeps_management_running`, `test_explicit_acknowledgement_reaches_shutdown_gate` | Offline continued management and explicit acknowledgement |
| T34 | H `test_t34_unknown_model_order_field_fails_schema`, `test_t22_worker_cannot_cite_future_or_outside_snapshot` | Offline strict output schema and source allowlist |
| T35 | E backup/WAL/hash/schema/path tests; U `test_persisted_replay_client_renders_without_runtime_start`; F `test_console_alert_failure_preserves_durable_client_visibility`, `test_incident_is_committed_before_console_delivery` | Offline restore, UI reconnect and persisted alerts despite console-delivery failure; no outbound notification transport exercised |
| T36 | A `test_t36_buffered_delayed_cutoff` | Offline cutoff and entitlement buffer |
| T37 | A `test_t37_completed_bar_end_controls_release`; R `test_timeframe_resampling_requires_every_completed_minute`, `test_daily_resampling_uses_actual_early_close` | Offline complete intervals required |
| T38 | M `test_t38_delayed_fixture_never_uses_historical_signal_as_submission_time` | Offline current decision time; no historical-quote fills asserted |
| T39 | G late-news service and assignment/exercise tests, executed under `FREE_DELAYED` on the next operational cycle | Offline operational updates receive no signal-delay hold |
| T40 | A `test_t40_late_receipt_does_not_introduce_second_delay`; D delayed-history test | Offline receipt does not restart intentional delay |
| T41 | R `test_subscribed_entitlement_loss_retains_holdings_without_profile_fallback` | Offline entitlement loss blocks entries and preserves profile/holdings |
| T42 | A `test_t42_indicative_quote_never_becomes_genuine_execution_quote`; O indicative/skewed quote test | Offline indicative quality remains nonexecutable |
| T43 | P expired-trigger test; O `test_indicative_or_skewed_quotes_never_pass_execution` | Offline evidence readiness and execution deadline/quote gates |
| T44 | R `test_operator_profile_switch_revalidates_plans_and_retains_real_holdings` | Offline authenticated profile change preserves actual exposure |
| T45 | A `test_t45_current_safety_news_is_excluded_from_delayed_signal`; G late-news service test | Offline aligned research and current safety remain separate |
| T46 | P `test_two_session_profile_account_matrix`; M `test_t60_two_session_account_profile_matrix` | Offline explicit capital/cash assumptions and scoped outputs |
| T47 | A invalid-balance tests; S `test_invalid_funds_do_not_fabricate_account_but_preserve_verified_identity`; G `test_t04_invalid_financial_data_preserves_operational_holdings_and_close_authority` | Offline unknown financial facts block entries while holdings persist |
| T48 | A `test_t48_equal_equity_different_cash_changes_feasibility`; P cash-poor equal-equity cases | Offline equity alone does not establish feasibility |
| T49 | O `test_spread_permission_and_minimum_size_and_capacity` | Offline boundary values, whole contracts and permissions |
| T50 | G `test_t50_pre_gauss_reassesses_revoked_options_permission_from_frozen_plan`, competing-reservation tests; H current-budget-account test | Offline PreGauss rechecks options permission 3→2, preserves frozen plan/report, defers eligibility and final RiskGate rejects stale sizing |
| T51 | A `test_t51_larger_capital_does_not_exceed_liquidity_capacity`; P capacity-limited cases | Offline recomputed capacity-constrained quantities |
| T52 | G `test_t14_concurrent_candidates_cannot_claim_same_position_slot` | Offline current reservations dominate competing proposals |
| T53 | A `test_t53_insufficient_evidence_returns_no_trade`; P no-feasible-size branches | Offline no trade with binding reasons |
| T54 | A `test_supported_ranking_compares_expected_value_per_unit_risk`, `test_t53_insufficient_evidence_returns_no_trade` | Offline comparable supported ranking and rejected unsupported expectancy |
| T55 | A `test_t55_comparison_rejects_actual_scope_and_preserves_input_balance`; G `test_t55_foreign_account_plan_cannot_reserve_or_reach_gateway`, forged/mutated-authority test | Offline actual/scenario and foreign-plan identity checks block reservations and broker access |
| T56 | O permission/minimum-size test; S `test_actual_permissions_option_buying_power_and_out_of_scope_exposure`; B `test_paid_profile_does_not_change_execution` | Offline capital/subscription labels grant no permissions |
| T57 | H `test_t57_paid_budget_reserves_then_settles_actual_usage`; E `test_repeated_shared_cost_id_is_rejected`, `test_actual_fill_costs_are_deducted_once` | Offline reservation settlement and unique cost allocation |
| T58 | H `test_t58_cache_keys_include_actual_account_constraints`; R profile-switch test | Offline account-specific cache and plan revalidation |
| T59 | `tests/ui/test_session_scenario_browser.py::test_t59_client_scenario_view_reads_preserve_actual_account_and_reservations`; `test_t59_interactive_scenario_switch_preserves_actual_account_and_reservations` | CI client view-state checks plus offline Chromium Actual → Hypothetical → Actual tab switches; explicit scenario/profile/as-of labels; actual account and reservation rows unchanged before comparison and after every switch; browser proof explicitly skips without local Playwright/Chromium |
| T60 | P `test_two_session_profile_account_matrix`; M `test_t60_two_session_account_profile_matrix` | Offline all roles, causal lineage, both profiles and capital/cash/capacity matrix; zero broker writes |

The acceptance boundary remains explicit: an offline passing case does not establish
deployment endpoint permission, actual option order mechanics, measured operational
uptime or profitable strategy selection. Each deployment records those results for its
account, data profile, strategy and policy versions.
