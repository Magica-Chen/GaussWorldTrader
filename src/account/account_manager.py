"""
Core account manager for Alpaca trading.

Handles account information, authentication, and basic account operations.
"""

from datetime import datetime
import logging
import os
from typing import Any, Dict, List

import requests


class AccountAPIError(RuntimeError):
    """Raised when an Alpaca account API request fails."""


class PositionNotFoundError(AccountAPIError):
    """Raised when a requested position does not exist."""


class AccountManager:
    """Main account management interface."""

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        base_url: str = None,
        paper: bool = True,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        # Use paper trading URL by default for safety
        if base_url:
            self.base_url = base_url
        elif paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.logger = logging.getLogger(__name__)

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Dict[str, Any] | None = None,
        json: Dict[str, Any] | None = None,
        timeout: int = 10,
        allow_empty: bool = False,
    ) -> Any:
        """Perform an Alpaca request and raise on any transport or API failure."""
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                params=params,
                json=json,
                timeout=timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            detail = exc.response.text.strip() if exc.response is not None else str(exc)
            self.logger.exception("%s failed with HTTP %s", action, status_code)
            raise AccountAPIError(f"{action} failed with HTTP {status_code}: {detail}") from exc
        except requests.RequestException as exc:
            self.logger.exception("%s failed", action)
            raise AccountAPIError(f"{action} failed: {exc}") from exc

        if response.status_code == 204 or not response.content.strip():
            if allow_empty:
                return None
            raise AccountAPIError(f"{action} returned an empty response")

        try:
            return response.json()
        except ValueError as exc:
            self.logger.exception("%s returned invalid JSON", action)
            raise AccountAPIError(f"{action} returned invalid JSON") from exc

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account_data = self._request_json(
            "GET",
            "/v2/account",
            action="Retrieve account information",
        )
        self.logger.info("Account information retrieved successfully")
        return account_data

    def get_account_activities(
        self,
        activity_types: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get account activities."""
        params = {
            "page_size": page_size,
        }

        if activity_types:
            params["activity_types"] = ",".join(activity_types)
        if start_date:
            params["after"] = start_date
        if end_date:
            params["until"] = end_date

        activities = self._request_json(
            "GET",
            "/v2/account/activities",
            action="Retrieve account activities",
            params=params,
        )
        self.logger.info("Retrieved %s account activities", len(activities))
        return activities

    def get_portfolio_history(
        self,
        period: str = "1D",
        timeframe: str = "1Min",
        extended_hours: bool = True,
    ) -> Dict[str, Any]:
        """Get portfolio history."""
        params = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": extended_hours,
        }

        history = self._request_json(
            "GET",
            "/v2/account/portfolio/history",
            action="Retrieve portfolio history",
            params=params,
        )
        self.logger.info("Portfolio history retrieved successfully")
        return history

    def get_trading_account_status(self) -> Dict[str, Any]:
        """Get detailed trading account status."""
        account = self.get_account()

        status = {
            "account_id": account.get("id"),
            "account_number": account.get("account_number"),
            "status": account.get("status"),
            "trading_blocked": account.get("trading_blocked"),
            "transfers_blocked": account.get("transfers_blocked"),
            "account_blocked": account.get("account_blocked"),
            "pattern_day_trader": account.get("pattern_day_trader"),
            "day_trading_buying_power": float(account.get("daytrading_buying_power", 0)),
            "cash": float(account.get("cash", 0)),
            "buying_power": float(account.get("buying_power", 0)),
            "portfolio_value": float(account.get("portfolio_value", 0)),
            "equity": float(account.get("equity", 0)),
            "last_equity": float(account.get("last_equity", 0)),
            "multiplier": account.get("multiplier"),
            "currency": account.get("currency"),
            "created_at": account.get("created_at"),
            "updated_at": account.get("updated_at"),
        }

        # Calculate derived metrics
        if status["portfolio_value"] > 0:
            status["cash_percentage"] = status["cash"] / status["portfolio_value"] * 100
        else:
            status["cash_percentage"] = 0
        status["equity_change"] = status["equity"] - status["last_equity"]
        if status["last_equity"] > 0:
            status["equity_change_percentage"] = (
                status["equity_change"] / status["last_equity"] * 100
            )
        else:
            status["equity_change_percentage"] = 0

        return status

    def get_market_calendar(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict[str, Any]]:
        """Get market calendar."""
        params = {}
        if start_date:
            params["start"] = start_date
        if end_date:
            params["end"] = end_date

        calendar = self._request_json(
            "GET",
            "/v2/calendar",
            action="Retrieve market calendar",
            params=params,
        )
        self.logger.info("Retrieved market calendar for %s days", len(calendar))
        return calendar

    def get_market_clock(self) -> Dict[str, Any]:
        """Get market clock information."""
        clock = self._request_json(
            "GET",
            "/v2/clock",
            action="Retrieve market clock",
        )
        self.logger.info("Market clock retrieved successfully")
        return clock

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.get_market_clock()
        return bool(clock.get("is_open", False))

    def get_account_summary(self) -> str:
        """Generate formatted account summary."""
        status = self.get_trading_account_status()
        clock = self.get_market_clock()
        market_status = "OPEN" if clock.get("is_open", False) else "CLOSED"

        summary = f"""
🌍 GAUSS WORLD TRADER - ACCOUNT SUMMARY
======================================
Account ID: {status.get('account_id', 'N/A')}
Status: {status.get('status', 'N/A')}
Market Status: {market_status}

FINANCIAL OVERVIEW:
------------------
• Portfolio Value: ${status.get('portfolio_value', 0):,.2f}
• Cash Available: ${status.get('cash', 0):,.2f} ({status.get('cash_percentage', 0):.1f}%)
• Buying Power: ${status.get('buying_power', 0):,.2f}
• Day Trading BP: ${status.get('day_trading_buying_power', 0):,.2f}

PERFORMANCE:
-----------
• Current Equity: ${status.get('equity', 0):,.2f}
• Previous Equity: ${status.get('last_equity', 0):,.2f}
• Daily Change: ${status.get('equity_change', 0):,.2f} ({status.get('equity_change_percentage', 0):+.2f}%)

ACCOUNT STATUS:
--------------
• Trading Blocked: {status.get('trading_blocked', False)}
• Transfers Blocked: {status.get('transfers_blocked', False)}
• Pattern Day Trader: {status.get('pattern_day_trader', False)}
• Account Multiplier: {status.get('multiplier', 'N/A')}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Using: {"Paper Trading" if "paper" in self.base_url else "Live Trading"}
"""

        return summary

    def validate_account(self) -> Dict[str, Any]:
        """Validate account credentials and status."""
        validation = {
            "credentials_valid": False,
            "account_active": False,
            "trading_enabled": False,
            "paper_trading": "paper" in self.base_url,
            "errors": [],
        }

        account = self.get_account()
        validation["credentials_valid"] = True

        # Check account status
        status = account.get("status")
        if status == "ACTIVE":
            validation["account_active"] = True
        else:
            validation["errors"].append(f"Account status: {status}")

        # Check trading permissions
        if not account.get("trading_blocked", True):
            validation["trading_enabled"] = True
        else:
            validation["errors"].append("Trading is blocked")

        # Additional checks
        if account.get("account_blocked", False):
            validation["errors"].append("Account is blocked")

        if account.get("transfers_blocked", False):
            validation["errors"].append("Transfers are blocked")

        return validation

    def get_account_configurations(self) -> Dict[str, Any]:
        """Get account configurations."""
        configurations = self._request_json(
            "GET",
            "/v2/account/configurations",
            action="Retrieve account configurations",
        )
        self.logger.info("Account configurations retrieved successfully")
        return configurations

    def update_account_configurations(self, configurations: Dict[str, Any]) -> Dict[str, Any]:
        """Update account configurations."""
        updated_config = self._request_json(
            "PATCH",
            "/v2/account/configurations",
            action="Update account configurations",
            json=configurations,
        )
        self.logger.info("Account configurations updated successfully")
        return updated_config
