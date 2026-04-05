"""
Position Management for Alpaca Trading

Handles position tracking, analysis, and management
"""

from datetime import datetime
import logging
from typing import Any, Dict, List

import requests

from .account_manager import AccountAPIError, PositionNotFoundError


def convert_crypto_symbol_for_display(symbol: str) -> str:
    """
    Convert crypto symbols to consistent display format.
    Converts BTCUSD (position format) to BTC/USD (display/API format).
    """
    if not isinstance(symbol, str):
        return symbol

    # Known crypto symbol mappings (position format -> display format)
    crypto_mappings = {
        'BTCUSD': 'BTC/USD',
        'ETHUSD': 'ETH/USD',
        'LTCUSD': 'LTC/USD',
        'BCHUSD': 'BCH/USD',
        'ADAUSD': 'ADA/USD',
        'DOTUSD': 'DOT/USD',
        'UNIUSD': 'UNI/USD',
        'LINKUSD': 'LINK/USD',
        'XLMUSD': 'XLM/USD',
        'ALGOUSD': 'ALGO/USD'
    }

    # Convert if it's a known crypto symbol, otherwise return as-is
    return crypto_mappings.get(symbol.upper(), symbol)


class PositionManager:
    """Manages trading positions"""
    
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.logger = logging.getLogger(__name__)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions"""
        positions = self.account_manager._request_json(
            "GET",
            "/v2/positions",
            action="Retrieve positions",
        )

        # Convert crypto symbols for consistent display
        for position in positions:
            if "symbol" in position:
                position["symbol"] = convert_crypto_symbol_for_display(position["symbol"])

        self.logger.info("Retrieved %s positions", len(positions))
        return positions
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for specific symbol"""
        try:
            response = requests.get(
                f"{self.account_manager.base_url}/v2/positions/{symbol}",
                headers=self.account_manager.headers,
                timeout=10,
            )
            response.raise_for_status()

            try:
                position = response.json()
            except ValueError as exc:
                raise AccountAPIError(
                    f"Failed to decode position payload for {symbol}"
                ) from exc

            # Convert crypto symbol for consistent display
            if "symbol" in position:
                position["symbol"] = convert_crypto_symbol_for_display(position["symbol"])

            self.logger.info("Retrieved position for %s", symbol)
            return position

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise PositionNotFoundError(
                    f"No position found for {convert_crypto_symbol_for_display(symbol)}"
                ) from exc
            self.logger.exception("Error retrieving position for %s", symbol)
            raise AccountAPIError(f"Failed to retrieve position for {symbol}: {exc}") from exc
        except requests.RequestException as exc:
            self.logger.exception("Error retrieving position for %s", symbol)
            raise AccountAPIError(f"Failed to retrieve position for {symbol}: {exc}") from exc
    
    def close_position(self, symbol: str, qty: str = None, percentage: str = None) -> Dict[str, Any]:
        """Close position (all or partial)"""
        params = {}
        if qty:
            params["qty"] = qty
        if percentage:
            params["percentage"] = percentage

        result = self.account_manager._request_json(
            "DELETE",
            f"/v2/positions/{symbol}",
            action=f"Close position for {symbol}",
            params=params,
        )
        self.logger.info("Position close order submitted for %s", symbol)
        return result
    
    def close_all_positions(self, cancel_orders: bool = True) -> Dict[str, Any]:
        """Close all positions"""
        params = {}
        if cancel_orders:
            params["cancel_orders"] = "true"

        results = self.account_manager._request_json(
            "DELETE",
            "/v2/positions",
            action="Close all positions",
            params=params,
        )
        self.logger.info("All positions close orders submitted")
        return {"success": True, "orders": results}
    
    def analyze_positions(self) -> Dict[str, Any]:
        """Analyze current positions"""
        positions = self.get_all_positions()
        analysis = {
            "total_positions": len(positions),
            "long_positions": 0,
            "short_positions": 0,
            "total_market_value": 0,
            "total_unrealized_pnl": 0,
            "total_unrealized_pnl_percent": 0,
            "positions_by_sector": {},
            "top_winners": [],
            "top_losers": [],
            "largest_positions": [],
            "risk_metrics": {},
        }

        if not positions:
            return analysis

        position_details = []

        for pos in positions:
            try:
                qty = float(pos.get("qty", 0))
                market_value = float(pos.get("market_value", 0))
                unrealized_pnl = float(pos.get("unrealized_pl", 0))
                cost_basis = float(pos.get("cost_basis", 0))

                # Basic counts
                if qty > 0:
                    analysis["long_positions"] += 1
                elif qty < 0:
                    analysis["short_positions"] += 1

                # Totals
                analysis["total_market_value"] += abs(market_value)
                analysis["total_unrealized_pnl"] += unrealized_pnl

                # Position details for further analysis
                pos_detail = {
                    "symbol": pos.get("symbol"),
                    "qty": qty,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": (
                        unrealized_pnl / abs(cost_basis) * 100 if cost_basis != 0 else 0
                    ),
                    "cost_basis": cost_basis,
                    "current_price": float(pos.get("current_price", 0)),
                    "avg_entry_price": float(pos.get("avg_entry_price", 0)),
                }

                position_details.append(pos_detail)

            except (ValueError, TypeError) as exc:
                symbol = pos.get("symbol", "unknown")
                raise ValueError(f"Invalid position data for {symbol}: {exc}") from exc

        # Calculate total unrealized P&L percentage
        if analysis["total_market_value"] > 0:
            total_cost_basis = analysis["total_market_value"] - analysis["total_unrealized_pnl"]
            if total_cost_basis > 0:
                analysis["total_unrealized_pnl_percent"] = (
                    analysis["total_unrealized_pnl"] / total_cost_basis
                ) * 100

        # Sort and get top/bottom performers
        position_details.sort(key=lambda x: x["unrealized_pnl"], reverse=True)
        analysis["top_winners"] = position_details[:5]
        analysis["top_losers"] = position_details[-5:]

        # Sort by position size
        position_details.sort(key=lambda x: abs(x["market_value"]), reverse=True)
        analysis["largest_positions"] = position_details[:10]

        # Risk metrics
        if position_details:
            pnl_values = [pos["unrealized_pnl_percent"] for pos in position_details]
            analysis["risk_metrics"] = {
                "max_gain_percent": max(pnl_values) if pnl_values else 0,
                "max_loss_percent": min(pnl_values) if pnl_values else 0,
                "avg_pnl_percent": sum(pnl_values) / len(pnl_values) if pnl_values else 0,
                "positions_profitable": len([p for p in pnl_values if p > 0]),
                "positions_losing": len([p for p in pnl_values if p < 0]),
            }

        return analysis
    
    def get_positions_summary(self) -> str:
        """Generate formatted positions summary"""
        analysis = self.analyze_positions()
        summary = f"""
🌍 GAUSS WORLD TRADER - POSITIONS SUMMARY
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
--------
• Total Positions: {analysis['total_positions']}
• Long Positions: {analysis['long_positions']}
• Short Positions: {analysis['short_positions']}
• Total Market Value: ${analysis['total_market_value']:,.2f}

PERFORMANCE:
-----------
• Total Unrealized P&L: ${analysis['total_unrealized_pnl']:,.2f}
• Total Unrealized P&L %: {analysis['total_unrealized_pnl_percent']:+.2f}%
"""
        
        # Risk metrics
        if analysis['risk_metrics']:
            risk = analysis['risk_metrics']
            summary += f"""
RISK METRICS:
------------
• Profitable Positions: {risk['positions_profitable']}/{analysis['total_positions']}
• Losing Positions: {risk['positions_losing']}/{analysis['total_positions']}
• Best Performer: {risk['max_gain_percent']:+.2f}%
• Worst Performer: {risk['max_loss_percent']:+.2f}%
• Average P&L: {risk['avg_pnl_percent']:+.2f}%
"""
        
        # Top winners
        if analysis['top_winners']:
            summary += """
TOP WINNERS:
-----------
"""
            for i, pos in enumerate(analysis['top_winners'][:5], 1):
                summary += f"{i}. {pos['symbol']:>6}: ${pos['unrealized_pnl']:>8,.2f} ({pos['unrealized_pnl_percent']:+.2f}%)\n"
        
        # Top losers
        if analysis['top_losers']:
            summary += """
TOP LOSERS:
----------
"""
            for i, pos in enumerate(analysis['top_losers'][:5], 1):
                summary += f"{i}. {pos['symbol']:>6}: ${pos['unrealized_pnl']:>8,.2f} ({pos['unrealized_pnl_percent']:+.2f}%)\n"
        
        # Largest positions
        if analysis['largest_positions']:
            summary += """
LARGEST POSITIONS:
-----------------
"""
            for i, pos in enumerate(analysis['largest_positions'][:5], 1):
                summary += f"{i}. {pos['symbol']:>6}: ${abs(pos['market_value']):>10,.2f} ({pos['qty']:>8.0f} shares)\n"
        
        return summary
    
    def get_position_details(self, symbol: str) -> str:
        """Get detailed information for a specific position"""
        position = self.get_position(symbol)
        qty = float(position.get('qty', 0))
        market_value = float(position.get('market_value', 0))
        unrealized_pnl = float(position.get('unrealized_pl', 0))
        cost_basis = float(position.get('cost_basis', 0))
        current_price = float(position.get('current_price', 0))
        avg_entry_price = float(position.get('avg_entry_price', 0))

        unrealized_pnl_percent = (unrealized_pnl / abs(cost_basis) * 100) if cost_basis != 0 else 0
        side = "LONG" if qty > 0 else "SHORT" if qty < 0 else "NONE"
        price_change_pct = 0 if avg_entry_price == 0 else (
            (current_price - avg_entry_price) / avg_entry_price * 100
        )

        details = f"""
🌍 POSITION DETAILS: {symbol}
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

POSITION OVERVIEW:
-----------------
• Symbol: {symbol}
• Side: {side}
• Quantity: {qty:,.0f} shares
• Average Entry Price: ${avg_entry_price:.2f}
• Current Price: ${current_price:.2f}

VALUATION:
---------
• Market Value: ${market_value:,.2f}
• Cost Basis: ${abs(cost_basis):,.2f}
• Unrealized P&L: ${unrealized_pnl:,.2f}
• Unrealized P&L %: {unrealized_pnl_percent:+.2f}%

PRICE MOVEMENT:
--------------
• Price Change: ${current_price - avg_entry_price:+.2f}
• Price Change %: {price_change_pct:+.2f}%
"""

        return details
